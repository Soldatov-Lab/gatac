"""
GPU-accelerated motif enrichment analysis for ATAC-seq data.

This module provides GPU-accelerated motif scanning using CuPy for vectorized
PWM scoring across sequences. It implements functionality similar to SnapATAC2's
motif_enrichment but with CUDA acceleration.
"""

from __future__ import annotations

import logging
import tempfile
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Literal, Optional, Union

import cupy as cp
import numpy as np
import pandas as pd
import polars as pl
from numba import njit

from ._bias_matching import sample_bias_matched_indices

logger = logging.getLogger(__name__)

# CuPy memory pools for efficient memory management
mempool = cp.get_default_memory_pool()
pinned_mempool = cp.get_default_pinned_memory_pool()


# =============================================================================
# FASTA Handling with gzip support
# =============================================================================


@contextmanager
def _open_fasta(fasta_path: Union[str, Path]):
    """
    Open a FASTA file, handling gzip compression if needed.
    
    pyfaidx doesn't support regular gzip, only BGZF. This helper
    decompresses gzip files using rapidgzip to a temporary file.
    
    Parameters
    ----------
    fasta_path : str or Path
        Path to FASTA file (supports .fa, .fasta, .fa.gz, .fasta.gz)
        
    Yields
    ------
    pyfaidx.Fasta
        Opened FASTA object
    """
    from pyfaidx import Fasta
    
    fasta_path = Path(fasta_path)
    
    if str(fasta_path).endswith('.gz'):
        # Decompress gzip file to temporary location
        import shutil
        try:
            import rapidgzip
            _use_rapidgzip = True
        except ImportError:
            import gzip
            _use_rapidgzip = False
        
        logger.info(f"Decompressing {fasta_path.name}...")
        
        # Create temp file with same base name for pyfaidx indexing
        temp_dir = tempfile.mkdtemp(prefix="gatac_fasta_")
        temp_fasta = Path(temp_dir) / fasta_path.name.replace('.gz', '')
        
        try:
            if _use_rapidgzip:
                # Parallel decompression with rapidgzip
                with rapidgzip.open(str(fasta_path)) as f_in:
                    with open(temp_fasta, 'wb') as f_out:
                        while True:
                            chunk = f_in.read(64 * 1024 * 1024)
                            if not chunk:
                                break
                            f_out.write(chunk)
            else:
                with gzip.open(str(fasta_path), 'rb') as f_in:
                    with open(temp_fasta, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
            
            logger.info(f"Decompressed to {temp_fasta}")
            
            # Open with pyfaidx (will create .fai index)
            genome = Fasta(str(temp_fasta), one_based_attributes=False)
            yield genome
            
        finally:
            # Cleanup temp files
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp dir {temp_dir}: {e}")
    else:
        # Uncompressed FASTA - open directly
        genome = Fasta(str(fasta_path), one_based_attributes=False)
        yield genome


def _ordered_unique_regions(region_groups: list[list[str]]) -> list[str]:
    """Collect regions once while preserving input order."""
    ordered_regions = []
    seen = set()

    for group in region_groups:
        for region in group:
            if region not in seen:
                seen.add(region)
                ordered_regions.append(region)

    return ordered_regions


def _fetch_region_sequences(
    regions: list[str],
    genome_fasta: Union[str, Path],
) -> list[str]:
    """Fetch region sequences from a FASTA file."""
    logger.info(f"Fetching {len(regions)} sequences...")

    sequences = []
    with _open_fasta(genome_fasta) as genome:
        for region in regions:
            try:
                chrom, coords = region.split(":")
                start, end = coords.split("-")
                start, end = int(start), int(end)
                seq = str(genome[chrom][start:end].seq)
                sequences.append(seq)
            except Exception as e:
                logger.warning(f"Failed to fetch sequence for {region}: {e}")
                sequences.append("")

    return sequences


def _compute_gc_content_from_sequences(sequences: list[str]) -> np.ndarray:
    """Compute per-sequence GC fraction on CPU."""
    gc_content = np.zeros(len(sequences), dtype=np.float64)

    for i, seq in enumerate(sequences):
        seq = seq.upper()
        seq_len = len(seq)
        if seq_len == 0:
            continue

        gc_count = seq.count("G") + seq.count("C")
        gc_content[i] = gc_count / seq_len

    return gc_content


def _compute_region_gc_content(
    regions: list[str],
    genome_fasta: Union[str, Path],
) -> np.ndarray:
    """Compute GC fraction for a list of genomic regions."""
    sequences = _fetch_region_sequences(regions, genome_fasta)
    return _compute_gc_content_from_sequences(sequences)


def _compute_bg_probs_from_sequences(
    sequences: list[str],
) -> tuple[float, float, float, float]:
    """Estimate background nucleotide probabilities from sequences."""
    counts = np.zeros(4, dtype=np.int64)

    for seq in sequences:
        seq_upper = seq.upper()
        counts[0] += seq_upper.count("A")
        counts[1] += seq_upper.count("C")
        counts[2] += seq_upper.count("G")
        counts[3] += seq_upper.count("T")

    total = counts.sum()
    if total == 0:
        return (0.25, 0.25, 0.25, 0.25)

    probs = counts / total
    return tuple(float(x) for x in probs)


def _resolve_bg_probs(
    bg_probs: Union[str, tuple[float, float, float, float]],
    sequences: list[str],
) -> tuple[float, float, float, float]:
    """Resolve string background modes to explicit nucleotide probabilities."""
    if isinstance(bg_probs, str):
        if bg_probs in {"auto", "subject"}:
            resolved = _compute_bg_probs_from_sequences(sequences)
            logger.info(
                "Background from sequences: "
                f"A={resolved[0]:.4f} C={resolved[1]:.4f} "
                f"G={resolved[2]:.4f} T={resolved[3]:.4f}"
            )
            return resolved
        if bg_probs == "even":
            return (0.25, 0.25, 0.25, 0.25)
        raise ValueError(
            f"Unknown bg mode: {bg_probs}. "
            "Use 'auto', 'subject', 'even', or a tuple of 4 floats."
        )

    if len(bg_probs) != 4:
        raise ValueError(
            "Background nucleotide probabilities must contain exactly 4 values "
            "for (A, C, G, T)."
        )

    probs = tuple(float(x) for x in bg_probs)
    total = sum(probs)
    if total <= 0:
        raise ValueError("Background nucleotide probabilities must sum to a positive value.")

    return tuple(x / total for x in probs)



# =============================================================================
# DNA Motif Classes
# =============================================================================


class DNAMotif:
    """
    DNA motif represented as a position weight matrix (PWM).
    
    Attributes
    ----------
    id : str
        Unique identifier for the motif
    name : str, optional
        Human-readable name
    family : str, optional
        Transcription factor family
    pwm : np.ndarray
        Position weight matrix of shape (length, 4) with columns [A, C, G, T]
    """
    
    def __init__(
        self,
        id: str,
        pwm: np.ndarray,
        name: Optional[str] = None,
        family: Optional[str] = None,
        pfm: Optional[np.ndarray] = None,
    ):
        """
        Initialize a DNAMotif.
        
        Parameters
        ----------
        id : str
            Unique identifier
        pwm : np.ndarray
            Position weight matrix, shape (length, 4)
        name : str, optional
            Human-readable name
        family : str, optional
            Transcription factor family
        pfm : np.ndarray, optional
            Raw position frequency (count) matrix, shape (length, 4).
            When provided, ``to_log_odds(mode="motifmatchr")`` applies a
            MOODS-compatible pseudocount to the counts before computing
            log-odds, matching R's motifmatchr scoring exactly.
        """
        self.id = id
        self.name = name
        self.family = family
        
        # Ensure PWM is 2D with shape (length, 4)
        pwm = np.asarray(pwm, dtype=np.float64)
        if pwm.ndim != 2 or pwm.shape[1] != 4:
            raise ValueError(f"PWM must have shape (length, 4), got {pwm.shape}")
        
        # Add pseudocount to avoid log(0)
        pwm = np.clip(pwm, 1e-10, 1.0)
        # Normalize rows to sum to 1
        pwm = pwm / pwm.sum(axis=1, keepdims=True)
        self.pwm = pwm
        
        # Store raw count matrix for MOODS-compatible pseudocount handling
        if pfm is not None:
            pfm = np.asarray(pfm, dtype=np.float64)
            if pfm.shape != self.pwm.shape:
                raise ValueError(
                    f"PFM shape {pfm.shape} must match PWM shape {self.pwm.shape}"
                )
        self.pfm = pfm
    
    def __repr__(self) -> str:
        name_str = f", name={self.name}" if self.name else ""
        return f"DNAMotif(id={self.id}{name_str}, length={len(self)})"
    
    def __len__(self) -> int:
        return self.pwm.shape[0]
    
    def info_content(self) -> float:
        """Compute total information content of the motif in bits."""
        ic = 0.0
        for row in self.pwm:
            entropy = -np.sum(row * np.log2(row + 1e-10))
            ic += 2.0 - entropy
        return ic
    
    def to_log_odds(
        self, 
        bg_probs: tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25),
        mode: Literal["gatac", "motifmatchr"] = "gatac",
        pseudocount: float = 0.8,
    ) -> np.ndarray:
        """
        Convert PWM to log-odds scores.
        
        Parameters
        ----------
        bg_probs : tuple
            Background nucleotide probabilities (A, C, G, T)
        mode : {"gatac", "motifmatchr"}, default "gatac"
            - "gatac": Natural log-odds with minimal pseudocount
            - "motifmatchr": Log2-odds matching MOODS/motifmatchr scoring.
              When raw counts (``pfm``) are available, applies a
              MOODS-compatible pseudocount:
              ``prob = (count + pseudocount * bg) / (row_sum + pseudocount)``
        pseudocount : float, default 0.8
            Pseudocount multiplier used in motifmatchr mode when raw counts
            (``self.pfm``) are available.  Ignored for probability-only motifs
            (e.g. from MEME files where the pseudocount is already baked in).
            
        Returns
        -------
        np.ndarray
            Log-odds matrix of shape (length, 4)
        """
        bg = np.array(bg_probs, dtype=np.float64)
        
        if mode == "motifmatchr":
            if self.pfm is not None:
                # MOODS-compatible log-odds from raw counts with pseudocount.
                # Formula: prob = (count + ps*bg) / (row_sum + ps)
                # score  = log2(prob / even) + log2(bg / even)
                smoothed = self.pfm + pseudocount * bg
                prob = smoothed / smoothed.sum(axis=1, keepdims=True)
            else:
                # PPM input (e.g. MEME): pseudocount already baked in
                prob = self.pwm
            
            even = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float64)
            log_odds = np.log2(prob / even)
            adj = np.log2(bg) - np.log2(even)
            return log_odds + adj
        else:
            # Original GATAC mode: simple natural log-odds
            return np.log(self.pwm / bg)
    
    def reverse_complement(self) -> "DNAMotif":
        """
        Return reverse complement of this motif.
        
        Returns
        -------
        DNAMotif
            New motif with reversed and complemented PWM
        """
        # Reverse rows and swap A<->T, C<->G columns
        rc_pwm = self.pwm[::-1, ::-1].copy()
        rc_pfm = self.pfm[::-1, ::-1].copy() if self.pfm is not None else None
        return DNAMotif(
            id=f"{self.id}_rc",
            pwm=rc_pwm,
            name=f"{self.name}_rc" if self.name else None,
            family=self.family,
            pfm=rc_pfm,
        )


# =============================================================================
# MEME File Parser
# =============================================================================


def read_motifs(filename: Union[str, Path], unique: bool = True) -> list[DNAMotif]:
    """
    Read motifs from a MEME format file.
    
    Parameters
    ----------
    filename : str or Path
        Path to MEME format file
    unique : bool, default True
        A transcription factor may have multiple motifs. If True, 
        only the motifs with the highest information content will be selected.
        This matches snapatac2's cis_bp(unique=True) behavior.
        
    Returns
    -------
    list[DNAMotif]
        List of parsed motifs
    """
    path = Path(filename)
    with open(path, 'r') as f:
        content = f.read()
    
    motifs = parse_meme(content)
    
    # Extract TF name from motif ID (format: TF_NAME+MOTIF_ID+D)
    for motif in motifs:
        motif.name = motif.id.split('+')[0]
    
    if unique:
        # Keep only motif with highest information content per TF name
        unique_motifs = {}
        for motif in motifs:
            name = motif.name
            if name not in unique_motifs or unique_motifs[name].info_content() < motif.info_content():
                unique_motifs[name] = motif
        motifs = list(unique_motifs.values())
    
    return motifs


def parse_meme(content: str) -> list[DNAMotif]:
    """
    Parse MEME format content into motifs.
    
    Parameters
    ----------
    content : str
        MEME format file content
        
    Returns
    -------
    list[DNAMotif]
        List of parsed motifs
    """
    motifs = []
    
    # Split by MOTIF keyword
    parts = content.split("MOTIF")
    
    for part in parts[1:]:  # Skip header before first MOTIF
        lines = part.strip().split("\n")
        if not lines:
            continue
        
        # First line contains motif ID (and optionally name)
        id_line = lines[0].strip()
        id_parts = id_line.split()
        motif_id = id_parts[0] if id_parts else "unknown"
        motif_name = id_parts[1] if len(id_parts) > 1 else None
        
        # Find the letter-probability matrix line
        pwm_start = None
        motif_length = None
        for i, line in enumerate(lines):
            if "letter-probability matrix" in line.lower():
                # Parse w= parameter for motif length
                # Handle formats like "w=10" or "w= 10"
                import re
                w_match = re.search(r'w\s*=\s*(\d+)', line)
                if w_match:
                    motif_length = int(w_match.group(1))
                pwm_start = i + 1
                break
        
        if pwm_start is None:
            continue
        
        # Parse PWM rows
        pwm_rows = []
        for i in range(pwm_start, len(lines)):
            line = lines[i].strip()
            if not line or line.startswith("URL") or line.startswith("MOTIF"):
                break
            
            values = line.split()
            if len(values) >= 4:
                try:
                    row = [float(v) for v in values[:4]]
                    pwm_rows.append(row)
                except ValueError:
                    break
            
            if motif_length and len(pwm_rows) >= motif_length:
                break
        
        if pwm_rows:
            pwm = np.array(pwm_rows, dtype=np.float64)
            motifs.append(DNAMotif(id=motif_id, pwm=pwm, name=motif_name))
    
    return motifs


# =============================================================================
# Sequence Encoding
# =============================================================================


# Nucleotide to integer mapping: A=0, C=1, G=2, T=3, other=-1
_NUC_TO_INT = np.full(256, -1, dtype=np.int8)
_NUC_TO_INT[ord('A')] = 0
_NUC_TO_INT[ord('a')] = 0
_NUC_TO_INT[ord('C')] = 1
_NUC_TO_INT[ord('c')] = 1
_NUC_TO_INT[ord('G')] = 2
_NUC_TO_INT[ord('g')] = 2
_NUC_TO_INT[ord('T')] = 3
_NUC_TO_INT[ord('t')] = 3


def _encode_sequence(seq: str) -> np.ndarray:
    """Encode a DNA sequence to integer array."""
    seq_bytes = np.frombuffer(seq.encode('ascii'), dtype=np.uint8)
    return _NUC_TO_INT[seq_bytes]


def _encode_sequences_batch(sequences: list[str]) -> tuple[cp.ndarray, cp.ndarray]:
    """
    Batch encode DNA sequences for GPU processing.
    
    Parameters
    ----------
    sequences : list[str]
        List of DNA sequences
        
    Returns
    -------
    tuple[cp.ndarray, cp.ndarray]
        - encoded: CuPy array of shape (n_seqs, max_len) with nucleotide codes
        - lengths: CuPy array of shape (n_seqs,) with sequence lengths
    """
    if not sequences:
        return cp.zeros((0, 0), dtype=cp.int8), cp.zeros(0, dtype=cp.int32)
    
    lengths = np.array([len(s) for s in sequences], dtype=np.int32)
    max_len = lengths.max()
    n_seqs = len(sequences)
    
    # Create padded array on CPU
    encoded = np.full((n_seqs, max_len), -1, dtype=np.int8)
    for i, seq in enumerate(sequences):
        encoded[i, :len(seq)] = _encode_sequence(seq)
    
    return cp.asarray(encoded), cp.asarray(lengths)


def _reverse_complement_encoded(encoded: cp.ndarray) -> cp.ndarray:
    """
    Compute reverse complement of encoded sequences.
    
    Parameters
    ----------
    encoded : cp.ndarray
        Encoded sequences with A=0, C=1, G=2, T=3, N=-1
        
    Returns
    -------
    cp.ndarray
        Reverse complemented sequences
    """
    # Complement: A<->T (0<->3), C<->G (1<->2)
    # For valid bases: complement = 3 - x
    # For N (-1): keep as -1
    rc = cp.where(encoded >= 0, 3 - encoded, encoded)
    # Reverse along sequence axis
    return rc[:, ::-1]


# =============================================================================
# GPU-Accelerated PWM Scanning
# =============================================================================


@njit(cache=True)
def _compute_score_threshold_jit(
    pwm_log_odds: np.ndarray,
    bg: np.ndarray,
    pvalue: float,
    precision: float,
) -> float:
    """
    Numba JIT-compiled score threshold computation.
    
    This is the performance-critical inner function that uses dynamic programming
    to compute the score distribution and find the threshold for a given p-value.
    """
    motif_len = pwm_log_odds.shape[0]
    
    # Compute score range
    min_scores = np.empty(motif_len, dtype=np.float64)
    max_scores = np.empty(motif_len, dtype=np.float64)
    
    for i in range(motif_len):
        min_val = pwm_log_odds[i, 0]
        max_val = pwm_log_odds[i, 0]
        for j in range(1, 4):
            if pwm_log_odds[i, j] < min_val:
                min_val = pwm_log_odds[i, j]
            if pwm_log_odds[i, j] > max_val:
                max_val = pwm_log_odds[i, j]
        min_scores[i] = min_val
        max_scores[i] = max_val
    
    total_min = 0.0
    total_max = 0.0
    for i in range(motif_len):
        total_min += min_scores[i]
        total_max += max_scores[i]
    
    if total_min >= total_max:
        return 0.0
    
    # Create score bins
    num_bins_float = (total_max - total_min) / precision
    num_bins = int(num_bins_float + 0.999999)  # ceil
    if num_bins > 100000:
        num_bins = 100000
    step = (total_max - total_min) / num_bins
    
    # Initialize probability distribution
    accum = np.zeros(num_bins + 1, dtype=np.float64)
    accum[0] = 1.0
    new_accum = np.zeros(num_bins + 1, dtype=np.float64)
    
    # Process each position
    for pos in range(motif_len):
        # Reset new_accum
        for i in range(num_bins + 1):
            new_accum[i] = 0.0
        
        pos_min = min_scores[pos]
        
        # For each nucleotide
        for j in range(4):
            score_diff = pwm_log_odds[pos, j] - pos_min
            shift = int(score_diff / step)
            if shift < 0:
                shift = 0
            elif shift > num_bins:
                shift = num_bins
            
            weight = bg[j]
            
            if shift == 0:
                for i in range(num_bins + 1):
                    new_accum[i] += accum[i] * weight
            else:
                # Shift probabilities
                for i in range(shift, num_bins + 1):
                    new_accum[i] += accum[i - shift] * weight
        
        # Swap accumulators
        accum, new_accum = new_accum, accum
    
    # Compute min_delta: smallest gap between max and second-max log-odds
    # at any position (used by MOODS when P(max_score) > pvalue)
    min_delta = np.inf
    for pos in range(motif_len):
        max1 = -np.inf
        max2 = -np.inf
        for j in range(4):
            v = pwm_log_odds[pos, j]
            if v > max1:
                max2 = max1
                max1 = v
            elif v > max2:
                max2 = v
        delta = max1 - max2
        if delta < min_delta:
            min_delta = delta

    # Find rightmost non-zero bin (max score may not land on last bin
    # due to floating-point shift truncation in the DP)
    max_nonzero_bin = num_bins
    for i in range(num_bins, -1, -1):
        if accum[i] > 0.0:
            max_nonzero_bin = i
            break

    # Scan from right tail (MOODS convention): accumulate P(score >= s)
    right_sum = accum[max_nonzero_bin]
    if right_sum > pvalue:
        # P(max_score) alone exceeds pvalue — use MOODS heuristic:
        # place threshold halfway between max and second-highest unique score
        return total_max - min_delta / 2.0

    for r in range(max_nonzero_bin - 1, -1, -1):
        right_sum += accum[r]
        if right_sum > pvalue:
            # Threshold at next bin boundary above r
            return total_min + (r + 1) * step

    return total_min


def _compute_score_threshold(
    pwm_log_odds: np.ndarray,
    bg_probs: tuple[float, float, float, float],
    pvalue: float = 1e-5,
    precision: float = 1e-4,
) -> float:
    """
    Compute score threshold for given p-value using optimized JIT DP.
    
    Parameters
    ----------
    pwm_log_odds : np.ndarray
        Log-odds PWM matrix (length, 4)
    bg_probs : tuple
        Background nucleotide probabilities
    pvalue : float
        Desired p-value threshold
    precision : float
        Precision for score binning (larger = faster but less accurate)
        
    Returns
    -------
    float
        Score threshold
    """
    bg = np.array(bg_probs, dtype=np.float64)
    return _compute_score_threshold_jit(pwm_log_odds, bg, pvalue, precision)


def _scan_motif_gpu(
    encoded_seqs: cp.ndarray,
    seq_lengths: cp.ndarray,
    pwm_log_odds: cp.ndarray,
    threshold: float,
    check_rc: bool = True,
) -> cp.ndarray:
    """
    Scan sequences for motif matches using GPU.
    
    Parameters
    ----------
    encoded_seqs : cp.ndarray
        Encoded sequences, shape (n_seqs, max_len)
    seq_lengths : cp.ndarray
        Actual lengths of each sequence
    pwm_log_odds : cp.ndarray
        Log-odds PWM, shape (motif_len, 4)
    threshold : float
        Score threshold for match
    check_rc : bool
        Whether to also check reverse complement
        
    Returns
    -------
    cp.ndarray
        Boolean array of shape (n_seqs,) indicating motif presence
    """
    n_seqs, max_len = encoded_seqs.shape
    motif_len = pwm_log_odds.shape[0]
    
    if max_len < motif_len:
        return cp.zeros(n_seqs, dtype=cp.bool_)
    
    n_positions = max_len - motif_len + 1
    
    # Create sliding window view for efficient position processing
    # We'll compute scores for all positions at once
    
    # Initialize scores array
    scores = cp.zeros((n_seqs, n_positions), dtype=cp.float32)
    valid_mask = cp.ones((n_seqs, n_positions), dtype=cp.bool_)
    
    # Process each position in the motif
    for pos_in_motif in range(motif_len):
        # Get nucleotide at this motif position for all sequences and start positions
        # Shape: (n_seqs, n_positions)
        nucs = encoded_seqs[:, pos_in_motif:pos_in_motif + n_positions]
        
        # Mark invalid positions (containing N)
        valid_mask &= (nucs >= 0)
        
        # Safe indexing for PWM lookup (replace -1 with 0 temporarily)
        nucs_safe = cp.clip(nucs, 0, 3)
        
        # Look up scores from PWM
        pos_scores = pwm_log_odds[pos_in_motif, nucs_safe]
        scores += pos_scores
    
    # Apply validity mask and check threshold
    scores = cp.where(valid_mask, scores, -cp.inf)
    
    # Check if sequence length is sufficient for each position
    position_indices = cp.arange(n_positions, dtype=cp.int32)[None, :]  # (1, n_positions)
    seq_len_mask = (position_indices + motif_len) <= seq_lengths[:, None]
    scores = cp.where(seq_len_mask, scores, -cp.inf)
    
    # Check forward strand
    forward_match = (scores >= threshold).any(axis=1)
    
    if not check_rc:
        return forward_match
    
    # Check reverse complement
    rc_seqs = _reverse_complement_encoded(encoded_seqs)
    
    rc_scores = cp.zeros((n_seqs, n_positions), dtype=cp.float32)
    rc_valid_mask = cp.ones((n_seqs, n_positions), dtype=cp.bool_)
    
    for pos_in_motif in range(motif_len):
        nucs = rc_seqs[:, pos_in_motif:pos_in_motif + n_positions]
        rc_valid_mask &= (nucs >= 0)
        nucs_safe = cp.clip(nucs, 0, 3)
        pos_scores = pwm_log_odds[pos_in_motif, nucs_safe]
        rc_scores += pos_scores
    
    rc_scores = cp.where(rc_valid_mask, rc_scores, -cp.inf)
    rc_scores = cp.where(seq_len_mask, rc_scores, -cp.inf)
    
    rc_match = (rc_scores >= threshold).any(axis=1)
    
    return forward_match | rc_match


def _scan_motifs_batch_gpu(
    encoded_seqs: cp.ndarray,
    seq_lengths: cp.ndarray,
    pwm_list: list[np.ndarray],
    thresholds: np.ndarray,
    rc_seqs: Optional[cp.ndarray] = None,
    motif_batch_size: int = 32,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Batch scan multiple motifs on GPU for improved throughput.
    
    Groups motifs by length and processes them together to minimize
    memory allocations and maximize GPU utilization.
    
    Parameters
    ----------
    encoded_seqs : cp.ndarray
        Pre-encoded sequences on GPU, shape (n_seqs, max_len)
    seq_lengths : cp.ndarray
        Sequence lengths on GPU
    pwm_list : list[np.ndarray]
        List of log-odds PWM matrices (one per motif)
    thresholds : np.ndarray
        Score thresholds for each motif
    rc_seqs : cp.ndarray, optional
        Pre-computed reverse complement sequences. If None, RC not checked.
    motif_batch_size : int
        Number of motifs to process together per length group
    show_progress : bool
        Whether to show progress bar
        
    Returns
    -------
    np.ndarray
        Boolean array of shape (n_motifs, n_seqs) indicating motif presence
    """
    from tqdm.auto import tqdm
    
    n_seqs, max_len = encoded_seqs.shape
    n_motifs = len(pwm_list)
    
    # Pre-allocate result array
    all_bound = np.zeros((n_motifs, n_seqs), dtype=np.bool_)
    
    # Group motifs by length for efficient batch processing
    length_to_motifs = {}
    for i, pwm in enumerate(pwm_list):
        motif_len = pwm.shape[0]
        if motif_len not in length_to_motifs:
            length_to_motifs[motif_len] = []
        length_to_motifs[motif_len].append(i)
    
    # Create progress bar for motif processing
    pbar = tqdm(total=n_motifs, desc="Motifs", disable=not show_progress)
    
    # Process each length group
    for motif_len, motif_indices in length_to_motifs.items():
        if max_len < motif_len:
            pbar.update(len(motif_indices))
            continue
        
        n_positions = max_len - motif_len + 1
        
        # Precompute position indices and length mask for this motif length
        position_indices = cp.arange(n_positions, dtype=cp.int32)[None, :]
        seq_len_mask = (position_indices + motif_len) <= seq_lengths[:, None]
        
        # Precompute valid nucleotide masks and safe nucleotide slices for forward strand
        fw_valid_mask = cp.ones((n_seqs, n_positions), dtype=cp.bool_)
        fw_nucs_safe_list = []
        for pos_in_motif in range(motif_len):
            nucs = encoded_seqs[:, pos_in_motif:pos_in_motif + n_positions]
            fw_valid_mask &= (nucs >= 0)
            fw_nucs_safe_list.append(cp.clip(nucs, 0, 3))
        
        # Combine masks once
        fw_combined_mask = fw_valid_mask & seq_len_mask
        del fw_valid_mask
        
        # Precompute for reverse complement
        rc_nucs_safe_list = None
        rc_combined_mask = None
        if rc_seqs is not None:
            rc_valid_mask = cp.ones((n_seqs, n_positions), dtype=cp.bool_)
            rc_nucs_safe_list = []
            for pos_in_motif in range(motif_len):
                nucs = rc_seqs[:, pos_in_motif:pos_in_motif + n_positions]
                rc_valid_mask &= (nucs >= 0)
                rc_nucs_safe_list.append(cp.clip(nucs, 0, 3))
            rc_combined_mask = rc_valid_mask & seq_len_mask
            del rc_valid_mask
        
        # Process motifs in batches within this length group
        for batch_start in range(0, len(motif_indices), motif_batch_size):
            batch_end = min(batch_start + motif_batch_size, len(motif_indices))
            batch_motif_indices = motif_indices[batch_start:batch_end]
            batch_size = len(batch_motif_indices)
            
            # Stack PWMs for this batch - shape: (batch_size, motif_len, 4)
            batch_pwms = np.stack([pwm_list[i] for i in batch_motif_indices])
            batch_pwms_gpu = cp.asarray(batch_pwms, dtype=cp.float32)
            batch_thresholds = cp.asarray(thresholds[batch_motif_indices], dtype=cp.float32)
            
            # Forward scan - compute scores for all motifs in batch
            # Shape: (batch_size, n_seqs, n_positions)
            fw_scores = cp.zeros((batch_size, n_seqs, n_positions), dtype=cp.float32)
            
            for pos_in_motif in range(motif_len):
                nucs_safe = fw_nucs_safe_list[pos_in_motif]
                # batch_pwms_gpu[:, pos_in_motif, :] has shape (batch_size, 4)
                # nucs_safe has shape (n_seqs, n_positions)
                # Use advanced indexing for vectorized lookup
                pos_scores = batch_pwms_gpu[:, pos_in_motif, :][:, nucs_safe]
                fw_scores += pos_scores
            
            # Apply masks
            fw_scores = cp.where(fw_combined_mask[None, :, :], fw_scores, -cp.inf)
            
            # Check threshold - shape: (batch_size, n_seqs)
            fw_match = (fw_scores >= batch_thresholds[:, None, None]).any(axis=2)
            
            if rc_seqs is not None:
                # Reverse complement scan
                rc_scores = cp.zeros((batch_size, n_seqs, n_positions), dtype=cp.float32)
                
                for pos_in_motif in range(motif_len):
                    nucs_safe = rc_nucs_safe_list[pos_in_motif]
                    pos_scores = batch_pwms_gpu[:, pos_in_motif, :][:, nucs_safe]
                    rc_scores += pos_scores
                
                rc_scores = cp.where(rc_combined_mask[None, :, :], rc_scores, -cp.inf)
                rc_match = (rc_scores >= batch_thresholds[:, None, None]).any(axis=2)
                
                batch_bound = fw_match | rc_match
                del rc_scores, rc_match
            else:
                batch_bound = fw_match
            
            # Transfer batch results to CPU
            batch_bound_cpu = cp.asnumpy(batch_bound)
            for local_idx, global_idx in enumerate(batch_motif_indices):
                all_bound[global_idx] = batch_bound_cpu[local_idx]
            
            # Update progress bar
            pbar.update(batch_size)
            
            del batch_pwms_gpu, fw_scores, fw_match, batch_bound
        
        # Free memory after processing this length group
        del fw_nucs_safe_list, fw_combined_mask
        if rc_nucs_safe_list is not None:
            del rc_nucs_safe_list, rc_combined_mask
    
    pbar.close()
    return all_bound



# =============================================================================
# Statistical Tests
# =============================================================================


def _p_adjust_bh(p_values: np.ndarray) -> np.ndarray:
    """
    Benjamini-Hochberg p-value correction for multiple testing.
    
    Parameters
    ----------
    p_values : np.ndarray
        Array of p-values
        
    Returns
    -------
    np.ndarray
        Adjusted p-values
    """
    n = len(p_values)
    if n == 0:
        return p_values.copy()
    
    # Sort p-values
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]
    
    # Compute BH adjusted values
    rank = np.arange(1, n + 1)
    adjusted = sorted_p * n / rank
    
    # Ensure monotonicity (cumulative minimum from right)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    
    # Restore original order
    result = np.empty(n)
    result[sorted_idx] = adjusted
    return result


def _normalize_background_groups(
    regions: dict[str, list[str]],
    background: Optional[Union[list[str], dict[str, list[str]]]],
) -> Optional[dict[str, list[str]]]:
    """Normalize shared or per-group backgrounds to a per-group mapping."""
    if background is None:
        return None

    if isinstance(background, dict):
        missing = [group for group in regions if group not in background]
        extra = [group for group in background if group not in regions]
        if missing or extra:
            problems = []
            if missing:
                problems.append(f"missing keys: {missing}")
            if extra:
                problems.append(f"unexpected keys: {extra}")
            raise ValueError(
                "When 'background' is a dict, its keys must match 'regions'. "
                + "; ".join(problems)
            )
        background_groups = {group: list(background[group]) for group in regions}
    else:
        background_groups = {group: list(background) for group in regions}

    empty_groups = [group for group, group_background in background_groups.items() if len(group_background) == 0]
    if empty_groups:
        raise ValueError(
            "Background regions must be non-empty for every group. "
            f"Empty groups: {empty_groups}"
        )

    return background_groups


def _sample_gc_matched_background_single(
    target_regions: list[str],
    background_pool: list[str],
    gc_by_region: dict[str, float],
    *,
    n_background: Optional[int],
    n_bins: int,
    replace: bool,
    rng: np.random.Generator,
) -> list[str]:
    """Sample a background set whose GC distribution matches a target set."""
    if len(target_regions) == 0:
        raise ValueError("Target regions must be non-empty.")

    target_set = set(target_regions)
    unique_pool = [region for region in _ordered_unique_regions([background_pool]) if region not in target_set]
    if len(unique_pool) == 0:
        raise ValueError(
            "Background pool is empty after removing target regions. "
            "Provide a larger background pool."
        )

    target_size = len(target_regions) if n_background is None else int(n_background)
    if target_size < 1:
        raise ValueError("'n_background' must be at least 1.")

    target_gc = np.asarray([gc_by_region[region] for region in target_regions], dtype=np.float64)
    pool_gc = np.asarray([gc_by_region[region] for region in unique_pool], dtype=np.float64)

    sampled_indices = sample_bias_matched_indices(
        target_gc,
        pool_gc,
        n_samples=target_size,
        n_bins=n_bins,
        replace=replace,
        rng=rng,
    )
    return [unique_pool[idx] for idx in sampled_indices]


def sample_gc_matched_background(
    regions: Union[list[str], dict[str, list[str]]],
    genome_fasta: Union[str, Path],
    *,
    background_pool: Union[list[str], dict[str, list[str]]],
    n_background: Optional[int] = None,
    n_bins: int = 50,
    replace: bool = True,
    random_state: Optional[int] = 0,
) -> Union[list[str], dict[str, list[str]]]:
    """
    Sample background peaks whose GC-content distribution matches target peaks.

    Parameters
    ----------
    regions : list[str] or dict[str, list[str]]
        Target peaks to match. When a dict is provided, each group is sampled
        independently and the return value mirrors the same keys.
    genome_fasta : str or Path
        Path to the reference genome FASTA used to compute GC content.
    background_pool : list[str] or dict[str, list[str]]
        Candidate background peaks to sample from. When `regions` is a dict,
        this can be either one shared pool or a dict keyed like `regions`.
    n_background : int, optional
        Number of peaks to sample per target group. Defaults to the size of the
        corresponding target set.
    n_bins : int, default 50
        Matching resolution. Larger values enforce tighter GC matching.
    replace : bool, default True
        Whether sampled background peaks may be reused. Set to ``False`` to
        require unique sampled peaks within each returned background set.
    random_state : int, optional
        Seed for deterministic sampling.

    Returns
    -------
    list[str] or dict[str, list[str]]
        GC-matched background peaks with the same container shape as `regions`.

    Examples
    --------
    >>> matched_bg = ga.tl.sample_gc_matched_background(
    ...     da_peaks,
    ...     genome_fasta="genome.fa",
    ...     background_pool=all_peaks,
    ... )
    >>> matched_bg_by_group = ga.tl.sample_gc_matched_background(
    ...     marker_peaks,
    ...     genome_fasta="genome.fa",
    ...     background_pool=list(peak_adata.var_names),
    ...     replace=False,
    ... )
    """
    if n_bins < 1:
        raise ValueError("'n_bins' must be at least 1.")

    rng = np.random.default_rng(random_state)

    if isinstance(regions, dict):
        if isinstance(background_pool, dict):
            background_groups = _normalize_background_groups(regions, background_pool)
        else:
            background_groups = {group: list(background_pool) for group in regions}

        gc_regions = _ordered_unique_regions(
            list(regions.values()) + list(background_groups.values())
        )
        gc_values = _compute_region_gc_content(gc_regions, genome_fasta)
        gc_by_region = {
            region: float(gc)
            for region, gc in zip(gc_regions, gc_values, strict=False)
        }

        return {
            group: _sample_gc_matched_background_single(
                target_regions,
                background_groups[group],
                gc_by_region,
                n_background=n_background,
                n_bins=n_bins,
                replace=replace,
                rng=rng,
            )
            for group, target_regions in regions.items()
        }

    if isinstance(background_pool, dict):
        raise ValueError(
            "When 'regions' is a list, 'background_pool' must also be a list."
        )

    gc_regions = _ordered_unique_regions([list(regions), list(background_pool)])
    gc_values = _compute_region_gc_content(gc_regions, genome_fasta)
    gc_by_region = {
        region: float(gc)
        for region, gc in zip(gc_regions, gc_values, strict=False)
    }

    return _sample_gc_matched_background_single(
        list(regions),
        list(background_pool),
        gc_by_region,
        n_background=n_background,
        n_bins=n_bins,
        replace=replace,
        rng=rng,
    )


# =============================================================================
# Main API
# =============================================================================


def motif_enrichment(
    motifs: list[DNAMotif],
    regions: dict[str, list[str]],
    genome_fasta: Union[str, Path],
    background: Optional[Union[list[str], dict[str, list[str]]]] = None,
    method: Optional[Literal["binomial", "hypergeometric"]] = None,
    pvalue: float = 1e-5,
    check_rc: bool = True,
    bg_probs: Union[
        Literal["auto", "subject", "even"],
        tuple[float, float, float, float],
    ] = (0.25, 0.25, 0.25, 0.25),
    motif_batch_size: int = 16,
) -> dict[str, pd.DataFrame]:
    """
    Identify enriched transcription factor motifs using GPU acceleration.
    
    This function scans genomic regions for motif matches and performs
    statistical enrichment testing against a background set.
    
    Parameters
    ----------
    motifs : list[DNAMotif]
        List of transcription factor motifs to test
    regions : dict[str, list[str]]
        Groups of genomic regions to test. Keys are group names,
        values are lists of region strings in "chr:start-end" format.
        Each group is tested independently.
    genome_fasta : str or Path
        Path to genome FASTA file for sequence extraction
    background : list[str] or dict[str, list[str]], optional
        Background regions. Pass a single list to use one shared background for
        all groups, or a dict keyed like `regions` to use per-group matched
        backgrounds. If None, the union of all tested regions is used.
    method : {"binomial", "hypergeometric"}, optional
        Statistical test method. If None, uses "hypergeometric" when
        background is None (subset testing), else "binomial".
    pvalue : float, default 1e-5
        P-value threshold for motif matching
    check_rc : bool, default True
        Whether to check both strands (forward and reverse complement)
    bg_probs : {"auto", "subject", "even"} or tuple, default (0.25, 0.25, 0.25, 0.25)
        Background nucleotide probabilities (A, C, G, T) used when converting
        motifs to log-odds scores and computing match thresholds. Use
        ``"auto"`` to estimate base frequencies from all scanned sequences
        (foreground plus any provided background), ``"subject"`` as an alias
        for the same behavior, ``"even"`` for a uniform background, or pass a
        custom 4-tuple.
    motif_batch_size : int, default 16
        Number of motifs of the same length to process together on GPU.
        Higher values increase GPU parallelism but use more memory.
        
    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary mapping group names to DataFrames with columns:
        - id: Motif ID
        - name: Motif name
        - family: Motif family
        - log2(fold change): Log2 fold enrichment
        - p-value: Raw p-value
        - adjusted p-value: BH-corrected p-value
        
    Examples
    --------
    >>> import gatac
    >>> motifs = gatac.tl.read_motifs("motifs.meme")
    >>> regions = {
    ...     "cluster1": ["chr1:1000-1500", "chr1:5000-5500"],
    ...     "cluster2": ["chr2:2000-2500"],
    ... }
    >>> results = gatac.tl.motif_enrichment(
    ...     motifs, regions, "genome.fa"
    ... )
    >>> results["cluster1"]  # DataFrame with enrichment results
    >>> matched_bg = gatac.tl.sample_gc_matched_background(
    ...     regions,
    ...     genome_fasta="genome.fa",
    ...     background_pool=all_peaks,
    ... )
    >>> results = gatac.tl.motif_enrichment(
    ...     motifs,
    ...     regions,
    ...     "genome.fa",
    ...     background=matched_bg,
    ...     bg_probs="auto",
    ... )
    """
    from scipy.stats import binom, hypergeom
    from tqdm.auto import tqdm
    
    # Determine method
    if method is None:
        method = "hypergeometric" if background is None else "binomial"

    background_groups = _normalize_background_groups(regions, background)
    
    # Collect all unique regions
    all_region_groups = list(regions.values())
    if background_groups is not None:
        all_region_groups.extend(background_groups.values())
    all_regions = _ordered_unique_regions(all_region_groups)
    region_to_idx = {r: i for i, r in enumerate(all_regions)}

    sequences = _fetch_region_sequences(all_regions, genome_fasta)
    
    # OPTIMIZATION: Encode sequences ONCE upfront and keep on GPU
    logger.info("Encoding sequences for GPU...")
    encoded_seqs, seq_lengths = _encode_sequences_batch(sequences)
    
    # Precompute reverse complement if needed
    if check_rc:
        rc_seqs = _reverse_complement_encoded(encoded_seqs)
    else:
        rc_seqs = None
    
    fg_indices_dict = {}
    for group_name, group_regions in regions.items():
        fg_indices_dict[group_name] = np.array(
            [region_to_idx[r] for r in group_regions], dtype=np.int32
        )

    bg_indices_dict = None
    if background_groups is not None:
        bg_indices_dict = {}
        for group_name, group_background in background_groups.items():
            bg_indices_dict[group_name] = np.array(
                [region_to_idx[r] for r in group_background], dtype=np.int32
            )
    
    resolved_bg_probs = _resolve_bg_probs(bg_probs, sequences)

    # OPTIMIZATION: Precompute all thresholds using Numba JIT
    logger.info("Precomputing score thresholds for all motifs...")
    
    pwm_list = [motif.to_log_odds(resolved_bg_probs) for motif in motifs]
    bg_array = np.array(resolved_bg_probs, dtype=np.float64)
    
    # Compute thresholds using JIT-compiled function (parallelizable)
    thresholds = np.empty(len(motifs), dtype=np.float64)
    for i, pwm in enumerate(pwm_list):
        thresholds[i] = _compute_score_threshold_jit(pwm, bg_array, pvalue, 1e-4)
    
    logger.info(f"Scanning {len(motifs)} motifs across {len(all_regions)} regions...")
    
    # OPTIMIZATION: Batch scan all motifs at once using GPU
    # This reduces GPU memory transfers and enables better parallelism
    # Larger batch size = more GPU parallelism but more memory usage
    all_bound = _scan_motifs_batch_gpu(
        encoded_seqs, seq_lengths, pwm_list, thresholds, rc_seqs,
        motif_batch_size=motif_batch_size
    )
    
    # Free GPU memory after scanning
    del encoded_seqs, seq_lengths, rc_seqs
    mempool.free_all_blocks()
    
    # Compute statistics for all motifs at once using vectorized operations
    n_motifs = len(motifs)
    n_groups = len(regions)
    n_seqs = all_bound.shape[1]

    default_bound_bg = all_bound.sum(axis=1) if background_groups is None else None
    
    # Preallocate result arrays
    total_results = n_motifs * n_groups
    motif_ids = []
    motif_names = []
    motif_families = []
    group_names_list = []
    fold_changes = np.empty(total_results, dtype=np.float64)
    n_fg_arr = np.empty(total_results, dtype=np.int32)
    N_fg_arr = np.empty(total_results, dtype=np.int32)
    n_bg_arr = np.empty(total_results, dtype=np.int32)
    N_bg_arr = np.empty(total_results, dtype=np.int32)
    
    result_idx = 0
    for group_name, group_regions in regions.items():
        fg_indices = fg_indices_dict[group_name]
        total_fg = len(fg_indices)

        if background_groups is None:
            total_bg = n_seqs
            bound_bg = default_bound_bg
        else:
            bg_indices = bg_indices_dict[group_name]
            total_bg = len(bg_indices)
            bound_bg = all_bound[:, bg_indices].sum(axis=1)
        
        # Vectorized foreground computation for all motifs
        bound_fg = all_bound[:, fg_indices].sum(axis=1)  # Shape: (n_motifs,)
        
        for i, motif in enumerate(motifs):
            bf = bound_fg[i]
            bb = bound_bg[i]
            
            # Compute fold change
            if bf == 0:
                log_fc = 0.0 if bb == 0 else float("-inf")
            elif bb == 0:
                log_fc = float("inf")
            else:
                fc = (bf / total_fg) / (bb / total_bg)
                log_fc = np.log2(fc) if fc > 0 else float("-inf")
            
            motif_ids.append(motif.id)
            motif_names.append(motif.name)
            motif_families.append(motif.family)
            group_names_list.append(group_name)
            fold_changes[result_idx] = log_fc
            n_fg_arr[result_idx] = int(bf)
            N_fg_arr[result_idx] = total_fg
            n_bg_arr[result_idx] = int(bb)
            N_bg_arr[result_idx] = total_bg
            result_idx += 1
    
    # Compute p-values (vectorized)
    p_values = np.zeros(total_results)
    
    up_idx = fold_changes >= 0
    down_idx = fold_changes < 0
    
    if method == "binomial":
        # Binomial test
        bg_prob = np.clip(n_bg_arr / N_bg_arr, 1e-10, 1 - 1e-10)
        p_values[up_idx] = binom.sf(n_fg_arr[up_idx] - 1, N_fg_arr[up_idx], bg_prob[up_idx])
        p_values[down_idx] = binom.cdf(n_fg_arr[down_idx], N_fg_arr[down_idx], bg_prob[down_idx])
    elif method == "hypergeometric":
        # Hypergeometric test
        p_values[up_idx] = hypergeom.sf(
            n_fg_arr[up_idx] - 1, N_bg_arr[up_idx], n_bg_arr[up_idx], N_fg_arr[up_idx]
        )
        p_values[down_idx] = hypergeom.cdf(
            n_fg_arr[down_idx], N_bg_arr[down_idx], n_bg_arr[down_idx], N_fg_arr[down_idx]
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use 'binomial' or 'hypergeometric'")
    
    p_values = np.clip(p_values, 1e-300, 1.0)
    
    # Organize results by group
    result = {}
    unique_groups = list(regions.keys())
    
    for group in unique_groups:
        group_mask = np.array([g == group for g in group_names_list])
        group_pvals = p_values[group_mask]
        adjusted_pvals = _p_adjust_bh(group_pvals)
        
        group_df = pl.DataFrame({
            "id": [motif_ids[i] for i, m in enumerate(group_mask) if m],
            "name": [motif_names[i] for i, m in enumerate(group_mask) if m],
            "family": [motif_families[i] for i, m in enumerate(group_mask) if m],
            "log2(fold change)": fold_changes[group_mask].tolist(),
            "p-value": group_pvals.tolist(),
            "adjusted p-value": adjusted_pvals.tolist(),
        })
        result[group] = group_df
    
    logger.info("Motif enrichment analysis complete")
    return {k: v.to_pandas() for k, v in result.items()}


# =============================================================================
# GSEA-based Motif Enrichment
# =============================================================================


def gsea_motif_enrichment(
    adata,
    rankings: "Union[pd.DataFrame, dict[str, pd.DataFrame]]",
    logfc_col: str = "log2fc",
    *,
    motif_key: str = "motif_match",
    permutation_num: int = 1000,
    min_size: int = 15,
    max_size: int = 2000,
    seed: int = 42,
    threads: int = 1,
    backend: Literal["gpu", "gseapy"] = "gpu",
    gs_batch_size: int = 4,
) -> "Union[pd.DataFrame, dict[str, pd.DataFrame]]":
    """
    Run preranked GSEA to identify enriched TF motifs from a LogFC-ranked peak list.

    Unlike Fisher/hypergeometric motif enrichment, GSEA does not require a hard
    significance threshold to define "marker peaks". Instead it ranks *all* peaks
    by log2 fold change and asks whether motif-containing peaks cluster at the top
    (or bottom) of that ranking. This provides statistical power even in shallow
    ATAC-seq data where no individual peak may reach significance.

    Motif gene sets are built from the binary peak×motif matrix stored at
    ``adata.varm[motif_key]`` (populated by :func:`gatac.tl.scan_motifs`).

    Parameters
    ----------
    adata : AnnData
        Peak-level AnnData object with ``varm[motif_key]`` (sparse bool
        matrix of shape n_peaks × n_motifs) and ``uns["motif_name"]``
        (list of motif names). Populated by ``ga.tl.scan_motifs``.
    rankings : pd.DataFrame or dict[str, pd.DataFrame]
        Per-peak ranking table(s).

        * **Single DataFrame** – index must be peak names matching
          ``adata.var_names``; ``logfc_col`` specifies the log2FC column.
          Returns a single :class:`pandas.DataFrame`.
        * **Dict of DataFrames** – keys are group labels; each value is a
          DataFrame as above. Returns ``dict[str, pandas.DataFrame]``.
    logfc_col : str, default ``"log2fc"``
        Name of the column in each DataFrame that contains log2 fold change
        values. Peaks are ranked descending by this column before GSEA.
    motif_key : str, default ``"motif_match"``
        Key in ``adata.varm`` that stores the peak×motif binary matrix.
    permutation_num : int, default 1000
        Number of GSEA permutations. Increase for more precise FDR estimates.
    min_size : int, default 15
        Minimum number of ranked peaks a motif gene set must contain (after
        intersection with the ranked list) to be tested. Smaller sets yield
        noisy NES estimates.
    max_size : int, default 2000
        Maximum motif gene set size.
    seed : int, default 42
        Random seed for permutation reproducibility.
    threads : int, default 1
        Number of threads passed to ``gseapy.prerank`` (only used when
        ``backend="gseapy"``).
    backend : {"gpu", "gseapy"}, default "gpu"
        Which backend to use for the enrichment score computation.

        * ``"gpu"`` – CuPy-based GPU implementation. Much faster for large
          numbers of motifs (10-50× speedup). Requires a CUDA-capable GPU.
        * ``"gseapy"`` – Delegates to ``gseapy.prerank`` (Rust backend).
          No GPU required.
    gs_batch_size : int, default 4
        Number of gene sets processed simultaneously on the GPU per kernel
        call (GPU backend only). Larger values increase throughput at the
        cost of more VRAM. Reduce if you encounter out-of-memory errors.

    Returns
    -------
    pandas.DataFrame or dict[str, pandas.DataFrame]
        GSEA results with columns:

        * ``motif``           – motif name
        * ``NES``             – normalised enrichment score (positive = enriched
          at the top / high-logFC end)
        * ``pval``            – nominal p-value
        * ``fdr``             – FDR q-value (Benjamini–Hochberg)
        * ``lead_edge_n``     – number of peaks in the leading edge
        * ``set_size``        – number of ranked peaks containing the motif
        * ``lead_edge_frac``  – ``lead_edge_n / set_size``; fraction of the
          motif set in the leading edge (0–1)

        Sorted descending by NES. Returns a single DataFrame when *rankings*
        is a single DataFrame, or a dict when it is a dict.

    Raises
    ------
    ImportError
        If ``gseapy`` is not installed (when ``backend="gseapy"``).
    KeyError
        If ``motif_key`` is not found in ``adata.varm``, or ``logfc_col``
        is not found in a rankings DataFrame.

    Examples
    --------
    >>> import gatac as ga
    >>> import pandas as pd

    >>> # rankings is a DataFrame with peaks as index and a 'log2fc' column
    >>> ranked = pd.DataFrame({"log2fc": logfc_values}, index=peak_names)
    >>> result = ga.tl.gsea_motif_enrichment(peak_adata, ranked)

    >>> # Multiple groups at once
    >>> group_rankings = {
    ...     "CD4_Memory": pd.DataFrame({"log2fc": logfc_cd4}, index=peak_names),
    ...     "NK":         pd.DataFrame({"log2fc": logfc_nk},  index=peak_names),
    ... }
    >>> results = ga.tl.gsea_motif_enrichment(peak_adata, group_rankings)
    >>> results["NK"].head()
    """
    import pandas as pd
    import scipy.sparse as sp

    if backend == "gseapy":
        try:
            import gseapy as gp
        except ImportError as exc:
            raise ImportError(
                "gseapy is required for backend='gseapy'. "
                "Install it with: pip install gseapy"
            ) from exc
    elif backend == "gpu":
        from gatac.tl.gsea import prerank_gpu
    else:
        raise ValueError(f"Unknown backend: {backend!r}. Use 'gpu' or 'gseapy'.")

    # ------------------------------------------------------------------
    # Build motif gene sets from adata.varm[motif_key]
    # ------------------------------------------------------------------
    if motif_key not in adata.varm:
        raise KeyError(
            f"'{motif_key}' not found in adata.varm. "
            "Run ga.tl.scan_motifs first to populate the motif match matrix."
        )

    motif_match = adata.varm[motif_key]   # peaks × motifs  (sparse bool)
    motif_names_list = list(adata.uns["motif_name"])
    peak_names = adata.var_names

    if sp.issparse(motif_match):
        motif_match_csc = sp.csc_matrix(motif_match)
    else:
        motif_match_csc = sp.csc_matrix(motif_match)

    feature_sets: dict[str, list] = {}
    for i, name in enumerate(motif_names_list):
        peak_idx = motif_match_csc.getcol(i).nonzero()[0]
        if len(peak_idx) >= min_size:
            feature_sets[str(name)] = list(peak_names[peak_idx])

    if not feature_sets:
        logger.warning(
            f"No motif feature set has ≥{min_size} peaks. "
            "Try lowering min_size or re-scanning with a more lenient pvalue."
        )

    logger.info(
        f"Built {len(feature_sets):,} motif feature sets "
        f"(≥{min_size} peaks; filter to ≥{min_size})"
    )

    # ------------------------------------------------------------------
    # Internal helpers: run GSEA for a single ranked DataFrame
    # ------------------------------------------------------------------
    def _run_one_gseapy(df: pd.DataFrame, group_label: str = "") -> pl.DataFrame:
        import gseapy as gp

        if logfc_col not in df.columns:
            raise KeyError(
                f"Column '{logfc_col}' not found in rankings DataFrame"
                + (f" for group '{group_label}'" if group_label else "")
                + f". Available columns: {list(df.columns)}"
            )

        ranked_series = df[logfc_col].sort_values(ascending=False)

        res = gp.prerank(
            rnk=ranked_series,
            gene_sets=feature_sets,
            permutation_num=permutation_num,
            seed=seed,
            min_size=min_size,
            max_size=max_size,
            threads=threads,
            no_plot=True,
            verbose=False,
        )

        raw = res.res2d.rename(columns={
            "Term": "motif",
            "NOM p-val": "pval",
            "FDR q-val": "fdr",
            "Lead_genes": "lead_edge",
        })

        raw["lead_edge_n"] = raw["lead_edge"].apply(
            lambda x: len(x.split(";")) if isinstance(x, str) and x else 0
        )
        raw["set_size"] = raw["motif"].map(lambda m: len(feature_sets.get(m, [])))
        raw["lead_edge_frac"] = raw.apply(
            lambda row: row["lead_edge_n"] / row["set_size"] if row["set_size"] > 0 else 0.0,
            axis=1,
        )

        result_df = (
            raw[["motif", "NES", "pval", "fdr", "lead_edge_n", "set_size", "lead_edge_frac"]]
            .sort_values("NES", ascending=False)
            .reset_index(drop=True)
        )

        return pl.from_pandas(result_df)

    def _run_one_gpu(df: pd.DataFrame, group_label: str = "") -> pl.DataFrame:
        if logfc_col not in df.columns:
            raise KeyError(
                f"Column '{logfc_col}' not found in rankings DataFrame"
                + (f" for group '{group_label}'" if group_label else "")
                + f". Available columns: {list(df.columns)}"
            )

        ranked_series = df[logfc_col].sort_values(ascending=False)

        results = prerank_gpu(
            feature_names=list(ranked_series.index),
            ranking_values=ranked_series.values,
            feature_sets=feature_sets,
            weight=1.0,
            min_size=min_size,
            max_size=max_size,
            permutation_num=permutation_num,
            seed=seed,
            gs_batch_size=gs_batch_size,
        )

        if not results:
            return pl.DataFrame({
                "motif": [],
                "NES": [],
                "pval": [],
                "fdr": [],
                "lead_edge_n": [],
                "set_size": [],
                "lead_edge_frac": [],
            })

        result_df = pl.DataFrame({
            "motif": [r["term"] for r in results],
            "NES": [r["nes"] for r in results],
            "pval": [r["pval"] for r in results],
            "fdr": [r["fdr"] for r in results],
            "lead_edge_n": [r["lead_edge_n"] for r in results],
            "set_size": [len(r["hits"]) for r in results],
            "lead_edge_frac": [
                r["lead_edge_n"] / len(r["hits"]) if len(r["hits"]) > 0 else 0.0
                for r in results
            ],
        }).sort("NES", descending=True)

        return result_df

    _run_one = _run_one_gpu if backend == "gpu" else _run_one_gseapy

    # ------------------------------------------------------------------
    # Dispatch: single DataFrame or dict of DataFrames
    # ------------------------------------------------------------------
    if isinstance(rankings, dict):
        output: dict[str, pd.DataFrame] = {}
        for group, df in rankings.items():
            logger.info(f"Running GSEA ({backend}) for group '{group}'...")
            output[group] = _run_one(df, group_label=str(group)).to_pandas()
        return output
    else:
        logger.info(f"Running GSEA ({backend}) on provided rankings...")
        return _run_one(rankings).to_pandas()
