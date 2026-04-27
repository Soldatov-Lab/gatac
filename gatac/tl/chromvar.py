"""
GPU-accelerated chromVAR implementation for ATAC-seq data.

This module provides chromVAR-style TF deviation analysis using GPU acceleration
via CuPy and cuML. Adapted from scPrinter's chromvar implementation with gatac's
infrastructure.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Optional, Union

import cupy as cp
import cupyx.scipy.sparse as cupy_sparse
import numpy as np
import scipy.sparse as sp
from anndata import AnnData
from scipy.sparse import csr_matrix as scipy_csr_matrix
from tqdm.auto import tqdm, trange

from ._bias_matching import compute_bias_knn_indices, normalize_bias_matrix
from .motif import DNAMotif, _encode_sequences_batch, _open_fasta

logger = logging.getLogger(__name__)

# CuPy memory pools for efficient memory management
mempool = cp.get_default_memory_pool()
pinned_mempool = cp.get_default_pinned_memory_pool()


# =============================================================================
# Helper Functions
# =============================================================================


def scipy_to_cupy_sparse(sparse_matrix: scipy_csr_matrix) -> cupy_sparse.csr_matrix:
    """
    Convert a SciPy sparse matrix to a CuPy sparse matrix.
    
    Parameters
    ----------
    sparse_matrix : scipy.sparse.csr_matrix
        SciPy CSR matrix to convert
        
    Returns
    -------
    cupyx.scipy.sparse.csr_matrix
        CuPy CSR matrix on GPU
        
    Raises
    ------
    ValueError
        If input matrix is not a SciPy CSR matrix
    """
    if not isinstance(sparse_matrix, scipy_csr_matrix):
        raise ValueError("Input matrix must be a SciPy CSR matrix")

    # Get the CSR components of the SciPy sparse matrix
    data = sparse_matrix.data.astype(np.float32)
    indices = sparse_matrix.indices
    indptr = sparse_matrix.indptr
    shape = sparse_matrix.shape

    # Convert the components to CuPy arrays
    data_cp = cp.array(data)
    indices_cp = cp.array(indices)
    indptr_cp = cp.array(indptr)

    # Create a CuPy CSR matrix with these components
    cupy_sparse_matrix = cupy_sparse.csr_matrix(
        (data_cp, indices_cp, indptr_cp), shape=shape
    )
    return cupy_sparse_matrix


def _compute_gc_content_gpu(
    peak_regions: list[str],
    genome_fasta: Union[str, Path],
    coordinate_system: Literal["0-based", "1-based"] = "0-based",
    batch_size: int = 10000,
) -> np.ndarray:
    """
    Compute GC content using GPU-accelerated batch processing.

    Fetches sequences from the FASTA file in one pass, encodes them as a
    padded 2-D byte matrix, and counts G/C bases on the GPU via CuPy.
    This is significantly faster than per-peak Python string `.count()` calls
    for large peak sets.

    Parameters
    ----------
    peak_regions : list[str]
        Peak names in "chr:start-end" format.
    genome_fasta : str or Path
        Path to genome FASTA file.
    coordinate_system : {"0-based", "1-based"}, default "0-based"
        Coordinate system of the peak names.
    batch_size : int, default 10000
        Number of peaks to process per GPU batch.

    Returns
    -------
    np.ndarray
        GC content per peak, shape (n_peaks,), dtype float32.
    """
    n_peaks = len(peak_regions)

    # --- 1. Fetch all sequences (CPU / I/O bound) ---
    sequences: list[str] = []
    with _open_fasta(genome_fasta) as genome:
        for region in tqdm(peak_regions, desc="Fetching sequences for GC content"):
            try:
                chrom, coords = region.split(":")
                start, end = int(coords.split("-")[0]), int(coords.split("-")[1])
                if coordinate_system == "1-based":
                    seq = str(genome[chrom][start - 1 : end].seq).upper()
                else:
                    seq = str(genome[chrom][start:end].seq).upper()
                sequences.append(seq)
            except Exception as e:
                logger.warning(f"Failed to fetch {region}: {e}")
                sequences.append("")

    lengths = np.array([len(s) for s in sequences], dtype=np.int32)
    gc_content = np.zeros(n_peaks, dtype=np.float32)

    # --- 2. GPU-vectorised GC counting in batches ---
    for batch_start in range(0, n_peaks, batch_size):
        batch_end = min(batch_start + batch_size, n_peaks)
        batch_seqs = sequences[batch_start:batch_end]
        batch_lengths = lengths[batch_start:batch_end]

        max_len = int(batch_lengths.max()) if batch_lengths.max() > 0 else 0
        if max_len == 0:
            continue

        # Encode: pad sequences into a (batch, max_len) uint8 matrix
        seq_array = np.zeros((len(batch_seqs), max_len), dtype=np.uint8)
        for i, seq in enumerate(batch_seqs):
            n = len(seq)
            if n > 0:
                seq_array[i, :n] = np.frombuffer(seq.encode(), dtype=np.uint8)

        # Transfer to GPU and count G (71) and C (67) in parallel
        seq_gpu = cp.asarray(seq_array)
        gc_mask = (seq_gpu == 71) | (seq_gpu == 67)
        gc_counts = gc_mask.sum(axis=1).astype(cp.float32)
        batch_len_gpu = cp.asarray(batch_lengths, dtype=cp.float32)
        batch_gc = cp.where(batch_len_gpu > 0, gc_counts / batch_len_gpu, 0.0)
        gc_content[batch_start:batch_end] = cp.asnumpy(batch_gc)

        del seq_gpu, gc_mask, gc_counts, batch_gc
        mempool.free_all_blocks()

    return gc_content


def compute_peak_bias(
    adata: AnnData,
    genome_fasta: Union[str, Path],
    *,
    add_gc_content: bool = True,
    add_cpg_density: bool = False,
) -> None:
    """
    Compute peak biases (GC content and/or CpG density) for background sampling.
    
    This function adds bias columns to `adata.var` that are used by `sample_bg_peaks`
    to match foreground and background peaks.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object with peak matrix. Peak names in `adata.var_names` should be
        in "chr:start-end" format.
    genome_fasta : str or Path
        Path to genome FASTA file (supports .fa, .fasta, .fa.gz, .fasta.gz)
    add_gc_content : bool, default True
        Whether to compute GC content
    add_cpg_density : bool, default False
        Whether to compute CpG density
        
    Returns
    -------
    None
        Adds columns to `adata.var`:
        - "gc_content": GC content (if add_gc_content=True)
        - "cpg_density": CpG density (if add_cpg_density=True)
        
    Examples
    --------
    >>> import gatac as ga
    >>> ga.tl.compute_peak_bias(peak_adata, "genome.fa")
    >>> peak_adata.var["gc_content"]  # GC content per peak
    """
    logger.info(f"Computing peak biases from {genome_fasta}...")

    peak_regions = list(adata.var_names)
    n_peaks = len(peak_regions)

    # GC content: GPU-accelerated batch computation
    if add_gc_content:
        adata.var["gc_content"] = _compute_gc_content_gpu(peak_regions, genome_fasta)
        logger.info("Added 'gc_content' to adata.var")

    # CpG density: sequential (requires per-sequence string scan)
    if add_cpg_density:
        cpg_density = np.zeros(n_peaks, dtype=np.float32)
        with _open_fasta(genome_fasta) as genome:
            for i, region in enumerate(tqdm(peak_regions, desc="Computing CpG density")):
                try:
                    chrom, coords = region.split(":")
                    start, end = int(coords.split("-")[0]), int(coords.split("-")[1])
                    seq = str(genome[chrom][start:end].seq).upper()
                    cpg_density[i] = seq.count("CG") / len(seq) if len(seq) > 0 else 0.0
                except Exception as e:
                    logger.warning(f"Failed to process {region}: {e}")
        adata.var["cpg_density"] = cpg_density
        logger.info("Added 'cpg_density' to adata.var")


# =============================================================================
# Background Peak Sampling
# =============================================================================


def _chromvar_binning(
    trans_norm_mat: np.ndarray,
    *,
    bs: int = 50,
    w: float = 0.1,
    niterations: int = 50,
) -> np.ndarray:
    """
    Original chromVAR binning-based background sampling.
    
    Translated from the chromVAR R package. Creates bins based on bias features
    and samples background peaks from bins with similar properties.
    
    Parameters
    ----------
    trans_norm_mat : np.ndarray
        Normalized and transformed bias matrix (n_peaks, n_features)
    bs : int, default 50
        Bin size for creating bins
    w : float, default 0.1
        Width parameter for Gaussian kernel density estimation
    niterations : int, default 50
        Number of background peaks to sample per peak
        
    Returns
    -------
    np.ndarray
        Background peak indices of shape (n_peaks, niterations)
    """
    from scipy.spatial.distance import cdist
    from scipy.stats import norm
    
    # Create bins
    bins1 = np.linspace(
        np.min(trans_norm_mat[:, 0]), np.max(trans_norm_mat[:, 0]), bs
    )
    bins2 = np.linspace(
        np.min(trans_norm_mat[:, 1]), np.max(trans_norm_mat[:, 1]), bs
    )

    # Create bin_data
    bin_data = np.array(np.meshgrid(bins1, bins2)).T.reshape(-1, 2)

    # Calculate Euclidean distances between bins
    bin_dist = cdist(bin_data, bin_data, "euclidean")

    # Calculate probabilities using Gaussian kernel
    bin_p = norm.pdf(bin_dist, 0, w)
    
    # Find nearest bin for each peak using cuML (GPU)
    try:
        from cuml.neighbors import NearestNeighbors
        
        logger.info("Finding nearest bins using cuML...")
        knn = NearestNeighbors(n_neighbors=1, metric="euclidean")
        knn.fit(bin_data)
        distances, indices = knn.kneighbors(trans_norm_mat)
        bin_membership = indices.flatten()
    except ImportError:
        logger.warning("cuML not available, falling back to scipy for binning")
        from scipy.spatial.distance import cdist
        distance = cdist(trans_norm_mat, bin_data)
        bin_membership = np.argmin(distance, axis=1)
    
    # Calculate bin density
    unique, counts = np.unique(bin_membership, return_counts=True)
    bin_density = np.zeros(bs**2)
    bin_density[unique] = counts

    # Sample background peaks
    background_peaks = _bg_sample_helper(
        bin_membership, bin_p, bin_density, niterations
    )

    return background_peaks


def _bg_sample_helper(
    bin_membership: np.ndarray,
    bin_p: np.ndarray,
    bin_density: np.ndarray,
    niterations: int,
) -> np.ndarray:
    """
    Helper function for chromVAR-style background sampling.
    
    For each bin, samples background peaks according to bin probabilities.
    
    Parameters
    ----------
    bin_membership : np.ndarray
        Bin assignment for each peak
    bin_p : np.ndarray
        Probability matrix between bins (n_bins, n_bins)
    bin_density : np.ndarray
        Number of peaks in each bin
    niterations : int
        Number of background peaks to sample
        
    Returns
    -------
    np.ndarray
        Background peak indices of shape (n_peaks, niterations)
    """
    n = len(bin_membership)
    out = np.zeros((n, niterations), dtype=np.int32)

    for i in trange(len(bin_density), desc="Sampling background peaks"):
        ix = np.where(bin_membership == i)[0]
        if len(ix) == 0:  # Skip if no members in bin
            continue
        p_tmp = bin_p[i, :]
        p = (p_tmp / bin_density)[bin_membership]
        p /= p.sum()
        # Sampling with replacement according to probabilities
        sampled_indices = np.random.choice(
            np.arange(len(p)), size=niterations * len(ix), replace=True, p=p
        )
        out[ix, :] = sampled_indices.reshape((len(ix), niterations))

    return out


def sample_bg_peaks(
    adata: AnnData,
    *,
    method: Literal["knn", "chromvar"] = "knn",
    n_iterations: int = 50,
    bg_columns: list[str] = ["gc_content", "reads_per_peak"],
    genome_fasta: Optional[Union[str, Path]] = None,
    n_neighbors: int = 50,
    bs: int = 50,
    w: float = 0.1,
) -> None:
    """
    Sample background peaks for chromVAR analysis.
    
    This function matches foreground peaks with background peaks that have similar
    biases (e.g., GC content and accessibility). Two methods are supported:
    
    1. **"knn"** (default): GPU-accelerated k-NN using cuML. Faster and recommended.
    2. **"chromvar"**: Original chromVAR binning method. Slower but faithful to R package.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object with peak matrix. Must have bias columns in `adata.var`
        (e.g., from `compute_peak_bias`).
    method : {"knn", "chromvar"}, default "knn"
        Background sampling method:
        - "knn": cuML nearest neighbors (GPU, faster)
        - "chromvar": Original chromVAR binning (CPU, slower)
    n_iterations : int, default 50
        Number of background peaks to sample per peak
    bg_columns : list[str], default ["gc_content", "reads_per_peak"]
        Columns in `adata.var` to use for bias matching. Any column listed here
        that is absent from `adata.var` will be computed automatically when
        `genome_fasta` is provided.
    genome_fasta : str or Path, optional
        Path to genome FASTA file. Required when `bg_columns` contains
        "gc_content" and it has not been precomputed.
    n_neighbors : int, default 50
        Number of neighbors for k-NN method (only used if method="knn")
    bs : int, default 50
        Bin size for chromVAR method (only used if method="chromvar")
    w : float, default 0.1
        Gaussian kernel width for chromVAR method (only used if method="chromvar")

    Returns
    -------
    None
        Adds `adata.varm["bg_peaks"]` with shape (n_peaks, n_iterations) containing
        background peak indices for each peak.

    Examples
    --------
    >>> import gatac as ga
    >>> # Option A: precompute biases separately
    >>> ga.tl.compute_peak_bias(peak_adata, "genome.fa")
    >>> ga.tl.sample_bg_peaks(peak_adata, method="knn")
    >>>
    >>> # Option B: let sample_bg_peaks compute gc_content on the fly
    >>> ga.tl.sample_bg_peaks(peak_adata, method="knn", genome_fasta="genome.fa")
    >>> peak_adata.varm["bg_peaks"]  # Background indices
    """
    # Compute reads per peak (log10 transformed)
    reads_per_peak = np.asarray(adata.X.sum(axis=0)).flatten()
    if np.min(reads_per_peak) <= 0:
        raise ValueError("Some peaks have no reads. Filter peaks before sampling.")
    reads_per_peak = np.log10(reads_per_peak)
    adata.var["reads_per_peak"] = reads_per_peak

    # Auto-compute missing bias columns that can be derived from the genome
    if "gc_content" in bg_columns and "gc_content" not in adata.var.columns:
        if genome_fasta is None:
            raise ValueError(
                "'gc_content' is not in adata.var and no genome_fasta was provided. "
                "Either run ga.tl.compute_peak_bias(adata, genome_fasta) first, "
                "or pass genome_fasta to sample_bg_peaks()."
            )
        logger.info("'gc_content' not found in adata.var — computing automatically...")
        adata.var["gc_content"] = _compute_gc_content_gpu(
            list(adata.var_names), genome_fasta
        )

    # Prepare bias matrix
    if len(bg_columns) > 0:
        bias_matrix = np.asarray(adata.var[bg_columns].values, dtype=np.float64)
        trans_norm_mat = normalize_bias_matrix(bias_matrix)
    else:
        trans_norm_mat = reads_per_peak.reshape(-1, 1)
    
    logger.info(f"Sampling background peaks using method '{method}'...")
    
    if method == "knn":
        # GPU-accelerated k-NN using cuML
        try:
            logger.info("Using cuML for k-NN background sampling...")
            knn_idx = compute_bias_knn_indices(
                trans_norm_mat,
                n_neighbors=n_iterations,
                exclude_self=True,
            )
        except ImportError:
            raise ImportError(
                "cuML is required for method='knn'. Install via: "
                "pip install cuml-cu12 (or appropriate CUDA version). "
                "Alternatively, use method='chromvar'."
            )
            
    elif method == "chromvar":
        # Original chromVAR binning method
        if trans_norm_mat.shape[1] != 2:
            raise ValueError(
                "chromVAR method requires exactly 2 bias features. "
                f"Got {trans_norm_mat.shape[1]}. Use method='knn' for other cases."
            )
        knn_idx = _chromvar_binning(
            trans_norm_mat, bs=bs, w=w, niterations=n_iterations
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Store in adata
    adata.varm["bg_peaks"] = knn_idx.astype(np.int32)
    logger.info(
        f"Sampled {n_iterations} background peaks for {adata.n_vars} peaks. "
        f"Stored in adata.varm['bg_peaks']"
    )


# =============================================================================
# Motif Scanning
# =============================================================================


def scan_motifs(
    adata: AnnData,
    motifs: list[DNAMotif],
    genome_fasta: Union[str, Path],
    *,
    pvalue: float = 5e-5,
    check_rc: bool = True,
    bg: Union[str, tuple[float, float, float, float]] = "subject",
    mode: Literal["gatac", "motifmatchr"] = "gatac",
    key_added: str = "motif_match",
    peak_batch_size: int = 50000,
    coordinate_system: Literal["0-based", "1-based"] = "0-based",
) -> None:
    """
    Scan peaks for motif matches and create a sparse motif match matrix.
    
    This function wraps the existing GPU motif scanning infrastructure from
    `gatac.tl.motif` and produces a boolean matrix indicating which peaks
    contain each motif.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object with peak matrix. Peak names in `adata.var_names` should be
        in "chr:start-end" format.
    motifs : list[DNAMotif]
        List of motifs to scan (from `read_motifs` or `parse_meme`)
    genome_fasta : str or Path
        Path to genome FASTA file (supports .fa, .fasta, .fa.gz, .fasta.gz)
    pvalue : float, default 5e-5
        P-value threshold for motif matching (matches R/motifmatchr default)
    check_rc : bool, default True
        Whether to check both strands (forward and reverse complement)
    bg : str or tuple, default "subject"
        Background nucleotide probabilities ``(A, C, G, T)``. Use
        ``"subject"`` to compute them from extracted peak sequences, which
        matches motifmatchr's default. Use ``"even"`` for a uniform
        background ``(0.25, 0.25, 0.25, 0.25)``, or pass a custom 4-tuple.
    key_added : str, default "motif_match"
        Key to store motif match matrix in `adata.varm`
    peak_batch_size : int, default 50000
        Number of peaks to process at once on GPU. Reduce if running out of
        GPU memory.
    coordinate_system : {"0-based", "1-based"}, default "0-based"
        Coordinate system of peak names in ``adata.var_names``. ``"0-based"``
        is BED-style half-open indexing ``[start, end)`` and is what GATAC
        peak callers produce, so sequences are extracted as
        ``genome[start:end]``. ``"1-based"`` is the closed interval format
        used by R/GenomicRanges and chromVAR, so sequences are extracted as
        ``genome[start-1:end]``.
    mode : {"gatac", "motifmatchr"}, default "gatac"
        Motif scoring mode. ``"gatac"`` uses the standard natural-log odds
        ``ln(p / bg)``. ``"motifmatchr"`` reproduces motifmatchr/scPrinter
        scoring with ``log2(p / 0.25) - (log2(0.25) - log2(bg))``.
        
    Returns
    -------
    None
        Adds to `adata.varm[key_added]` a sparse boolean matrix of shape
        (n_peaks, n_motifs) and stores motif names in `adata.uns["motif_name"]`.
        
    Examples
    --------
    >>> import gatac as ga
    >>> motifs = ga.tl.read_motifs("motifs.meme")
    >>> 
    >>> # For GATAC-generated peaks (0-based BED format)
    >>> ga.tl.scan_motifs(peak_adata, motifs, "genome.fa")
    >>> 
    >>> # For R/chromVAR peaks (1-based GenomicRanges format)
    >>> ga.tl.scan_motifs(
    ...     peak_adata, motifs, "genome.fa",
    ...     coordinate_system="1-based"
    ... )
    """
    from .motif import (
        _compute_score_threshold,
        _reverse_complement_encoded,
        _scan_motifs_batch_gpu,
    )
    from tqdm.auto import tqdm
    
    logger.info(f"Scanning {len(motifs)} motifs in {adata.n_vars} peaks...")
    logger.info(f"Using {coordinate_system} coordinate system")
    
    peak_regions = list(adata.var_names)
    n_peaks = len(peak_regions)
    n_motifs = len(motifs)
    
    # Fetch all sequences first (on CPU)
    logger.info("Fetching sequences from genome...")
    with _open_fasta(genome_fasta) as genome:
        sequences = []
        for region in tqdm(peak_regions, desc="Fetching sequences"):
            try:
                chrom, coords = region.split(":")
                start, end = coords.split("-")
                start, end = int(start), int(end)
                
                # Handle coordinate system conversion
                if coordinate_system == "0-based":
                    # BED format: 0-based, half-open [start, end)
                    # pyfaidx uses 0-based indexing, so use directly
                    seq = str(genome[chrom][start:end].seq)
                elif coordinate_system == "1-based":
                    # R/GenomicRanges: 1-based, closed [start, end]
                    # Convert to 0-based for pyfaidx
                    seq = str(genome[chrom][start-1:end].seq)
                else:
                    raise ValueError(f"Unknown coordinate_system: {coordinate_system}")
                
                sequences.append(seq)
            except Exception as e:
                logger.warning(f"Failed to fetch sequence for {region}: {e}")
                sequences.append("")
    
    # Resolve background probabilities
    if isinstance(bg, str):
        if bg == "even":
            bg_probs = (0.25, 0.25, 0.25, 0.25)
        elif bg == "subject":
            # Compute bg from extracted sequences (matches R/motifmatchr default)
            counts = np.zeros(4, dtype=np.int64)
            for seq in sequences:
                seq_upper = seq.upper()
                counts[0] += seq_upper.count('A')
                counts[1] += seq_upper.count('C')
                counts[2] += seq_upper.count('G')
                counts[3] += seq_upper.count('T')
            total = counts.sum()
            if total > 0:
                bg_probs = tuple(counts / total)
            else:
                bg_probs = (0.25, 0.25, 0.25, 0.25)
            logger.info(f"Background from sequences: A={bg_probs[0]:.4f} C={bg_probs[1]:.4f} G={bg_probs[2]:.4f} T={bg_probs[3]:.4f}")
        else:
            raise ValueError(f"Unknown bg mode: {bg}. Use 'subject', 'even', or a tuple of 4 floats.")
    else:
        bg_probs = bg

    # Precompute thresholds and PWM list (these stay on CPU)
    logger.info("Computing score thresholds...")
    bg_probs_np = np.array(bg_probs, dtype=np.float64)
    pwm_list = []
    thresholds = []
    for motif in tqdm(motifs, desc="Computing thresholds"):
        pwm_log_odds = motif.to_log_odds(bg_probs, mode=mode)
        threshold = _compute_score_threshold(pwm_log_odds, bg_probs_np, pvalue)
        pwm_list.append(pwm_log_odds)
        thresholds.append(threshold)
    
    thresholds = np.array(thresholds, dtype=np.float32)
    
    # Process peaks in batches to avoid GPU OOM
    n_batches = (n_peaks + peak_batch_size - 1) // peak_batch_size
    logger.info(f"Scanning motifs on GPU in {n_batches} batch(es) of {peak_batch_size} peaks...")
    
    all_matches = []
    
    for batch_idx in tqdm(range(n_batches), desc="Peak batches"):
        batch_start = batch_idx * peak_batch_size
        batch_end = min(batch_start + peak_batch_size, n_peaks)
        batch_seqs = sequences[batch_start:batch_end]
        
        # Encode batch sequences
        encoded_seqs, seq_lengths = _encode_sequences_batch(batch_seqs)
        
        if check_rc:
            rc_seqs = _reverse_complement_encoded(encoded_seqs)
        else:
            rc_seqs = None
        
        # Scan this batch
        batch_matches = _scan_motifs_batch_gpu(
            encoded_seqs,
            seq_lengths,
            pwm_list,
            thresholds,
            rc_seqs=rc_seqs,
            motif_batch_size=16,
            show_progress=(n_batches == 1),  # Only show inner progress if single batch
        )
        
        # batch_matches has shape (n_motifs, batch_size) - transpose for peaks × motifs
        all_matches.append(batch_matches.T)  # Now (batch_size, n_motifs)
        
        # Free GPU memory
        del encoded_seqs, seq_lengths, rc_seqs
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
    
    # Concatenate all batches
    motif_matches_array = np.vstack(all_matches)  # (n_peaks, n_motifs)
    
    # Convert to sparse matrix
    motif_match_matrix = sp.csr_matrix(motif_matches_array, dtype=bool)
    
    # Store in adata
    adata.varm[key_added] = motif_match_matrix
    adata.uns["motif_name"] = np.array(
        [m.name if m.name is not None else m.id for m in motifs]
    )
    
    logger.info(
        f"Found {motif_match_matrix.nnz} motif matches "
        f"({100 * motif_match_matrix.nnz / (n_peaks * n_motifs):.2f}% density). "
        f"Stored in adata.varm['{key_added}']"
    )
    
    # Cleanup GPU memory
    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()


# =============================================================================
# Custom CUDA Kernels for chromVAR
# =============================================================================
# 
# Memory analysis (compared to original CPU Welford implementation):
# 
# ORIGINAL per cell batch:
#   - bg_mean, bg_m2: 2 × (n_cells_chunk × n_motifs_chunk) × 8 bytes on CPU
#   - bg_dev_iter: (n_cells_chunk × n_motifs_chunk) × 4 bytes on GPU
#   - GPU→CPU transfer: n_bg_peaks × (n_cells_chunk × n_motifs_chunk) × 4 bytes
# 
# OPTIMIZED per cell batch:
#   - bg_mean_gpu, bg_m2_gpu: 2 × (n_cells_chunk × n_motifs_chunk) × 8 bytes on GPU
#   - bg_dev_iter: (n_cells_chunk × n_motifs_chunk) × 4 bytes on GPU (reused)
#   - bg_peaks_gpu: (n_peaks × n_bg_peaks) × 4 bytes on GPU (one-time transfer)
#   - GPU→CPU transfer: 3 × (n_cells_chunk × n_motifs_chunk) × 4 bytes (final only)
# 
# Net GPU memory delta: +16 bytes per element for Welford accumulators (float64)
#                       +bg_peaks array (typically small: ~peaks × 50 × 4 bytes)
# Net transfer reduction: ~50× fewer GPU→CPU transfers (n_bg_peaks → 1)
#
# The additional GPU memory for Welford accumulators is minimal compared to
# the count matrix and motif match matrix already on GPU.
# =============================================================================

# Fused deviation kernel: computes (observed - expected) / expected in one pass
# Memory: reads observed, var_match, exp_obs; writes to out (no intermediates)
_fused_deviation_kernel = cp.ElementwiseKernel(
    'float32 observed, float32 var_match, float32 exp_obs',
    'float32 out',
    '''
    float expected = exp_obs * var_match;
    out = (expected != 0.0f) ? (observed - expected) / expected : 0.0f;
    ''',
    'fused_deviation_kernel'
)

# Welford online update kernel: updates mean and M2 in-place on GPU
# Memory: same as CPU version but avoids GPU→CPU transfer
# Note: CUDA uses 'double' not 'float64'
_welford_update_kernel = cp.ElementwiseKernel(
    'float32 new_value, float64 count',
    'float64 mean, float64 m2',
    '''
    double delta = (double)new_value - mean;
    mean += delta / count;
    double delta2 = (double)new_value - mean;
    m2 += delta * delta2;
    ''',
    'welford_update_kernel'
)

# Z-score normalization kernel: (obs - mean) / std with NaN handling
# Memory: reads obs_dev, mean, std; writes to out (in-place capable)
_zscore_kernel = cp.ElementwiseKernel(
    'float32 obs_dev, float32 bg_mean, float32 bg_std',
    'float32 out',
    '''
    if (bg_std > 0.0f) {
        out = (obs_dev - bg_mean) / bg_std;
    } else {
        out = 0.0f;
    }
    ''',
    'zscore_kernel'
)

# Batched Welford update kernel: processes multiple samples at once
# Updates mean and m2 for a batch of new values using parallel algorithm
_welford_batch_finalize_kernel = cp.ElementwiseKernel(
    'float64 batch_mean, float64 batch_m2, float64 batch_count, float64 total_count',
    'float64 mean, float64 m2',
    '''
    // Parallel algorithm for combining Welford statistics
    // See: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    double delta = batch_mean - mean;
    double new_count = total_count;
    double old_count = new_count - batch_count;
    mean = (old_count * mean + batch_count * batch_mean) / new_count;
    m2 = m2 + batch_m2 + delta * delta * old_count * batch_count / new_count;
    ''',
    'welford_batch_finalize_kernel'
)


# =============================================================================
# chromVAR Deviation Computation
# =============================================================================


def _compute_deviations(
    motif_match: cupy_sparse.csr_matrix,
    count: cupy_sparse.csr_matrix,
    expectation_obs: cp.ndarray,
    expectation_var: cp.ndarray,
) -> cp.ndarray:
    """
    Compute deviation scores for a chunk of cells.
    
    Parameters
    ----------
    motif_match : cupyx.scipy.sparse.csr_matrix
        Motif match matrix on GPU (n_peaks, n_motifs)
    count : cupyx.scipy.sparse.csr_matrix
        Count matrix on GPU (n_cells, n_peaks)
    expectation_obs : cp.ndarray
        Expected total reads per cell (n_cells, 1)
    expectation_var : cp.ndarray
        Expected accessibility per peak (1, n_peaks)
        
    Returns
    -------
    cp.ndarray
        Deviation scores (n_cells, n_motifs)
    """
    # Observed counts: cells × motifs (sparse @ sparse = dense for small result)
    observed = count @ motif_match
    
    # Expected counts: (n_cells, 1) @ ((1, n_peaks) @ (n_peaks, n_motifs))
    # Compute inner product first to keep operations sparse
    var_match = expectation_var @ motif_match  # (1, n_motifs) - small, can be dense
    
    # Convert to dense only if needed
    if cupy_sparse.issparse(observed):
        observed = observed.toarray()
    if cupy_sparse.issparse(var_match):
        var_match = var_match.toarray()
    
    # Use fused kernel to compute deviation in one pass
    # This avoids creating intermediate 'expected' array
    # Broadcast var_match (1, n_motifs) across cells
    out = _fused_deviation_kernel(
        observed.astype(cp.float32),
        var_match.astype(cp.float32),  # broadcasts to (n_cells, n_motifs)
        expectation_obs.astype(cp.float32),  # broadcasts to (n_cells, n_motifs)
    )
    
    return out


def compute_deviations(
    adata: AnnData,
    *,
    batch_size: int = 5000,
    motif_batch_size: int = -1,
    key_added: str = "chromvar",
    return_adata: bool = False,
) -> Optional[AnnData]:
    """
    Compute chromVAR TF deviation scores.

    Computes per-cell, per-motif deviation scores normalized by background
    expectation. Requires prior setup:

    1. `sample_bg_peaks()` to generate `adata.varm["bg_peaks"]`
    2. `scan_motifs()` to generate `adata.varm["motif_match"]`

    The algorithm:
    - For each cell, computes observed motif accessibility
    - Computes expected accessibility based on overall peak accessibility and cell depth
    - For background peaks, computes deviation
    - Z-score normalizes: (observed_dev - mean_bg_dev) / std_bg_dev

    Parameters
    ----------
    adata : AnnData
        AnnData object with peak matrix (cells × peaks). Must have:
        - `adata.varm["bg_peaks"]`: Background peak indices from `sample_bg_peaks()`
        - `adata.varm["motif_match"]`: Motif match matrix from `scan_motifs()`
        - `adata.uns["motif_name"]`: Motif names from `scan_motifs()`
    batch_size : int, default 5000
        Number of cells to process at once. Reduce if GPU memory is limited.
    motif_batch_size : int, default -1
        Number of motifs to process at once. If -1, uses default of 100 motifs
        to balance memory usage and speed. Reduce further for very large datasets.
    key_added : str, default "chromvar"
        Key under which the deviation DataFrame is stored in `adata.obsm`.
    return_adata : bool, default False
        If True, also return a new AnnData with deviations as `.X`.

    Returns
    -------
    None or AnnData
        Always stores deviations as a DataFrame in `adata.obsm[key_added]`.
        Returns an AnnData (cells × motifs) only when `return_adata=True`.

    Examples
    --------
    >>> import gatac as ga
    >>>
    >>> # 1. Create peak matrix
    >>> peak_adata = ga.tl.make_peak_matrix(tile_adata, parquet_path)
    >>>
    >>> # 2. Compute biases and sample background
    >>> ga.tl.compute_peak_bias(peak_adata, "genome.fa")
    >>> ga.tl.sample_bg_peaks(peak_adata)
    >>>
    >>> # 3. Scan motifs
    >>> motifs = ga.tl.read_motifs("motifs.meme")
    >>> ga.tl.scan_motifs(peak_adata, motifs, "genome.fa")
    >>>
    >>> # 4. Compute deviations (stored in peak_adata.obsm["chromvar"])
    >>> ga.tl.compute_deviations(peak_adata)
    >>> peak_adata.obsm["chromvar"]  # DataFrame (cells × motifs)
    """
    # Validate inputs
    if "bg_peaks" not in adata.varm:
        raise ValueError(
            "Cannot find 'bg_peaks' in adata.varm. "
            "Please run sample_bg_peaks() first."
        )
    
    if "motif_match" not in adata.varm:
        raise ValueError(
            "Cannot find 'motif_match' in adata.varm. "
            "Please run scan_motifs() first."
        )
    
    if "motif_name" not in adata.uns:
        raise ValueError(
            "Cannot find 'motif_name' in adata.uns. "
            "Please run scan_motifs() first."
        )
    
    logger.info("Computing chromVAR deviations on GPU...")
    
    # Compute expectations
    logger.info("Computing expectation reads per cell and peak...")
    expectation_var = cp.asarray(
        adata.X.sum(0), dtype=cp.float32
    ).reshape((1, adata.X.shape[1]))
    expectation_var /= expectation_var.sum()
    
    expectation_obs = np.asarray(
        adata.X.sum(1), dtype=np.float32
    ).reshape((adata.X.shape[0], 1))
    
    # Get motif match matrix
    motif_match = adata.varm["motif_match"]
    if motif_batch_size < 0:
        # Default to processing 100 motifs at a time to reduce memory
        motif_batch_size = min(100, motif_match.shape[1])
    
    dev_all = []
    
    for motif_start in tqdm(
        list(range(0, motif_match.shape[1], motif_batch_size)),
        desc="Processing motif chunks",
    ):
        motif_end = min(motif_start + motif_batch_size, motif_match.shape[1])
        motif_match_chunk = motif_match[:, motif_start:motif_end]
        n_motifs_chunk = motif_match_chunk.shape[1]
        
        # Convert to CuPy sparse
        if sp.issparse(motif_match_chunk):
            motif_match_gpu = scipy_to_cupy_sparse(motif_match_chunk.tocsr())
        else:
            motif_match_gpu = cp.asarray(motif_match_chunk, dtype=cp.float32)
        
        # Initialize deviation arrays
        obs_dev = np.zeros((adata.n_obs, n_motifs_chunk), dtype=np.float32)
        mean_bg_dev = np.zeros_like(obs_dev)
        std_bg_dev = np.zeros_like(obs_dev)
        n_bg_peaks = adata.varm["bg_peaks"].shape[1]
        
        # Convert motif_match to dense for efficient background indexing
        # This avoids expensive sparse fancy indexing that causes memory spikes
        # Memory: n_peaks × n_motifs_chunk × 4 bytes (e.g., 100k × 100 × 4 = 40MB)
        if cupy_sparse.issparse(motif_match_gpu):
            motif_match_dense = motif_match_gpu.toarray().astype(cp.float32)
        else:
            motif_match_dense = motif_match_gpu.astype(cp.float32)
        
        # Pre-compute var_match once (used for all backgrounds with reindexing)
        # Shape: (1, n_peaks) @ (n_peaks, n_motifs) -> (1, n_motifs)
        var_match_all = (expectation_var @ motif_match_dense).astype(cp.float32)
        
        # Pre-transfer bg_peaks to GPU once (small: n_peaks × n_bg_peaks int32)
        bg_peaks_gpu = cp.asarray(adata.varm["bg_peaks"])
        
        # Process cells in batches
        for cell_start in tqdm(
            range(0, adata.n_obs, batch_size),
            desc="Processing cell chunks",
            leave=False,
        ):
            cell_end = min(cell_start + batch_size, adata.n_obs)
            n_cells_chunk = cell_end - cell_start
            
            # Get count matrix chunk
            X_chunk = adata.X[cell_start:cell_end]
            expectation_obs_chunk = cp.asarray(expectation_obs[cell_start:cell_end])
            
            # Convert to CuPy sparse
            if sp.issparse(X_chunk):
                X_chunk_gpu = scipy_to_cupy_sparse(X_chunk.tocsr())
            else:
                X_chunk_gpu = cp.asarray(X_chunk, dtype=cp.float32)
            
            # Compute observed deviation (stays on GPU)
            obs_dev_chunk = _compute_deviations(
                motif_match_gpu,
                X_chunk_gpu,
                expectation_obs_chunk,
                expectation_var,
            )
            
            # GPU-side Welford accumulation for background deviations
            # Memory: same as CPU version (2 arrays of shape n_cells_chunk × n_motifs_chunk)
            # but avoids n_bg_peaks GPU→CPU transfers per cell batch
            bg_mean_gpu = cp.zeros((n_cells_chunk, n_motifs_chunk), dtype=cp.float64)
            bg_m2_gpu = cp.zeros((n_cells_chunk, n_motifs_chunk), dtype=cp.float64)
            
            # Process all background iterations
            for bg_iter in range(n_bg_peaks):
                bg_peak_idx = bg_peaks_gpu[:, bg_iter]
                
                # Use dense indexing (much more memory efficient than sparse fancy indexing)
                bg_motif_match = motif_match_dense[bg_peak_idx, :]
                
                # Compute observed: X @ bg_motif_match
                observed = X_chunk_gpu @ bg_motif_match
                
                # Compute var_match for background
                exp_var_reindexed = expectation_var.flatten()[bg_peak_idx]
                var_match_bg = (exp_var_reindexed @ bg_motif_match).reshape(1, -1)
                
                # Fused deviation computation
                bg_dev_iter = _fused_deviation_kernel(
                    observed.astype(cp.float32),
                    var_match_bg.astype(cp.float32),
                    expectation_obs_chunk.astype(cp.float32),
                )
                
                # Welford update on GPU
                _welford_update_kernel(
                    bg_dev_iter,
                    cp.float64(bg_iter + 1),
                    bg_mean_gpu,
                    bg_m2_gpu,
                )
            
            # Transfer final statistics to CPU (only once per cell batch)
            obs_dev[cell_start:cell_end, :] = cp.asnumpy(obs_dev_chunk)
            mean_bg_dev[cell_start:cell_end, :] = cp.asnumpy(bg_mean_gpu).astype(np.float32)
            bg_var_gpu = bg_m2_gpu / n_bg_peaks
            std_bg_dev[cell_start:cell_end, :] = cp.asnumpy(cp.sqrt(bg_var_gpu)).astype(np.float32)
            
            # Cleanup
            del X_chunk_gpu, obs_dev_chunk, bg_mean_gpu, bg_m2_gpu, bg_var_gpu
            mempool.free_all_blocks()
            pinned_mempool.free_all_blocks()
        
        # Cleanup dense motif matrix and bg_peaks
        del motif_match_dense, var_match_all, bg_peaks_gpu
        
        # Z-score normalization using fused GPU kernel
        # Transfer back to GPU for final computation, then back to CPU
        obs_dev_gpu = cp.asarray(obs_dev)
        mean_gpu = cp.asarray(mean_bg_dev)
        std_gpu = cp.asarray(std_bg_dev)
        dev = cp.asnumpy(_zscore_kernel(obs_dev_gpu, mean_gpu, std_gpu))
        del obs_dev_gpu, mean_gpu, std_gpu
        
        dev_all.append(dev)
        
        # Cleanup
        del motif_match_gpu
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
    
    # Concatenate all motif chunks
    dev = np.concatenate(dev_all, axis=1) if len(dev_all) > 1 else dev_all[0]

    import pandas as pd

    motif_names = adata.uns["motif_name"]
    adata.obsm[key_added] = pd.DataFrame(
        dev, index=adata.obs_names, columns=motif_names
    )

    logger.info(
        f"Computed chromVAR deviations: {dev.shape}. "
        f"Stored in adata.obsm['{key_added}']"
    )

    if return_adata:
        dev_adata = AnnData(dev, dtype=np.float32, obs=adata.obs.copy())
        dev_adata.var_names = motif_names
        return dev_adata
    return None


# =============================================================================
# Full chromVAR Pipeline
# =============================================================================


def chromvar(
    adata: AnnData,
    genome_fasta: Union[str, Path],
    motifs_path: Union[str, Path],
    *,
    method: Literal["knn", "chromvar"] = "chromvar",
    n_iterations: int = 50,
    pvalue: float = 5e-5,
    check_rc: bool = True,
    bg: Union[str, tuple] = "subject",
    coordinate_system: Literal["0-based", "1-based"] = "0-based",
    batch_size: int = 5000,
    motif_batch_size: int = -1,
    key_added: str = "chromvar",
    return_adata: bool = False,
) -> Optional[AnnData]:
    """
    Run the full chromVAR pipeline in a single call.

    Executes the following steps in order:

    1. `compute_peak_bias` — GC content from genome FASTA
    2. `sample_bg_peaks` — background peak sampling
    3. `read_motifs` + `scan_motifs` — motif matching
    4. `compute_deviations` — TF deviation scores

    Parameters
    ----------
    adata : AnnData
        Peak-level AnnData (cells × peaks).
    genome_fasta : str or Path
        Path to genome FASTA file.
    motifs_path : str or Path
        Path to motif file in MEME format.
    method : {"knn", "chromvar"}, default "chromvar"
        Background sampling method passed to `sample_bg_peaks`.
    n_iterations : int, default 50
        Number of background peaks to sample per peak.
    pvalue : float, default 5e-5
        P-value threshold for motif matching.
    check_rc : bool, default True
        Whether to scan both strands.
    bg : str or tuple, default "subject"
        Background nucleotide probabilities for motif scoring.
    coordinate_system : {"0-based", "1-based"}, default "0-based"
        Coordinate system of peak names in `adata.var_names`.
    batch_size : int, default 5000
        Number of cells per GPU batch in `compute_deviations`.
    motif_batch_size : int, default -1
        Number of motifs per chunk in `compute_deviations`.
    key_added : str, default "chromvar"
        Key under which the deviation DataFrame is stored in `adata.obsm`.
    return_adata : bool, default False
        If True, also return a new AnnData with deviations as `.X`.

    Returns
    -------
    None or AnnData
        Always stores deviations as a DataFrame in `adata.obsm[key_added]`.
        Returns an AnnData (cells × motifs) only when `return_adata=True`.

    Examples
    --------
    >>> import gatac as ga
    >>> ga.tl.chromvar(
    ...     peak_adata,
    ...     "../resources/GRCh38.p13.genome.fa",
    ...     "../resources/cisBP_human.meme",
    ... )
    >>> peak_adata.obsm["chromvar"]  # DataFrame (cells × motifs)
    """
    from .motif import read_motifs

    logger.info("=== chromVAR pipeline ===")

    logger.info("Step 1/4: Computing peak biases...")
    compute_peak_bias(adata, genome_fasta)

    logger.info("Step 2/4: Sampling background peaks...")
    sample_bg_peaks(adata, method=method, n_iterations=n_iterations)

    logger.info("Step 3/4: Reading motifs and scanning peaks...")
    motifs = read_motifs(motifs_path)
    scan_motifs(
        adata,
        motifs,
        genome_fasta,
        pvalue=pvalue,
        check_rc=check_rc,
        bg=bg,
        coordinate_system=coordinate_system,
    )

    logger.info("Step 4/4: Computing deviations...")
    return compute_deviations(
        adata,
        batch_size=batch_size,
        motif_batch_size=motif_batch_size,
        key_added=key_added,
        return_adata=return_adata,
    )
