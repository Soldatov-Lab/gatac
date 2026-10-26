"""
Peak calling, merging and the cell × peak matrix.

``call`` runs MACS3's narrow-peak caller per group of cells on the GPU (see
``_macs``), with SnapATAC2's settings by default; ``merge`` and
``make_matrix`` follow SnapATAC2's ``merge_peaks`` and peak counting.
"""
from __future__ import annotations

import gc
import logging
from collections import defaultdict
from math import log
from pathlib import Path
from typing import Literal, Optional, Union

import cudf
import cupy as cp
import numpy as np
import pandas as pd
from numba import njit, prange

from .._readers import read_bed
from ..io.fragments import FragmentFile
from ..pp.genome import get_chrom_sizes
from ..pp.gene import optimize_sparse_matrix_dtype
from . import _macs

logger = logging.getLogger(__name__)

# GPU memory pool references
mempool = cp.get_default_memory_pool()
pinned_mempool = cp.get_default_pinned_memory_pool()


# =============================================================================
# Reading the fragments of the groups
# =============================================================================

# Host memory a group's fragments take while read: uint32 starts and ends
_HOST_BYTES_PER_FRAGMENT = 8


def _source_parquet_name(source_file: str) -> str:
    """
    The fragment file of a ``source_file`` value: a Parquet file name, as
    tile matrices of several files record it, is kept; an h5ad name maps
    ``{sample}_tile.h5ad`` or ``{sample}.h5ad`` to ``{sample}.parquet``.
    """
    if source_file.endswith(".parquet"):
        return source_file
    name = source_file.replace(".h5ad", "")
    if name.endswith("_tile"):
        name = name[:-5]
    return f"{name}.parquet"


def _available_host_memory() -> int:
    """
    Bytes of host memory still available: the system's available memory,
    capped by the limit of each cgroup the process is in (a SLURM job, a
    container) less the memory its processes use.
    """
    with open("/proc/meminfo") as f:
        info = dict(line.split(":", 1) for line in f)
    available = int(info["MemAvailable"].split()[0]) * 1024
    try:
        with open("/proc/self/cgroup") as f:
            lines = f.read().splitlines()
    except OSError:
        return available
    for line in lines:
        _, controllers, path = line.split(":", 2)
        if controllers == "":
            root, limit_file, used_key = Path("/sys/fs/cgroup"), "memory.max", "anon"
        elif "memory" in controllers.split(","):
            root, limit_file, used_key = Path("/sys/fs/cgroup/memory"), "memory.limit_in_bytes", "total_rss"
        else:
            continue
        # The limit may be set on any ancestor (SLURM sets it on the job)
        d = root / path.lstrip("/")
        while True:
            try:
                limit = int((d / limit_file).read_text())
                stat = dict(l.split() for l in (d / "memory.stat").read_text().splitlines())
                available = min(available, limit - int(stat[used_key]))
            except (OSError, ValueError, KeyError):
                pass
            if d == root:
                break
            d = d.parent
    return max(available, 0)


def _group_fragment_counts(sources: list, n_groups: int) -> np.ndarray:
    """Fragments of each group, from the barcode tables of the files (all chromosomes)."""
    n = np.zeros(n_groups, np.int64)
    for fragments, cells, cell_group in sources:
        counts = fragments.barcode_counts()
        per_cell = pd.Series(counts["n_unique"].to_numpy(), index=counts["barcode"].to_numpy())
        per_cell = per_cell.reindex(cells).fillna(0).to_numpy()
        n += np.bincount(cell_group, weights=per_cell, minlength=n_groups).astype(np.int64)
    return n


def _group_passes(n_fragments: np.ndarray, cap: int) -> list:
    """
    Groups read in each pass over the files, at most ``cap`` fragments per
    pass (first fit, largest groups first; a group above ``cap`` gets its
    own pass).
    """
    passes, room = [], []
    for i in np.argsort(-n_fragments, kind="stable"):
        n = int(n_fragments[i])
        k = next((k for k, r in enumerate(room) if n <= r), None)
        if k is None:
            passes.append([])
            room.append(cap)
            k = len(passes) - 1
        passes[k].append(int(i))
        room[k] -= n
    return passes


def _read_groups(
    fragments: FragmentFile,
    cells: np.ndarray,
    cell_group: np.ndarray,
    out: list,
    chroms: Optional[list],
    batch_size: int,
) -> None:
    """
    Add the fragments of ``cells`` to ``out[group][chrom]``, a list of
    ``(starts, ends)`` host arrays, ``group`` being ``cell_group`` of the cell.

    Each batch is sorted by (group, chromosome) on the GPU and copied to the
    host once, so the file is read a single time whatever the number of groups.
    Each group keeps its own copy, freed once its peaks are called.
    """
    names = list(chroms) if chroms is not None else list(fragments.chroms)
    lut = cp.asarray(cell_group, dtype=cp.int64)
    n_chroms = len(names)
    for df in fragments.iter_batches(["chrom", "start", "end"], batch_size, chroms=chroms, cells=cells):
        if len(df) == 0:
            continue
        key = lut[df["cell"].values] * n_chroms + df["chrom"].values.astype(cp.int64)
        order = cp.argsort(key)
        key = key[order].get()
        starts = df["start"].values[order].get()
        ends = df["end"].values[order].get()
        del df, order
        bounds = np.flatnonzero(key[1:] != key[:-1]) + 1
        for lo, hi in zip(np.concatenate(([0], bounds)), np.concatenate((bounds, [key.size]))):
            group, chrom = divmod(int(key[lo]), n_chroms)
            out[group][names[chrom]].append((starts[lo:hi].copy(), ends[lo:hi].copy()))
        mempool.free_all_blocks()


def _concat_group(chunks: dict) -> dict:
    """``{chrom: [(starts, ends), ...]}`` to ``{chrom: (starts, ends)}``, emptying ``chunks`` as it goes."""
    out = {}
    for c in list(chunks):
        parts = chunks.pop(c)
        out[c] = (np.concatenate([s for s, _ in parts]), np.concatenate([e for _, e in parts]))
    return out


# =============================================================================
# Main Public API
# =============================================================================


def call(
    adata: "AnnData",
    groupby: str,
    parquet_path: Union[str, Path],
    *,
    genome: Union[str, dict] = "hg38",
    q_thresh: float = 0.05,
    d_treat: int = 200,
    shift: Optional[int] = None,
    d_ctrl: int = 10000,
    max_gap: int = 30,
    peak_amp: int = 200,
    fe_cutoff: float = 1.0,
    call_summits: bool = True,
    macs3_version: Literal["3.0", "3.1"] = "3.0",
    key_added: str = "gmacs",
    inplace: bool = True,
    verbose: bool = True,
    batch_size: int = 10,
    filter_chromosomes: bool = True,
    host_memory_gb: Optional[float] = None,
) -> Optional[dict]:
    """
    Call peaks per group of cells with MACS3, on the GPU.

    Gives the peaks of SnapATAC2's ``macs3`` :cite:p:`Zhang2024`, which runs
    MACS3 :cite:p:`Zhang2008` on each group: the Tn5 insertions of the
    group's fragments (``start`` on the + strand, ``end - 1`` on the -
    strand) are piled up after a shift of ``shift`` and an extension to
    ``d_treat``; the local lambda is the pileup in a ``d_ctrl`` window,
    floored at the genome background; runs whose q-value passes
    ``q_thresh`` are joined across gaps of at most ``max_gap`` into peaks of
    at least ``peak_amp`` bp, and summits are called inside each peak. The
    computation is MACS3's, on run-length pileups on the GPU, so peaks,
    summits and scores are the ones MACS3 reports, at a fraction of the
    time. Fragments are read once for all the groups that fit in
    `host_memory_gb`.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix (typically a tiled matrix).
        If `source_file` is in obs, it's used to map cells to parquet files.
    groupby : str
        Key in `adata.obs` containing cluster/group assignments.
    parquet_path : str or Path
        Path to parquet file or directory containing parquet files.
        If directory, uses `source_file` obs to find specific files.
    genome : str or dict
        Genome name (e.g., 'hg38', 'mm10') or dict of chromosome sizes.
        Its total length is MACS3's genome size (``-g``), as SnapATAC2 sets
        it, and its chromosomes are those kept with `filter_chromosomes`.
    q_thresh : float
        Q-value cutoff (MACS3 ``-q``, default: 0.05).
    d_treat : int
        Fragment size the insertions are extended to, in bp (MACS3
        ``--extsize``, default: 200).
    shift : int, optional
        Shift of the insertions before extension (MACS3 ``--shift``).
        Default: ``-(d_treat // 2)``, which centers the extension on the
        insertion (-100 for the default ``d_treat``, as in SnapATAC2).
    d_ctrl : int
        Window of the local lambda, in bp (MACS3 ``--llocal``, default: 10000).
    max_gap : int
        Largest gap between significant runs joined into one peak (MACS3
        ``--max-gap``, default: 30, as in SnapATAC2).
    peak_amp : int
        Minimum peak length in bp (MACS3 ``--min-length``, default: 200),
        also the smoothing window of the summit caller.
    fe_cutoff : float
        Fold-enrichment cutoff at the summit (MACS3 ``--fe-cutoff``, default: 1.0).
    call_summits : bool
        Call every summit of a peak (MACS3 ``--call-summits``), one row per
        summit, as SnapATAC2 does (default: True). With False, one summit
        per peak.
    macs3_version : {"3.0", "3.1"}
        Summit filtering of MACS3 3.0, which SnapATAC2 (<= 2.9) uses, or of
        MACS3 3.1, which fixed it (default: "3.0"). Only summits differ.
    key_added : str
        Key in `adata.uns` where peaks will be stored (default: 'gmacs')
    inplace : bool
        If True, store results in `adata.uns[key_added]` (default: True)
    verbose : bool
        If True, show tqdm progress bars (default: True)
    batch_size : int
        Number of parquet row groups to load at once (default: 10).
        Increase for faster processing, decrease to reduce memory usage.
    filter_chromosomes : bool
        If True, only keep fragments from chromosomes recognized by the 
        provided genome (keys if genome is a dict, or standard chromosomes 
        for the named genome) (default: True).
    host_memory_gb : float, optional
        Host memory for the fragments held at once, in GB (8 bytes per
        fragment). Groups that do not fit together are read in further
        passes over the files; results are the same. Default: half the
        available memory, within the limit of the job or container if any.

    Returns
    -------
    dict[str, pd.DataFrame] or None
        If `inplace=False`, returns dict mapping group names to peak DataFrames.
        Each DataFrame has the columns of SnapATAC2's ``macs3``: chrom, start,
        end, name, score, strand, signal_value (fold enrichment), p_value and
        q_value (-log10) and peak (summit offset from start), one row per
        summit.
        
    Examples
    --------
    >>> import gatac
    >>> # Load tiled AnnData with cluster assignments
    >>> adata = sc.read_h5ad("tiled_matrix.h5ad")
    >>> # Call peaks per cluster (default settings for ATAC-seq)
    >>> gatac.peak.call(
    ...     adata,
    ...     groupby="leiden",
    ...     parquet_path="/path/to/fragments.parquet",
    ... )
    >>> # Access peaks for cluster "0"
    >>> peaks_cluster0 = adata.uns["gmacs"]["0"]
    """
    parquet_path = Path(parquet_path)
    if macs3_version not in ("3.0", "3.1"):
        raise ValueError(f"macs3_version must be '3.0' or '3.1', got {macs3_version!r}")

    # Get genome length
    if isinstance(genome, str):
        chrom_sizes = get_chrom_sizes(genome)
    else:
        chrom_sizes = genome
    genome_length = sum(chrom_sizes.values())
    logger.info(f"Using genome length: {genome_length:,} bp")

    # Get cluster assignments
    if groupby not in adata.obs.columns:
        raise ValueError(f"'{groupby}' not found in adata.obs")

    labels = adata.obs[groupby]
    groups = [g for g in labels.unique() if not pd.isna(g)]
    logger.info(f"Calling peaks for {len(groups)} groups in '{groupby}'")
    group_index = pd.Index(groups)
    cell_group = group_index.get_indexer(labels)
    has_group = cell_group >= 0

    params = _macs.Params(
        gsize=genome_length,
        log_qvalue=log(q_thresh, 10) * -1,
        extsize=d_treat,
        shift=-(d_treat // 2) if shift is None else shift,
        llocal=d_ctrl,
        max_gap=max_gap,
        min_length=peak_amp,
        fe_cutoff=fe_cutoff,
        call_summits=call_summits,
        macs3_version=macs3_version,
    )
    chroms = sorted(chrom_sizes) if filter_chromosomes else None

    # (file, cells, their groups) to read
    if parquet_path.is_dir():
        # Directory mode - use source_file obs
        if 'source_file' not in adata.obs.columns:
            raise ValueError(
                "parquet_path is a directory but 'source_file' not in adata.obs. "
                "Please provide a single parquet file path instead."
            )
        source = adata.obs['source_file'].to_numpy()
        source_files = pd.unique(source[has_group])
        logger.info(f"Using {len(source_files)} source files from source_file obs")
        sources = []
        for src_file in source_files:
            src_path = parquet_path / _source_parquet_name(src_file)
            if not src_path.exists():
                logger.warning(f"Parquet file not found: {src_path}")
                continue
            sel = has_group & (source == src_file)
            sources.append((FragmentFile(src_path), adata.obs_names.to_numpy()[sel], cell_group[sel]))
    else:
        sources = [(FragmentFile(parquet_path), adata.obs_names.to_numpy()[has_group], cell_group[has_group])]

    # Groups are read together, in as few passes over the files as the host
    # memory allows, and called smallest first so that each group's call
    # runs with the fragments of fewer groups still held
    n_frag = _group_fragment_counts(sources, len(groups))
    if host_memory_gb is None:
        budget = 0.5 * _available_host_memory()
    else:
        budget = host_memory_gb * 1e9
    passes = _group_passes(n_frag, int(budget // _HOST_BYTES_PER_FRAGMENT))
    if len(passes) > 1:
        logger.info(f"Reading the groups in {len(passes)} passes to fit {budget / 1e9:.0f} GB of host memory")

    if verbose:
        from tqdm.auto import tqdm
    peaks_dict = {}
    for k, members in enumerate(passes):
        chunks = {i: defaultdict(list) for i in members}
        lut = np.full(len(groups), -1, np.int64)
        lut[members] = np.arange(len(members))
        step = "" if len(passes) == 1 else f" ({k + 1}/{len(passes)})"
        files = tqdm(sources, desc=f"Reading files{step}") if verbose else sources
        for ff, cells, cg in files:
            sel = lut[cg] >= 0
            if sel.any():
                _read_groups(ff, cells[sel], lut[cg[sel]], [chunks[i] for i in members], chroms, batch_size)
        gc.collect()

        order = sorted(members, key=lambda i: n_frag[i])
        for i in tqdm(order, desc=f"Groups{step}") if verbose else order:
            group_str = str(groups[i])
            fragments = _concat_group(chunks.pop(i))
            num_fragments = sum(len(s) for s, _ in fragments.values())
            if num_fragments == 0:
                logger.warning(f"  No fragments found for group {group_str}")
                peaks_dict[group_str] = _macs._empty()
                continue
            logger.info(f"Group {group_str}: {num_fragments:,} fragments, calling peaks...")
            peaks = _macs.callpeak(fragments, params)
            del fragments
            peaks_dict[group_str] = peaks
            logger.info(f"  Found {len(peaks):,} summits")
            mempool.free_all_blocks()
            pinned_mempool.free_all_blocks()
    peaks_dict = {str(g): peaks_dict[str(g)] for g in groups}

    if inplace:
        adata.uns[key_added] = peaks_dict
        return None
    else:
        return peaks_dict


@njit(cache=True, parallel=True)
def _iterative_merge_numba(
    starts: np.ndarray,
    ends: np.ndarray,
    p_values: np.ndarray,
    group_ids: np.ndarray,
    n_groups: int,
) -> np.ndarray:
    """
    Numba-accelerated iterative merge for all groups at once.
    
    For each group of overlapping peaks, repeatedly select the most significant
    peak (highest p_value = -log10(p)), remove all overlapping peaks, repeat.
    
    Parameters
    ----------
    starts : np.ndarray
        Peak start positions (sorted by position within each chromosome)
    ends : np.ndarray
        Peak end positions
    p_values : np.ndarray
        Peak p-values (-log10(p), higher = more significant)
    group_ids : np.ndarray
        Group ID for each peak (peaks with same ID are overlapping)
    n_groups : int
        Total number of groups
        
    Returns
    -------
    np.ndarray
        Boolean mask where True indicates peaks to keep
    """
    n = len(starts)
    keep_mask = np.zeros(n, dtype=np.bool_)
    
    # Pre-compute group boundaries for faster access
    # Find first and last index of each group
    group_start_idx = np.zeros(n_groups, dtype=np.int64)
    group_end_idx = np.zeros(n_groups, dtype=np.int64)
    
    # Since data is sorted by position and groups are contiguous, 
    # we can find boundaries efficiently
    if n > 0:
        group_start_idx[0] = 0
        for i in range(1, n):
            if group_ids[i] != group_ids[i-1]:
                group_end_idx[group_ids[i-1]] = i
                group_start_idx[group_ids[i]] = i
        group_end_idx[group_ids[n-1]] = n
    
    # Process each group in parallel
    for group_id in prange(n_groups):
        start_i = group_start_idx[group_id]
        end_i = group_end_idx[group_id]
        n_group = end_i - start_i
        
        if n_group == 0:
            continue
        
        if n_group == 1:
            keep_mask[start_i] = True
            continue
        
        # Track which peaks in the group are still available
        available = np.ones(n_group, dtype=np.bool_)
        
        while True:
            # Find the best available peak (highest p_value); among equal
            # p-values the last in position order, as Rust's max_by in SnapATAC2
            best_local_idx = -1
            best_pval = -np.inf
            
            for i in range(n_group):
                if available[i]:
                    idx = start_i + i
                    if p_values[idx] >= best_pval:
                        best_pval = p_values[idx]
                        best_local_idx = i
            
            if best_local_idx == -1:
                break
            
            # Keep this peak
            best_idx = start_i + best_local_idx
            keep_mask[best_idx] = True
            
            # Remove all overlapping peaks
            best_start = starts[best_idx]
            best_end = ends[best_idx]
            
            for i in range(n_group):
                if available[i]:
                    idx = start_i + i
                    # Check overlap: start < best_end AND end > best_start
                    if starts[idx] < best_end and ends[idx] > best_start:
                        available[i] = False
    
    return keep_mask


@njit(cache=True)
def _find_overlapping_groups_with_chroms_numba(
    starts: np.ndarray, 
    ends: np.ndarray, 
    chroms: np.ndarray
) -> tuple:
    """
    Numba-accelerated function to find groups of overlapping intervals with chromosome awareness.
    
    Given sorted intervals by (chromosome, start), assigns a group ID to each interval.
    Intervals that overlap (or are adjacent) within the same chromosome get the same group ID.
    Chromosome changes always start a new group.
    
    Parameters
    ----------
    starts : np.ndarray
        Start positions (must be sorted by chrom, then start)
    ends : np.ndarray
        End positions
    chroms : np.ndarray
        Chromosome codes (integers)
        
    Returns
    -------
    tuple
        (group_ids array, number of groups)
    """
    n = len(starts)
    if n == 0:
        return np.zeros(0, dtype=np.int64), 0
    
    group_ids = np.zeros(n, dtype=np.int64)
    current_group = 0
    current_end = ends[0]
    current_chrom = chroms[0]
    
    for i in range(1, n):
        # New group if chromosome changes or no overlap
        if chroms[i] != current_chrom:
            # Chromosome change - start new group
            current_group += 1
            current_chrom = chroms[i]
            current_end = ends[i]
        elif starts[i] <= current_end:
            # Same chromosome, overlapping - extend group
            if ends[i] > current_end:
                current_end = ends[i]
        else:
            # Same chromosome, no overlap - new group
            current_group += 1
            current_end = ends[i]
        group_ids[i] = current_group
    
    return group_ids, current_group + 1


def merge(
    peaks: Union[dict[str, pd.DataFrame], "AnnData"],
    chrom_sizes: Optional[Union[str, dict[str, int]]] = None,
    half_width: int = 250,
    use_rep: str = "gmacs",
    key_added: str = "peaks",
    inplace: bool = True,
) -> Optional[pd.DataFrame]:
    """Merge peaks from different groups into fixed-width, non-overlapping peaks.

    This mirrors the behavior of SnapATAC2's ``merge`` :cite:p:`Zhang2024`.

    The algorithm expands each peak summit by ``half_width`` on both sides,
    sorts peaks by genomic position, groups overlapping or adjacent intervals,
    and then iteratively keeps the most significant peak in each overlap group
    while discarding the overlapping alternatives.
     
    This algorithm matches SnapATAC2's Rust implementation which uses merge_sorted_bed_with
    to group overlapping intervals before applying iterative_merge.

    Parameters
    ----------
    peaks
        Peak information from different groups. Either a dict mapping group names
        to pandas DataFrames with peak info, or an AnnData object containing
        peaks in `.uns[use_rep]`.
    chrom_sizes
        Chromosome sizes. If a string is provided, it is interpreted as a genome
        name passed to `get_chrom_sizes`. If peaks is an AnnData and chrom_sizes
        is None, will try to infer from adata.uns['reference_sequences'].
    half_width
        Half width of the merged peaks.
    use_rep
        When peaks is an AnnData, key in `.uns` containing peak information.
    key_added
        When peaks is an AnnData and inplace=True, key in `.uns` to store merged peaks.
    inplace
        When peaks is an AnnData, whether to store results in `.uns[key_added]`.

    Returns
    -------
    pd.DataFrame or None
        A dataframe with merged, fixed-width, non-overlapping peaks.
        If peaks is an AnnData and inplace=True, returns None and stores in
        `.uns[key_added]`.

    Examples
    --------
    >>> import gatac as ga
    >>> # Operate in-place on an AnnData: stores results in adata.uns["peaks"]
    >>> ga.peak.merge(adata, use_rep="gmacs", key_added="peaks")
    >>> # Or pass a dict of per-group peak DataFrames:
    >>> peaks_df = ga.peak.merge(
    ...     adata.uns["gmacs"],
    ...     chrom_sizes="hg38",
    ...     half_width=250,
    ... )
    """

    # Handle AnnData input
    if hasattr(peaks, 'uns'):
        adata = peaks
        if use_rep not in adata.uns:
            raise ValueError(f"'{use_rep}' not found in adata.uns")
        peaks_dict = adata.uns[use_rep]
        
        # Try to infer chrom_sizes if not provided
        if chrom_sizes is None:
            if 'reference_sequences' in adata.uns:
                ref_seqs = adata.uns['reference_sequences']
                if isinstance(ref_seqs, pd.DataFrame):
                    chrom_sizes = dict(zip(
                        ref_seqs['reference_seq_name'],
                        ref_seqs['reference_seq_length']
                    ))
                else:
                    raise ValueError(
                        "chrom_sizes must be provided when reference_sequences "
                        "is not a DataFrame"
                    )
            else:
                raise ValueError(
                    "chrom_sizes must be provided when AnnData lacks "
                    "reference_sequences in uns"
                )
    else:
        peaks_dict = peaks
        adata = None
        if chrom_sizes is None:
            raise ValueError("chrom_sizes must be provided when peaks is a dict")
    
    if isinstance(chrom_sizes, str):
        chrom_sizes = get_chrom_sizes(chrom_sizes)

    if not peaks_dict:
        result = pd.DataFrame()
        if adata is not None and inplace:
            adata.uns[key_added] = result
            return None
        return result

    required_cols = {"chrom", "start", "end", "p_value"}
    expanded_peaks = []

    # Process each group's peaks - expand summits to fixed width
    for _, df in peaks_dict.items():
        if df is None or len(df) == 0:
            continue
        missing = required_cols.difference(df.columns)
        if missing:
            raise ValueError(f"Missing required columns in peaks: {sorted(missing)}")

        df_copy = df.copy()
        
        # Calculate summit position using numpy for speed
        start_vals = df_copy["start"].values.astype(np.int64)
        if "peak" in df_copy.columns:
            peak_vals = df_copy["peak"].values.astype(np.int64)
            summit = start_vals + peak_vals
        else:
            end_vals = df_copy["end"].values.astype(np.int64)
            summit = (start_vals + end_vals) // 2

        # Expand to fixed width - match SnapATAC2's half-open interval convention
        start = np.maximum(summit - half_width, 0)
        df_copy["start"] = start
        df_copy["end"] = summit + half_width + 1  # +1 for half-open interval
        df_copy["peak"] = summit - start

        expanded_peaks.append(df_copy)

    if len(expanded_peaks) == 0:
        result = pd.DataFrame()
        if adata is not None and inplace:
            adata.uns[key_added] = result
            return None
        return result

    # Concatenate all peaks and move to GPU
    all_peaks = pd.concat(expanded_peaks, ignore_index=True)
    all_peaks_gpu = cudf.DataFrame(all_peaks)
    
    # Get arrays on GPU
    starts_cp = all_peaks_gpu['start'].to_cupy().astype(cp.int64)
    ends_cp = all_peaks_gpu['end'].to_cupy().astype(cp.int64)
    p_values_cp = all_peaks_gpu['p_value'].to_cupy().astype(cp.float64)
    
    # Create chromosome encoding on GPU
    chrom_cat = all_peaks_gpu['chrom'].astype('category')
    chrom_codes_cp = chrom_cat.cat.codes.to_cupy().astype(cp.int32)
    unique_chroms = chrom_cat.cat.categories.to_pandas().tolist()
    
    # Peaks are merged unclipped (start saturating at 0) and clipped to the
    # chromosome afterwards, as SnapATAC2 does
    starts_cp = cp.maximum(starts_cp, 0)

    # Global sort: by chromosome code, then by start, then by end
    # Use composite key for stable sorting on GPU
    max_pos = int(cp.max(ends_cp)) + 1
    sort_key = (chrom_codes_cp.astype(cp.int64) * (max_pos * max_pos) + 
                starts_cp * max_pos + ends_cp)
    global_sort_idx = cp.argsort(sort_key)
    
    # Apply sort to all arrays
    sorted_starts = starts_cp[global_sort_idx]
    sorted_ends = ends_cp[global_sort_idx]
    sorted_pvals = p_values_cp[global_sort_idx]
    sorted_chroms = chrom_codes_cp[global_sort_idx]
    
    # Process ALL chromosomes in ONE batch using combined approach:
    # 1. Find overlapping groups (treating chromosome changes as group boundaries)
    # 2. Run iterative merge on all groups in parallel
    
    # Find overlapping groups with chromosome boundaries
    # A new group starts when: (a) chromosome changes OR (b) no overlap with previous
    n = len(sorted_starts)
    
    # Transfer to CPU once for the grouping/merge algorithm
    sorted_starts_np = cp.asnumpy(sorted_starts)
    sorted_ends_np = cp.asnumpy(sorted_ends)
    sorted_pvals_np = cp.asnumpy(sorted_pvals)
    sorted_chroms_np = cp.asnumpy(sorted_chroms)
    
    # Use optimized numba function for grouping with chromosome awareness
    group_ids_np, n_groups = _find_overlapping_groups_with_chroms_numba(
        sorted_starts_np, sorted_ends_np, sorted_chroms_np
    )
    
    # Run parallel iterative merge across all groups
    keep_mask_np = _iterative_merge_numba(
        sorted_starts_np, sorted_ends_np, sorted_pvals_np, group_ids_np, n_groups
    )
    
    # Transfer result back to GPU
    keep_mask_cp = cp.asarray(keep_mask_np)
    
    # Get kept indices in original order
    kept_sorted_indices = cp.where(keep_mask_cp)[0]
    kept_original_indices = global_sort_idx[kept_sorted_indices]
    
    # Filter on GPU and convert back
    if len(kept_original_indices) == 0:
        result = pd.DataFrame(columns=all_peaks.columns)
    else:
        # Use GPU filtering
        result_gpu = all_peaks_gpu.iloc[cp.asnumpy(kept_original_indices)]
        result = result_gpu.to_pandas()
    
    # Cleanup GPU memory
    del all_peaks_gpu, starts_cp, ends_cp, p_values_cp, chrom_codes_cp
    del sorted_starts, sorted_ends, sorted_pvals, sorted_chroms, global_sort_idx
    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()
    
    # Deduplicate exact duplicates (same chrom, start, end) keeping most significant
    if len(result) > 0 and 'p_value' in result.columns:
        logger.debug(f"Deduplicating {len(result):,} peaks...")
        
        # Convert to cuDF for GPU-accelerated groupby
        result_gpu = cudf.DataFrame(result)
        
        # Sort by p-value to ensure most significant is first in each group
        # p-values are -log10(p), so higher values are MORE significant
        result_gpu = result_gpu.sort_values('p_value', ascending=False)
        
        # Drop duplicates based on (chrom, start, end), keeping first (most significant)
        result_gpu = result_gpu.drop_duplicates(
            subset=['chrom', 'start', 'end'],
            keep='first'
        )
        
        # Convert back to pandas
        result = result_gpu.to_pandas()
        
        del result_gpu
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
        
        logger.debug(f"After deduplication: {len(result):,} peaks")

    # Clip to the chromosomes, as SnapATAC2's clip_peak
    if len(result) > 0:
        size = result["chrom"].map(chrom_sizes).to_numpy(dtype=np.float64)
        known = ~np.isnan(size)
        start = result["start"].to_numpy(dtype=np.int64)
        end = result["end"].to_numpy(dtype=np.int64)
        size = np.where(known, size, np.iinfo(np.int64).max).astype(np.int64)
        new_start = np.minimum(start, size)
        new_end = np.minimum(end, size)
        result["peak"] = np.minimum(new_start + result["peak"].to_numpy(dtype=np.int64), new_end) - new_start
        result["start"] = new_start
        result["end"] = new_end

    # Store in AnnData if requested
    if adata is not None and inplace:
        adata.uns[key_added] = result
        return None
    
    return result


def _expand_ranges(lo: cp.ndarray, hi: cp.ndarray) -> tuple[cp.ndarray, cp.ndarray]:
    """``(i, j)`` for every ``j`` in ``[lo[i], hi[i])``, as two index arrays."""
    n = cp.maximum(hi - lo, 0)
    total = int(n.sum())
    if total == 0:
        return cp.empty(0, dtype=cp.int64), cp.empty(0, dtype=cp.int64)
    ends = cp.cumsum(n)
    k = cp.arange(total, dtype=cp.int64)
    i = cp.searchsorted(ends, k, side='right')
    return i, k - (ends - n)[i] + lo[i]


def _count_fragments_in_peaks_gpu(
    fragments_df: cudf.DataFrame,
    peaks_by_chrom: dict,
    cell_index: cp.ndarray,
    chrom_names: list,
    n_peaks: int,
    counting_strategy: str = "paired-insertion",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Count fragment insertions overlapping with peaks using GPU-accelerated operations.
    
    Uses optimized searchsorted for O(N log M) overlap detection.
    
    Parameters
    ----------
    fragments_df : cudf.DataFrame
        Fragments with 'chrom' (index into ``chrom_names``), 'start', 'end'
        and 'cell' (index into ``cell_index``) columns
    peaks_by_chrom : dict
        Pre-grouped peaks dict: {chrom: (starts_cp, ends_cp, peak_indices_cp)}
    cell_index : cp.ndarray
        Output row (cell index) of each value of the 'cell' column
    chrom_names : list of str
        Chromosome names indexed by the 'chrom' column
    n_peaks : int
        Total number of peaks
    counting_strategy : str
        Counting strategy: 'paired-insertion' (default), 'insertion', or 'fragment'
        
    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (rows, cols, data) as numpy arrays for sparse matrix construction
    """
    all_results = []
    
    # Add global fragment index column to preserve indices through chromosome filtering
    fragments_df = fragments_df.reset_index(drop=True)
    fragments_df['_global_frag_idx'] = cp.arange(len(fragments_df), dtype=cp.int64)
    
    # Process each chromosome
    for ci in cp.unique(fragments_df['chrom'].values).get():
        chrom = chrom_names[ci]
        if chrom not in peaks_by_chrom:
            continue
            
        chrom_frags = fragments_df[fragments_df['chrom'] == ci]
        peak_starts, peak_ends, peak_indices = peaks_by_chrom[chrom]
        
        if len(chrom_frags) == 0:
            continue
        
        # Get fragment arrays on GPU
        frag_starts = chrom_frags['start'].to_cupy()
        frag_ends = chrom_frags['end'].to_cupy()
        global_frag_indices = chrom_frags['_global_frag_idx'].to_cupy()
        
        def emit(f_local, p_local):
            if len(f_local) > 0:
                all_results.append(cudf.DataFrame({
                    'frag_idx': global_frag_indices[f_local],
                    'peak_idx': peak_indices[p_local],
                }))

        # Peaks from peak.merge never overlap, so the peak holding a position
        # is the last one starting at or before it. A peak_file may hold
        # overlapping peaks: then every peak starting within the widest peak's
        # width before the position is a candidate.
        max_width = int((peak_ends - peak_starts).max())
        disjoint = bool(cp.all(peak_starts[1:] >= peak_ends[:-1]))

        if counting_strategy == "fragment":
            # Frag overlaps peak if: frag_start < peak_end AND frag_end > peak_start
            lo = cp.searchsorted(peak_starts, frag_starts.astype(cp.int64) - max_width, side='right')
            hi = cp.searchsorted(peak_starts, frag_ends, side='left')
            f_local, p_local = _expand_ranges(lo, hi)
            keep = peak_ends[p_local] > frag_starts[f_local]
            emit(f_local[keep], p_local[keep])
        else:
            # Insertion-based counting (paired-insertion or insertion), at
            # start and end - 1. Insertion at p is in peak if:
            # peak_start <= p < peak_end
            ins2_pos = frag_ends - 1
            if disjoint:
                idx1 = cp.searchsorted(peak_starts, frag_starts, side='right') - 1
                valid1 = (idx1 >= 0) & (frag_starts < peak_ends[idx1])
                idx2 = cp.searchsorted(peak_starts, ins2_pos, side='right') - 1
                valid2 = (idx2 >= 0) & (ins2_pos < peak_ends[idx2])
                if counting_strategy == "paired-insertion":
                    # Both insertions in one peak: count the fragment once
                    valid2 &= ~valid1 | (idx1 != idx2)
                v1 = cp.where(valid1)[0]
                v2 = cp.where(valid2)[0]
                emit(v1, idx1[v1])
                emit(v2, idx2[v2])
            else:
                for pos in (frag_starts, ins2_pos):
                    lo = cp.searchsorted(peak_starts, pos.astype(cp.int64) - max_width, side='right')
                    hi = cp.searchsorted(peak_starts, pos, side='right')
                    f_local, p_local = _expand_ranges(lo, hi)
                    keep = pos[f_local] < peak_ends[p_local]
                    # paired-insertion: the drop_duplicates below counts a
                    # fragment with both insertions in one peak once
                    emit(f_local[keep], p_local[keep])
        
        mempool.free_all_blocks()
    
    if len(all_results) == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32), np.array([], dtype=np.int32)
    
    # Concatenate all overlap results
    all_overlaps = cudf.concat(all_results, ignore_index=True)
    del all_results
    
    # Cell index of each overlapping fragment (frag_idx is the row in fragments_df)
    frag_cells = fragments_df['cell'].values
    all_overlaps['cell_idx'] = cell_index[frag_cells[all_overlaps['frag_idx'].values]]
    
    # For paired-insertion, count each (fragment, peak) pair once
    if counting_strategy == "paired-insertion":
        all_overlaps = all_overlaps.drop_duplicates(subset=['frag_idx', 'peak_idx'])
    
    # Aggregate by (cell_idx, peak_idx) using GPU groupby
    counts = all_overlaps.groupby(['cell_idx', 'peak_idx']).size().reset_index(name='count')
    
    # Convert to numpy arrays
    rows = counts['cell_idx'].to_pandas().values
    cols = counts['peak_idx'].to_pandas().values
    data = counts['count'].to_pandas().values
    
    del all_overlaps, counts, frag_cells
    mempool.free_all_blocks()
    
    return rows, cols, data


def _read_and_count_fragments_batched(
    parquet_path: Path,
    barcodes: set,
    peaks_gpu: cudf.DataFrame,
    barcode_to_idx: dict,
    n_peaks: int,
    batch_size: int = 10,
    counting_strategy: str = "paired-insertion",
    filter_chromosomes: bool = True,
    chrom_names: Optional[list[str]] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read fragments from parquet in batches and count insertions in peaks.
    
    Uses chunked row group reading to avoid OOM when processing large files.
    
    Parameters
    ----------
    parquet_path : Path
        Path to the fragment parquet file
    barcodes : set
        Set of barcodes to include
    peaks_gpu : cudf.DataFrame
        Peaks DataFrame on GPU
    barcode_to_idx : dict
        Mapping from barcode to cell index
    n_peaks : int
        Total number of peaks
    batch_size : int
        Number of row groups to load at once
    counting_strategy : str
        Counting strategy: 'paired-insertion' (default), 'insertion', or 'fragment'
    filter_chromosomes : bool
        If True, only keep fragments from chromosomes present in `chrom_names` (default: True).
    chrom_names : list[str], optional
        List of allowed chromosome names for filtering.
        
    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (rows, cols, data) as numpy arrays for sparse matrix construction
    """
    fragments = FragmentFile(parquet_path)
    logger.debug(f"Reading {fragments.path.name}: {fragments.num_row_groups} row groups")
    
    # Output row of each barcode of this file (once)
    cells = list(barcodes)
    cell_index = cp.asarray([barcode_to_idx[b] for b in cells], dtype=cp.int64)
    
    # Pre-group peaks by chromosome (once)
    peaks_by_chrom = {}
    for chrom in peaks_gpu['chrom'].to_pandas().unique():
        chrom_peaks = peaks_gpu[peaks_gpu['chrom'] == chrom]
        peaks_by_chrom[chrom] = (
            chrom_peaks['start'].to_cupy(),
            chrom_peaks['end'].to_cupy(),
            chrom_peaks['peak_idx'].to_cupy(),
        )
    
    # Only fragments on chromosomes with peaks can overlap one
    allowed = set(chrom_names) if filter_chromosomes and chrom_names is not None else None
    chroms = [c for c in peaks_by_chrom if allowed is None or c in allowed]
    
    all_rows = []
    all_cols = []
    all_data = []
    
    # Process row groups in batches
    for filtered in fragments.iter_batches(
        ['chrom', 'start', 'end'], batch_size, chroms=chroms, cells=cells,
    ):
        if len(filtered) > 0:
            rows, cols, data = _count_fragments_in_peaks_gpu(
                filtered, peaks_by_chrom, cell_index, chroms, n_peaks, counting_strategy
            )
            all_rows.append(rows)
            all_cols.append(cols)
            all_data.append(data)
        
        del filtered
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
    
    # Concatenate all batches
    if len(all_rows) > 0:
        return np.concatenate(all_rows), np.concatenate(all_cols), np.concatenate(all_data)
    return np.array([], dtype=np.int32), np.array([], dtype=np.int32), np.array([], dtype=np.int32)


def make_matrix(
    adata: "AnnData",
    parquet_path: Union[str, Path, list[Union[str, Path]]],
    *,
    use_rep: str = "peaks",
    peak_file: Optional[Path] = None,
    genome: Union[str, dict] = "hg38",
    counting_strategy: str = "paired-insertion",
    inplace: bool = False,
    batch_size: int = 50,
    verbose: Union[bool, str] = True,
    filter_chromosomes: bool = True,
) -> Optional["AnnData"]:
    """Generate cell by peak count matrix.

    This function counts fragment insertions overlapping with peak regions to create
    a cell × peak count matrix. Efficiently processes multiple parquet files
    using batched row group reading to minimize memory usage.

    Parameters
    ----------
    adata
        The annotated data matrix (typically a tiled matrix).
        If `source_file` is in obs, it's used to map cells to parquet files.
    parquet_path
        Path to parquet file, directory containing parquet files, or a list
        of parquet file paths.
        - If a single file: all cells are read from this file.
        - If a directory: uses `source_file` obs to find specific files.
        - If a list of files: processes all files, matching barcodes to cells.
    use_rep
        Key in `.uns` containing peak information. The peaks can be a DataFrame
        with 'chrom', 'start', 'end' columns.
    peak_file
        BED file containing peaks. If provided, peak information will be read
        from this file instead of `use_rep`.
    genome
        Genome name (e.g., 'hg38', 'mm10') or dict of chromosome sizes.
        Used for validation and chromosome filtering.
    counting_strategy
        Counting strategy for peak matrix generation. ``"paired-insertion"``
        counts Tn5 insertions at fragment ends and counts a fragment once if
        both ends land in the same peak; this matches SnapATAC2's default.
        ``"insertion"`` counts each insertion separately, so a fragment with
        both ends in a peak contributes 2. ``"fragment"`` is the legacy mode
        that counts any fragment overlapping the peak.
    inplace
        Whether to add the peak matrix to the AnnData object (not recommended,
        will replace .X). If False, returns a new AnnData object.
    batch_size
        Number of parquet row groups to load at once (default: 50).
        Larger values are faster but use more GPU memory.
    verbose
        Controls progress reporting. ``False`` disables progress output.
        ``True`` or ``"tqdm"`` shows a tqdm progress bar. ``"log"`` prints
        one line per file with ETA for non-interactive environments.
    filter_chromosomes : bool
        If True, only keep fragments from chromosomes recognized by the 
        provided genome (keys if genome is a dict, or standard chromosomes 
        for the named genome) (default: True).

    Returns
    -------
    AnnData or None
        If inplace=False, returns a new AnnData object with the peak matrix.
        Otherwise returns None and modifies adata in place.

    Examples
    --------
    >>> import gatac
    >>> # After calling peaks and merging
    >>> gatac.peak.merge(adata, use_rep="gmacs", key_added="peaks")
    >>> # Create peak matrix from a single file
    >>> peak_mat = gatac.peak.make_matrix(
    ...     adata,
    ...     parquet_path="/path/to/fragments.parquet",
    ...     use_rep="peaks",
    ...     inplace=False,
    ... )
    >>> # Create peak matrix from multiple files
    >>> peak_mat = gatac.peak.make_matrix(
    ...     adata,
    ...     parquet_path=["/path/to/sample1.parquet", "/path/to/sample2.parquet"],
    ...     use_rep="peaks",
    ...     inplace=False,
    ... )
    >>> # Create peak matrix from directory (uses source_file obs)
    >>> peak_mat = gatac.peak.make_matrix(
    ...     adata,
    ...     parquet_path="/path/to/parquet_dir/",
    ...     use_rep="peaks",
    ...     inplace=False,
    ... )
    """
    import scipy.sparse as sp
    from anndata import AnnData as AD
    
    # Get peaks: a peak_file takes precedence over use_rep
    if peak_file is not None:
        # Read from BED file
        peaks_df = read_bed(peak_file)
    elif use_rep is not None:
        if use_rep not in adata.uns:
            raise ValueError(f"'{use_rep}' not found in adata.uns")
        peaks_df = adata.uns[use_rep]
        if not isinstance(peaks_df, pd.DataFrame):
            raise ValueError(f"adata.uns['{use_rep}'] must be a DataFrame")
        if not all(col in peaks_df.columns for col in ['chrom', 'start', 'end']):
            raise ValueError("Peaks DataFrame must have 'chrom', 'start', 'end' columns")
        peaks_df = peaks_df[['chrom', 'start', 'end']].copy()
    else:
        raise ValueError("Must specify either peak_file or use_rep")
    
    # Get chrom sizes for filtering
    if isinstance(genome, str):
        chrom_sizes = get_chrom_sizes(genome)
    else:
        chrom_sizes = genome
    chrom_names = list(chrom_sizes.keys()) if chrom_sizes is not None else None
    
    logger.info(f"Counting fragments in {len(peaks_df):,} peaks")
    
    # Sort peaks and add index
    peaks_df = peaks_df.sort_values(['chrom', 'start']).reset_index(drop=True)
    
    # Convert peaks to cuDF for GPU processing
    peaks_gpu = cudf.DataFrame(peaks_df)
    peaks_gpu['peak_idx'] = cp.arange(len(peaks_gpu))
    
    # Build sparse matrix dimensions
    n_cells = len(adata.obs)
    n_peaks = len(peaks_df)
    
    # Create cell index mapping (barcode -> row index)
    barcode_to_idx = {cell: idx for idx, cell in enumerate(adata.obs.index)}
    all_barcodes = set(adata.obs.index)
    
    # Determine parquet files to process
    if isinstance(parquet_path, list):
        # List of parquet files provided directly
        parquet_files = [Path(p) for p in parquet_path]
        file_to_barcodes = {pf: all_barcodes for pf in parquet_files}
        logger.info(f"Processing {len(parquet_files)} parquet files (list mode)")
    else:
        parquet_path = Path(parquet_path)
        
        if parquet_path.is_dir():
            # Directory mode - use source_file obs to map cells to files
            if 'source_file' not in adata.obs.columns:
                raise ValueError(
                    "parquet_path is a directory but 'source_file' not in adata.obs"
                )
            
            # Group barcodes by source file
            file_to_barcodes = defaultdict(set)
            for barcode, src_file in adata.obs['source_file'].items():
                parquet_name = _source_parquet_name(src_file)
                parquet_file = parquet_path / parquet_name
                file_to_barcodes[parquet_file].add(barcode)
            
            parquet_files = list(file_to_barcodes.keys())
            logger.info(f"Processing {len(parquet_files)} parquet files (directory mode)")
        else:
            # Single file mode
            parquet_files = [parquet_path]
            file_to_barcodes = {parquet_path: all_barcodes}
            logger.info("Processing single parquet file")
    
    # Build CSR matrix incrementally to avoid OOM from holding COO + CSR simultaneously
    # Initialize accumulator as None (will be created with first batch)
    count_matrix = None
    matrix_dtype = np.int32  # Will be refined later
    
    # Process each parquet file
    n_files = len(parquet_files)
    if verbose == "tqdm" or verbose is True:
        from tqdm.auto import tqdm
        file_iter = tqdm(parquet_files, desc="Parquet files")
    else:
        file_iter = parquet_files
    
    import time
    start_time = time.time()
    
    for file_idx, parquet_file in enumerate(file_iter):
        if not parquet_file.exists():
            logger.warning(f"Parquet file not found: {parquet_file}")
            continue
        
        barcodes_for_file = file_to_barcodes[parquet_file]
        
        if len(barcodes_for_file) == 0:
            continue
        
        # Log progress if verbose="log"
        if verbose == "log":
            elapsed = time.time() - start_time
            if file_idx > 0:
                eta = elapsed / file_idx * (n_files - file_idx)
                eta_str = f", ETA: {eta:.1f}s"
            else:
                eta_str = ""
            print(
                f"[{file_idx + 1}/{n_files}] Processing {parquet_file.name}: "
                f"{len(barcodes_for_file):,} barcodes{eta_str}",
                flush=True,
            )
        
        logger.debug(f"Processing {parquet_file.name}: {len(barcodes_for_file):,} barcodes")
        
        # Read and count fragments in batches
        rows, cols, data = _read_and_count_fragments_batched(
            parquet_file,
            barcodes_for_file,
            peaks_gpu,
            barcode_to_idx,
            n_peaks,
            batch_size=batch_size,
            counting_strategy=counting_strategy,
            filter_chromosomes=filter_chromosomes,
            chrom_names=chrom_names,
        )
        
        # Convert batch to CSR immediately and add to accumulator
        # This avoids holding all COO data in memory until the end
        if len(rows) > 0:
            # Create CSR for this batch (coo -> csr aggregates duplicates)
            batch_matrix = sp.coo_matrix(
                (data, (rows, cols)),
                shape=(n_cells, n_peaks),
                dtype=np.int32,
            ).tocsr()
            
            # Free the COO data immediately
            del rows, cols, data
            
            if count_matrix is None:
                count_matrix = batch_matrix
            else:
                # CSR addition is efficient and in-place friendly
                count_matrix = count_matrix + batch_matrix
            
            del batch_matrix
        
        # Cleanup
        gc.collect()
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
    
    # Handle empty case
    if count_matrix is None:
        count_matrix = sp.csr_matrix((n_cells, n_peaks), dtype=np.int32)
    
    # Optimize dtype to save memory
    if count_matrix.nnz > 0:
        max_count = int(count_matrix.data.max())
        target_dtype = optimize_sparse_matrix_dtype(max_count)
        if target_dtype != count_matrix.dtype:
            count_matrix = count_matrix.astype(target_dtype)
    
    logger.info(f"Peak matrix: {n_cells:,} cells × {n_peaks:,} peaks")
    if n_cells * n_peaks > 0:
        logger.info(f"Sparsity: {100 * (1 - count_matrix.nnz / (n_cells * n_peaks)):.2f}%")
    
    # Create peak names (vectorized)
    peak_names = (peaks_df['chrom'].astype(str) + ':' + 
                  peaks_df['start'].astype(str) + '-' + 
                  peaks_df['end'].astype(str)).tolist()
    
    if inplace:
        # Replace .X with peak matrix (not recommended)
        adata.X = count_matrix
        # Update var to match peaks
        adata.var = pd.DataFrame(index=peak_names)
        for col in peaks_df.columns:
            if col not in ['chrom', 'start', 'end']:
                adata.var[col] = peaks_df[col].values
        return None
    else:
        # Create new AnnData
        var_df = pd.DataFrame(index=peak_names)
        for col in peaks_df.columns:
            if col not in ['chrom', 'start', 'end']:
                var_df[col] = peaks_df[col].values
        
        new_adata = AD(
            X=count_matrix,
            obs=adata.obs.copy(),
            var=var_df,
        )
        return new_adata
