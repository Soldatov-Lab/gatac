"""
GPU-accelerated peak calling using gmacs algorithm.

This module integrates gmacs peak calling directly into GATAC,
adapted to use GATAC's parquet handling and genome inference.
"""

import gc
import logging
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Optional, Union

import cudf
import cupy as cp
import numpy as np
import pandas as pd
from cupyx.scipy.ndimage import maximum_filter1d
from cupyx.scipy.special import pdtr, pdtrc
from numba import njit, prange

from ..pp.genome import get_chrom_sizes

logger = logging.getLogger(__name__)

# GPU memory pool references
mempool = cp.get_default_memory_pool()
pinned_mempool = cp.get_default_pinned_memory_pool()


# =============================================================================
# GMACS Core Functions (integrated from gmacs package)
# =============================================================================


def _pileup(starts, ends, chr_length, offset):
    """
    Compute pileup (coverage) at each position along a chromosome.
    
    Parameters
    ----------
    starts : cp.ndarray
        Fragment start positions
    ends : cp.ndarray
        Fragment end positions
    chr_length : int
        Length of chromosome
    offset : int
        Extension offset for fragments
        
    Returns
    -------
    cp.ndarray
        Coverage vector for the chromosome
    """
    start_poss = cp.concatenate((starts - offset, ends - offset))
    end_poss = cp.concatenate((starts + offset, ends + offset))
    cov_vec = cp.zeros(chr_length, dtype=np.int32)
    start_poss = cp.clip(start_poss, 0, chr_length - 1)
    end_poss = cp.clip(end_poss, 0, chr_length - 1)
    cp.add.at(cov_vec, start_poss, 1)
    cp.add.at(cov_vec, end_poss, -1)
    cov_vec = cp.cumsum(cov_vec)
    return cov_vec


def _compute_peaks(q_vals, thresh=0.1):
    """Filter peaks based on q-values. Return indices where q-value >= thresh."""
    peaks = cp.where((q_vals >= thresh))[0]
    return peaks


def _merge_consecutive(arr, max_gap=30):
    """Merge overlapping and consecutive peak calls."""
    if len(arr) == 0:
        return cp.zeros((0, 2), dtype=cp.int64)
    boundaries = cp.where(cp.diff(arr) >= max_gap)[0] + 1
    split_indices = cp.concatenate(
        (cp.asarray([0]), boundaries, cp.asarray([len(arr)]))
    )
    starts = arr[split_indices[:-1]]
    ends = arr[split_indices[1:] - 1]
    M = cp.stack([starts, ends], axis=1)
    return M


def _filter_peaks(peaks, peak_amplitude=150):
    """Remove short peaks smaller than peak_amplitude bases."""
    if len(peaks) == 0:
        return peaks
    peak_lengths = peaks[:, 1] - peaks[:, 0]
    peak_ind = cp.where(peak_lengths >= peak_amplitude)[0]
    merged_filt = peaks[peak_ind]
    return merged_filt


@njit(cache=True, parallel=True)
def _calculate_peak_summits_numba(starts, ends, signal):
    """
    Numba-accelerated peak summit calculation with parallel execution.
    
    Finds the position of minimum signal value within each peak region.
    Uses prange for multi-core parallelization.
    """
    n_peaks = len(starts)
    arg_max_indices = np.zeros(n_peaks, dtype=np.int64)
    
    for i in prange(n_peaks):
        start = starts[i]
        end = ends[i]
        if end > start:
            # Find argmin within this range
            min_val = signal[start]
            min_idx = start
            for j in range(start + 1, end):
                if signal[j] < min_val:
                    min_val = signal[j]
                    min_idx = j
            arg_max_indices[i] = min_idx
        else:
            arg_max_indices[i] = start
    
    return arg_max_indices


def _calculate_peak_summits(peaks, signal):
    """
    Compute peak summits (highest point in each peak).
    
    The highest point corresponds to the lowest q-value (most significant).
    Uses numba JIT for efficiency.
    """
    if len(peaks) == 0:
        return np.array([], dtype=np.int64)
    
    starts = peaks[:, 0].astype(np.int64)
    ends = peaks[:, 1].astype(np.int64)
    
    return _calculate_peak_summits_numba(starts, ends, signal)


def _fdr(unique_p_values, unique_p_counts):
    """
    Compute FDR corrected values using Benjamini-Hochberg procedure.
    
    Parameters
    ----------
    unique_p_values : cp.ndarray
        Vector of unique p-values (log10 transformed)
    unique_p_counts : cp.ndarray
        Counts for each unique p-value
        
    Returns
    -------
    tuple
        (sorted_keys, sorted_values) as cupy arrays, pre-sorted for efficient lookup
    """
    unique_p_values = -1 * unique_p_values
    sorted_indices = cp.argsort(unique_p_values)[::-1]
    sorted_unique_p_values = unique_p_values[sorted_indices]
    sorted_counts = unique_p_counts[sorted_indices]
    cumulative_k = cp.cumsum(sorted_counts) - sorted_counts + 1
    sorted_unique_p_values = cp.where(
        sorted_unique_p_values == -cp.inf, -cp.inf, sorted_unique_p_values
    )
    total_counts = cp.sum(unique_p_counts)

    f = cp.log10(total_counts)
    q_values = cp.asnumpy(sorted_unique_p_values + (cp.log10(cumulative_k) - f))

    # Vectorized monotonic decreasing enforcement
    # q-values must be non-increasing and clamped to [0, inf)
    # Reverse, take cumulative minimum, reverse back
    q_values_clipped = np.maximum(q_values, 0)  # Clamp to >= 0
    q_values_np = np.minimum.accumulate(q_values_clipped)  # Monotonic non-increasing

    # Return as sorted cupy arrays for efficient lookup
    # Keys are p-values (original, not negated), sorted for searchsorted
    p_values_original = -1 * sorted_unique_p_values
    
    # Sort by keys for efficient searchsorted lookup
    sort_indices = cp.argsort(p_values_original)
    sorted_keys = p_values_original[sort_indices]
    sorted_values = cp.asarray(q_values_np)[sort_indices]
    
    return (sorted_keys, sorted_values)


def _replace_with_dict(array, pq_table):
    """
    Replace array elements with values from pq_table lookup.
    
    Parameters
    ----------
    array : cp.ndarray
        Array of p-values to look up
    pq_table : tuple
        (sorted_keys, sorted_values) as cupy arrays, pre-sorted for searchsorted
    """
    sorted_keys, sorted_values = pq_table
    
    idx = cp.searchsorted(sorted_keys, array)
    # Ensure idx is within bounds for comparison
    idx_clipped = cp.minimum(idx, len(sorted_keys) - 1)
    valid = (idx < len(sorted_keys)) & (sorted_keys[idx_clipped] == array)

    result = cp.where(valid, sorted_values[idx_clipped], array)
    return result


def _compute_poisson_cdfs(observations, lambdas):
    """Compute upper tail of Poisson distribution (p-values)."""
    p_vals = pdtrc(observations, lambdas)
    return cp.log10(p_vals)


def _make_pq_table_from_groups(chrom_groups, num_reads, d_treat=150, d_ctrl=10000, genome_length=3088286401):
    """
    Compute the pq-table from pre-grouped chromosome data.
    
    Parameters
    ----------
    chrom_groups : dict
        Pre-grouped {chrom: (starts_cupy, ends_cupy)} data
    num_reads : int
        Total number of reads
    d_treat : int
        Treatment extension distance (default: 150)
    d_ctrl : int
        Large local control extension distance (default: 10000)
    genome_length : int
        Total genome length for background calculation
    """
    unique_p_values = cp.asarray([])
    unique_p_counts = cp.asarray([])
    
    scale_llocal = d_treat / d_ctrl
    lambda_bg = 2 * d_treat * num_reads / genome_length
    
    for chrom, (starts, ends) in chrom_groups.items():
        chrom_length = cp.max(ends).item()

        pileup_treat = _pileup(starts, ends, chrom_length, int(d_treat / 2))
        
        # Large local pileup
        pileup_llocal = _pileup(starts, ends, chrom_length, int(d_ctrl / 2))
        pileup_ctrl = pileup_llocal * scale_llocal
        
        # Take maximum with genome-wide background
        pileup_ctrl = cp.maximum(pileup_ctrl, lambda_bg)
        
        p_values = _compute_poisson_cdfs(pileup_treat, pileup_ctrl)
        p_values, p_counts = cp.unique(p_values, return_counts=True)
        all_values = cp.concatenate((unique_p_values, p_values))
        all_counts = cp.concatenate((unique_p_counts, p_counts))
        merged_values, inverse_indices = cp.unique(all_values, return_inverse=True)
        merged_counts = cp.zeros_like(merged_values, dtype=all_counts.dtype)
        cp.add.at(merged_counts, inverse_indices, all_counts)

        unique_p_values = merged_values
        unique_p_counts = merged_counts

        del pileup_ctrl, pileup_treat, pileup_llocal, p_values, merged_values
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()

    pq_table = _fdr(unique_p_values, unique_p_counts)
    return pq_table


def _make_pq_table(result_df, num_reads, d_treat=150, d_ctrl=10000, genome_length=3088286401):
    """Legacy wrapper for backward compatibility."""
    # Pre-group chromosomes
    chrom_groups = {}
    chroms = result_df['chrom'].unique().to_pandas()
    for chrom in chroms:
        chrom_df = result_df[result_df['chrom'] == chrom]
        chrom_groups[chrom] = (chrom_df['start'].to_cupy(), chrom_df['end'].to_cupy())
    
    return _make_pq_table_from_groups(chrom_groups, num_reads, d_treat, d_ctrl, genome_length)


def _call_peaks_chrom(
    starts,
    ends,
    pq_table,
    num_reads,
    q_thresh=0.1,
    d=150,
    d_ctrl=10000,
    genome_length=3088286401,
    max_gap=30,
    peak_amp=150,
    fe_cutoff=1.0,
):
    """
    Call peaks for a single chromosome.
    
    Parameters
    ----------
    d : int
        Treatment extension distance (extsize in MACS3)
    d_ctrl : int
        Large local control extension distance (llocal in MACS3)
    fe_cutoff : float
        Fold enrichment cutoff (default: 1.0)
        Peaks with signal_value < fe_cutoff are filtered out
    
    Returns
    -------
    pd.DataFrame
        Peak information for this chromosome
    """
    chrom_length = cp.max(ends).item()

    scale_llocal = d / d_ctrl
    lambda_bg = 2 * d * num_reads / genome_length
    q_thresh_log = -cp.log10(q_thresh)
    peak_amp = peak_amp - 1

    pileup_treat = _pileup(starts, ends, chrom_length, int(d / 2))
    
    # Large local pileup
    pileup_llocal = _pileup(starts, ends, chrom_length, int(d_ctrl / 2))
    pileup_ctrl = pileup_llocal * scale_llocal
    
    # Take maximum with genome-wide background
    pileup_ctrl = cp.maximum(pileup_ctrl, lambda_bg)
    del pileup_llocal

    p_values = _compute_poisson_cdfs(pileup_treat, pileup_ctrl)
    q_values = _replace_with_dict(p_values, pq_table)

    p_values[p_values == -cp.inf] = -1000
    q_values[q_values == cp.inf] = 1000

    peaks = _compute_peaks(q_values, q_thresh_log)

    if len(peaks) == 0:
        return pd.DataFrame()

    m = _merge_consecutive(peaks, max_gap)
    filtered_peaks = _filter_peaks(m, peak_amp)
    if len(filtered_peaks) == 0:
        return pd.DataFrame()

    merged_peaks = cp.asnumpy(filtered_peaks)
    peak_summits_args = _calculate_peak_summits(merged_peaks, cp.asnumpy(p_values))
    
    def extract_values(indexes, vec):
        return vec[indexes]
    
    q_summit = cp.asnumpy(extract_values(peak_summits_args, q_values))
    p_summit = cp.asnumpy(extract_values(peak_summits_args, p_values))
    treat_summit = cp.asnumpy(extract_values(peak_summits_args, pileup_treat))
    ctrl_summit = cp.asnumpy(extract_values(peak_summits_args, pileup_ctrl))

    df_op = pd.DataFrame(
        data={
            "start": merged_peaks[:, 0],
            "end": merged_peaks[:, 1],
            "peak": peak_summits_args - merged_peaks[:, 0],
            "signal_value": (treat_summit + 1) / (ctrl_summit + 1),
            "p_value": -1 * p_summit,
            "q_value": q_summit,
            "pileup": treat_summit,
        }
    )
    df_op["name"] = "."
    # Suppress pandas RuntimeWarning about invalid value in cast
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in cast')
        df_op["score"] = np.minimum(np.array(df_op["q_value"] * 10, dtype=np.int64), 1000)
    df_op["strand"] = "."
    
    # Filter by fold enrichment cutoff
    if fe_cutoff > 0:
        df_op = df_op[df_op["signal_value"] >= fe_cutoff].copy()

    del q_values, p_values, pileup_treat, m, peaks, pileup_ctrl, filtered_peaks
    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()
    cp._default_memory_pool.free_all_blocks()

    return df_op


def _gmacs_core(
    result_df,
    num_reads,
    q_thresh=0.1,
    d_treat=150,
    d_ctrl=10000,
    genome_length=3088286401,
    max_gap=30,
    peak_amp=150,
    fe_cutoff=1.0,
):
    """
    Core gmacs peak calling on a cuDF DataFrame.
    
    Parameters
    ----------
    result_df : cudf.DataFrame
        Sorted DataFrame with 'chrom', 'start', 'end' columns
    num_reads : int
        Total number of reads
    q_thresh : float
        Q-value threshold for peak calling
    d_treat : int
        Treatment extension distance (extsize in MACS3, default: 150)
    d_ctrl : int
        Large local control extension distance (llocal in MACS3, default: 10000)
    genome_length : int
        Total genome length
    max_gap : int
        Maximum gap for merging peaks
    peak_amp : int
        Minimum peak amplitude
    fe_cutoff : float
        Fold enrichment cutoff (default: 1.0)
        
    Returns
    -------
    pd.DataFrame
        Called peaks
    """
    logger.debug("Pre-grouping chromosomes...")
    chrom_groups = {}
    chroms = result_df['chrom'].unique().to_pandas()
    for chrom in chroms:
        chrom_df = result_df[result_df['chrom'] == chrom]
        chrom_groups[chrom] = (chrom_df['start'].to_cupy(), chrom_df['end'].to_cupy())
    
    logger.debug("Computing PQ table...")
    pq_table = _make_pq_table_from_groups(
        chrom_groups,
        num_reads,
        genome_length=genome_length,
        d_treat=d_treat,
        d_ctrl=d_ctrl,
    )

    peaks = pd.DataFrame()
    
    for chrom, (starts, ends) in chrom_groups.items():
        df_chr = _call_peaks_chrom(
            starts,
            ends,
            pq_table,
            num_reads,
            max_gap=max_gap,
            q_thresh=q_thresh,
            peak_amp=peak_amp,
            genome_length=genome_length,
            d=d_treat,
            d_ctrl=d_ctrl,
            fe_cutoff=fe_cutoff,
        )
        
        # Cleanup per-chromosome GPU memory
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
        cp._default_memory_pool.free_all_blocks()
        
        if len(df_chr) > 0:
            df_chr["chrom"] = chrom
            peaks = pd.concat([peaks, df_chr], axis=0)
    
    # Cleanup grouped data
    del chrom_groups

    if len(peaks) > 0:
        peaks = peaks[
            [
                "chrom", "start", "end", "name", "score", "strand",
                "signal_value", "p_value", "q_value", "peak",
            ]
        ]
        peaks['end'] = peaks['end'].astype(int)
        peaks['start'] = peaks['start'].astype(int)
    
    return peaks


# =============================================================================
# Chunked Parquet Reading for Cell Subset Extraction
# =============================================================================


def _read_fragments_for_barcodes_chunked(
    parquet_path: Path,
    barcodes: set,
    batch_size: int = 10,
) -> cudf.DataFrame:
    """
    Read fragments from parquet file, filtering to specific barcodes.
    
    Uses chunked reading (row groups) to avoid OOM when extracting
    a small subset of cells from a large parquet file.
    
    Parameters
    ----------
    parquet_path : Path
        Path to parquet file
    barcodes : set
        Set of barcodes to extract
    batch_size : int
        Number of row groups to load at once (default: 10)
        
    Returns
    -------
    cudf.DataFrame
        Filtered fragments DataFrame with chrom, start, end columns
    """
    import pyarrow.parquet as pq
    
    parquet_path = Path(parquet_path)
    
    # Get parquet file metadata
    pf = pq.ParquetFile(str(parquet_path))
    n_row_groups = pf.metadata.num_row_groups
    
    logger.debug(f"Reading {parquet_path.name}: {n_row_groups} row groups")
    
    chunks = []
    total_frags = 0
    
    # Process row groups in batches
    for batch_start in range(0, n_row_groups, batch_size):
        batch_end = min(batch_start + batch_size, n_row_groups)
        row_group_batch = list(range(batch_start, batch_end))
        
        # Read batch of row groups at once
        try:
            chunk_df = cudf.read_parquet(
                str(parquet_path),
                columns=['chrom', 'start', 'end', 'barcode'],
                row_groups=row_group_batch,
            )
        except TypeError:
            # Fallback for older cuDF versions - read entire file
            logger.warning("Row group reading not supported, reading full file")
            chunk_df = cudf.read_parquet(
                str(parquet_path),
                columns=['chrom', 'start', 'end', 'barcode'],
            )
            # Filter and return immediately
            barcodes_series = cudf.Series(list(barcodes))
            filtered = chunk_df[chunk_df['barcode'].isin(barcodes_series)]
            return filtered[['chrom', 'start', 'end']].sort_values(['chrom', 'start'])
        
        # Filter to target barcodes
        barcodes_series = cudf.Series(list(barcodes))
        filtered = chunk_df[chunk_df['barcode'].isin(barcodes_series)]
        
        if len(filtered) > 0:
            chunks.append(filtered[['chrom', 'start', 'end']])
            total_frags += len(filtered)
        
        del chunk_df, filtered
        mempool.free_all_blocks()
    
    if len(chunks) == 0:
        return cudf.DataFrame({'chrom': [], 'start': [], 'end': []})
    
    result = cudf.concat(chunks, ignore_index=True)
    result = result.sort_values(['chrom', 'start'])
    
    logger.debug(f"Extracted {total_frags:,} fragments for {len(barcodes):,} barcodes")
    
    return result


# =============================================================================
# Main Public API
# =============================================================================


def call_peaks(
    adata: "AnnData",
    groupby: str,
    parquet_path: Union[str, Path],
    *,
    genome: Union[str, dict] = "hg38",
    q_thresh: float = 0.05,
    d_treat: int = 200,
    d_ctrl: int = 10000,
    max_gap: int = 30,
    peak_amp: int = 200,
    fe_cutoff: float = 1.0,
    key_added: str = "gmacs",
    inplace: bool = True,
    verbose: bool = True,
    batch_size: int = 10,
) -> Optional[dict]:
    """
    GPU-accelerated peak calling per cluster using gmacs algorithm.
    
    Calls peaks for each group of cells defined by `groupby`, using the
    fragment data from parquet files. Similar to SnapATAC2's macs3 function
    but using GPU-accelerated gmacs.
    
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
        Used to compute genome_length for significance calculations.
    q_thresh : float
        Q-value threshold for peak calling (default: 0.05)
    d_treat : int
        Treatment extension distance in bp (default: 200)
    d_ctrl : int
        Large local control extension distance in bp (default: 10000).
        Corresponds to MACS3's --llocal parameter.
    max_gap : int
        Maximum gap for merging peaks (default: 30)
    peak_amp : int
        Minimum peak amplitude in bp (default: 200)
    fe_cutoff : float
        Fold enrichment cutoff (default: 1.0)
        Peaks with signal_value < fe_cutoff are filtered out
    key_added : str
        Key in `adata.uns` where peaks will be stored (default: 'gmacs')
    inplace : bool
        If True, store results in `adata.uns[key_added]` (default: True)
    verbose : bool
        If True, show tqdm progress bars (default: True)
    batch_size : int
        Number of parquet row groups to load at once (default: 10).
        Increase for faster processing, decrease to reduce memory usage.
        
    Returns
    -------
    dict[str, pd.DataFrame] or None
        If `inplace=False`, returns dict mapping group names to peak DataFrames.
        Each DataFrame has columns: chrom, start, end, name, score, strand,
        signal_value, p_value, q_value, peak.
        
    Examples
    --------
    >>> import gatac
    >>> # Load tiled AnnData with cluster assignments
    >>> adata = sc.read_h5ad("tiled_matrix.h5ad")
    >>> # Call peaks per cluster (default settings for ATAC-seq)
    >>> gatac.tl.call_peaks(
    ...     adata,
    ...     groupby="leiden",
    ...     parquet_path="/path/to/fragments.parquet",
    ... )
    >>> # Access peaks for cluster "0"
    >>> peaks_cluster0 = adata.uns["gmacs"]["0"]
    """
    parquet_path = Path(parquet_path)
    
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
    
    groups = adata.obs[groupby].unique()
    logger.info(f"Calling peaks for {len(groups)} groups in '{groupby}'")
    
    # Determine parquet file mapping
    if parquet_path.is_dir():
        # Directory mode - use source_file obs
        if 'source_file' not in adata.obs.columns:
            raise ValueError(
                "parquet_path is a directory but 'source_file' not in adata.obs. "
                "Please provide a single parquet file path instead."
            )
        parquet_dir = parquet_path
        source_files = adata.obs['source_file'].unique()
        logger.info(f"Using {len(source_files)} source files from source_file obs")
    else:
        # Single file mode
        parquet_dir = None
    
    def _h5ad_to_parquet_name(h5ad_name: str) -> str:
        """
        Convert h5ad filename to parquet filename.
        
        Maps {sample}_tile.h5ad -> {sample}.parquet
        """
        # Remove .h5ad extension
        name = h5ad_name.replace('.h5ad', '')
        # Remove _tile suffix if present
        if name.endswith('_tile'):
            name = name[:-5]
        # Add .parquet extension
        return f"{name}.parquet"
    
    peaks_dict = {}
    
    # Import tqdm for progress bars
    if verbose:
        from tqdm.auto import tqdm
        groups_iter = tqdm(groups, desc="Groups", position=0)
    else:
        groups_iter = groups
    
    for group in groups_iter:
        group_str = str(group)
        logger.info(f"Processing group: {group_str}")
        
        # Get barcodes for this group
        mask = adata.obs[groupby] == group
        group_barcodes = set(adata.obs.index[mask])
        logger.debug(f"  {len(group_barcodes):,} cells in group")
        
        # Collect fragments from parquet file(s)
        if parquet_dir is not None:
            # Multi-file mode: read from each source file
            all_fragments = []
            source_files_for_group = adata.obs.loc[mask, 'source_file'].unique()
            
            if verbose:
                file_iter = tqdm(
                    source_files_for_group, 
                    desc=f"  Files ({group_str})", 
                    position=1, 
                    leave=False
                )
            else:
                file_iter = source_files_for_group
            
            for src_file in file_iter:
                # Convert h5ad name to parquet name
                parquet_name = _h5ad_to_parquet_name(src_file)
                src_path = parquet_dir / parquet_name
                if not src_path.exists():
                    logger.warning(f"Parquet file not found: {src_path}")
                    continue
                # Get barcodes from this file that are in this group
                file_mask = mask & (adata.obs['source_file'] == src_file)
                file_barcodes = set(adata.obs.index[file_mask])
                frags = _read_fragments_for_barcodes_chunked(src_path, file_barcodes, batch_size=batch_size)
                if len(frags) > 0:
                    all_fragments.append(frags)
            
            if len(all_fragments) == 0:
                logger.warning(f"  No fragments found for group {group_str}")
                peaks_dict[group_str] = pd.DataFrame()
                continue
            
            fragments_df = cudf.concat(all_fragments, ignore_index=True)
            fragments_df = fragments_df.sort_values(['chrom', 'start'])
        else:
            # Single file mode
            fragments_df = _read_fragments_for_barcodes_chunked(parquet_path, group_barcodes, batch_size=batch_size)
        
        if len(fragments_df) == 0:
            logger.warning(f"  No fragments found for group {group_str}")
            peaks_dict[group_str] = pd.DataFrame()
            continue
        
        num_reads = len(fragments_df)
        logger.info(f"  {num_reads:,} fragments, calling peaks...")
        
        # Call peaks using gmacs
        peaks = _gmacs_core(
            fragments_df,
            num_reads,
            q_thresh=q_thresh,
            d_treat=d_treat,
            d_ctrl=d_ctrl,
            genome_length=genome_length,
            max_gap=max_gap,
            peak_amp=peak_amp,
            fe_cutoff=fe_cutoff,
        )
        
        peaks_dict[group_str] = peaks
        logger.info(f"  Found {len(peaks):,} peaks")
        
        # Cleanup
        del fragments_df
        gc.collect()
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
    
    if inplace:
        adata.uns[key_added] = peaks_dict
        return None
    else:
        return peaks_dict


def _remove_overlapping_peaks_gpu(starts_cp, ends_cp, keep_mask_cp):
    """
    GPU-accelerated removal of overlapping peaks.
    
    Uses vectorized operations to check if each peak overlaps with any
    previously kept peak. Processes peaks in order of significance (p-value).
    
    Parameters
    ----------
    starts_cp : cp.ndarray
        Peak start positions (sorted by p-value)
    ends_cp : cp.ndarray
        Peak end positions (sorted by p-value)
    keep_mask_cp : cp.ndarray (bool)
        Boolean mask, initially all True, updated to False for overlapping peaks
        
    Returns
    -------
    cp.ndarray (bool)
        Updated mask where True indicates peaks to keep
    """
    n_peaks = len(starts_cp)
    
    # Process in chunks to avoid memory explosion with large peak sets
    chunk_size = min(5000, n_peaks)
    
    for i in range(0, n_peaks, chunk_size):
        chunk_end = min(i + chunk_size, n_peaks)
        
        # For each peak in this chunk, check against all previously kept peaks
        chunk_starts = starts_cp[i:chunk_end]
        chunk_ends = ends_cp[i:chunk_end]
        chunk_size_actual = len(chunk_starts)
        
        if i > 0:
            # Get all previously kept peaks
            prev_kept_mask = keep_mask_cp[:i]
            prev_kept_indices = cp.where(prev_kept_mask)[0]
            
            if len(prev_kept_indices) > 0:
                prev_starts = starts_cp[prev_kept_indices]
                prev_ends = ends_cp[prev_kept_indices]
                
                # Check overlap: chunk_peak overlaps prev_peak if
                # chunk_start < prev_end AND chunk_end > prev_start
                # Shape: (chunk_size, n_prev_kept)
                overlaps = (
                    (chunk_starts[:, None] < prev_ends[None, :]) &
                    (chunk_ends[:, None] > prev_starts[None, :])
                )
                
                # If any overlap exists for a peak, mark it as not kept
                has_overlap = cp.any(overlaps, axis=1)
                keep_mask_cp[i:chunk_end] = keep_mask_cp[i:chunk_end] & ~has_overlap
                
                del overlaps, has_overlap, prev_starts, prev_ends
    
    return keep_mask_cp


def merge_peaks(
    peaks: Union[dict[str, pd.DataFrame], "AnnData"],
    chrom_sizes: Optional[Union[str, dict[str, int]]] = None,
    half_width: int = 250,
    use_rep: str = "gmacs",
    key_added: str = "peaks",
    inplace: bool = True,
) -> Optional[pd.DataFrame]:
    """Merge peaks from different groups into fixed-width, non-overlapping peaks.

    This mirrors the behavior of SnapATAC2's `merge_peaks` by expanding each peak
    summit by `half_width` on both sides, then iteratively keeping the most
    significant peak (smallest p-value) and discarding any overlapping peaks.
    
    GPU-accelerated for improved performance on large peak sets.

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

    # Process each group's peaks
    for _, df in peaks_dict.items():
        if df is None or len(df) == 0:
            continue
        missing = required_cols.difference(df.columns)
        if missing:
            raise ValueError(f"Missing required columns in peaks: {sorted(missing)}")

        # Convert to cuDF for GPU operations
        df_gpu = cudf.DataFrame(df)
        
        # Calculate summit position
        if "peak" in df_gpu.columns:
            summit = df_gpu["start"].astype(int) + df_gpu["peak"].astype(int)
        else:
            summit = ((df_gpu["start"].astype(int) + df_gpu["end"].astype(int)) // 2).astype(int)

        # Expand to fixed width
        df_gpu["start"] = summit - half_width
        df_gpu["end"] = summit + half_width
        df_gpu["peak"] = half_width

        # Clamp to chromosome sizes on GPU
        for chrom in df_gpu['chrom'].unique().to_pandas():
            chrom_size = chrom_sizes.get(chrom)
            if chrom_size is None:
                continue
            
            chrom_mask = (df_gpu['chrom'] == chrom).to_cupy()
            
            # Work directly with cupy arrays to avoid index alignment issues
            starts_all = df_gpu['start'].to_cupy()
            ends_all = df_gpu['end'].to_cupy()
            
            # Clip only the chromosome-specific values
            starts_all[chrom_mask] = cp.clip(starts_all[chrom_mask], 0, max(chrom_size - 1, 0))
            ends_all[chrom_mask] = cp.clip(ends_all[chrom_mask], 1, chrom_size)
            ends_all[chrom_mask] = cp.maximum(ends_all[chrom_mask], starts_all[chrom_mask] + 1)
            
            # Assign back the entire columns
            df_gpu['start'] = cudf.Series(starts_all)
            df_gpu['end'] = cudf.Series(ends_all)

        expanded_peaks.append(df_gpu.to_pandas())
        
        del df_gpu
        mempool.free_all_blocks()

    if len(expanded_peaks) == 0:
        result = pd.DataFrame()
        if adata is not None and inplace:
            adata.uns[key_added] = result
            return None
        return result

    # Concatenate all peaks
    all_peaks = pd.concat(expanded_peaks, ignore_index=True)
    
    # Convert to cuDF for GPU-accelerated merging
    all_peaks_gpu = cudf.DataFrame(all_peaks)
    
    merged_chunks = []
    
    # Process each chromosome separately
    for chrom in all_peaks_gpu['chrom'].unique().to_pandas():
        chrom_df = all_peaks_gpu[all_peaks_gpu['chrom'] == chrom]
        
        if len(chrom_df) == 0:
            continue
        
        # Sort by p-value (ascending = most significant first)
        chrom_df = chrom_df.sort_values("p_value", ascending=True)
        
        # Get GPU arrays
        starts_cp = chrom_df['start'].to_cupy().astype(cp.int64)
        ends_cp = chrom_df['end'].to_cupy().astype(cp.int64)
        
        # Initialize keep mask (all True initially)
        keep_mask_cp = cp.ones(len(starts_cp), dtype=bool)
        
        # Remove overlapping peaks using GPU
        keep_mask_cp = _remove_overlapping_peaks_gpu(starts_cp, ends_cp, keep_mask_cp)
        
        # Filter to kept peaks
        keep_mask_cpu = cp.asnumpy(keep_mask_cp)
        chrom_df_filtered = chrom_df.to_pandas()[keep_mask_cpu]
        
        if len(chrom_df_filtered) > 0:
            merged_chunks.append(chrom_df_filtered)
        
        del starts_cp, ends_cp, keep_mask_cp, chrom_df
        mempool.free_all_blocks()
    
    if len(merged_chunks) == 0:
        result = pd.DataFrame(columns=all_peaks.columns)
    else:
        result = pd.concat(merged_chunks, ignore_index=True)
    
    # Cleanup
    del all_peaks_gpu
    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()
    
    # Store in AnnData if requested
    if adata is not None and inplace:
        adata.uns[key_added] = result
        return None
    
    return result


def _count_fragments_in_peaks_gpu(
    fragments_df: cudf.DataFrame,
    peaks_gpu: cudf.DataFrame,
    barcode_to_idx: dict,
    n_peaks: int,
) -> tuple[list, list, list]:
    """
    Count fragment overlaps with peaks using GPU-accelerated operations.
    
    Uses binary search for efficient interval overlap detection and
    GPU-native aggregation with cudf groupby.
    
    Parameters
    ----------
    fragments_df : cudf.DataFrame
        Fragments with 'chrom', 'start', 'end', 'barcode' columns
    peaks_gpu : cudf.DataFrame
        Peaks with 'chrom', 'start', 'end', 'peak_idx' columns, sorted by chrom/start
    barcode_to_idx : dict
        Mapping from barcode to cell index
    n_peaks : int
        Total number of peaks
        
    Returns
    -------
    tuple[list, list, list]
        (rows, cols, data) for sparse matrix construction
    """
    # Pre-create barcode -> cell_idx mapping on GPU
    barcode_mapping = cudf.DataFrame({
        'barcode': list(barcode_to_idx.keys()),
        'cell_idx': list(barcode_to_idx.values()),
    })
    
    all_results = []
    
    # Process each chromosome
    chroms = fragments_df['chrom'].unique().to_pandas()
    
    for chrom in chroms:
        chrom_frags = fragments_df[fragments_df['chrom'] == chrom]
        chrom_peaks = peaks_gpu[peaks_gpu['chrom'] == chrom]
        
        if len(chrom_peaks) == 0 or len(chrom_frags) == 0:
            continue
        
        n_frags = len(chrom_frags)
        n_chrom_peaks = len(chrom_peaks)
        
        # Get peak arrays on GPU (peaks are sorted by start)
        peak_starts = chrom_peaks['start'].to_cupy()
        peak_ends = chrom_peaks['end'].to_cupy()
        peak_indices = chrom_peaks['peak_idx'].to_cupy()
        
        # Get fragment arrays on GPU
        frag_starts = chrom_frags['start'].to_cupy()
        frag_ends = chrom_frags['end'].to_cupy()
        
        # Use searchsorted for efficient overlap detection
        # For each fragment, find candidate peaks using binary search:
        # - right_bound: first peak where peak_start >= frag_end (no overlap possible)
        # - left_bound: first peak where peak_end > frag_start (overlap possible)
        
        # Process fragments in batches to manage memory
        frag_batch_size = min(50000, n_frags)
        
        for frag_batch_start in range(0, n_frags, frag_batch_size):
            frag_batch_end = min(frag_batch_start + frag_batch_size, n_frags)
            batch_starts = frag_starts[frag_batch_start:frag_batch_end]
            batch_ends = frag_ends[frag_batch_start:frag_batch_end]
            batch_size = len(batch_starts)
            
            # Pre-create batch fragment indices (once per batch, not per chunk)
            batch_frag_indices = cp.arange(frag_batch_start, frag_batch_end, dtype=cp.int64)
            
            # Binary search to find candidate peak ranges
            # right_bounds[i] = first peak where peak_start >= frag_end[i]
            right_bounds = cp.searchsorted(peak_starts, batch_ends, side='left')
            
            # For each fragment, we only need to check peaks in [0, right_bound)
            # But we still need to verify peak_end > frag_start
            
            # Use chunked broadcasting for remaining candidates
            # This is much smaller than full broadcast due to searchsorted pruning
            peak_chunk_size = min(50000, n_chrom_peaks)
            
            for peak_chunk_start in range(0, n_chrom_peaks, peak_chunk_size):
                peak_chunk_end = min(peak_chunk_start + peak_chunk_size, n_chrom_peaks)
                chunk_peak_starts = peak_starts[peak_chunk_start:peak_chunk_end]
                chunk_peak_ends = peak_ends[peak_chunk_start:peak_chunk_end]
                chunk_peak_indices = peak_indices[peak_chunk_start:peak_chunk_end]
                
                # Only check fragments where right_bound > peak_chunk_start
                # This prunes fragments that can't overlap this peak chunk
                frag_mask = right_bounds > peak_chunk_start
                if not cp.any(frag_mask):
                    continue
                
                masked_starts = batch_starts[frag_mask]
                masked_ends = batch_ends[frag_mask]
                masked_frag_indices = batch_frag_indices[frag_mask]
                
                # Compute overlap matrix (smaller due to masking)
                overlaps = (
                    (masked_starts[:, None] < chunk_peak_ends[None, :]) &
                    (masked_ends[:, None] > chunk_peak_starts[None, :])
                )
                
                # Find overlapping pairs
                frag_local_idx, peak_local_idx = cp.where(overlaps)
                
                if len(frag_local_idx) == 0:
                    del overlaps
                    continue
                
                # Map to global fragment indices
                global_frag_idx = masked_frag_indices[frag_local_idx]
                global_peak_idx = chunk_peak_indices[peak_local_idx]
                
                # Store overlap pairs (will aggregate later)
                overlap_df = cudf.DataFrame({
                    'frag_idx': cudf.Series(global_frag_idx),
                    'peak_idx': cudf.Series(global_peak_idx),
                })
                all_results.append(overlap_df)
                
                del overlaps, frag_local_idx, peak_local_idx
            
            del batch_starts, batch_ends, right_bounds
        
        del frag_starts, frag_ends, peak_starts, peak_ends, peak_indices
        mempool.free_all_blocks()
    
    if len(all_results) == 0:
        return [], [], []
    
    # Concatenate all overlap results
    all_overlaps = cudf.concat(all_results, ignore_index=True)
    del all_results
    
    # Get barcodes for overlapping fragments (GPU join)
    # Create a temporary column for join
    frag_barcodes = fragments_df[['barcode']].reset_index(drop=True)
    frag_barcodes['frag_idx'] = cp.arange(len(frag_barcodes))
    
    # Join to get barcodes
    all_overlaps = all_overlaps.merge(frag_barcodes, on='frag_idx', how='left')
    
    # Join to get cell indices
    all_overlaps = all_overlaps.merge(barcode_mapping, on='barcode', how='inner')
    
    # Aggregate by (cell_idx, peak_idx) using GPU groupby
    counts = all_overlaps.groupby(['cell_idx', 'peak_idx']).size().reset_index(name='count')
    
    # Convert to CPU for sparse matrix construction
    rows = counts['cell_idx'].to_pandas().tolist()
    cols = counts['peak_idx'].to_pandas().tolist()
    data = counts['count'].to_pandas().tolist()
    
    del all_overlaps, counts, frag_barcodes
    mempool.free_all_blocks()
    
    return rows, cols, data
    
    return rows, cols, data


def _read_and_count_fragments_batched(
    parquet_path: Path,
    barcodes: set,
    peaks_gpu: cudf.DataFrame,
    barcode_to_idx: dict,
    n_peaks: int,
    batch_size: int = 10,
) -> tuple[list, list, list]:
    """
    Read fragments from parquet in batches and count overlaps with peaks.
    
    Uses chunked row group reading to avoid OOM when processing large files.
    
    Parameters
    ----------
    parquet_path : Path
        Path to parquet file
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
        
    Returns
    -------
    tuple[list, list, list]
        (rows, cols, data) for sparse matrix construction
    """
    import pyarrow.parquet as pq
    
    parquet_path = Path(parquet_path)
    
    # Get parquet file metadata
    pf = pq.ParquetFile(str(parquet_path))
    n_row_groups = pf.metadata.num_row_groups
    
    logger.debug(f"Reading {parquet_path.name}: {n_row_groups} row groups")
    
    all_rows = []
    all_cols = []
    all_data = []
    
    # Process row groups in batches
    for batch_start in range(0, n_row_groups, batch_size):
        batch_end = min(batch_start + batch_size, n_row_groups)
        row_group_batch = list(range(batch_start, batch_end))
        
        # Read batch of row groups
        try:
            chunk_df = cudf.read_parquet(
                str(parquet_path),
                columns=['chrom', 'start', 'end', 'barcode'],
                row_groups=row_group_batch,
            )
        except TypeError:
            # Fallback for older cuDF versions - read entire file once
            logger.warning("Row group reading not supported, reading full file")
            chunk_df = cudf.read_parquet(
                str(parquet_path),
                columns=['chrom', 'start', 'end', 'barcode'],
            )
            # Filter to target barcodes
            barcodes_series = cudf.Series(list(barcodes))
            filtered = chunk_df[chunk_df['barcode'].isin(barcodes_series)]
            
            if len(filtered) > 0:
                rows, cols, data = _count_fragments_in_peaks_gpu(
                    filtered, peaks_gpu, barcode_to_idx, n_peaks
                )
                return rows, cols, data
            return [], [], []
        
        # Filter to target barcodes
        barcodes_series = cudf.Series(list(barcodes))
        filtered = chunk_df[chunk_df['barcode'].isin(barcodes_series)]
        
        if len(filtered) > 0:
            rows, cols, data = _count_fragments_in_peaks_gpu(
                filtered, peaks_gpu, barcode_to_idx, n_peaks
            )
            all_rows.extend(rows)
            all_cols.extend(cols)
            all_data.extend(data)
        
        del chunk_df, filtered
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
    
    return all_rows, all_cols, all_data


def make_peak_matrix(
    adata: "AnnData",
    parquet_path: Union[str, Path, list[Union[str, Path]]],
    *,
    use_rep: str = "peaks",
    peak_file: Optional[Path] = None,
    genome: Union[str, dict] = "hg38",
    inplace: bool = False,
    batch_size: int = 50,
    verbose: bool = True,
) -> Optional["AnnData"]:
    """Generate cell by peak count matrix.

    This function counts fragments overlapping with peak regions to create
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
        Used for validation.
    inplace
        Whether to add the peak matrix to the AnnData object (not recommended,
        will replace .X). If False, returns a new AnnData object.
    batch_size
        Number of parquet row groups to load at once (default: 50).
        Larger values are faster but use more GPU memory.
    verbose
        If True, show progress bars.

    Returns
    -------
    AnnData or None
        If inplace=False, returns a new AnnData object with the peak matrix.
        Otherwise returns None and modifies adata in place.

    Examples
    --------
    >>> import gatac
    >>> # After calling peaks and merging
    >>> gatac.tl.merge_peaks(adata, use_rep="gmacs", key_added="peaks")
    >>> # Create peak matrix from a single file
    >>> peak_mat = gatac.tl.make_peak_matrix(
    ...     adata,
    ...     parquet_path="/path/to/fragments.parquet",
    ...     use_rep="peaks",
    ...     inplace=False,
    ... )
    >>> # Create peak matrix from multiple files
    >>> peak_mat = gatac.tl.make_peak_matrix(
    ...     adata,
    ...     parquet_path=["/path/to/sample1.parquet", "/path/to/sample2.parquet"],
    ...     use_rep="peaks",
    ...     inplace=False,
    ... )
    >>> # Create peak matrix from directory (uses source_file obs)
    >>> peak_mat = gatac.tl.make_peak_matrix(
    ...     adata,
    ...     parquet_path="/path/to/parquet_dir/",
    ...     use_rep="peaks",
    ...     inplace=False,
    ... )
    """
    import scipy.sparse as sp
    from anndata import AnnData as AD
    
    # Get peaks
    if peak_file is not None and use_rep is not None:
        raise ValueError("Cannot specify both peak_file and use_rep")
    
    if peak_file is not None:
        # Read from BED file
        peaks_df = pd.read_csv(
            peak_file,
            sep='\t',
            header=None,
            usecols=[0, 1, 2],
            names=['chrom', 'start', 'end'],
        )
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
    
    def _h5ad_to_parquet_name(h5ad_name: str) -> str:
        """Convert h5ad filename to parquet filename."""
        name = h5ad_name.replace('.h5ad', '')
        if name.endswith('_tile'):
            name = name[:-5]
        return f"{name}.parquet"
    
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
                parquet_name = _h5ad_to_parquet_name(src_file)
                parquet_file = parquet_path / parquet_name
                file_to_barcodes[parquet_file].add(barcode)
            
            parquet_files = list(file_to_barcodes.keys())
            logger.info(f"Processing {len(parquet_files)} parquet files (directory mode)")
        else:
            # Single file mode
            parquet_files = [parquet_path]
            file_to_barcodes = {parquet_path: all_barcodes}
            logger.info("Processing single parquet file")
    
    # Collect sparse matrix components
    all_rows = []
    all_cols = []
    all_data = []
    
    # Process each parquet file
    if verbose:
        from tqdm.auto import tqdm
        file_iter = tqdm(parquet_files, desc="Parquet files")
    else:
        file_iter = parquet_files
    
    for parquet_file in file_iter:
        if not parquet_file.exists():
            logger.warning(f"Parquet file not found: {parquet_file}")
            continue
        
        barcodes_for_file = file_to_barcodes[parquet_file]
        
        if len(barcodes_for_file) == 0:
            continue
        
        logger.debug(f"Processing {parquet_file.name}: {len(barcodes_for_file):,} barcodes")
        
        # Read and count fragments in batches
        rows, cols, data = _read_and_count_fragments_batched(
            parquet_file,
            barcodes_for_file,
            peaks_gpu,
            barcode_to_idx,
            n_peaks,
            batch_size=batch_size,
        )
        
        all_rows.extend(rows)
        all_cols.extend(cols)
        all_data.extend(data)
        
        # Cleanup
        gc.collect()
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
    
    # Aggregate duplicate (row, col) entries by summing
    # This handles cases where same cell-peak pair appears in multiple batches
    if len(all_rows) > 0:
        from collections import Counter
        aggregated = Counter()
        for r, c, d in zip(all_rows, all_cols, all_data):
            aggregated[(r, c)] += d
        
        all_rows = [k[0] for k in aggregated.keys()]
        all_cols = [k[1] for k in aggregated.keys()]
        all_data = list(aggregated.values())
    
    # Create sparse matrix
    count_matrix = sp.csr_matrix(
        (all_data, (all_rows, all_cols)),
        shape=(n_cells, n_peaks),
        dtype=np.int32,
    )
    
    logger.info(f"Peak matrix: {n_cells:,} cells × {n_peaks:,} peaks")
    if n_cells * n_peaks > 0:
        logger.info(f"Sparsity: {100 * (1 - count_matrix.nnz / (n_cells * n_peaks)):.2f}%")
    
    # Create peak names
    peak_names = [
        f"{row.chrom}:{row.start}-{row.end}"
        for row in peaks_df.itertuples(index=False)
    ]
    
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
