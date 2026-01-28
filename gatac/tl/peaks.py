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
