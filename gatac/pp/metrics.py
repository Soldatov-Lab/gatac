"""
GPU-accelerated metrics for ATAC-seq data.

Implementation:
1. compute_metrics: streams the fragments onto the GPU in batches of row
   groups to bound peak memory, on small and large datasets alike.
"""
from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, List, Sequence, Union

import cudf
import cupy as cp
import cupyx
import numpy as np
import pandas as pd

from ._utils import async_memory_pool, cleanup_gpu_memory
from .._readers import read_gtf
from ..io.fragments import FragmentFile

logger = logging.getLogger(__name__)


_TSSE_KERNEL = cp.RawKernel(r'''
extern "C" __global__
void compute_tsse_kernel(
    const int* ins_pos,
    const int* ins_cell_idx,
    const int* left_bounds,
    const int* right_bounds,
    const int* tss_pos,
    const signed char* tss_strand,
    int n_insertions,
    int window_size,
    int half_smooth,
    float* out_data  // shape (n_cells, 3)
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_insertions) return;

    int pos = ins_pos[i];
    int cell = ins_cell_idx[i];
    int lb = left_bounds[i];
    int rb = right_bounds[i];

    for (int t_idx = lb; t_idx < rb; ++t_idx) {
        int t_pos = tss_pos[t_idx];
        bool negative = (tss_strand[t_idx] != 0);
        
        int dist = pos - t_pos;
        int offset = negative ? -dist : dist;
        int offset_idx = offset + window_size;

        if (offset_idx >= 0 && offset_idx < 100) {
            atomicAdd(&out_data[cell * 3 + 0], 1.0f);
        } else if (offset_idx >= window_size - half_smooth && offset_idx <= window_size + half_smooth) {
            atomicAdd(&out_data[cell * 3 + 1], 1.0f);
        } else if (offset_idx > window_size * 2 - 100 && offset_idx <= window_size * 2) {
            atomicAdd(&out_data[cell * 3 + 2], 1.0f);
        }
    }
}
''', 'compute_tsse_kernel')

def _load_tss_from_gtf(gtf_path: str | Path) -> pd.DataFrame:
    """Distinct transcript TSSs of a GTF file: ``chrom``, ``tss`` (0-based), ``strand``.

    Private helper used by :func:`compute_metrics` when the caller passes a
    ``gtf_path`` instead of a pre-loaded TSS DataFrame.
    """
    logger.info(f"Loading TSS from {gtf_path}")
    df = read_gtf(gtf_path, "transcript")

    # TSS is start-1 for + strand, end-1 for - strand (0-based)
    tss = np.where(df['strand'] == '-', df['end'] - 1, df['start'] - 1)
    tss_df = pd.DataFrame({'chrom': df['chrom'], 'tss': tss, 'strand': df['strand']})
    tss_df = tss_df.drop_duplicates(ignore_index=True)

    logger.info(f"Loaded {len(tss_df):,} unique TSSs")
    return tss_df


def compute_metrics(
    parquet_path: str | Path | Sequence[str | Path],
    tss_source: Union[str, Path, pd.DataFrame],
    window_size: int = 2000,
    smooth_window: int = 11,
    min_unique_frags: int = 100,
    chrom_sizes: dict[str, int] | None = None,
    exclude_chroms: list[str] | None = ['chrM', 'M'],
    row_groups_per_batch: int = 64,
    workers: int = 4,
) -> cudf.DataFrame:
    """
    Compute TSS enrichment scores using GPU-accelerated streaming via row groups.

    This function streams fragment data in batches of row groups to balance
    speed and GPU memory usage.

    Parameters
    ----------
    parquet_path : str, Path or list of them
        Fragment parquet file(s). With several files the TSS table is
        loaded once, the files are processed a few at a time, and their
        rows are concatenated in the order given.
    tss_source : str, Path, or DataFrame
        Either a path to a GTF file (the TSS table is loaded internally)
        or a pre-built pandas, cuDF or Polars DataFrame of TSS positions
        (columns: ``chrom``, ``tss`` (0-based), ``strand``).  When a pre-built
        DataFrame is passed, the underlying GTF file is not re-read.
    window_size : int
        Distance around TSS to consider (default: 2000)
    smooth_window : int
        Window size for smoothing the TSS signal (default: 11)
    min_unique_frags : int
        Minimum unique fragments per cell to include in output (default: 100)
    chrom_sizes : dict[str, int] | None
        If provided, only include fragments on these chromosomes.
    exclude_chroms : list[str] | None
        Chromosomes to exclude from TSS enrichment calculation (default: ["chrM", "M"])
    row_groups_per_batch : int
        Number of parquet row groups to process in each GPU batch (default: 64)
    workers : int
        Number of files processed at once (default: 4).

    Returns
    -------
    results : cudf.DataFrame
        DataFrame with columns: ['barcode', 'tsse_score', 'n_unique', 'duplicate_fraction', 'mito_fraction']

    Examples
    --------
    >>> import gatac as ga
    >>> # Pass a GTF path directly
    >>> metrics = ga.pp.compute_metrics(
    ...     "pbmc.parquet",
    ...     "GRCh38.gtf.gz",
    ...     min_unique_frags=100,
    ...     exclude_chroms=["chrM", "M"],
    ... )
    >>> metrics.columns.tolist()
    ['barcode', 'tsse_score', 'n_unique', 'duplicate_fraction', 'mito_fraction']
    >>> # Several samples: the GTF is read once
    >>> metrics = ga.pp.compute_metrics(["s1.parquet", "s2.parquet"], "GRCh38.gtf.gz")
    """
    # Resolve TSS source: accept either a GTF path or a pre-loaded DataFrame.
    if isinstance(tss_source, (str, os.PathLike)):
        tss_df = _load_tss_from_gtf(tss_source)
    elif isinstance(tss_source, pd.DataFrame):
        tss_df = tss_source
    elif hasattr(tss_source, "to_pandas"):     # cuDF or Polars
        tss_df = tss_source.to_pandas()
    else:
        raise TypeError(
            "tss_source must be a GTF path or a pandas, cuDF or Polars DataFrame, "
            f"not {type(tss_source).__name__}"
        )

    single = isinstance(parquet_path, (str, os.PathLike))
    paths = [parquet_path] if single else list(parquet_path)
    # Several files at once from threads, so that one file's reads overlap
    # another's kernels, under a stream-ordered memory pool. Largest first,
    # so that they do not run alone at the end.
    order = sorted(range(len(paths)), key=lambda i: -os.path.getsize(paths[i]))
    with async_memory_pool(), ThreadPoolExecutor(max(1, min(len(paths), workers))) as pool:
        futures = {
            i: pool.submit(
                _compute_metrics_file, paths[i], tss_df, window_size, smooth_window,
                min_unique_frags, chrom_sizes, exclude_chroms, row_groups_per_batch,
            )
            for i in order
        }
        results = [futures[i].result() for i in range(len(paths))]
    cleanup_gpu_memory()
    if single:
        return results[0]
    return cudf.concat(results, ignore_index=True)


def _compute_metrics_file(
    parquet_path, tss_df, window_size, smooth_window, min_unique_frags,
    chrom_sizes, exclude_chroms, row_groups_per_batch,
) -> cudf.DataFrame:
    """:func:`compute_metrics` of one fragment file."""
    logger.info(f"Computing metrics of {Path(parquet_path).name} "
                f"(streaming mode, row_groups_per_batch={row_groups_per_batch})")
    fragments = FragmentFile(parquet_path)

    # =========================================================================
    # Phase 1: Per-barcode fragment and read counts
    # =========================================================================
    valid_chroms_qc = list(chrom_sizes.keys()) if chrom_sizes else None
    qc_summary = fragments.barcode_counts(chroms=valid_chroms_qc, mito_chroms=['chrM', 'M'])

    if len(qc_summary) == 0:
        logger.warning("No cells passed the minimum fragment filter!")
        return cudf.DataFrame({
            'barcode': [], 'tsse_score': [], 'n_unique': [], 
            'duplicate_fraction': [], 'mito_fraction': []
        })
    qc_summary = cudf.from_pandas(qc_summary.sort_values('barcode', ignore_index=True))

    # Filter by min_unique_frags and compute final QC metrics
    qc_metrics = qc_summary[qc_summary['n_unique'] >= min_unique_frags].copy()
    qc_metrics['duplicate_fraction'] = (qc_metrics['n_total'] - qc_metrics['n_unique']) / (qc_metrics['n_total'] + 1e-9)
    qc_metrics['mito_fraction'] = qc_metrics['n_mito'] / (qc_metrics['n_total'] + 1e-9)
    
    n_cells = len(qc_metrics)
    logger.info(f"Found {n_cells:,} cells with >= {min_unique_frags} unique fragments")
    
    # Clean up summary
    del qc_summary

    # =========================================================================
    # Phase 2: Prepare TSS data on GPU with chromosome encoding
    # =========================================================================
    # Filter TSS
    if exclude_chroms:
        tss_filtered = tss_df[~tss_df['chrom'].isin(exclude_chroms)]
    else:
        tss_filtered = tss_df

    # Group TSS by chromosome for efficient lookup. The kernel adds whole
    # counts over every TSS in a window, so the order of tied TSSs is free.
    tss_by_chrom = {}
    for chrom, tss_sub in tss_filtered.groupby('chrom', sort=True):
        tss_sub = tss_sub.sort_values('tss', kind='stable')
        tss_pos = cp.asarray(tss_sub['tss'].to_numpy(), dtype='int32')
        tss_strand = cp.asarray((tss_sub['strand'] == '-').to_numpy(), dtype='int8')
        tss_by_chrom[chrom] = (tss_pos, tss_strand)
    tss_chroms = list(tss_by_chrom)

    # =========================================================================
    # Phase 3: Stream fragments and compute TSSe using row group batches
    # =========================================================================
    # Initialize accumulator on GPU (3 bins per cell)
    data_cp = cp.zeros((n_cells, 3), dtype='float32')
    half_smooth = smooth_window // 2
    
    stream_chroms = [c for c in tss_chroms if valid_chroms_qc is None or c in valid_chroms_qc]
    cells = qc_metrics['barcode'].to_arrow().to_pylist()
    num_row_groups = fragments.num_row_groups

    logger.info(f"Processing {num_row_groups} row groups in batches of {row_groups_per_batch}...")
    
    batches = fragments.iter_batches(
        ['chrom', 'start', 'end'], row_groups_per_batch, chroms=stream_chroms, cells=cells,
    )
    for i, chunk_df in zip(range(0, num_row_groups, row_groups_per_batch), batches):
        if len(chunk_df) == 0:
            continue
        
        # Process fragments, one chromosome at a time
        chunk_chrom = chunk_df['chrom'].values
        tpb = 256
        
        for ci in cp.unique(chunk_chrom).get():
            tss_pos, tss_strand = tss_by_chrom[stream_chroms[ci]]
            on_chrom = chunk_chrom == ci
            
            cell_idx = chunk_df['cell'].values[on_chrom].astype('int32')
            starts = chunk_df['start'].values[on_chrom].astype('int32')
            ends = (chunk_df['end'].values[on_chrom] - 1).astype('int32')
            
            n_frags = len(starts)
            bpg = (n_frags + tpb - 1) // tpb
            
            # Process starts
            lb = tss_pos.searchsorted(starts - window_size, side='left').astype('int32')
            rb = tss_pos.searchsorted(starts + window_size + 1, side='left').astype('int32')
            _TSSE_KERNEL(
                (bpg,), (tpb,),
                (starts, cell_idx, lb, rb, tss_pos, tss_strand,
                 n_frags, window_size, half_smooth, data_cp)
            )
            
            # Process ends
            lb = tss_pos.searchsorted(ends - window_size, side='left').astype('int32')
            rb = tss_pos.searchsorted(ends + window_size + 1, side='left').astype('int32')
            _TSSE_KERNEL(
                (bpg,), (tpb,),
                (ends, cell_idx, lb, rb, tss_pos, tss_strand,
                 n_frags, window_size, half_smooth, data_cp)
            )
        
        total_processed = min(i + row_groups_per_batch, num_row_groups)
        if (i // row_groups_per_batch + 1) % 5 == 0 or total_processed == num_row_groups:
            logger.info(f"  Processed row groups {total_processed}/{num_row_groups}")
        
        del chunk_df

    logger.info(f"Completed TSSe computation")
    
    # =========================================================================
    # Phase 4: Compute final scores
    # =========================================================================
    tss_signal = data_cp[:, 1] / smooth_window
    bg_signal = (data_cp[:, 0] / 100 + data_cp[:, 2] / 100) / 2
    tsse_scores = tss_signal / (bg_signal + 0.1)
    
    # Build results as cuDF DataFrame
    results = qc_metrics[['barcode', 'n_unique', 'duplicate_fraction', 'mito_fraction']].copy()
    results['tsse_score'] = tsse_scores
    
    # Reorder columns to match regular compute_metrics
    results = results[['barcode', 'tsse_score', 'n_unique', 'duplicate_fraction', 'mito_fraction']]
    
    return results
