"""
GPU-accelerated metrics for ATAC-seq data.

Implementation:
1. compute_metrics: Polars-based, streams data in chunks to reduce peak memory.
   This is the standard implementation for both small and large datasets.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, List

import cudf
import cupy as cp
import cupyx
import numpy as np

from ._utils import cleanup_gpu_memory

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

def _load_tss_from_gtf(gtf_path: str | Path) -> 'pl.DataFrame':
    """Load TSS locations from a GTF file using Polars.

    Private helper used by :func:`compute_metrics` when the caller passes a
    ``gtf_path`` instead of a pre-loaded TSS DataFrame.
    """

    import polars as pl

    logger.info(f"Loading TSS from {gtf_path} (Polars)")

    # GTF columns
    cols = ['chrom', 'source', 'feature', 'start', 'end', 'score', 'strand', 'frame', 'attribute']

    # Read GTF (tab-separated, ignore lines starting with #)
    df = pl.read_csv(
        gtf_path,
        separator='\t',
        comment_prefix='#',
        has_header=False,
        new_columns=cols,
    ).select(['chrom', 'feature', 'start', 'end', 'strand'])

    # Filter for transcripts and compute TSS
    df = df.filter(pl.col('feature') == 'transcript')

    # TSS is start-1 for + strand, end-1 for - strand (0-based)
    df = df.with_columns([
        pl.when(pl.col('strand') == '-')
          .then(pl.col('end') - 1)
          .otherwise(pl.col('start') - 1)
          .alias('tss')
    ])

    # Keep unique TSS positions
    tss_df = df.select(['chrom', 'tss', 'strand']).unique()

    logger.info(f"Loaded {len(tss_df):,} unique TSSs (Polars)")
    return tss_df


def compute_metrics(
    parquet_path: str | Path,
    tss_source: Union[str, Path, 'pl.DataFrame'],
    window_size: int = 2000,
    smooth_window: int = 11,
    min_unique_frags: int = 100,
    chrom_sizes: dict[str, int] | None = None,
    exclude_chroms: list[str] | None = ['chrM', 'M'],
    row_groups_per_batch: int = 64,
) -> cudf.DataFrame:
    """
    Compute TSS enrichment scores using GPU-accelerated streaming via row groups.

    This function streams fragment data in batches of row groups to balance
    speed and GPU memory usage.

    Parameters
    ----------
    parquet_path : str or Path
        Path to the parquet file containing ATAC fragments.
    tss_source : str, Path, or pl.DataFrame
        Either a path to a GTF file (the TSS table is loaded internally)
        or a pre-built ``polars.DataFrame`` of TSS positions
        (columns: ``chrom``, ``tss``, ``strand``).  When a pre-built
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
    """

    import polars as pl
    import pyarrow.parquet as pq

    # Resolve TSS source: accept either a GTF path or a pre-loaded DataFrame.
    if isinstance(tss_source, (str, Path)):
        tss_df = _load_tss_from_gtf(tss_source)
    else:
        tss_df = tss_source

    logger.info(f"Computing metrics (streaming mode, row_groups_per_batch={row_groups_per_batch})")
    parquet_path = Path(parquet_path)
    
    # =========================================================================
    # Phase 1: Compute cell QC metrics using row groups
    # =========================================================================
    parquet_file = pq.ParquetFile(str(parquet_path))
    num_row_groups = parquet_file.metadata.num_row_groups
    
    qc_summary = None
    valid_chroms_qc = list(chrom_sizes.keys()) if chrom_sizes is not None else None
    
    logger.info(f"Aggregating QC metrics from {num_row_groups} row groups in batches of {row_groups_per_batch}...")
    
    for i in range(0, num_row_groups, row_groups_per_batch):
        batch_groups = list(range(i, min(i + row_groups_per_batch, num_row_groups)))
        
        # Load batch into GPU memory
        chunk_df = cudf.read_parquet(
            str(parquet_path), 
            row_groups=batch_groups, 
            columns=['chrom', 'barcode', 'count']
        )
        
        if valid_chroms_qc:
            chunk_df = chunk_df[chunk_df['chrom'].isin(valid_chroms_qc)]
        
        # Identify mitochondrial fragments
        chunk_df['n_mito'] = chunk_df['chrom'].isin(['chrM', 'M']).astype('uint32') * chunk_df['count']
        
        # Group by barcode for this chunk
        chunk_agg = chunk_df.groupby('barcode').agg({
            'count': ['sum', 'size'],
            'n_mito': 'sum'
        })
        chunk_agg.columns = ['n_total', 'n_unique', 'n_mito']
        chunk_agg = chunk_agg.reset_index()
        
        if qc_summary is None:
            qc_summary = chunk_agg
        else:
            qc_summary = cudf.concat([qc_summary, chunk_agg])
            # Re-aggregate to keep memory low
            qc_summary = qc_summary.groupby('barcode').sum().reset_index()
        
        del chunk_df, chunk_agg
        cleanup_gpu_memory()
        
    if qc_summary is None or len(qc_summary) == 0:
        logger.warning("No cells passed the minimum fragment filter!")
        return cudf.DataFrame({
            'barcode': [], 'tsse_score': [], 'n_unique': [], 
            'duplicate_fraction': [], 'mito_fraction': []
        })
        
    # Filter by min_unique_frags and compute final QC metrics
    qc_metrics = qc_summary[qc_summary['n_unique'] >= min_unique_frags].copy()
    qc_metrics['duplicate_fraction'] = (qc_metrics['n_total'] - qc_metrics['n_unique']) / (qc_metrics['n_total'] + 1e-9)
    qc_metrics['mito_fraction'] = qc_metrics['n_mito'] / (qc_metrics['n_total'] + 1e-9)
    
    n_cells = len(qc_metrics)
    logger.info(f"Found {n_cells:,} cells with >= {min_unique_frags} unique fragments")
    
    # Create barcode -> cell_idx mapping (on GPU)
    barcode_to_idx_gpu = qc_metrics[['barcode']].copy()
    barcode_to_idx_gpu['cell_idx'] = cp.arange(n_cells, dtype='int32')
    
    # Clean up summary
    del qc_summary

    # =========================================================================
    # Phase 2: Prepare TSS data on GPU with chromosome encoding
    # =========================================================================
    # Filter TSS
    if exclude_chroms:
        tss_filtered = tss_df.filter(~pl.col('chrom').is_in(exclude_chroms))
    else:
        tss_filtered = tss_df
    
    # Create chromosome -> integer encoding for faster processing
    tss_chroms = tss_filtered['chrom'].unique().sort().to_list()
    
    # Group TSS by chromosome for efficient lookup
    tss_by_chrom = {}
    for chrom in tss_chroms:
        tss_sub = tss_filtered.filter(pl.col('chrom') == chrom).sort('tss')
        tss_pos = cp.asarray(tss_sub['tss'].to_numpy(), dtype='int32')
        tss_strand = cp.asarray((tss_sub['strand'] == '-').to_numpy(), dtype='int8')
        tss_by_chrom[chrom] = (tss_pos, tss_strand)

    # =========================================================================
    # Phase 3: Stream fragments and compute TSSe using row group batches
    # =========================================================================
    # Initialize accumulator on GPU (3 bins per cell)
    data_cp = cp.zeros((n_cells, 3), dtype='float32')
    half_smooth = smooth_window // 2
    
    parquet_file = pq.ParquetFile(str(parquet_path))
    num_row_groups = parquet_file.metadata.num_row_groups
    
    logger.info(f"Processing {num_row_groups} row groups in batches of {row_groups_per_batch}...")
    
    for i in range(0, num_row_groups, row_groups_per_batch):
        batch_groups = list(range(i, min(i + row_groups_per_batch, num_row_groups)))
        
        # Load batch directly to GPU memory
        chunk_df = cudf.read_parquet(
            str(parquet_path), 
            row_groups=batch_groups, 
            columns=['chrom', 'start', 'end', 'barcode']
        )
        
        # Filter chromosomes
        chunk_df = chunk_df[chunk_df['chrom'].isin(tss_chroms)]
        if valid_chroms_qc:
            chunk_df = chunk_df[chunk_df['chrom'].isin(valid_chroms_qc)]
        
        # Join with barcodes to filter and get cell_idx
        chunk_df = chunk_df.merge(barcode_to_idx_gpu, on='barcode')
        
        if len(chunk_df) == 0:
            continue
        
        # Process fragments
        chunk_chroms = chunk_df['chrom'].unique().to_arrow().to_pylist()
        tpb = 256
        
        for chrom in chunk_chroms:
            if chrom not in tss_by_chrom:
                continue
                
            tss_pos, tss_strand = tss_by_chrom[chrom]
            chrom_df = chunk_df[chunk_df['chrom'] == chrom]
            
            cell_idx = chrom_df['cell_idx'].values.astype('int32')
            starts = chrom_df['start'].values.astype('int32')
            ends = (chrom_df['end'] - 1).values.astype('int32')
            
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
        
        # Clean up
        del chunk_df
        cleanup_gpu_memory()

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
