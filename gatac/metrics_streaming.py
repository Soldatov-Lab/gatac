"""
GPU-accelerated streaming metrics for ATAC-seq data.

Uses chunked parquet reading with cuDF for GPU-accelerated computation,
enabling processing of datasets larger than GPU VRAM.

Optimized for high GPU utilization with batched row groups and vectorized operations.
"""

import logging
from pathlib import Path
from typing import Literal

import polars as pl

logger = logging.getLogger(__name__)


def load_tss_from_gtf_polars(gtf_path: str | Path) -> pl.LazyFrame:
    """
    Load TSS locations from a GTF file using Polars lazy scanning.
    
    Parameters
    ----------
    gtf_path : str or Path
        Path to the GTF file.
        
    Returns
    -------
    pl.LazyFrame
        LazyFrame with columns: ['chrom', 'tss', 'strand']
    """
    logger.info(f"Loading TSS from {gtf_path}")
    
    cols = ['chrom', 'source', 'feature', 'start', 'end', 
            'score', 'strand', 'frame', 'attribute']
    
    return (
        pl.scan_csv(
            str(gtf_path),
            separator='\t',
            comment_prefix='#',
            has_header=False,
            new_columns=cols,
        )
        .select(['chrom', 'feature', 'start', 'end', 'strand'])
        .filter(pl.col('feature') == 'transcript')
        .with_columns([
            pl.col('chrom').cast(pl.String),
            pl.when(pl.col('strand') == '-')
            .then(pl.col('end'))
            .otherwise(pl.col('start'))
            .alias('tss')
        ])
        .select(['chrom', 'tss', 'strand'])
        .unique()
    )


def compute_metrics_streaming(
    fragments_path: str | Path,
    tss_df: pl.LazyFrame | pl.DataFrame,
    window_size: int = 2000,
    smooth_window: int = 11,
    engine_mode: Literal["in-memory", "streaming"] = "streaming",
    memory_resource: Literal["cuda-async", "managed", "managed-pool", "cuda"] = "managed-pool",
    device: int = 0,
    batch_row_groups: int = 5,  # Smaller batches for lower peak memory
    min_unique_frags: int = 100,  # Minimum unique fragments per cell to include in output
) -> pl.DataFrame:
    """
    Compute TSS enrichment metrics using chunked GPU processing.
    
    Optimized for high GPU utilization with batched processing.
    Only cells with >= min_unique_frags are included in the output.
    
    Parameters
    ----------
    fragments_path : str or Path
        Path to fragments Parquet file
    tss_df : pl.LazyFrame or pl.DataFrame
        TSS locations from load_tss_from_gtf_polars
    window_size : int
        Distance around TSS to consider (default: 2000)
    smooth_window : int
        Window size for smoothing the TSS signal (default: 11)
    engine_mode : str
        "streaming" for chunked processing, "in-memory" for full load
    memory_resource : str
        GPU memory resource type (default: "managed-pool")
    device : int
        GPU device ID (default: 0)
    batch_row_groups : int
        Number of row groups to process at once (default: 20)
    min_unique_frags : int
        Minimum unique fragments per cell to include in output (default: 100)
        
    Returns
    -------
    pl.DataFrame
        DataFrame with columns: ['barcode', 'tsse_score', 'n_unique', 
                                 'duplicate_fraction', 'mito_fraction']
    """
    import gc
    import cudf
    import cupy as cp
    import cupyx
    import pyarrow.parquet as pq
    
    logger.info(f"Computing metrics with cuDF GPU (chunked mode, batch={batch_row_groups})")
    
    # Configure RMM memory resource for UVM
    try:
        import rmm
        if memory_resource == "managed-pool":
            mr = rmm.mr.PrefetchResourceAdaptor(
                rmm.mr.PoolMemoryResource(rmm.mr.ManagedMemoryResource())
            )
            rmm.mr.set_current_device_resource(mr)
        elif memory_resource == "managed":
            mr = rmm.mr.ManagedMemoryResource()
            rmm.mr.set_current_device_resource(mr)
        elif memory_resource == "cuda-async":
            free_memory, _ = rmm.mr.available_device_memory()
            initial_pool_size = 256 * (int(free_memory * 0.8) // 256)
            mr = rmm.mr.CudaAsyncMemoryResource(initial_pool_size=initial_pool_size)
            rmm.mr.set_current_device_resource(mr)
        logger.info(f"Using {memory_resource} memory resource")
    except Exception as e:
        logger.warning(f"Could not configure memory resource: {e}")
    
    # Convert TSS to cuDF
    if isinstance(tss_df, pl.LazyFrame):
        tss_pl = tss_df.collect()
    else:
        tss_pl = tss_df
    tss_cudf = cudf.from_pandas(tss_pl.to_pandas())
    tss_cudf = tss_cudf.sort_values(['chrom', 'tss']).reset_index(drop=True)
    logger.info(f"Loaded {len(tss_cudf):,} TSS positions")
    
    # Open parquet file
    pq_file = pq.ParquetFile(str(fragments_path))
    total_rows = pq_file.metadata.num_rows
    num_row_groups = pq_file.metadata.num_row_groups
    logger.info(f"Processing {total_rows:,} fragments in {num_row_groups} row groups")
    
    n_offsets = 2 * window_size + 1
    
    def cleanup():
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
    
    # =========================================================================
    # Pass 1: Compute QC metrics (batched)
    # =========================================================================
    logger.info("Pass 1: Computing QC metrics...")
    qc_accum = []
    
    for batch_start in range(0, num_row_groups, batch_row_groups):
        batch_end = min(batch_start + batch_row_groups, num_row_groups)
        
        # Read multiple row groups at once
        tables = [pq_file.read_row_group(i, columns=['chrom', 'barcode', 'count']) 
                  for i in range(batch_start, batch_end)]
        import pyarrow as pa
        combined_table = pa.concat_tables(tables)
        df = cudf.DataFrame.from_arrow(combined_table)
        del tables, combined_table
        
        # Cast dtypes
        df['chrom'] = df['chrom'].astype('category')
        df['barcode'] = df['barcode'].astype('category')
        df['count'] = df['count'].astype('uint16')
        
        # QC metrics per barcode (fully on GPU)
        chunk_qc = df.groupby('barcode', observed=True).agg({
            'count': ['sum', 'size']
        })
        chunk_qc.columns = ['n_total', 'n_unique']
        chunk_qc = chunk_qc.reset_index()
        
        # Mito counts
        is_mito = df['chrom'].isin(['chrM', 'M'])
        mito_df = df[is_mito]
        if len(mito_df) > 0:
            mito_counts = mito_df.groupby('barcode', observed=True)['count'].sum().reset_index()
            mito_counts.columns = ['barcode', 'n_mito']
            chunk_qc = chunk_qc.merge(mito_counts, on='barcode', how='left').fillna(0)
        else:
            chunk_qc['n_mito'] = 0
        
        qc_accum.append(chunk_qc)
        del df
        cleanup()
    
    # Aggregate QC on GPU
    qc_all = cudf.concat(qc_accum, ignore_index=True)
    del qc_accum
    
    qc_final = qc_all.groupby('barcode', observed=True).agg({
        'n_total': 'sum',
        'n_unique': 'sum', 
        'n_mito': 'sum'
    }).reset_index()
    qc_final['duplicate_fraction'] = (qc_final['n_total'] - qc_final['n_unique']) / (qc_final['n_total'] + 1e-9)
    qc_final['mito_fraction'] = qc_final['n_mito'] / (qc_final['n_total'] + 1e-9)
    
    # Filter to cells with minimum unique fragments
    total_barcodes = len(qc_final)
    qc_final = qc_final[qc_final['n_unique'] >= min_unique_frags]
    n_cells = len(qc_final)
    logger.info(f"Filtered to {n_cells:,} cells with >= {min_unique_frags} unique fragments (from {total_barcodes:,} total)")
    
    if n_cells == 0:
        logger.warning("No cells passed the minimum fragment filter!")
        return pl.DataFrame({
            'barcode': [],
            'tsse_score': [],
            'n_unique': [],
            'duplicate_fraction': [],
            'mito_fraction': [],
        })
    
    # Create barcode index mapping ON GPU for filtered cells
    unique_barcodes = qc_final['barcode'].reset_index(drop=True)
    
    barcode_idx_df = cudf.DataFrame({
        'barcode': unique_barcodes,
        'cell_idx': cp.arange(n_cells, dtype='int32')
    })
    
    # Estimate profile matrix memory
    profile_mem_gb = (n_cells * n_offsets * 4) / (1024**3)
    logger.info(f"Profile matrix size: {n_cells:,} cells × {n_offsets} offsets = {profile_mem_gb:.2f} GB")
    
    # Initialize profile matrix on GPU
    data_cp = cp.zeros((n_cells, n_offsets), dtype='float32')
    
    # =========================================================================
    # Pass 2: Compute TSS enrichment profiles (batched, fully vectorized)
    # =========================================================================
    logger.info("Pass 2: Computing TSS enrichment profiles...")
    
    for batch_start in range(0, num_row_groups, batch_row_groups):
        batch_end = min(batch_start + batch_row_groups, num_row_groups)
        
        # Read multiple row groups at once
        tables = [pq_file.read_row_group(i, columns=['chrom', 'start', 'end', 'barcode', 'count']) 
                  for i in range(batch_start, batch_end)]
        import pyarrow as pa
        combined_table = pa.concat_tables(tables)
        df = cudf.DataFrame.from_arrow(combined_table)
        del tables, combined_table
        
        df['chrom'] = df['chrom'].astype('category')
        df['barcode'] = df['barcode'].astype('category')
        
        # Early filter: only keep fragments from barcodes that passed min_frags filter
        df = df[df['barcode'].isin(unique_barcodes)]
        if len(df) == 0:
            del df
            cleanup()
            continue
        
        # Process each chromosome
        chroms = df['chrom'].unique().to_arrow().to_pylist()
        
        for chrom in chroms:
            f_chrom = df[df['chrom'] == chrom]
            if len(f_chrom) == 0:
                continue
                
            t_chrom = tss_cudf[tss_cudf['chrom'] == chrom]
            if len(t_chrom) == 0:
                continue
            
            # Expand to insertions (vectorized)
            df_start = f_chrom[['start', 'barcode', 'count']].rename(columns={'start': 'pos'})
            df_end = f_chrom[['end', 'barcode', 'count']].rename(columns={'end': 'pos'})
            insertions = cudf.concat([df_start, df_end], ignore_index=True)
            insertions = insertions.sort_values('pos')
            
            # Vectorized nearest TSS lookup (fully on GPU)
            idx = t_chrom['tss'].searchsorted(insertions['pos'], side='left')
            if hasattr(idx, 'values'):
                idx = idx.values
            
            idx_r = cp.clip(idx, 0, len(t_chrom) - 1)
            idx_l = cp.clip(idx - 1, 0, len(t_chrom) - 1)
            
            val_r = t_chrom['tss'].take(idx_r).values
            val_l = t_chrom['tss'].take(idx_l).values
            pos_vals = insertions['pos'].values
            
            dist_r = cp.abs(val_r - pos_vals)
            dist_l = cp.abs(pos_vals - val_l)
            take_idx = cp.where(dist_r < dist_l, idx_r, idx_l)
            
            nearest_tss = t_chrom['tss'].take(take_idx).values
            nearest_strand = t_chrom['strand'].take(take_idx)
            
            dist = pos_vals - nearest_tss
            mask = cp.abs(dist) <= window_size
            n_valid = int(mask.sum())
            
            if n_valid > 0:
                # Filter to valid insertions
                valid_indices = cp.where(mask)[0]
                
                # Apply strand adjustment
                strand_is_neg = (nearest_strand == '-').values[mask]
                offset = cp.where(strand_is_neg, -dist[mask], dist[mask])
                offset_idx = (offset + window_size).astype('int32')
                
                # Get barcodes and counts (stay on GPU)
                bc_series = insertions['barcode'].take(valid_indices)
                count_vals = insertions['count'].take(valid_indices).values.astype('float32')
                
                # Vectorized barcode-to-index lookup using GPU merge
                bc_df = cudf.DataFrame({'barcode': bc_series})
                bc_df = bc_df.merge(barcode_idx_df, on='barcode', how='left')
                cell_indices = bc_df['cell_idx'].values
                
                # Filter out unmapped barcodes
                valid = ~cp.isnan(cell_indices) if cell_indices.dtype.kind == 'f' else (cell_indices >= 0)
                
                if valid.any():
                    cupyx.scatter_add(data_cp, 
                                     (cell_indices[valid].astype('int32'), offset_idx[valid]), 
                                     count_vals[valid])
            
            del insertions, df_start, df_end
        
        del df
        cleanup()
        logger.debug(f"Processed row groups {batch_start+1}-{batch_end}/{num_row_groups}")
    
    # =========================================================================
    # Compute TSSe scores from profiles
    # =========================================================================
    logger.info("Computing TSSe scores...")
    center_idx = window_size
    half_smooth = smooth_window // 2
    
    tss_signal = data_cp[:, (center_idx - half_smooth):(center_idx + half_smooth + 1)].mean(axis=1)
    bg_signal = (data_cp[:, 0:100].mean(axis=1) + data_cp[:, -100:].mean(axis=1)) / 2
    tsse_scores = tss_signal / (bg_signal + 0.1)
    
    # Build final result from filtered QC data
    unique_barcodes_list = unique_barcodes.astype('str').to_pandas().tolist()
    
    result_df = pl.DataFrame({
        'barcode': unique_barcodes_list,
        'tsse_score': cp.asnumpy(tsse_scores),
        'n_unique': qc_final['n_unique'].values.get().astype('int64'),
        'duplicate_fraction': qc_final['duplicate_fraction'].values.get(),
        'mito_fraction': qc_final['mito_fraction'].values.get(),
    })
    
    logger.info(f"Computed metrics for {len(result_df):,} cells")
    return result_df

