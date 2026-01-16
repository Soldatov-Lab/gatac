"""
GPU-accelerated metrics for ATAC-seq data.
"""

import logging
from pathlib import Path
from typing import Optional, List

import cudf
import cupy as cp
import cupyx
import numpy as np

logger = logging.getLogger(__name__)

def load_tss_from_gtf(gtf_path: str | Path) -> cudf.DataFrame:
    """
    Load TSS locations from a GTF file using cuDF.
    
    Parameters
    ----------
    gtf_path : str or Path
        Path to the GTF file.
        
    Returns
    -------
    tss_df : cudf.DataFrame
        DataFrame with columns: ['chrom', 'tss', 'strand']
    """
    logger.info(f"Loading TSS from {gtf_path}")
    
    # GTF columns
    cols = [
        'chrom', 'source', 'feature', 'start', 'end', 
        'score', 'strand', 'frame', 'attribute'
    ]
    
    # Read GTF (tab-separated, ignore lines starting with #)
    df = cudf.read_csv(
        gtf_path,
        sep='\t',
        comment='#',
        header=None,
        names=cols,
        usecols=['chrom', 'feature', 'start', 'end', 'strand']
    )
    
    # Filter for transcripts
    df = df[df['feature'] == 'transcript']
    
    # Determine TSS based on strand
    # If strand is +, TSS is start. If -, TSS is end.
    # Note: GTF coordinates are 1-based.
    df['tss'] = df['start']
    neg_strand = df['strand'] == '-'
    df.loc[neg_strand, 'tss'] = df.loc[neg_strand, 'end']
    
    # Keep unique TSS positions per chromosome/strand
    tss_df = df[['chrom', 'tss', 'strand']].drop_duplicates().reset_index(drop=True)
    
    logger.info(f"Loaded {len(tss_df):,} unique TSSs")
    return tss_df

def _merge_asof_nearest(
    left_df: cudf.DataFrame, 
    right_df: cudf.DataFrame, 
    left_on: str, 
    right_on: str
) -> cudf.DataFrame:
    """
    Custom implementation of merge_asof(direction='nearest') for cuDF.
    
    Assumes right_df is already sorted by right_on.
    """
    if len(right_df) == 0:
        # Return left_df with empty columns from right_df
        res = left_df.copy()
        for col in right_df.columns:
            if col != right_on:
                res[col] = None
        return res

    # Find insertion points (first element >= left_val)
    idx = right_df[right_on].searchsorted(left_df[left_on], side='left')
    
    # Ensure we are working with cupy arrays for numeric calculations
    if hasattr(idx, 'values'):
        idx = idx.values
    
    # Candidate indices are idx and idx - 1
    idx_r = cp.clip(idx, 0, len(right_df) - 1)
    idx_l = cp.clip(idx - 1, 0, len(right_df) - 1)
    
    # Get values at these indices to compare distances
    val_r = right_df[right_on].take(idx_r).values
    val_l = right_df[right_on].take(idx_l).values
    pos_left = left_df[left_on].values
    
    dist_r = cp.abs(val_r - pos_left)
    dist_l = cp.abs(pos_left - val_l)
    
    # Choose nearest index
    take_idx = cp.where(dist_r < dist_l, idx_r, idx_l)
    
    # Gather samples from right_df
    right_sampled = right_df.take(take_idx).reset_index(drop=True)
    
    # Drop right_on from right_sampled if it's redundant/matches left_on
    # (though in our case tss and pos are different)
    
    # Combine columns
    res = left_df.reset_index(drop=True)
    for col in right_df.columns:
        if col not in res.columns:
            res[col] = right_sampled[col]
            
    return res

def compute_metrics(
    fragments_df: cudf.DataFrame,
    tss_df: cudf.DataFrame,
    window_size: int = 2000,
    smooth_window: int = 11,
    min_unique_frags: int = 100,
) -> cudf.DataFrame:
    """
    Compute TSS enrichment scores and quality metrics per cell using GPU acceleration 
    with memory-efficient chromosomal chunking.
    
    Parameters
    ----------
    fragments_df : cudf.DataFrame
        Fragment data with columns: ['chrom', 'start', 'end', 'barcode', 'count']
    tss_df : cudf.DataFrame
        TSS data from load_tss_from_gtf
    window_size : int
        Distance around TSS to consider (default: 2000)
    smooth_window : int
        Window size for smoothing the TSS signal (default: 11)
    min_unique_frags : int
        Minimum unique fragments per cell to include in output (default: 100)
        
    Returns
    -------
    results : cudf.DataFrame
        DataFrame with columns: ['barcode', 'tsse_score', 'n_unique', 'duplicate_fraction', 'mito_fraction']
    """
    import gc
    logger.info(f"Computing metrics (TSSe, fragments, mito) for cells with >= {min_unique_frags} frags")
    
    # 1. Calculate cell-level QC metrics
    # Unique fragments = number of rows in fragment file
    # Total fragments = sum of 'count' column (number of reads/duplicates)
    
    # Group by barcode to get both row count and sum of counts
    agg_df = fragments_df.groupby('barcode', observed=True).agg({
        'count': ['sum', 'size']
    })
    agg_df.columns = ['n_total', 'n_unique']
    agg_df = agg_df.reset_index()
    
    # Filter to cells with minimum unique fragments
    total_barcodes = len(agg_df)
    agg_df = agg_df[agg_df['n_unique'] >= min_unique_frags]
    n_cells = len(agg_df)
    logger.info(f"Filtered to {n_cells:,} cells with >= {min_unique_frags} unique fragments (from {total_barcodes:,} total)")
    
    if n_cells == 0:
        logger.warning("No cells passed the minimum fragment filter!")
        return cudf.DataFrame({
            'barcode': [],
            'tsse_score': [],
            'n_unique': [],
            'duplicate_fraction': [],
            'mito_fraction': [],
        })

    # Filter fragments to only include valid barcodes
    valid_barcodes = agg_df['barcode']
    fragments_df = fragments_df[fragments_df['barcode'].isin(valid_barcodes)]
    
    # Calculate mitochondrial fraction (on filtered fragments)
    is_mito = fragments_df['chrom'].isin(['chrM', 'M'])
    mito_counts = fragments_df[is_mito].groupby('barcode', observed=True)['count'].sum().reset_index()
    mito_counts.columns = ['barcode', 'n_mito']
    
    # Merge QC metrics
    qc_metrics = agg_df.merge(mito_counts, on='barcode', how='left').fillna(0)
    qc_metrics['duplicate_fraction'] = (qc_metrics['n_total'] - qc_metrics['n_unique']) / (qc_metrics['n_total'] + 1e-9)
    qc_metrics['mito_fraction'] = qc_metrics['n_mito'] / (qc_metrics['n_total'] + 1e-9)
    
    # Prepare barcode mapping
    unique_barcodes = qc_metrics['barcode'].unique().sort_values()
    n_cells = len(unique_barcodes)
    n_offsets = 2 * window_size + 1
    
    barcode_to_idx = cudf.DataFrame({
        'barcode': unique_barcodes,
        'cell_idx': cp.arange(n_cells, dtype='int32')
    })
    
    # Initialize dense profile matrix on GPU
    # (n_cells x n_offsets) ~ (10k x 4001) x 4 bytes = 160MB
    data_cp = cp.zeros((n_cells, n_offsets), dtype='float32')
    
    # Process chromosome by chromosome to save memory
    # This avoids doubling memory by concatenating start/end for all fragments at once
    chromosomes = fragments_df['chrom'].unique().to_arrow().to_pylist()
    
    # Pre-sort TSS once
    tss_df = tss_df.sort_values(['chrom', 'tss'])

    for chrom in chromosomes:
        logger.debug(f"Processing chromosome {chrom}")
        
        # Filter fragments for this chromosome
        f_chrom = fragments_df[fragments_df['chrom'] == chrom]
        if len(f_chrom) == 0:
            continue
            
        # Filter TSS for this chromosome
        t_chrom = tss_df[tss_df['chrom'] == chrom]
        if len(t_chrom) == 0:
            continue
            
        # 2. Expand fragments to insertions (start and end) for TSSe
        df_start = f_chrom[['start', 'barcode', 'count']].rename(columns={'start': 'pos'})
        df_end = f_chrom[['end', 'barcode', 'count']].rename(columns={'end': 'pos'})
        insertions = cudf.concat([df_start, df_end], ignore_index=True)
        del f_chrom
        
        # 3. Join insertions with nearest TSS
        insertions = insertions.sort_values('pos')
        
        mapped = _merge_asof_nearest(
            insertions, 
            t_chrom, 
            left_on='pos', 
            right_on='tss'
        )
        del insertions
        
        # 4. Filter and calculate offsets
        mapped['dist'] = mapped['pos'] - mapped['tss']
        mask = mapped['dist'].abs() <= window_size
        mapped = mapped[mask]
        
        if len(mapped) > 0:
            # Adjust for strand orientation
            mapped['offset'] = mapped['dist']
            neg_mask = mapped['strand'] == '-'
            mapped.loc[neg_mask, 'offset'] = -mapped.loc[neg_mask, 'offset']
            
            # Map barcodes to indices and shift offset
            mapped['offset_idx'] = (mapped['offset'] + window_size).astype('int32')
            
            # Aggregate to Cell x Offset profiles for this chromosome
            profiles = mapped.groupby(['barcode', 'offset_idx'], observed=True)['count'].sum().reset_index()
            del mapped
            
            profiles = profiles.merge(barcode_to_idx, on='barcode')
            
            # Scatter counts into global matrx
            cell_idx = profiles['cell_idx'].values
            offset_idx = profiles['offset_idx'].values
            counts = profiles['count'].values.astype('float32')
            
            cupyx.scatter_add(data_cp, (cell_idx, offset_idx), counts)
            del profiles
            
        # Explicit cleanup
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
    
    # 6. Calculate TSSe Score
    center_idx = window_size
    half_smooth = smooth_window // 2
    
    # Smoothed TSS signal (center 11bp)
    tss_signal = data_cp[:, (center_idx - half_smooth):(center_idx + half_smooth + 1)].mean(axis=1)
    
    # Background signal (average of first 100bp and last 100bp)
    bg_signal = (data_cp[:, 0:100].mean(axis=1) + data_cp[:, -100:].mean(axis=1)) / 2
    
    # TSSe = tss_signal / (bg_signal + 0.1)
    tsse_scores = tss_signal / (bg_signal + 0.1)
    
    # 7. Collect results
    results = cudf.DataFrame({
        'barcode': unique_barcodes,
        'tsse_score': tsse_scores
    })
    
    # Merge with QC metrics
    final_cols = ['barcode', 'tsse_score', 'n_unique', 'duplicate_fraction', 'mito_fraction']
    results = results.merge(qc_metrics, on='barcode')[final_cols]
    
    return results
