"""
GPU-accelerated gene activity matrix generation from ATAC fragment data.

Implements paired-insertion counting strategy matching SnapATAC2.
"""

import logging
import time
import gc
from pathlib import Path
from typing import Optional, Tuple, Literal, List

import cudf
import cupy as cp
import cupyx.scipy.sparse as cusp
import numpy as np
import polars as pl
import scipy.sparse as sp

logger = logging.getLogger(__name__)


def optimize_sparse_matrix_dtype(max_val: int) -> np.dtype:
    """
    Select optimal count dtype based on maximum value.
    
    Automatically selects uint8, uint16, or int32 dtype to minimize memory usage
    while ensuring sufficient range for the data.
    
    Parameters
    ----------
    max_val : int
        Maximum count value in the data
        
    Returns
    -------
    np.dtype
        The optimized dtype
    """
    if max_val <= np.iinfo(np.uint8).max:
        target_dtype = np.uint8
        logger.info(f"Using uint8 dtype (max count: {max_val})")
    elif max_val <= np.iinfo(np.uint16).max:
        target_dtype = np.uint16
        logger.info(f"Using uint16 dtype (max count: {max_val})")
    else:
        target_dtype = np.int32
        logger.info(f"Using int32 dtype (max count: {max_val})")
    
    return target_dtype


def load_gene_annotation(
    gtf_path: str | Path,
    id_type: Literal["gene", "transcript"] = "gene",
    upstream: int = 2000,
    downstream: int = 0,
    include_gene_body: bool = True,
    gene_name_key: str = "gene_name",
    gene_id_key: str = "gene_id",
    transcript_name_key: str = "transcript_name",
    transcript_id_key: str = "transcript_id",
) -> pl.DataFrame:
    """
    Load gene annotation from a GTF file and compute regulatory domains.

    Parameters
    ----------
    gtf_path : str or Path
        Path to the GTF file.
    id_type : str
        "gene" or "transcript" - which feature type to use.
    upstream : int
        Base pairs upstream of TSS to include (default: 2000).
    downstream : int
        Base pairs downstream of regulatory domain (default: 0).
    include_gene_body : bool
        Whether to include the gene body in the regulatory domain.
    gene_name_key : str
        Key for gene name in GTF attributes (default: "gene_name").
    gene_id_key : str
        Key for gene ID in GTF attributes (default: "gene_id").
    transcript_name_key : str
        Key for transcript name in GTF attributes (default: "transcript_name").
    transcript_id_key : str
        Key for transcript ID in GTF attributes (default: "transcript_id").

    Returns
    -------
    pl.DataFrame
        DataFrame with columns: ['chrom', 'start', 'end', 'name', 'id', 'strand']
        where start/end define the regulatory domain.
    """
    logger.info(f"Loading gene annotation from {gtf_path}")
    
    # Detect file format from extension
    gtf_path = Path(gtf_path)
    is_gff3 = '.gff' in gtf_path.name.lower() or '.gff3' in gtf_path.name.lower()
    
    # GTF/GFF columns (same for both formats)
    cols = ['chrom', 'source', 'feature', 'start', 'end', 'score', 'strand', 'frame', 'attribute']
    
    # Read GTF/GFF (tab-separated, ignore lines starting with #)
    df = pl.read_csv(
        str(gtf_path),
        separator='\t',
        comment_prefix='#',
        has_header=False,
        new_columns=cols,
    ).select(['chrom', 'feature', 'start', 'end', 'strand', 'attribute'])
    
    # Filter for transcripts - SnapATAC2 always counts at transcript level
    # then aggregates to gene level using MAX (for id_type="gene")
    df = df.filter(pl.col('feature') == 'transcript')
    
    # Parse attributes to extract gene name/id and transcript name/id
    # We always need gene_name for gene-level aggregation
    if is_gff3:
        # GFF3: key=value;key2=value2
        df = df.with_columns([
            pl.col('attribute')
                .str.extract(rf'{gene_name_key}=([^;]+)')
                .alias('gene_name'),
            pl.col('attribute')
                .str.extract(rf'{gene_id_key}=([^;]+)')
                .alias('gene_id'),
            pl.col('attribute')
                .str.extract(rf'{transcript_name_key}=([^;]+)')
                .alias('transcript_name'),
            pl.col('attribute')
                .str.extract(rf'{transcript_id_key}=([^;]+)')
                .alias('transcript_id'),
        ])
    else:
        # GTF: key "value";
        df = df.with_columns([
            pl.col('attribute')
                .str.extract(rf'{gene_name_key} "([^"]+)"')
                .alias('gene_name'),
            pl.col('attribute')
                .str.extract(rf'{gene_id_key} "([^"]+)"')
                .alias('gene_id'),
            pl.col('attribute')
                .str.extract(rf'{transcript_name_key} "([^"]+)"')
                .alias('transcript_name'),
            pl.col('attribute')
                .str.extract(rf'{transcript_id_key} "([^"]+)"')
                .alias('transcript_id'),
        ])
    
    # Set name/id based on id_type
    if id_type == "gene":
        df = df.with_columns([
            pl.col('gene_name').alias('name'),
            pl.col('gene_id').alias('id'),
        ])
    else:
        df = df.with_columns([
            pl.col('transcript_name').alias('name'),
            pl.col('transcript_id').alias('id'),
        ])
    
    # Drop rows with missing name or id
    df = df.filter(pl.col('name').is_not_null() & pl.col('id').is_not_null())
    
    # Compute regulatory domain
    # SnapATAC2 converts 1-based GTF to 0-based positions:
    #   left = GTF_start - 1 (0-based start)
    #   right = GTF_end - 1 (0-based position of last base)
    # Then computes regulatory domain using these positions
    if include_gene_body:
        # Domain = [TSS - upstream, TES + downstream]
        # For + strand: TSS = left, TES = right
        # For - strand: TSS = right, TES = left
        df = df.with_columns([
            pl.when(pl.col('strand') == '+')
                .then(pl.col('start') - 1 - upstream)  # left - upstream
                .otherwise(pl.col('start') - 1 - downstream)  # left - downstream
                .clip(lower_bound=0)
                .alias('reg_start'),
            pl.when(pl.col('strand') == '+')
                .then(pl.col('end') - 1 + downstream)  # right + downstream (fixed: was end + downstream)
                .otherwise(pl.col('end') - 1 + upstream)  # right + upstream (fixed: was end + upstream)
                .alias('reg_end'),
        ])
    else:
        # Domain = [TSS - upstream, TSS + downstream]
        df = df.with_columns([
            pl.when(pl.col('strand') == '+')
                .then(pl.col('start') - 1 - upstream)
                .otherwise(pl.col('end') - 1 - downstream)
                .clip(lower_bound=0)
                .alias('reg_start'),
            pl.when(pl.col('strand') == '+')
                .then(pl.col('start') - 1 + downstream)
                .otherwise(pl.col('end') - 1 + upstream)
                .alias('reg_end'),
        ])
    
    # Select and rename columns - include gene_name for MAX aggregation
    result = df.select([
        'chrom',
        pl.col('reg_start').alias('start'),
        pl.col('reg_end').alias('end'),
        'name',
        'id',
        'gene_name',  # For MAX aggregation across transcripts of same gene
        'strand',
    ])
    
    # Deduplicate by (chrom, start, end, id) - transcripts with same region
    result = result.unique(subset=['chrom', 'start', 'end', 'id'], keep='first')
    
    # Sort by chromosome and start position
    result = result.sort(['chrom', 'start'])
    
    logger.info(f"Loaded {len(result):,} transcript regulatory domains (for {id_type}-level output)")
    return result


def create_gene_matrix_gpu(
    fragments_df: cudf.DataFrame,
    gene_regions: pl.DataFrame,
    exclude_chroms: Optional[list] = None,
    min_fragments_per_cell: int = 100,
    cell_metadata: Optional[cudf.DataFrame] = None,
    filter_query: Optional[str] = None,
    cell_batch_size: int = 500,
) -> Tuple[sp.csr_matrix, cudf.DataFrame, cudf.DataFrame]:
    """
    Generate a gene activity matrix from ATAC fragment data using GPU acceleration.

    Uses paired-insertion counting strategy: each fragment contributes insertions
    at start and end positions. If both insertions fall within the same gene,
    count +1 for that gene. If they fall in different genes, each gene gets +1.
    
    Memory-efficient implementation that processes cells in batches to avoid OOM.

    Parameters
    ----------
    fragments_df : cudf.DataFrame
        Fragment data with columns: 'chrom', 'start', 'end', 'barcode', 'count'
    gene_regions : pl.DataFrame
        Gene regulatory domains from load_gene_annotation()
    exclude_chroms : list, optional
        List of chromosomes to exclude. (default: ["chrM", "chrY", "M", "Y"])
    min_fragments_per_cell : int
        Minimum fragments required per barcode to include (default: 100)
    cell_metadata : cudf.DataFrame, optional
        Optional cell metadata for filtering.
    filter_query : str, optional
        Additional query string for filtering cells.
    cell_batch_size : int
        Number of cells to process per batch (default: 500). Lower values
        reduce GPU memory usage but may be slower.

    Returns
    -------
    matrix : scipy.sparse.csr_matrix
        Gene matrix with shape (n_cells, n_genes)
    cell_metadata : cudf.DataFrame
        Metadata for cells (barcodes)
    gene_metadata : cudf.DataFrame
        Metadata for genes
    """
    # Convert gene regions to cudf for GPU processing
    gene_df = cudf.from_pandas(gene_regions.to_pandas())
    
    # Get valid chromosomes (those in gene annotation, minus excluded ones)
    valid_chroms = set(gene_df['chrom'].unique().to_arrow().to_pylist())
    
    if exclude_chroms is not None:
        if isinstance(exclude_chroms, str):
            exclude_chroms = [exclude_chroms]
        valid_chroms = valid_chroms - set(exclude_chroms)
    
    valid_chroms = sorted(list(valid_chroms))  # Sort for reproducibility
    
    if len(valid_chroms) == 0:
        raise ValueError("No valid chromosomes in gene annotation after exclusion")
    
    # Filter fragments to valid chromosomes
    # Use categories to avoid ValueError in cudf if some valid_chroms are missing from fragments
    cats = fragments_df['chrom'].dtype.categories.to_arrow().to_pylist()
    fetch_chroms = [c for c in valid_chroms if c in cats]
    fragments_for_counting = fragments_df[fragments_df['chrom'].isin(fetch_chroms)]
    
    # Cell filtering (same logic as tile matrix)
    if cell_metadata is None:
        logger.debug("Filtering cells by unique fragment count")
        barcode_counts = fragments_for_counting.groupby('barcode', observed=True).agg({
            'count': ['sum', 'size']
        })
        barcode_counts.columns = ['n_total', 'n_unique']
        barcode_counts = barcode_counts.reset_index()

        valid_barcodes = barcode_counts[
            barcode_counts['n_unique'] >= min_fragments_per_cell
        ]['barcode']
        cell_metadata = barcode_counts[barcode_counts['barcode'].isin(valid_barcodes)]
    else:
        logger.debug("Using provided cell metadata for filtering")
        if filter_query:
            cell_metadata = cell_metadata.query(filter_query)
        
        if 'n_unique' in cell_metadata.columns:
            cell_metadata = cell_metadata[cell_metadata['n_unique'] >= min_fragments_per_cell]
        else:
            subset_frags = fragments_for_counting[fragments_for_counting['barcode'].isin(cell_metadata['barcode'])]
            barcode_counts = subset_frags.groupby('barcode', observed=True).agg({
                'count': ['sum', 'size']
            })
            barcode_counts.columns = ['n_total', 'n_unique']
            barcode_counts = barcode_counts.reset_index()
            
            valid_bc = barcode_counts[barcode_counts['n_unique'] >= min_fragments_per_cell]['barcode']
            cell_metadata = cell_metadata[cell_metadata['barcode'].isin(valid_bc)]
            cell_metadata = cell_metadata.merge(barcode_counts[['barcode', 'n_unique']], on='barcode', how='left')

        valid_barcodes = cell_metadata['barcode']

    # Filter fragments to valid barcodes and chromosomes  
    # Use categories to avoid ValueError in cudf
    barcode_cats = fragments_df['barcode'].dtype.categories.to_arrow().to_pylist()
    fetch_bc = [b for b in valid_barcodes.to_arrow().to_pylist() if b in barcode_cats]
    fragments_df = fragments_df[fragments_df['barcode'].isin(fetch_bc)]
    
    chrom_cats = fragments_df['chrom'].dtype.categories.to_arrow().to_pylist()
    fetch_chroms = [c for c in valid_chroms if c in chrom_cats]
    fragments_df = fragments_df[fragments_df['chrom'].isin(fetch_chroms)]
    
    # Get unique barcodes and create mapping
    unique_barcodes = fragments_df['barcode'].unique().reset_index(drop=True)
    n_cells = len(unique_barcodes)
    logger.debug(f"Retained {n_cells} cells")
    
    # Create barcode to global index mapping (on CPU for memory efficiency)
    barcode_list = unique_barcodes.to_arrow().to_pylist()
    barcode_to_global_idx = {bc: i for i, bc in enumerate(barcode_list)}

    # Filter and prepare gene data
    gene_df = gene_df[gene_df['chrom'].isin(valid_chroms)]
    gene_df = gene_df.reset_index(drop=True)
    gene_df['gene_idx'] = cp.arange(len(gene_df))
    
    # Check if we should aggregate by gene_name (SnapATAC2 default for id_type='gene')
    # If the user requested transcript-level (id_type='transcript'), we don't aggregate.
    # We can detect this by checking if 'id' and 'gene_name' are different for any row,
    # or more simply, if gene_regions has 'transcript_id' info but was filtered.
    # Actually, load_gene_annotation already sets 'id' and 'name' appropriately.
    # Consistency with SnapATAC2: if id_type='gene', we take MAX across transcripts.
    # If id_type='transcript', 'id' is transcript_id.
    
    # Create feature mapping
    # By default, use 'id' as the unique feature identifier
    unique_features = gene_df[['id']].drop_duplicates().sort_values('id').reset_index(drop=True)
    unique_features['final_feature_idx'] = cp.arange(len(unique_features))
    n_final_features = len(unique_features)
    
    # Build gene_idx -> final_feature_idx mapping
    gene_to_final = gene_df[['gene_idx', 'id']].merge(
        unique_features, on='id', how='left'
    )
    
    logger.debug(f"Computing matrix: {n_cells} cells × {n_final_features} features (batch size: {cell_batch_size})")

    # Prepare chromosome-level gene data (sorted by start for searchsorted)
    chrom_gene_data = {}
    for chrom in valid_chroms:
        chrom_genes = gene_df[gene_df['chrom'] == chrom].sort_values('start').reset_index(drop=True)
        if len(chrom_genes) > 0:
            chrom_gene_data[chrom] = {
                'starts': chrom_genes['start'].values,
                'ends': chrom_genes['end'].values,
                'indices': chrom_genes['gene_idx'].values,
            }

    # Process cells in batches to reduce memory
    n_batches = (n_cells + cell_batch_size - 1) // cell_batch_size
    all_batch_matrices = []
    max_count_overall = 0
    
    for batch_idx in range(n_batches):
        batch_start = batch_idx * cell_batch_size
        batch_end = min((batch_idx + 1) * cell_batch_size, n_cells)
        batch_barcodes = barcode_list[batch_start:batch_end]
        batch_size = len(batch_barcodes)
        
        logger.debug(f"Processing batch {batch_idx + 1}/{n_batches} ({batch_size} cells)")
        
        # Filter fragments for this batch
        batch_barcodes_gpu = cudf.Series(batch_barcodes)
        # Safe isin for categorical barcode
        fetch_batch_bc = [b for b in batch_barcodes if b in barcode_cats]
        batch_frags = fragments_df[fragments_df['barcode'].isin(fetch_batch_bc)]
        
        if len(batch_frags) == 0:
            # Empty batch - create zero matrix
            all_batch_matrices.append(sp.csr_matrix((batch_size, n_final_genes), dtype=np.uint8))
            continue
        
        # Create local barcode to batch index mapping
        batch_barcode_to_idx = cudf.DataFrame({
            'barcode': batch_barcodes_gpu,
            'cell_idx': cp.arange(batch_size, dtype=cp.int32)
        })
        batch_frags = batch_frags.merge(batch_barcode_to_idx, on='barcode', how='left')
        
        # Process each chromosome
        batch_transcript_counts = []
        
        for chrom in valid_chroms:
            if chrom not in chrom_gene_data or chrom not in chrom_cats:
                continue
                
            chrom_frags = batch_frags[batch_frags['chrom'] == chrom]
            if len(chrom_frags) == 0:
                continue
            
            gene_data = chrom_gene_data[chrom]
            
            # Find overlapping genes
            transcript_counts = _find_fragment_gene_overlaps_streaming(
                chrom_frags['start'].values,
                chrom_frags['end'].values,
                chrom_frags['cell_idx'].values,
                gene_data['starts'],
                gene_data['ends'],
                gene_data['indices'],
                chunk_size=200_000,  # Smaller chunks for memory efficiency
            )
            
            if len(transcript_counts) > 0:
                batch_transcript_counts.append(transcript_counts)
            
            del chrom_frags
            cp._default_memory_pool.free_all_blocks()
        
        # Build batch matrix
        if len(batch_transcript_counts) > 0:
            # Concatenate all chromosome results
            all_counts = cudf.concat(batch_transcript_counts, ignore_index=True)
            del batch_transcript_counts
            
            # Aggregate by (cell_idx, gene_idx) first
            all_counts = all_counts.groupby(['cell_idx', 'gene_idx'])['count'].sum().reset_index()
            
            # Map gene_idx to final_feature_idx
            all_counts = all_counts.merge(gene_to_final, on='gene_idx', how='left')
            
            # MAX aggregation: for each (cell, feature_id), take max count across transcripts
            # This handles transcripts -> gene aggregation if id_type was 'gene'
            # because 'id' would be the gene_id for multiple transcripts.
            # If id_type was 'transcript', 'id' is unique to each transcript, so max is just the value.
            feature_counts = all_counts.groupby(['cell_idx', 'final_feature_idx'])['count'].max().reset_index()
            del all_counts
            
            # Track max for dtype selection
            batch_max = int(feature_counts['count'].max())
            max_count_overall = max(max_count_overall, batch_max)
            
            # Build sparse matrix for this batch (on CPU)
            feature_counts_pd = feature_counts.to_pandas()
            del feature_counts
            cp._default_memory_pool.free_all_blocks()
            
            batch_matrix = sp.coo_matrix(
                (feature_counts_pd['count'].values.astype(np.int32),
                 (feature_counts_pd['cell_idx'].values, feature_counts_pd['final_feature_idx'].values)),
                shape=(batch_size, n_final_features),
                dtype=np.int32
            ).tocsr()
            del feature_counts_pd
        else:
            batch_matrix = sp.csr_matrix((batch_size, n_final_features), dtype=np.int32)
        
        all_batch_matrices.append(batch_matrix)
        
        # Clean up batch data
        del batch_frags, batch_barcode_to_idx, batch_barcodes_gpu
        cp._default_memory_pool.free_all_blocks()

    # Stack all batch matrices
    logger.debug("Combining batch matrices...")
    matrix = sp.vstack(all_batch_matrices, format='csr')
    del all_batch_matrices
    
    # Optimize dtype based on actual max value
    if max_count_overall > 0:
        matrix_dtype = optimize_sparse_matrix_dtype(max_count_overall)
        matrix = matrix.astype(matrix_dtype)
    else:
        matrix = matrix.astype(np.uint8)

    # Prepare gene metadata (unique features)
    gene_metadata = gene_df.groupby('id').first().reset_index()
    gene_metadata = gene_metadata.merge(unique_features, on='id', how='left')
    gene_metadata = gene_metadata.sort_values('final_feature_idx').reset_index(drop=True)
    gene_metadata = gene_metadata[['chrom', 'start', 'end', 'name', 'id', 'strand', 'gene_name', 'final_feature_idx']]
    gene_metadata = gene_metadata.rename(columns={'final_feature_idx': 'gene_idx'})

    logger.debug(f"Matrix: {n_cells} cells × {n_final_features} features, nnz: {matrix.nnz:,}")

    # Prepare cell metadata
    barcode_to_idx = cudf.DataFrame({
        'barcode': cudf.Series(barcode_list),
        'cell_idx': cp.arange(n_cells)
    })
    cell_metadata = barcode_to_idx.merge(cell_metadata, on='barcode', how='left')
    cell_metadata = cell_metadata.sort_values('cell_idx').reset_index(drop=True)
    if cell_metadata['barcode'].dtype != 'object':
        cell_metadata['barcode'] = cell_metadata['barcode'].astype(str)

    return matrix, cell_metadata, gene_metadata


def _find_fragment_gene_overlaps_streaming(
    frag_starts: cp.ndarray,
    frag_ends: cp.ndarray,
    cell_indices: cp.ndarray,
    gene_starts: cp.ndarray,
    gene_ends: cp.ndarray,
    gene_indices: cp.ndarray,
    chunk_size: int = 500_000,
) -> cudf.DataFrame:
    """
    Find all (cell, gene) overlaps using streaming 2D window approach.
    
    For each insertion (start and end-1), finds overlapping gene regions using
    searchsorted to find an anchor point, then looks backward through a window
    of candidate genes and validates overlaps.
    
    Returns aggregated (cell_idx, gene_idx, count) DataFrame.
    """
    n_fragments = len(frag_starts)
    n_genes = len(gene_starts)
    
    if n_fragments == 0 or n_genes == 0:
        return cudf.DataFrame({'cell_idx': cp.array([], dtype=cp.int32),
                               'gene_idx': cp.array([], dtype=cp.int32),
                               'count': cp.array([], dtype=cp.int32)})
    
    all_pairs_cells = []
    all_pairs_frags = []
    all_pairs_genes = []
    
    # Small window - SnapATAC2 uses 100
    max_check = 100
    offsets = cp.arange(max_check, dtype=cp.int32)
    
    for chunk_start in range(0, n_fragments, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_fragments)
        
        starts = frag_starts[chunk_start:chunk_end]
        ends = frag_ends[chunk_start:chunk_end] - 1  # Convert to insertion position
        cells = cell_indices[chunk_start:chunk_end]
        frag_idxs = cp.arange(chunk_start, chunk_end, dtype=cp.int32)
        
        # Both insertions
        positions = cp.concatenate([starts, ends])
        pos_cells = cp.concatenate([cells, cells])
        pos_frags = cp.concatenate([frag_idxs, frag_idxs])
        n_pos = len(positions)
        
        # searchsorted on gene_starts: index of first gene with start > position
        right_idx = cp.searchsorted(gene_starts, positions, side='right')
        
        # Create 2D candidate indices: (n_pos, max_check)
        # candidate[i,j] = right_idx[i] - 1 - j
        cand_idx = right_idx[:, None] - 1 - offsets[None, :]  # (n_pos, max_check)
        
        # Clip to valid range and mark invalid
        valid = (cand_idx >= 0) & (cand_idx < n_genes)
        cand_idx_safe = cp.clip(cand_idx, 0, n_genes - 1)
        
        # Look up gene ends for candidates
        cand_ends = gene_ends[cand_idx_safe]
        
        # Overlap: gene_start <= pos (guaranteed by searchsorted) AND gene_end > pos
        overlaps = valid & (cand_ends > positions[:, None])
        
        # Extract overlapping pairs
        row_idx, col_idx = cp.where(overlaps)
        if len(row_idx) > 0:
            all_pairs_cells.append(pos_cells[row_idx])
            all_pairs_frags.append(pos_frags[row_idx])
            all_pairs_genes.append(gene_indices[cand_idx_safe[row_idx, col_idx]])
        
        del positions, pos_cells, pos_frags, right_idx, cand_idx, cand_idx_safe, cand_ends, overlaps
        cp._default_memory_pool.free_all_blocks()
    
    if len(all_pairs_cells) == 0:
        return cudf.DataFrame({'cell_idx': cp.array([], dtype=cp.int32),
                               'gene_idx': cp.array([], dtype=cp.int32),
                               'count': cp.array([], dtype=cp.int32)})
    
    # Combine and deduplicate all pairs, then count
    pairs_df = cudf.DataFrame({
        'cell_idx': cp.concatenate(all_pairs_cells),
        'frag_idx': cp.concatenate(all_pairs_frags),
        'gene_idx': cp.concatenate(all_pairs_genes),
    })
    
    # Deduplicate: each (fragment, gene) pair counts as 1
    pairs_df = pairs_df.drop_duplicates()
    
    # Count per (cell, gene)
    counts = pairs_df.groupby(['cell_idx', 'gene_idx']).size().reset_index(name='count')
    
    del all_pairs_cells, all_pairs_frags, all_pairs_genes, pairs_df
    cp._default_memory_pool.free_all_blocks()
    
    return counts


def gene_matrix_to_anndata(
    matrix: sp.csr_matrix,
    cell_metadata: cudf.DataFrame,
    gene_metadata: cudf.DataFrame,
):
    """
    Convert gene matrix to AnnData object.

    Parameters
    ----------
    matrix : scipy.sparse.csr_matrix
        Gene matrix from create_gene_matrix_gpu (already on CPU)
    cell_metadata : cudf.DataFrame
        Cell metadata with barcodes
    gene_metadata : cudf.DataFrame
        Gene metadata with names and IDs

    Returns
    -------
    adata : AnnData
        AnnData object with gene activity matrix
    """
    import scanpy as sc

    # Matrix is already on CPU with optimal dtype
    obs = cell_metadata.to_pandas()
    obs.index = obs['barcode'].values

    var = gene_metadata.to_pandas()
    # Use gene name as index (with ID as backup for duplicates)
    var.index = var['name'].values
    # Handle duplicate gene names by appending ID
    if var.index.duplicated().any():
        dup_mask = var.index.duplicated(keep=False)
        var.loc[dup_mask, 'index'] = var.loc[dup_mask, 'name'] + '_' + var.loc[dup_mask, 'id']
        var.index = var['index'].fillna(var['name']).values

    adata = sc.AnnData(X=matrix, obs=obs, var=var)
    logger.debug(f"Created AnnData: {adata.shape[0]} cells × {adata.shape[1]} genes")

    return adata


def make_gene_matrix(
    input_parquet: str | Path,
    gene_anno: str | Path,
    output_path: Optional[str | Path] = None,
    id_type: Literal["gene", "transcript"] = "gene",
    upstream: int = 2000,
    downstream: int = 0,
    include_gene_body: bool = True,
    min_fragments_per_cell: int = 100,
    exclude_chroms: Optional[list] = None,
    metrics: Optional[str | Path | cudf.DataFrame] = None,
    filter_query: Optional[str] = None,
    barcode_prefix: Optional[str] = None,
    low_memory: bool = False,
    cell_batch_size: int = 500,
    gene_name_key: str = "gene_name",
    gene_id_key: str = "gene_id",
    transcript_name_key: str = "transcript_name",
    transcript_id_key: str = "transcript_id",
) -> 'sc.AnnData':
    """
    Process ATAC fragments parquet file and generate gene activity matrix.

    Uses paired-insertion counting strategy matching SnapATAC2: each fragment 
    contributes insertions at start and end positions. If both insertions fall 
    within the same gene's regulatory domain, count +1. If in different genes, 
    each gene gets +1.

    Parameters
    ----------
    input_parquet : str or Path
        Path to input parquet file containing ATAC fragments
    gene_anno : str or Path
        Path to GTF/GFF gene annotation file.
    output_path : str or Path, optional
        Path for output .h5ad file. If None, uses input filename.
    id_type : str
        "gene" or "transcript" - which feature type to use (default: "gene").
    upstream : int
        Base pairs upstream of TSS to include (default: 2000).
    downstream : int
        Base pairs downstream of regulatory domain (default: 0).
    include_gene_body : bool
        Whether to include the gene body in the regulatory domain (default: True).
    min_fragments_per_cell : int
        Minimum fragments required per barcode (default: 100)
    exclude_chroms : list, optional
        List of chromosomes to exclude. (default: None)
    metrics : str, Path, or cudf.DataFrame, optional
        Path to a CSV file or a cuDF DataFrame containing cell metrics for filtering.
    filter_query : str, optional
        Query string for filtering cells based on metrics (e.g. "tsse_score > 5").
    barcode_prefix : str, optional
        Prefix to add to barcodes
    low_memory : bool
        Use low memory mode for Parquet reading (default: False)
    cell_batch_size : int
        Number of cells to process per batch (default: 500). Lower values
        reduce GPU memory usage but may be slower.
    gene_name_key : str
        Key for gene name in GTF attributes (default: "gene_name").
    gene_id_key : str
        Key for gene ID in GTF attributes (default: "gene_id").
    transcript_name_key : str
        Key for transcript name in GTF attributes (default: "transcript_name").
    transcript_id_key : str
        Key for transcript ID in GTF attributes (default: "transcript_id").

    Returns
    -------
    adata : AnnData
        AnnData object with gene activity matrix
    """
    from .process import read_fragments_parquet
    import scanpy as sc

    input_parquet = Path(input_parquet)
    gene_anno = Path(gene_anno)
    
    if output_path is None:
        output_path = input_parquet.with_suffix('').with_name(
            input_parquet.stem + '_gene_matrix.h5ad'
        )
    else:
        output_path = Path(output_path)

    # Load gene annotation
    gene_regions = load_gene_annotation(
        gtf_path=gene_anno,
        id_type=id_type,
        upstream=upstream,
        downstream=downstream,
        include_gene_body=include_gene_body,
        gene_name_key=gene_name_key,
        gene_id_key=gene_id_key,
        transcript_name_key=transcript_name_key,
        transcript_id_key=transcript_id_key,
    )

    # Load metrics if provided
    cell_metadata_input = None
    if metrics is not None:
        if isinstance(metrics, cudf.DataFrame):
            cell_metadata_input = metrics
        else:
            metrics_path = Path(metrics)
            if metrics_path.exists():
                logger.info(f"Loading cell metrics from {metrics_path}")
                cell_metadata_input = cudf.read_csv(str(metrics_path))
            else:
                logger.warning(f"Metrics file {metrics_path} not found. Proceeding without it.")

    logger.info(f"Processing {input_parquet.name}")

    def _cleanup_memory():
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()

    def _read_and_process(use_low_mem: bool, exclude: Optional[List[str]] = None):
        df = read_fragments_parquet(input_parquet, low_memory=use_low_mem)
        
        df_sorted = df.sort_values('barcode')
        del df
        _cleanup_memory()

        matrix, cell_metadata, gene_metadata = create_gene_matrix_gpu(
            fragments_df=df_sorted,
            gene_regions=gene_regions,
            exclude_chroms=exclude,
            min_fragments_per_cell=min_fragments_per_cell,
            cell_metadata=cell_metadata_input,
            filter_query=filter_query,
            cell_batch_size=cell_batch_size,
        )
        return matrix, cell_metadata, gene_metadata

    start_time = time.perf_counter()
    try:
        matrix, cell_metadata, gene_metadata = _read_and_process(low_memory, exclude_chroms)
    except (MemoryError, RuntimeError) as e:
        err_msg = str(e).lower()
        is_oom = "out of memory" in err_msg or "std::bad_alloc" in err_msg or "cudaerrormemoryallocation" in err_msg
        
        if is_oom:
            if not low_memory:
                logger.warning(f"CUDA Out of Memory. Retrying with low_memory=True: {e}")
                _cleanup_memory()
                matrix, cell_metadata, gene_metadata = _read_and_process(True, exclude_chroms)
            else:
                logger.error(f"CUDA Out of Memory even with low_memory=True: {e}")
                raise e
        else:
            raise e

    # Convert to AnnData
    adata = gene_matrix_to_anndata(matrix, cell_metadata, gene_metadata)
    
    if barcode_prefix:
        adata.obs_names = [f"{barcode_prefix}{b}" for b in adata.obs_names]

    # Save
    adata.write_h5ad(str(output_path))
    total_time = time.perf_counter() - start_time
    logger.info(f"Created {output_path.name}: {adata.shape[0]:,} cells × {adata.shape[1]:,} genes ({total_time:.1f}s)")

    return adata
