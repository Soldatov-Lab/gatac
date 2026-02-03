"""
GPU-accelerated gene activity matrix generation from ATAC fragment data.

Implements paired-insertion counting strategy matching SnapATAC2.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Literal

import cudf
import cupy as cp
import cupyx.scipy.sparse as cusp
import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


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
) -> Tuple[cusp.csr_matrix, cudf.DataFrame, cudf.DataFrame]:
    """
    Generate a gene activity matrix from ATAC fragment data using GPU acceleration.

    Uses paired-insertion counting strategy: each fragment contributes insertions
    at start and end positions. If both insertions fall within the same gene,
    count +1 for that gene. If they fall in different genes, each gene gets +1.

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

    Returns
    -------
    matrix : cupyx.scipy.sparse.csr_matrix
        Gene matrix with shape (n_cells, n_genes)
    cell_metadata : cudf.DataFrame
        Metadata for cells (barcodes)
    gene_metadata : cudf.DataFrame
        Metadata for genes
    """
    # Convert gene regions to cudf for GPU processing
    gene_df = cudf.from_pandas(gene_regions.to_pandas())
    
    # Get chromosomes present in both gene annotation AND fragment data
    gene_chroms = set(gene_df['chrom'].unique().to_arrow().to_pylist())
    frag_chroms = set(fragments_df['chrom'].unique().to_arrow().to_pylist())
    
    # Intersect: only chromosomes in both datasets
    valid_chroms = gene_chroms & frag_chroms
    
    if exclude_chroms is not None:
        if isinstance(exclude_chroms, str):
            exclude_chroms = [exclude_chroms]
        valid_chroms = valid_chroms - set(exclude_chroms)
    
    valid_chroms = list(valid_chroms)
    
    if len(valid_chroms) == 0:
        raise ValueError("No common chromosomes between gene annotation and fragment data")
    
    # Filter fragments to valid chromosomes
    fragments_for_counting = fragments_df[fragments_df['chrom'].isin(valid_chroms)]
    
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
    fragments_df = fragments_df[fragments_df['barcode'].isin(valid_barcodes)]
    fragments_df = fragments_df[fragments_df['chrom'].isin(valid_chroms)]
    logger.debug(f"Retained {len(valid_barcodes)} cells")

    # Create barcode to index mapping
    unique_barcodes = fragments_df['barcode'].unique().reset_index(drop=True)
    barcode_to_idx = cudf.DataFrame({
        'barcode': unique_barcodes,
        'cell_idx': cp.arange(len(unique_barcodes))
    })
    fragments_df = fragments_df.merge(barcode_to_idx, on='barcode', how='left')

    # Filter and index genes
    gene_df = gene_df[gene_df['chrom'].isin(valid_chroms)]
    gene_df = gene_df.reset_index(drop=True)
    gene_df['gene_idx'] = cp.arange(len(gene_df))
    
    n_genes = len(gene_df)
    n_cells = len(unique_barcodes)
    logger.debug(f"Computing matrix: {n_cells} cells × {n_genes} genes")

    # Process each chromosome separately to manage memory
    all_cell_gene_pairs = []
    frag_offset = 0  # Global offset to ensure unique frag_idx across chromosomes

    for chrom in valid_chroms:
        chrom_frags = fragments_df[fragments_df['chrom'] == chrom]
        chrom_genes = gene_df[gene_df['chrom'] == chrom]
        
        if len(chrom_frags) == 0 or len(chrom_genes) == 0:
            continue
        
        n_chrom_frags = len(chrom_frags)
        
        # Get gene boundaries as cupy arrays for searchsorted
        gene_starts = chrom_genes['start'].values
        gene_ends = chrom_genes['end'].values
        gene_indices = chrom_genes['gene_idx'].values
        
        # Fragment insertions (start and end-1)
        frag_starts = chrom_frags['start'].values
        frag_ends = (chrom_frags['end'] - 1).values
        cell_indices = chrom_frags['cell_idx'].values
        frag_indices = cp.arange(n_chrom_frags) + frag_offset  # Globally unique fragment index
        
        # Find overlapping genes for BOTH insertions, then deduplicate per fragment
        # This matches SnapATAC2's paired-insertion counting strategy:
        # For each fragment, count each gene at most once (even if both insertions overlap it)
        
        pairs = _find_fragment_gene_pairs(
            frag_starts, frag_ends, cell_indices, frag_indices,
            gene_starts, gene_ends, gene_indices
        )
        if pairs is not None:
            all_cell_gene_pairs.append(pairs)
        
        frag_offset += n_chrom_frags

    # Build sparse matrix - first at transcript level, then MAX aggregate to gene level
    if len(all_cell_gene_pairs) > 0:
        all_pairs = cudf.concat(all_cell_gene_pairs, ignore_index=True)
        
        # Deduplicate: each (fragment, transcript) pair counts as 1
        # (handles case where both insertions overlap same transcript)
        all_pairs = all_pairs.drop_duplicates(subset=['frag_idx', 'gene_idx'])
        
        # Count per (cell, transcript) - this is transcript-level counting
        transcript_counts = all_pairs.groupby(['cell_idx', 'gene_idx']).size().reset_index(name='count')
        
        # MAX aggregation: for each cell, take max count across all transcripts of same gene
        # First, add gene_name to transcript_counts
        transcript_counts = transcript_counts.merge(
            gene_df[['gene_idx', 'gene_name']],
            on='gene_idx',
            how='left'
        )
        
        # Group by (cell_idx, gene_name) and take MAX count
        gene_counts = transcript_counts.groupby(['cell_idx', 'gene_name'])['count'].max().reset_index()
        
        # Create gene index mapping (unique genes)
        unique_genes = gene_df[['gene_name']].drop_duplicates().reset_index(drop=True)
        unique_genes['final_gene_idx'] = cp.arange(len(unique_genes))
        
        gene_counts = gene_counts.merge(unique_genes, on='gene_name', how='left')
        
        n_genes = len(unique_genes)
        
        coo_matrix = cusp.coo_matrix(
            (gene_counts['count'].values.astype(cp.float32), 
             (gene_counts['cell_idx'].values, gene_counts['final_gene_idx'].values)),
            shape=(n_cells, n_genes),
            dtype=cp.float32
        )
        matrix = coo_matrix.tocsr()
        
        # Prepare gene metadata (unique genes)
        gene_metadata = gene_df.groupby('gene_name').first().reset_index()
        gene_metadata = gene_metadata.merge(unique_genes, on='gene_name', how='left')
        gene_metadata = gene_metadata.sort_values('final_gene_idx').reset_index(drop=True)
        gene_metadata = gene_metadata[['chrom', 'start', 'end', 'name', 'id', 'strand', 'gene_name', 'final_gene_idx']]
        gene_metadata = gene_metadata.rename(columns={'final_gene_idx': 'gene_idx'})
    else:
        # Get unique genes for empty matrix
        unique_genes = gene_df[['gene_name']].drop_duplicates()
        n_genes = len(unique_genes)
        matrix = cusp.csr_matrix((n_cells, n_genes), dtype=cp.float32)
        gene_metadata = gene_df.groupby('gene_name').first().reset_index()
        gene_metadata['gene_idx'] = cp.arange(len(gene_metadata))
        gene_metadata = gene_metadata[['chrom', 'start', 'end', 'name', 'id', 'strand', 'gene_name', 'gene_idx']]

    logger.debug(f"Matrix: {n_cells} cells × {n_genes} genes, nnz: {matrix.nnz:,}")

    # Prepare cell metadata
    cell_metadata = barcode_to_idx.merge(cell_metadata, on='barcode', how='left')
    cell_metadata = cell_metadata.sort_values('cell_idx').reset_index(drop=True)
    if cell_metadata['barcode'].dtype != 'object':
        cell_metadata['barcode'] = cell_metadata['barcode'].astype(str)

    return matrix, cell_metadata, gene_metadata


def _find_fragment_gene_pairs(
    frag_starts: cp.ndarray,
    frag_ends: cp.ndarray,
    cell_indices: cp.ndarray,
    frag_indices: cp.ndarray,
    gene_starts: cp.ndarray,
    gene_ends: cp.ndarray,
    gene_indices: cp.ndarray,
) -> Optional[cudf.DataFrame]:
    """
    Find all (cell, fragment, gene) pairs where either insertion overlaps the gene.
    
    For paired-insertion counting matching SnapATAC2:
    - Find all genes overlapped by either the start or end insertion
    - Return DataFrame with cell_idx, frag_idx, gene_idx
    - Caller will deduplicate (frag_idx, gene_idx) pairs to ensure each gene
      is counted at most once per fragment
    """
    n_fragments = len(frag_starts)
    n_genes = len(gene_starts)
    
    if n_fragments == 0 or n_genes == 0:
        return None
    
    # Process both insertions together
    all_pairs_cells = []
    all_pairs_frags = []
    all_pairs_genes = []
    
    for positions in [frag_starts, frag_ends]:
        # Use searchsorted to find potential overlapping genes
        right_idx = cp.searchsorted(gene_starts, positions, side='right')
        
        # For each insertion, check genes in a window before right_idx
        max_check = 100  # Max overlapping genes to check per insertion (handles dense regions with up to 95 transcripts)
        check_range = cp.arange(max_check, dtype=cp.int32)
        
        # Candidate gene indices for each position
        candidate_gene_idx = right_idx[:, None] - max_check + check_range[None, :]
        candidate_gene_idx = cp.clip(candidate_gene_idx, 0, n_genes - 1)
        
        # Get gene boundaries for candidates
        cand_starts = gene_starts[candidate_gene_idx]
        cand_ends = gene_ends[candidate_gene_idx]
        cand_gene_ids = gene_indices[candidate_gene_idx]
        
        # Check overlap: gene_start <= position < gene_end
        positions_expanded = positions[:, None]
        overlaps = (cand_starts <= positions_expanded) & (positions_expanded < cand_ends)
        
        # Extract valid pairs
        overlap_idx = cp.where(overlaps)
        if len(overlap_idx[0]) > 0:
            insert_idx = overlap_idx[0]
            gene_local_idx = overlap_idx[1]
            
            all_pairs_cells.append(cell_indices[insert_idx])
            all_pairs_frags.append(frag_indices[insert_idx])
            all_pairs_genes.append(cand_gene_ids[insert_idx, gene_local_idx])
    
    if len(all_pairs_cells) == 0:
        return None
    
    # Combine all pairs
    return cudf.DataFrame({
        'cell_idx': cp.concatenate(all_pairs_cells),
        'frag_idx': cp.concatenate(all_pairs_frags),
        'gene_idx': cp.concatenate(all_pairs_genes),
    })


def gene_matrix_to_anndata(
    matrix: cusp.csr_matrix,
    cell_metadata: cudf.DataFrame,
    gene_metadata: cudf.DataFrame,
):
    """
    Convert GPU gene matrix to AnnData object.

    Parameters
    ----------
    matrix : cupyx.scipy.sparse.csr_matrix
        Gene matrix from create_gene_matrix_gpu
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
    import scipy.sparse as sp

    logger.debug("Converting GPU matrix to CPU")
    matrix_cpu = sp.csr_matrix(
        (matrix.data.get().astype(np.float32), matrix.indices.get(), matrix.indptr.get()),
        shape=matrix.shape
    )

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

    adata = sc.AnnData(X=matrix_cpu, obs=obs, var=var)
    logger.debug(f"Created AnnData: {adata.shape[0]} cells × {adata.shape[1]} genes")

    return adata
