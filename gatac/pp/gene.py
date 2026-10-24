"""
GPU-accelerated gene activity matrix generation from ATAC fragment data.

Implements paired-insertion counting strategy matching SnapATAC2.
"""
from __future__ import annotations

import logging
import time
import gc
from pathlib import Path
from typing import Optional, Tuple, Literal, List, Union

import cudf
import cupy as cp
import cupyx.scipy.sparse as cusp
import numpy as np
import pandas as pd
import scipy.sparse as sp

from .._readers import read_gtf, read_metrics

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
) -> pd.DataFrame:
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
    pd.DataFrame
        One row per transcript with columns ['chrom', 'start', 'end', 'name',
        'id', 'gene_name', 'strand', 'feature', 'order'], where start/end
        define the regulatory domain, 'feature' is what the transcript counts
        for (its gene name for ``id_type="gene"``, as SnapATAC2 keys genes by
        name; its transcript ID otherwise) and 'order' its row in the file.
    """
    logger.info(f"Loading gene annotation from {gtf_path}")

    # Transcripts only - SnapATAC2 always counts at transcript level
    # then aggregates to gene level using MAX (for id_type="gene").
    # We always need gene_name for gene-level aggregation.
    df = read_gtf(gtf_path, "transcript", {
        "gene_name": gene_name_key,
        "gene_id": gene_id_key,
        "transcript_name": transcript_name_key,
        "transcript_id": transcript_id_key,
    })

    # Set name/id based on id_type, and drop rows with missing name or id
    level = "gene" if id_type == "gene" else "transcript"
    df["name"], df["id"] = df[f"{level}_name"], df[f"{level}_id"]
    df["order"] = np.arange(len(df))
    df = df[df["name"].notna() & df["id"].notna()]

    # Compute regulatory domain
    # SnapATAC2 converts 1-based GTF to 0-based positions:
    #   left = GTF_start - 1 (0-based start)
    #   right = GTF_end - 1 (0-based position of last base)
    # Then computes regulatory domain using these positions
    plus = (df["strand"] == "+").to_numpy()
    left = df["start"].to_numpy() - 1
    right = df["end"].to_numpy() - 1
    if include_gene_body:
        # Domain = [TSS - upstream, TES + downstream]
        # For + strand: TSS = left, TES = right
        # For - strand: TSS = right, TES = left
        reg_start = np.where(plus, left - upstream, left - downstream)
        reg_end = np.where(plus, right + downstream, right + upstream)
    else:
        # Domain = [TSS - upstream, TSS + downstream]
        reg_start = np.where(plus, left - upstream, right - downstream)
        reg_end = np.where(plus, left + downstream, right + upstream)

    # gene_name is kept for the MAX aggregation across transcripts of a gene
    result = pd.DataFrame({
        "chrom": df["chrom"].to_numpy(),
        "start": np.maximum(reg_start, 0),
        "end": reg_end,
        "name": df["name"].to_numpy(),
        "id": df["id"].to_numpy(),
        "gene_name": df["gene_name"].to_numpy(),
        "strand": df["strand"].to_numpy(),
        "feature": df["name" if id_type == "gene" else "id"].to_numpy(),
        "order": df["order"].to_numpy(),
    })

    # Deduplicate by (chrom, start, end, id) - transcripts with same region
    result = result.drop_duplicates(subset=["chrom", "start", "end", "id"], keep="first")

    # Sort by chromosome and start position
    result = result.sort_values(["chrom", "start"], kind="stable", ignore_index=True)

    logger.info(f"Loaded {len(result):,} transcript regulatory domains (for {id_type}-level output)")
    return result


def create_gene_matrix_gpu(
    fragments_df: cudf.DataFrame,
    gene_regions: pd.DataFrame,
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
    gene_regions : pd.DataFrame
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
        Unused; kept for backwards compatibility (all cells are processed
        together, one chromosome at a time).

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
    gene_df = cudf.from_pandas(gene_regions)
    
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
            
            # The counts' barcodes are categorical, and cuDF's isin of a string
            # column against a categorical one matches nothing
            valid_bc = barcode_counts[barcode_counts['n_unique'] >= min_fragments_per_cell]['barcode']
            cell_metadata = cell_metadata[cell_metadata['barcode'].isin(valid_bc.astype(str))]
            cell_metadata = cell_metadata.merge(barcode_counts[['barcode', 'n_unique']], on='barcode', how='left')

        valid_barcodes = cell_metadata['barcode']

    # Filter fragments to valid barcodes and chromosomes  
    # Use categories to avoid ValueError in cudf
    barcode_cats = fragments_df['barcode'].dtype.categories.to_arrow().to_pylist()
    barcode_cat_pos = {b: i for i, b in enumerate(barcode_cats)}
    fetch_bc = [b for b in valid_barcodes.to_arrow().to_pylist() if b in barcode_cat_pos]
    fragments_df = fragments_df[fragments_df['barcode'].isin(fetch_bc)]

    chrom_cats = fragments_df['chrom'].dtype.categories.to_arrow().to_pylist()
    fetch_chroms = [c for c in valid_chroms if c in chrom_cats]
    fragments_df = fragments_df[fragments_df['chrom'].isin(fetch_chroms)]

    # Get unique barcodes and create mapping
    unique_barcodes = fragments_df['barcode'].unique().reset_index(drop=True)
    n_cells = len(unique_barcodes)
    logger.debug(f"Retained {n_cells} cells")

    # Cell index of each fragment, through the categorical codes of its barcode
    barcode_list = unique_barcodes.to_arrow().to_pylist()
    code_to_cell = np.full(len(barcode_cats), -1, dtype=np.int32)
    code_to_cell[[barcode_cat_pos[b] for b in barcode_list]] = np.arange(n_cells, dtype=np.int32)
    frag_cells = cp.asarray(code_to_cell)[fragments_df['barcode'].cat.codes.values]

    # Filter and prepare gene data
    gene_df = gene_df[gene_df['chrom'].isin(valid_chroms)]
    gene_df = gene_df.reset_index(drop=True)
    gene_df['gene_idx'] = cp.arange(len(gene_df))
    
    # Features as SnapATAC2 makes them: one per gene name (genes sharing a
    # name are one feature) or per transcript ID, in order of first
    # appearance in the annotation
    unique_features = (gene_df.groupby('feature').agg({'order': 'min'}).reset_index()
                       .sort_values('order').reset_index(drop=True)[['feature']])
    unique_features['final_feature_idx'] = cp.arange(len(unique_features))
    n_final_features = len(unique_features)
    
    # Build gene_idx -> final_feature_idx mapping
    gene_to_final = gene_df[['gene_idx', 'feature']].merge(
        unique_features, on='feature', how='left'
    )
    
    logger.debug(f"Computing matrix: {n_cells} cells × {n_final_features} features")

    gene_to_feature = cp.empty(len(gene_df), dtype=cp.int32)
    gene_to_feature[gene_to_final['gene_idx'].values] = gene_to_final['final_feature_idx'].values

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

    # One pass per chromosome over all cells. Transcripts are reduced to
    # their feature by MAX (SnapATAC2's gene aggregation; a no-op when 'id'
    # is the transcript), within each chromosome and then across them, for
    # features with transcripts on several.
    chrom_feature_counts = []
    for chrom in valid_chroms:
        if chrom not in chrom_gene_data or chrom not in chrom_cats:
            continue
        in_chrom = (fragments_df['chrom'] == chrom).values
        if not in_chrom.any():
            continue
        gene_data = chrom_gene_data[chrom]
        transcript_counts = _count_fragment_gene_overlaps(
            fragments_df['start'].values[in_chrom],
            fragments_df['end'].values[in_chrom],
            frag_cells[in_chrom],
            gene_data['starts'],
            gene_data['ends'],
            gene_data['indices'],
        )
        if len(transcript_counts) == 0:
            continue
        transcript_counts['final_feature_idx'] = gene_to_feature[transcript_counts['gene_idx'].values]
        chrom_feature_counts.append(
            transcript_counts.groupby(['cell_idx', 'final_feature_idx'])['count'].max().reset_index()
        )
        del transcript_counts

    # CSR from the (cell, feature) pairs sorted on the GPU
    if chrom_feature_counts:
        feature_counts = cudf.concat(chrom_feature_counts, ignore_index=True)
        del chrom_feature_counts
        feature_counts = (feature_counts.groupby(['cell_idx', 'final_feature_idx'])['count'].max()
                          .reset_index().sort_values(['cell_idx', 'final_feature_idx']))
        max_count_overall = int(feature_counts['count'].max())
        cells = feature_counts['cell_idx'].values
        indptr = cp.searchsorted(cells, cp.arange(n_cells + 1, dtype=cells.dtype))
        matrix = sp.csr_matrix(
            (feature_counts['count'].values.astype(cp.int32).get(),
             feature_counts['final_feature_idx'].values.get(),
             indptr.get()),
            shape=(n_cells, n_final_features),
        )
        del feature_counts, cells, indptr
    else:
        max_count_overall = 0
        matrix = sp.csr_matrix((n_cells, n_final_features), dtype=np.int32)

    # Optimize dtype based on actual max value
    if max_count_overall > 0:
        matrix_dtype = optimize_sparse_matrix_dtype(max_count_overall)
        matrix = matrix.astype(matrix_dtype)
    else:
        matrix = matrix.astype(np.uint8)

    # Prepare gene metadata (unique features): chrom, names, ID and strand of
    # the feature's first transcript in the annotation, and the span of its
    # transcripts' domains on that chromosome, the region whose fragments
    # count for it. (groupby.first picked an arbitrary transcript.)
    first = gene_df.groupby('feature').agg({'order': 'min'}).reset_index()
    gene_metadata = gene_df[['order', 'feature', 'chrom', 'name', 'id', 'strand', 'gene_name']].merge(
        first, on=['feature', 'order'])
    span = (gene_df[['feature', 'chrom', 'start', 'end']]
            .merge(gene_metadata[['feature', 'chrom']], on=['feature', 'chrom'])
            .groupby('feature').agg({'start': 'min', 'end': 'max'}).reset_index())
    gene_metadata = gene_metadata.merge(span, on='feature').merge(unique_features, on='feature', how='left')
    gene_metadata = gene_metadata.sort_values('final_feature_idx').reset_index(drop=True)
    gene_metadata = gene_metadata[['feature', 'chrom', 'start', 'end', 'name', 'id', 'strand', 'gene_name',
                                   'final_feature_idx']]
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


def _expand_segments(
    seg: cp.ndarray,
    cov_ptr: cp.ndarray,
    cov_transcript: cp.ndarray,
) -> Tuple[cp.ndarray, cp.ndarray]:
    """Row and transcript of every (row, transcript covering ``seg[row]``) pair."""
    n = cov_ptr[seg + 1] - cov_ptr[seg]
    ends = cp.cumsum(n)
    total = int(ends[-1]) if len(ends) else 0
    k = cp.arange(total, dtype=cp.int64)
    row = cp.searchsorted(ends, k, side='right')
    return row, cov_transcript[cov_ptr[seg[row]] + k - (ends - n)[row]]


def _count_fragment_gene_overlaps(
    frag_starts: cp.ndarray,
    frag_ends: cp.ndarray,
    cell_indices: cp.ndarray,
    gene_starts: cp.ndarray,
    gene_ends: cp.ndarray,
    gene_indices: cp.ndarray,
    chunk_size: int = 5_000_000,
) -> cudf.DataFrame:
    """
    Paired-insertion counts per (cell, transcript domain) on one chromosome.

    The domain starts and ends cut the chromosome into segments, each covered
    by a fixed set of domains, and every domain covers a contiguous run of
    them. An insertion (the fragment start, and its end - 1) finds its segment
    with one binary search; the fragments are counted per cell and segment,
    and each count goes to the domains covering the segment. A fragment counts
    once for a domain that holds both its insertions: the end insertion's
    segment adds only the domains that miss the start's. Fragments are taken
    ``chunk_size`` at a time.

    Returns aggregated (cell_idx, gene_idx, count) DataFrame.
    """
    empty = cudf.DataFrame({'cell_idx': cp.array([], dtype=cp.int32),
                            'gene_idx': cp.array([], dtype=cp.int32),
                            'count': cp.array([], dtype=cp.int32)})
    if len(frag_starts) == 0 or len(gene_starts) == 0:
        return empty

    # Segments [bounds[i], bounds[i + 1]); domain t covers segments
    # first[t] .. last[t] - 1, listed per segment in cov_transcript
    gene_starts = cp.asarray(gene_starts, dtype=cp.int64)
    gene_ends = cp.asarray(gene_ends, dtype=cp.int64)
    bounds = cp.unique(cp.concatenate([gene_starts, gene_ends]))
    n_seg = len(bounds) - 1
    first = cp.searchsorted(bounds, gene_starts)
    last = cp.searchsorted(bounds, gene_ends)
    span = last - first
    span_end = cp.cumsum(span)
    k = cp.arange(int(span_end[-1]), dtype=cp.int64)
    transcript = cp.searchsorted(span_end, k, side='right')
    seg = first[transcript] + k - (span_end - span)[transcript]
    order = cp.argsort(seg)
    cov_transcript = transcript[order]
    cov_ptr = cp.searchsorted(seg[order], cp.arange(n_seg + 1))
    del transcript, seg, order, k

    def segment_of(pos):
        s = cp.searchsorted(bounds, pos, side='right') - 1
        return cp.where((s >= 0) & (s < n_seg), s, -1)

    gene_indices = cp.asarray(gene_indices)
    counts = []
    for lo in range(0, len(frag_starts), chunk_size):
        hi = min(lo + chunk_size, len(frag_starts))
        cells = cp.asarray(cell_indices[lo:hi], dtype=cp.int64)
        s1 = segment_of(cp.asarray(frag_starts[lo:hi], dtype=cp.int64))
        s2 = segment_of(cp.asarray(frag_ends[lo:hi], dtype=cp.int64) - 1)

        # The domains holding the start insertion
        hit = s1 >= 0
        key, n = cp.unique(cells[hit] * n_seg + s1[hit], return_counts=True)
        row, t = _expand_segments(key % n_seg, cov_ptr, cov_transcript)
        cell_parts, t_parts, n_parts = [(key // n_seg)[row]], [t], [n[row]]

        # The domains holding the end insertion and not the start one
        hit = (s2 >= 0) & (s2 != s1)
        key, n = cp.unique((cells[hit] * (n_seg + 1) + s1[hit] + 1) * n_seg + s2[hit],
                           return_counts=True)
        row, t = _expand_segments(key % n_seg, cov_ptr, cov_transcript)
        start_seg = ((key // n_seg) % (n_seg + 1) - 1)[row]
        keep = (first[t] > start_seg) | (last[t] <= start_seg)
        cell_parts.append((key // n_seg // (n_seg + 1))[row][keep])
        t_parts.append(t[keep])
        n_parts.append(n[row][keep])
        del cells, s1, s2, hit, key, n, row, t, start_seg, keep

        counts.append(cudf.DataFrame({
            'cell_idx': cp.concatenate(cell_parts).astype(cp.int32),
            'gene_idx': gene_indices[cp.concatenate(t_parts)].astype(cp.int32),
            'count': cp.concatenate(n_parts).astype(cp.int32),
        }).groupby(['cell_idx', 'gene_idx']).sum().reset_index())

    if len(counts) == 1:
        return counts[0]
    return cudf.concat(counts, ignore_index=True).groupby(['cell_idx', 'gene_idx']).sum().reset_index()


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
    # Gene names, or transcript IDs, as SnapATAC2 names its features
    var.index = var.pop('feature').values

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
    metrics: Optional[str | Path | pd.DataFrame | cudf.DataFrame] = None,
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

    Uses paired-insertion counting strategy matching SnapATAC2
    :cite:p:`Zhang2024`: each fragment 
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
        Path for output .h5ad file. If None, the matrix is only returned.
    id_type : str
        "gene" or "transcript" - which feature type to use (default: "gene").
        As in SnapATAC2, genes are named and counted by gene name: genes
        sharing a name are one column, counting a cell's fragments in the
        most accessible of their transcripts. Transcripts are named by
        transcript ID.
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
    metrics : str, Path, pandas.DataFrame, cudf.DataFrame or polars.DataFrame, optional
        Cell metrics for filtering: the table returned by
        :func:`gatac.pp.compute_metrics`, or a CSV file of it.
    filter_query : str, optional
        Query string for filtering cells based on metrics (e.g. "tsse_score > 5").
    barcode_prefix : str, optional
        Prefix to add to barcodes
    low_memory : bool
        Use low memory mode for Parquet reading (default: False)
    cell_batch_size : int
        Unused; kept for backwards compatibility (all cells are processed
        together, one chromosome at a time).
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

    Examples
    --------
    >>> import gatac as ga
    >>> adata_gene = ga.pp.make_gene_matrix(
    ...     "pbmc_filtered.parquet",
    ...     gene_anno="GRCh38.gtf.gz",
    ...     id_type="gene",
    ...     upstream=2000,
    ...     downstream=0,
    ...     include_gene_body=True,
    ... )
    """

    from ..io.fragments import read_fragments_parquet
    import scanpy as sc

    input_parquet = Path(input_parquet)
    gene_anno = Path(gene_anno)
    
    if output_path is not None:
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
    cell_metadata_input = read_metrics(metrics, gpu=True)

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

    total_time = time.perf_counter() - start_time
    if output_path is not None:
        adata.write_h5ad(str(output_path))
        total_time = time.perf_counter() - start_time
        logger.info(f"Created {output_path.name}: {adata.shape[0]:,} cells × {adata.shape[1]:,} genes ({total_time:.1f}s)")
    else:
        logger.info(f"Gene matrix: {adata.shape[0]:,} cells × {adata.shape[1]:,} genes ({total_time:.1f}s)")

    return adata
