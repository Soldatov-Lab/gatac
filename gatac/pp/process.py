"""
Fragment processing pipeline for generating tile and gene matrices.
"""

import logging
import time
import gc
from pathlib import Path
from typing import Optional, Tuple, List, Literal

import cudf
import cupy as cp

from .tile import create_tile_matrix_gpu, tile_matrix_to_anndata
from .gene import load_gene_annotation, create_gene_matrix_gpu, gene_matrix_to_anndata

logger = logging.getLogger(__name__)

# Predefined dtypes to minimize memory footprint
FRAGMENT_DTYPES = {
    'chrom': 'category',
    'start': 'uint32',
    'end': 'uint32',
    'barcode': 'category',
    'count': 'uint16'
}

def read_fragments_parquet(
    input_path: str | Path,
    low_memory: bool = True,
    columns: Optional[List[str]] = None,
) -> cudf.DataFrame:
    """
    Read ATAC fragments from Parquet file optimized for GPU memory.

    Note: Parquet files contain intrinsic schema metadata. This function 
    expects the file to contain columns matching FRAGMENT_DTYPES keys:
    ['chrom', 'start', 'end', 'barcode', 'count'].
    """
    if columns is None:
        columns = list(FRAGMENT_DTYPES.keys())
        
    with cudf.option_context("io.parquet.low_memory", low_memory):
        df = cudf.read_parquet(str(input_path), columns=columns)
        
        # Ensure dtypes match our expectation to save memory
        for col, dtype in FRAGMENT_DTYPES.items():
            if col in df.columns and df[col].dtype != dtype:
                df[col] = df[col].astype(dtype)
                
    return df

def make_tile_matrix(
    input_parquet: str | Path,
    chrom_sizes: dict[str, int] | str,
    output_path: Optional[str | Path] = None,
    tile_size: int = 5000,
    min_fragments_per_cell: int = 100,
    exclude_chroms: Optional[list] = None,
    metrics: Optional[str | Path | cudf.DataFrame] = None,
    filter_query: Optional[str] = None,
    count_strategy: str = "unique",
    barcode_prefix: Optional[str] = None,
    low_memory: bool = False,
) -> 'sc.AnnData':
    """
    Process ATAC fragments parquet file and generate tile matrix.

    Parameters
    ----------
    input_parquet : str or Path
        Path to input parquet file containing ATAC fragments
    chrom_sizes : dict or str
        Dictionary of chromosome names and their sizes, or a genome name (e.g., 'hg38').
    output_path : str or Path, optional
        Path for output .h5ad file. If None, uses input filename.
    tile_size : int
        Size of genomic bins in base pairs (default: 5000)
    min_fragments_per_cell : int
        Minimum fragments required per barcode (default: 100)
    exclude_chroms : list, optional
        List of chromosomes to exclude. (default: None)
    metrics : str, Path, or cudf.DataFrame, optional
        Path to a CSV file or a cuDF DataFrame containing cell metrics for filtering.
    filter_query : str, optional
        Query string for filtering cells based on metrics (e.g. "tsse_score > 5").
    count_strategy : str
        Strategy for counting fragments in tiles. Options:
        - "unique": Count each unique fragment once (SnapATAC2 default)
        - "count": Use PCR duplicate counts from the 'count' column
        - "binarize": Convert counts to binary (0/1) per tile
        (default: "unique")
    barcode_prefix : str, optional
        Prefix to add to barcodes
    low_memory : bool
        Use low memory mode for Parquet reading (default: False)

    Returns
    -------
    adata : AnnData
        AnnData object with tile matrix
    """
    from .genome import get_chrom_sizes
    
    if isinstance(chrom_sizes, str):
        chrom_sizes = get_chrom_sizes(chrom_sizes)

    input_parquet = Path(input_parquet)
    if output_path is None:
        output_path = input_parquet.with_suffix('').with_name(
            input_parquet.stem + '_tile_matrix.h5ad'
        )
    else:
        output_path = Path(output_path)

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

        matrix, cell_metadata, tile_metadata = create_tile_matrix_gpu(
            fragments_df=df_sorted,
            chrom_sizes=chrom_sizes,
            tile_size=tile_size,
            exclude_chroms=exclude,
            min_fragments_per_cell=min_fragments_per_cell,
            cell_metadata=cell_metadata_input,
            filter_query=filter_query,
            return_sparse=True,
            count_strategy=count_strategy
        )
        return matrix, cell_metadata, tile_metadata

    start_time = time.perf_counter()
    try:
        matrix, cell_metadata, tile_metadata = _read_and_process(low_memory, exclude_chroms)
    except (MemoryError, RuntimeError) as e:
        err_msg = str(e).lower()
        is_oom = "out of memory" in err_msg or "std::bad_alloc" in err_msg or "cudaerrormemoryallocation" in err_msg
        
        if is_oom:
            if not low_memory:
                logger.warning(f"CUDA Out of Memory. Retrying with low_memory=True: {e}")
                _cleanup_memory()
                matrix, cell_metadata, tile_metadata = _read_and_process(True, exclude_chroms)
            else:
                logger.error(f"CUDA Out of Memory even with low_memory=True: {e}")
                raise e
        else:
            raise e

    # Convert to AnnData
    adata = tile_matrix_to_anndata(matrix, cell_metadata, tile_metadata)
    
    if barcode_prefix:
        adata.obs_names = [f"{barcode_prefix}{b}" for b in adata.obs_names]

    # Save
    adata.write_h5ad(str(output_path))
    total_time = time.perf_counter() - start_time
    logger.info(f"Created {output_path.name}: {adata.shape[0]:,} cells × {adata.shape[1]:,} tiles ({total_time:.1f}s)")

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
