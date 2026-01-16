"""
Fragment processing pipeline for generating tile matrices.
"""

import logging
import time
import gc
from pathlib import Path
from typing import Optional, Tuple, List

import cudf
import cupy as cp

from .tile import create_tile_matrix_gpu, tile_matrix_to_anndata

logger = logging.getLogger(__name__)

# Predefined dtypes to minimize memory footprint
FRAGMENT_DTYPES = {
    'chrom': 'category',
    'start': 'uint32',
    'end': 'uint32',
    'barcode': 'category',
    'count': 'uint16'
}

def make_tile_matrix(
    input_parquet: str | Path,
    chrom_sizes: dict[str, int] | str,
    output_path: Optional[str | Path] = None,
    tile_size: int = 5000,
    min_fragments_per_cell: int = 100,
    exclude_chroms: Optional[list] = ["chrM", "chrY", "M", "Y"],
    binarize: bool = False,
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
        List of chromosomes to exclude. (default: ["chrM", "chrY", "M", "Y"])
    binarize : bool
        Convert counts to binary (default: False)
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

    logger.info(f"Processing {input_parquet.name}")

    def _cleanup_memory():
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()

    def _read_and_process(use_low_mem: bool, exclude: Optional[List[str]] = None):
        with cudf.option_context("io.parquet.low_memory", use_low_mem):
            # Predefining columns slows down read if not careful, but ensure dtypes
            # cudf's read_parquet doesn't take dtype dict, so we cast immediately
            df = cudf.read_parquet(str(input_parquet), columns=list(FRAGMENT_DTYPES.keys()))
            
            # Ensure dtypes match our expectation to save memory
            for col, dtype in FRAGMENT_DTYPES.items():
                if df[col].dtype != dtype:
                    df[col] = df[col].astype(dtype)
            
            df_sorted = df.sort_values('barcode')
            del df
            _cleanup_memory()

            matrix, cell_metadata, tile_metadata = create_tile_matrix_gpu(
                fragments_df=df_sorted,
                chrom_sizes=chrom_sizes,
                tile_size=tile_size,
                exclude_chroms=exclude,
                min_fragments_per_cell=min_fragments_per_cell,
                return_sparse=True
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
