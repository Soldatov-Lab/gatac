"""
Fragment processing pipeline for generating tile matrices.
"""

import logging
import time
from pathlib import Path
from typing import Optional, Tuple

import cudf

from .tile import create_tile_matrix_gpu, tile_matrix_to_anndata

logger = logging.getLogger(__name__)


def make_tile_matrix(
    input_parquet: str | Path,
    output_path: Optional[str | Path] = None,
    tile_size: int = 5000,
    min_fragments_per_cell: int = 100,
    chromosomes: Optional[list] = None,
    binarize: bool = False,
) -> 'sc.AnnData':
    """
    Process ATAC fragments parquet file and generate tile matrix.

    Parameters
    ----------
    input_parquet : str or Path
        Path to input parquet file containing ATAC fragments
    output_path : str or Path, optional
        Path for output .h5ad file. If None, uses input filename.
    tile_size : int
        Size of genomic bins in base pairs (default: 5000)
    min_fragments_per_cell : int
        Minimum fragments required per barcode (default: 100)
    chromosomes : list, optional
        List of chromosomes to include. If None, uses all.
    binarize : bool
        Convert counts to binary (default: False)

    Returns
    -------
    adata : AnnData
        AnnData object with tile matrix
    """
    input_parquet = Path(input_parquet)
    if output_path is None:
        output_path = input_parquet.with_suffix('').with_name(
            input_parquet.stem + '_tile_matrix.h5ad'
        )
    else:
        output_path = Path(output_path)

    logger.info(f"Processing {input_parquet.name}")

    # Read and sort
    start_time = time.perf_counter()
    logger.debug("Reading parquet file")
    df = cudf.read_parquet(str(input_parquet))
    read_time = time.perf_counter() - start_time
    logger.debug(f"Read {len(df):,} rows in {read_time:.2f}s")

    start_sort = time.perf_counter()
    logger.debug("Sorting by barcode")
    df_sorted = df.sort_values('barcode')
    sort_time = time.perf_counter() - start_sort
    logger.debug(f"Sorted in {sort_time:.2f}s")

    # Generate tile matrix
    start_tile = time.perf_counter()
    matrix, cell_metadata, tile_metadata = create_tile_matrix_gpu(
        fragments_df=df_sorted,
        tile_size=tile_size,
        chromosomes=chromosomes,
        min_fragments_per_cell=min_fragments_per_cell,
        return_sparse=True
    )
    tile_time = time.perf_counter() - start_tile
    logger.debug(f"Tile matrix in {tile_time:.2f}s")

    # Convert to AnnData
    adata = tile_matrix_to_anndata(matrix, cell_metadata, tile_metadata)

    # Save
    adata.write_h5ad(str(output_path))
    total_time = time.perf_counter() - start_time
    logger.info(f"Created {output_path.name}: {adata.shape[0]:,} cells × {adata.shape[1]:,} tiles ({total_time:.1f}s)")

    return adata
