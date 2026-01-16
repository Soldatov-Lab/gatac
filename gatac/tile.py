"""
GPU-accelerated tile matrix generation from ATAC fragment data.
"""

import logging
from typing import Optional, Tuple

import cudf
import cupy as cp
import cupyx.scipy.sparse as cusp
import numpy as np

logger = logging.getLogger(__name__)


def create_tile_matrix_gpu(
    fragments_df: cudf.DataFrame,
    chrom_sizes: dict[str, int],
    tile_size: int = 5000,
    exclude_chroms: Optional[list] = ["chrM", "chrY", "M", "Y"],
    min_fragments_per_cell: int = 100,
    return_sparse: bool = True
) -> Tuple[cusp.csr_matrix, cudf.DataFrame, cudf.DataFrame]:
    """
    Generate a tile matrix from ATAC fragment data using GPU acceleration.

    Parameters
    ----------
    fragments_df : cudf.DataFrame
        Fragment data with columns: 'chrom', 'start', 'end', 'barcode', 'count'
    chrom_sizes : dict
        Dictionary of chromosome names and their sizes. Used to ensure consistent
        tile coordinates across different samples.
    tile_size : int
        Size of genomic bins in base pairs (default: 5000)
    exclude_chroms : list, optional
        List of chromosomes to exclude. (default: ["chrM", "chrY", "M", "Y"])
    min_fragments_per_cell : int
        Minimum fragments required per barcode to include (default: 100)
    return_sparse : bool
        Return sparse matrix (True) or dense array (False)

    Returns
    -------
    matrix : cupyx.scipy.sparse.csr_matrix or cupy.ndarray
        Tile matrix with shape (n_cells, n_tiles)
    cell_metadata : cudf.DataFrame
        Metadata for cells (barcodes) with total fragment counts
    tile_metadata : cudf.DataFrame
        Metadata for tiles with chromosome, start, end positions
    """
    if hasattr(chrom_sizes, 'chrom_sizes'):
        chrom_sizes = chrom_sizes.chrom_sizes

    logger.debug("Filtering cells by fragment count")
    barcode_counts = fragments_df.groupby('barcode')['count'].sum().reset_index()
    barcode_counts.columns = ['barcode', 'total_fragments']

    valid_barcodes = barcode_counts[
        barcode_counts['total_fragments'] >= min_fragments_per_cell
    ]['barcode']
    fragments_df = fragments_df[fragments_df['barcode'].isin(valid_barcodes)]
    logger.debug(f"Retained {len(valid_barcodes)} cells with >= {min_fragments_per_cell} fragments")

    if exclude_chroms is not None:
        if isinstance(exclude_chroms, str):
            exclude_chroms = [exclude_chroms]
        fragments_df = fragments_df[~fragments_df['chrom'].isin(exclude_chroms)]

    logger.debug("Creating genomic tiles")
    # Use chrom_sizes to determine which chromosomes to include and ensuring consistent order
    all_chroms = sorted(chrom_sizes.keys())
    if exclude_chroms is not None:
        included_chroms = [c for c in all_chroms if c not in exclude_chroms]
    else:
        included_chroms = all_chroms
    
    # Also ensure fragments only contain chromosomes we have sizes for
    fragments_df = fragments_df[fragments_df['chrom'].isin(included_chroms)]

    tiles_list = []
    chrom_to_offset = {}
    offset = 0
    for chrom in included_chroms:
        size = chrom_sizes[chrom]
        n_tiles = (size + tile_size - 1) // tile_size
        tile_starts = cp.arange(0, n_tiles * tile_size, tile_size)
        tile_ends = tile_starts + tile_size

        chrom_tiles = cudf.DataFrame({
            'chrom': chrom,
            'start': cudf.Series(tile_starts),
            'end': cudf.Series(tile_ends)
        })
        tiles_list.append(chrom_tiles)
        
        chrom_to_offset[chrom] = offset
        offset += n_tiles

    tile_metadata = cudf.concat(tiles_list, ignore_index=True)
    tile_metadata['tile_id'] = cp.arange(len(tile_metadata))
    logger.debug(f"Created {len(tile_metadata)} tiles across {len(included_chroms)} chromosomes")

    logger.debug("Assigning fragments to tiles")
    unique_barcodes = fragments_df['barcode'].unique().reset_index(drop=True)
    barcode_to_idx = cudf.DataFrame({
        'barcode': unique_barcodes,
        'cell_idx': cp.arange(len(unique_barcodes))
    })

    fragments_df = fragments_df.merge(barcode_to_idx, on='barcode', how='left')
    fragments_df['fragment_mid'] = (fragments_df['start'] + fragments_df['end']) // 2
    fragments_df['tile_idx'] = fragments_df['fragment_mid'] // tile_size

    chrom_offset_map = cudf.DataFrame({
        'chrom': list(chrom_to_offset.keys()),
        'offset': list(chrom_to_offset.values())
    })
    fragments_df = fragments_df.merge(chrom_offset_map, on='chrom', how='left')
    fragments_df['global_tile_idx'] = fragments_df['tile_idx'] + fragments_df['offset']

    logger.debug("Building sparse matrix")
    try:
        matrix_data = fragments_df.groupby(['cell_idx', 'global_tile_idx'])['count'].sum().reset_index()

        row_indices = matrix_data['cell_idx'].values
        col_indices = matrix_data['global_tile_idx'].values
        data = matrix_data['count'].values

        n_cells = len(unique_barcodes)
        n_tiles = len(tile_metadata)

        coo_matrix = cusp.coo_matrix(
            (data, (row_indices, col_indices)),
            shape=(n_cells, n_tiles),
            dtype=cp.float32
        )

        logger.debug(f"Matrix: {n_cells} cells × {n_tiles} tiles, density: {100 * coo_matrix.nnz / (n_cells * n_tiles):.4f}%")

        if return_sparse:
            matrix = coo_matrix.tocsr()
        else:
            matrix = coo_matrix.toarray()
            
    except (cp.cuda.memory.OutOfMemoryError, MemoryError) as e:
        logger.error(f"CUDA Out of Memory during matrix construction: {e}")
        # Re-raise as a generic error that process.py can catch and retry
        raise RuntimeError(f"CUDA Out of Memory: {e}") from e

    cell_metadata = barcode_to_idx.merge(
        barcode_counts[barcode_counts['barcode'].isin(valid_barcodes)],
        on='barcode',
        how='left'
    )
    cell_metadata = cell_metadata.sort_values('cell_idx').reset_index(drop=True)

    return matrix, cell_metadata, tile_metadata


def tile_matrix_to_anndata(
    matrix: cusp.csr_matrix,
    cell_metadata: cudf.DataFrame,
    tile_metadata: cudf.DataFrame,
):
    """
    Convert GPU tile matrix to AnnData object.

    Parameters
    ----------
    matrix : cupyx.scipy.sparse.csr_matrix
        Tile matrix from create_tile_matrix_gpu
    cell_metadata : cudf.DataFrame
        Cell metadata with barcodes and statistics
    tile_metadata : cudf.DataFrame
        Tile metadata with genomic coordinates

    Returns
    -------
    adata : AnnData
        AnnData object with tile matrix
    """
    import scanpy as sc
    import scipy.sparse as sp

    logger.debug("Converting GPU matrix to CPU")
    matrix_cpu = sp.csr_matrix(
        (matrix.data.get().astype(np.uint16), matrix.indices.get(), matrix.indptr.get()),
        shape=matrix.shape
    )

    obs = cell_metadata.to_pandas()
    obs.index = obs['barcode'].values

    var = tile_metadata.to_pandas()
    var.index = (var['chrom'].astype(str) + ':' +
                 var['start'].astype(str) + '-' +
                 var['end'].astype(str))

    adata = sc.AnnData(X=matrix_cpu, obs=obs, var=var)
    logger.debug(f"Created AnnData: {adata.shape[0]} cells × {adata.shape[1]} tiles")

    return adata
