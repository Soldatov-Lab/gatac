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
    cell_metadata: Optional[cudf.DataFrame] = None,
    filter_query: Optional[str] = None,
    return_sparse: bool = True,
    count_strategy: str = "unique"
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
    cell_metadata : cudf.DataFrame, optional
        Optional cell metadata for filtering (e.g. from quality metrics).
        If provided, fragment counting for initial filtering is skipped.
    filter_query : str, optional
        Additional query string for filtering cells based on cell_metadata.
    return_sparse : bool
        Return sparse matrix (True) or dense array (False)
    count_strategy : str
        Strategy for counting fragments in tiles. Options:
        - "unique": Count each unique fragment once (SnapATAC2 default)
        - "count": Use PCR duplicate counts from the 'count' column
        - "binarize": Convert counts to binary (0/1) per tile
        (default: "unique")

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

    # Get valid chromosomes (those in chrom_sizes, minus excluded ones)
    all_chroms = sorted(chrom_sizes.keys())
    if exclude_chroms is not None:
        if isinstance(exclude_chroms, str):
            exclude_chroms = [exclude_chroms]
        valid_chroms_for_counting = [c for c in all_chroms if c not in exclude_chroms]
    else:
        valid_chroms_for_counting = all_chroms
    
    # Filter fragments to valid chromosomes BEFORE counting
    # This matches SnapATAC2 behavior: only fragments on chromosomes in chrom_sizes are counted
    fragments_for_counting = fragments_df[fragments_df['chrom'].isin(valid_chroms_for_counting)]

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
        # 1. Apply user query if provided
        if filter_query:
            cell_metadata = cell_metadata.query(filter_query)
        
        # 2. Apply fragment count threshold
        # If n_unique is present in metadata, use it.
        # Otherwise, calculate it from fragments for the barcodes in metadata.
        if 'n_unique' in cell_metadata.columns:
            logger.debug(f"Applying threshold {min_fragments_per_cell} to n_unique in metadata")
            cell_metadata = cell_metadata[cell_metadata['n_unique'] >= min_fragments_per_cell]
        else:
            logger.debug(f"n_unique not found in metadata. Calculating for threshold {min_fragments_per_cell}")
            # Calculate counts only for barcodes currently in cell_metadata, using valid chromosomes
            subset_frags = fragments_for_counting[fragments_for_counting['barcode'].isin(cell_metadata['barcode'])]
            barcode_counts = subset_frags.groupby('barcode', observed=True).agg({
                'count': ['sum', 'size']
            })
            barcode_counts.columns = ['n_total', 'n_unique']
            barcode_counts = barcode_counts.reset_index()
            
            # Filter by threshold
            valid_bc = barcode_counts[barcode_counts['n_unique'] >= min_fragments_per_cell]['barcode']
            cell_metadata = cell_metadata[cell_metadata['barcode'].isin(valid_bc)]
            
            # Optionally merge n_unique back if you want it in the final obs
            cell_metadata = cell_metadata.merge(barcode_counts[['barcode', 'n_unique']], on='barcode', how='left')

        valid_barcodes = cell_metadata['barcode']

    # Now use all fragments for actual matrix construction (will be filtered by included_chroms below)
    fragments_df = fragments_df[fragments_df['barcode'].isin(valid_barcodes)]
    logger.debug(f"Retained {len(valid_barcodes)} cells")

    if exclude_chroms is not None:
        if isinstance(exclude_chroms, str):
            exclude_chroms = [exclude_chroms]
        fragments_df = fragments_df[~fragments_df['chrom'].isin(exclude_chroms)]

    logger.debug("Creating genomic tiles")
    # Use chrom_sizes to determine which chromosomes to include and ensuring consistent order
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
        tile_ends = cp.minimum(tile_starts + tile_size, size)

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

    # Match SnapATAC2 logic:
    # Each fragment has two insertions: (start + 4) and (end - 5).
    # If both fall in the same tile, the tile gets +1 (internal binarization per fragment).
    # If they fall in different tiles, each tile gets +1.
    fragments_df['tile_s'] = fragments_df['start'] // tile_size
    fragments_df['tile_e'] = (fragments_df['end'].astype(cp.int32) - 1) // tile_size
    # Ensure non-negative
    fragments_df['tile_e'] = fragments_df['tile_e'].clip(lower=0)

    chrom_offset_map = cudf.DataFrame({
        'chrom': list(chrom_to_offset.keys()),
        'offset': list(chrom_to_offset.values())
    })
    fragments_df = fragments_df.merge(chrom_offset_map, on='chrom', how='left')

    # Insertion 1 (start)
    df_ins1 = fragments_df[['cell_idx', 'tile_s', 'offset', 'count']].rename(
        columns={'tile_s': 'tile_idx'}
    )
    
    # Insertion 2 (end), only if it falls in a different tile
    df_ins2 = fragments_df[fragments_df['tile_s'] != fragments_df['tile_e']][
        ['cell_idx', 'tile_e', 'offset', 'count']
    ].rename(columns={'tile_e': 'tile_idx'})
    
    insertions = cudf.concat([df_ins1, df_ins2])
    insertions['global_tile_idx'] = insertions['tile_idx'] + insertions['offset']

    # Apply counting strategy
    if count_strategy not in ["unique", "count", "binarize"]:
        raise ValueError(f"Invalid count_strategy: {count_strategy}. Must be 'unique', 'count', or 'binarize'")
    
    if count_strategy in ["unique", "binarize"]:
        # For unique and binarize, treat each fragment as 1 initially
        insertions['count'] = 1

    logger.debug("Building sparse matrix")
    try:
        # Aggregate counts per tile
        matrix_data = insertions.groupby(['cell_idx', 'global_tile_idx'])['count'].sum().reset_index()
        
        # Apply binarization if requested
        if count_strategy == "binarize":
            matrix_data['count'] = (matrix_data['count'] > 0).astype(cp.uint8)

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
        cell_metadata,
        on='barcode',
        how='left'
    )
    cell_metadata = cell_metadata.sort_values('cell_idx').reset_index(drop=True)
    
    # Convert barcodes to strings only at the end for AnnData compatibility
    # This is done on the small cell_metadata DataFrame (unique barcodes only)
    if cell_metadata['barcode'].dtype != 'object':
        cell_metadata['barcode'] = cell_metadata['barcode'].astype(str)

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
    # Barcodes were converted to strings at end of create_tile_matrix_gpu
    obs.index = obs['barcode'].values

    var = tile_metadata.to_pandas()
    var.index = (var['chrom'].astype(str) + ':' +
                 var['start'].astype(str) + '-' +
                 var['end'].astype(str))

    adata = sc.AnnData(X=matrix_cpu, obs=obs, var=var)
    logger.debug(f"Created AnnData: {adata.shape[0]} cells × {adata.shape[1]} tiles")

    return adata
