"""
GPU-accelerated tile matrix generation from ATAC fragment data.

Memory strategy: process one chromosome at a time on the GPU, accumulate
COO triplets on the host, then assemble the final CSR matrix with SciPy.
Peak GPU footprint is therefore ~1/25 of the naïve approach (one chrom at a
time), making small tile sizes (e.g. 500 bp) tractable without CUDA OOM.
"""

import logging
import time
import gc
from pathlib import Path
from typing import Optional, Tuple, List, Union

import cudf
import cupy as cp
import cupyx.scipy.sparse as cusp
import numpy as np
import scipy.sparse as sp

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _process_chrom_on_gpu(
    chrom_frags: cudf.DataFrame,
    tile_size: int,
    count_strategy: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build COO components for one chromosome on the GPU and return them as
    plain NumPy arrays so the GPU scratch memory can be freed immediately.

    Parameters
    ----------
    chrom_frags : cudf.DataFrame
        Fragment subset for one chromosome.
        Required columns: 'cell_idx', 'start', 'end', 'count'.
    tile_size : int
        Bin size in base pairs.
    count_strategy : str
        One of "unique", "count", "binarize".

    Returns
    -------
    rows : np.ndarray[int32]   – cell indices
    cols : np.ndarray[int32]   – local tile indices (0-based within chrom)
    data : np.ndarray          – aggregated counts
    """
    frags = chrom_frags[['cell_idx', 'start', 'end', 'count']].copy()

    frags['tile_s'] = (frags['start'] // tile_size).astype('int32')
    frags['tile_e'] = ((frags['end'].astype('int32') - 1) // tile_size).clip(lower=0)

    # Insertion 1: always at start-tile
    df_ins1 = frags[['cell_idx', 'tile_s', 'count']].rename(columns={'tile_s': 'tile_idx'})

    # Insertion 2: only when fragment spans two tiles
    cross = frags['tile_s'] != frags['tile_e']
    df_ins2 = frags.loc[cross, ['cell_idx', 'tile_e', 'count']].rename(
        columns={'tile_e': 'tile_idx'}
    )

    insertions = cudf.concat([df_ins1, df_ins2], ignore_index=True)
    del df_ins1, df_ins2, frags

    if count_strategy in ('unique', 'binarize'):
        insertions['count'] = cp.ones(len(insertions), dtype=cp.uint8)

    # Aggregate per (cell, tile) on GPU
    agg = (
        insertions
        .groupby(['cell_idx', 'tile_idx'], sort=False)['count']
        .sum()
        .reset_index()
    )
    del insertions

    if count_strategy == 'binarize':
        agg['count'] = (agg['count'] > 0).astype(cp.uint8)

    # Pull to host and release GPU scratch
    rows = agg['cell_idx'].values.get().astype(np.int32)
    cols = agg['tile_idx'].values.get().astype(np.int32)
    data = agg['count'].values.get()
    del agg
    cp.get_default_memory_pool().free_all_blocks()

    return rows, cols, data


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_tile_matrix_gpu(
    fragments_df: cudf.DataFrame,
    chrom_sizes: dict[str, int],
    tile_size: int = 5000,
    exclude_chroms: Optional[list] = ["chrM", "chrY", "M", "Y"],
    min_fragments_per_cell: int = 100,
    cell_metadata: Optional[cudf.DataFrame] = None,
    filter_query: Optional[str] = None,
    return_sparse: bool = True,
    count_strategy: str = "unique",
) -> Tuple[Union[sp.csr_matrix, np.ndarray], cudf.DataFrame, cudf.DataFrame]:
    """
    Generate a tile matrix from ATAC fragment data using GPU acceleration.

    Chromosomes are processed one at a time to minimise peak GPU memory.
    COO components are accumulated on the host and the final CSR matrix is
    assembled with SciPy, so even very small tile sizes (e.g. 500 bp) do
    not cause CUDA OOM errors.

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
        Return sparse matrix (True) or dense ndarray (False). Dense output is
        only practical for small matrices.
    count_strategy : str
        Strategy for counting fragments in tiles. Options:
        - "unique": Count each unique fragment once (SnapATAC2 default)
        - "count": Use PCR duplicate counts from the 'count' column
        - "binarize": Convert counts to binary (0/1) per tile
        (default: "unique")

    Returns
    -------
    matrix : scipy.sparse.csr_matrix or numpy.ndarray
        Tile matrix with shape (n_cells, n_tiles)
    cell_metadata : cudf.DataFrame
        Metadata for cells (barcodes) with total fragment counts
    tile_metadata : cudf.DataFrame
        Metadata for tiles with chromosome, start, end positions
    """
    if hasattr(chrom_sizes, 'chrom_sizes'):
        chrom_sizes = chrom_sizes.chrom_sizes

    if count_strategy not in ('unique', 'count', 'binarize'):
        raise ValueError(
            f"Invalid count_strategy: '{count_strategy}'. "
            "Must be 'unique', 'count', or 'binarize'."
        )

    # ------------------------------------------------------------------ #
    # 1. Determine valid chromosomes                                       #
    # ------------------------------------------------------------------ #
    all_chroms = sorted(chrom_sizes.keys())
    if exclude_chroms is not None:
        if isinstance(exclude_chroms, str):
            exclude_chroms = [exclude_chroms]
        included_chroms = [c for c in all_chroms if c not in exclude_chroms]
    else:
        included_chroms = all_chroms

    # Fragments on included chromosomes only (matches SnapATAC2 counting)
    fragments_for_counting = fragments_df[fragments_df['chrom'].isin(included_chroms)]

    # ------------------------------------------------------------------ #
    # 2. Cell filtering                                                    #
    # ------------------------------------------------------------------ #
    if cell_metadata is None:
        logger.debug("Filtering cells by unique fragment count")
        barcode_counts = fragments_for_counting.groupby('barcode', observed=True).agg(
            {'count': ['sum', 'size']}
        )
        barcode_counts.columns = ['n_total', 'n_unique']
        barcode_counts = barcode_counts.reset_index()
        cell_metadata = barcode_counts[
            barcode_counts['n_unique'] >= min_fragments_per_cell
        ]
    else:
        logger.debug("Using provided cell metadata for filtering")
        if filter_query:
            cell_metadata = cell_metadata.query(filter_query)

        if 'n_unique' in cell_metadata.columns:
            cell_metadata = cell_metadata[
                cell_metadata['n_unique'] >= min_fragments_per_cell
            ]
        else:
            subset_frags = fragments_for_counting[
                fragments_for_counting['barcode'].isin(cell_metadata['barcode'])
            ]
            barcode_counts = subset_frags.groupby('barcode', observed=True).agg(
                {'count': ['sum', 'size']}
            )
            barcode_counts.columns = ['n_total', 'n_unique']
            barcode_counts = barcode_counts.reset_index()

            valid_bc = barcode_counts[
                barcode_counts['n_unique'] >= min_fragments_per_cell
            ]['barcode']
            cell_metadata = cell_metadata[cell_metadata['barcode'].isin(valid_bc)]
            cell_metadata = cell_metadata.merge(
                barcode_counts[['barcode', 'n_unique']], on='barcode', how='left'
            )

    valid_barcodes = cell_metadata['barcode']
    logger.debug(f"Retained {len(valid_barcodes)} cells")

    # ------------------------------------------------------------------ #
    # 3. Restrict fragments to valid cells and included chromosomes        #
    # ------------------------------------------------------------------ #
    fragments_df = fragments_df[
        fragments_df['barcode'].isin(valid_barcodes) &
        fragments_df['chrom'].isin(included_chroms)
    ]

    # ------------------------------------------------------------------ #
    # 4. Build barcode → integer index mapping                            #
    # ------------------------------------------------------------------ #
    unique_barcodes = fragments_df['barcode'].unique().reset_index(drop=True)
    barcode_to_idx = cudf.DataFrame({
        'barcode': unique_barcodes,
        'cell_idx': cp.arange(len(unique_barcodes), dtype=cp.int32),
    })
    n_cells = int(len(unique_barcodes))

    fragments_df = fragments_df.merge(barcode_to_idx, on='barcode', how='left')

    # ------------------------------------------------------------------ #
    # 5. Build tile metadata (lightweight, CPU-friendly via NumPy)        #
    # ------------------------------------------------------------------ #
    logger.debug("Creating genomic tiles")
    tiles_list = []
    chrom_tile_offset: dict[str, int] = {}
    offset = 0
    for chrom in included_chroms:
        size = chrom_sizes[chrom]
        n_tiles = (size + tile_size - 1) // tile_size
        starts = np.arange(0, n_tiles * tile_size, tile_size, dtype=np.uint32)
        ends = np.minimum(starts + tile_size, size).astype(np.uint32)
        tiles_list.append(
            cudf.DataFrame({
                'chrom': chrom,
                'start': cudf.Series(starts),
                'end': cudf.Series(ends),
            })
        )
        chrom_tile_offset[chrom] = offset
        offset += n_tiles

    tile_metadata = cudf.concat(tiles_list, ignore_index=True)
    tile_metadata['tile_id'] = cp.arange(len(tile_metadata), dtype=cp.int32)
    n_tiles_total = int(len(tile_metadata))
    del tiles_list
    logger.debug(
        f"Created {n_tiles_total:,} tiles across {len(included_chroms)} chromosomes"
    )

    # ------------------------------------------------------------------ #
    # 6. Per-chromosome streaming: GPU compute → host accumulation        #
    # ------------------------------------------------------------------ #
    logger.debug("Building sparse matrix chromosome by chromosome")
    host_rows: list[np.ndarray] = []
    host_cols: list[np.ndarray] = []
    host_data: list[np.ndarray] = []

    for chrom in included_chroms:
        chrom_frags = fragments_df[fragments_df['chrom'] == chrom]
        if len(chrom_frags) == 0:
            del chrom_frags
            continue

        rows, cols, data = _process_chrom_on_gpu(chrom_frags, tile_size, count_strategy)
        del chrom_frags

        if len(rows) == 0:
            continue

        col_offset = chrom_tile_offset[chrom]
        host_rows.append(rows)
        host_cols.append(cols + col_offset)
        host_data.append(data)
        logger.debug(f"  {chrom}: {len(rows):,} non-zero entries")

    # ------------------------------------------------------------------ #
    # 7. Assemble final sparse matrix on the host                         #
    # ------------------------------------------------------------------ #
    if host_rows:
        all_rows = np.concatenate(host_rows).astype(np.int32)
        all_cols = np.concatenate(host_cols).astype(np.int32)
        all_data = np.concatenate(host_data).astype(np.float32)
        del host_rows, host_cols, host_data
    else:
        all_rows = np.empty(0, dtype=np.int32)
        all_cols = np.empty(0, dtype=np.int32)
        all_data = np.empty(0, dtype=np.float32)

    coo = sp.coo_matrix(
        (all_data, (all_rows, all_cols)),
        shape=(n_cells, n_tiles_total),
        dtype=np.float32,
    )
    del all_rows, all_cols, all_data

    density = coo.nnz / (n_cells * n_tiles_total) * 100 if n_cells * n_tiles_total > 0 else 0
    logger.debug(
        f"Matrix: {n_cells:,} cells × {n_tiles_total:,} tiles, "
        f"density: {density:.4f}%"
    )

    if return_sparse:
        matrix = coo.tocsr()
    else:
        matrix = coo.toarray()
    del coo

    # ------------------------------------------------------------------ #
    # 8. Finalise cell metadata                                           #
    # ------------------------------------------------------------------ #
    cell_metadata = barcode_to_idx.merge(cell_metadata, on='barcode', how='left')
    cell_metadata = cell_metadata.sort_values('cell_idx').reset_index(drop=True)

    if cell_metadata['barcode'].dtype != 'object':
        cell_metadata['barcode'] = cell_metadata['barcode'].astype(str)

    return matrix, cell_metadata, tile_metadata


def tile_matrix_to_anndata(
    matrix: Union[sp.csr_matrix, cusp.csr_matrix, np.ndarray],
    cell_metadata: cudf.DataFrame,
    tile_metadata: cudf.DataFrame,
):
    """
    Convert a tile matrix to AnnData.

    Accepts scipy CSR, cupyx CSR (legacy GPU path), or a dense ndarray.

    Parameters
    ----------
    matrix : scipy.sparse.csr_matrix | cupyx.scipy.sparse.csr_matrix | ndarray
        Tile matrix with shape (n_cells, n_tiles)
    cell_metadata : cudf.DataFrame
        Cell metadata with barcodes and statistics
    tile_metadata : cudf.DataFrame
        Tile metadata with genomic coordinates

    Returns
    -------
    adata : AnnData
    """
    import scanpy as sc

    logger.debug("Converting matrix to AnnData")

    # Normalise to a scipy CSR matrix on the host
    if isinstance(matrix, cusp.csr_matrix):
        # Legacy GPU path: pull data to host
        matrix_cpu = sp.csr_matrix(
            (
                matrix.data.get().astype(np.uint16),
                matrix.indices.get(),
                matrix.indptr.get(),
            ),
            shape=matrix.shape,
        )
    elif sp.issparse(matrix):
        matrix_cpu = matrix.astype(np.uint16)
        if not isinstance(matrix_cpu, sp.csr_matrix):
            matrix_cpu = matrix_cpu.tocsr()
    else:
        # Dense numpy or cupy array
        if hasattr(matrix, 'get'):
            matrix = matrix.get()
        matrix_cpu = sp.csr_matrix(matrix.astype(np.uint16))

    obs = cell_metadata.to_pandas()
    obs.index = obs['barcode'].values

    var = tile_metadata.to_pandas()
    var.index = (
        var['chrom'].astype(str) + ':' +
        var['start'].astype(str) + '-' +
        var['end'].astype(str)
    )

    adata = sc.AnnData(X=matrix_cpu, obs=obs, var=var)
    logger.debug(f"Created AnnData: {adata.shape[0]:,} cells × {adata.shape[1]:,} tiles")

    return adata


def make_tile_matrix(
    input_parquet: str | Path,
    chrom_sizes: dict[str, int] | str,
    output_path: Optional[str | Path] = None,
    tile_size: int = 5000,
    min_fragments_per_cell: int = 100,
    exclude_chroms: Optional[list] = ["chrM", "chrY", "M", "Y"],
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
    from .process import read_fragments_parquet
    import scanpy as sc
    
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

    def _read_and_process(use_low_mem: bool):
        df = read_fragments_parquet(input_parquet, low_memory=use_low_mem)
        df_sorted = df.sort_values('barcode')
        del df
        _cleanup_memory()

        return create_tile_matrix_gpu(
            fragments_df=df_sorted,
            chrom_sizes=chrom_sizes,
            tile_size=tile_size,
            exclude_chroms=exclude_chroms,
            min_fragments_per_cell=min_fragments_per_cell,
            cell_metadata=cell_metadata_input,
            filter_query=filter_query,
            return_sparse=True,
            count_strategy=count_strategy,
        )

    start_time = time.perf_counter()

    # Chromosome-by-chromosome streaming already avoids most OOM conditions.
    # The fallback to low_memory Parquet reading handles very large files where
    # even loading the raw fragments DataFrame exhausts GPU memory.
    try:
        matrix, cell_metadata, tile_metadata = _read_and_process(low_memory)
    except (MemoryError, RuntimeError) as e:
        err_msg = str(e).lower()
        is_oom = (
            "out of memory" in err_msg
            or "std::bad_alloc" in err_msg
            or "cudaerrormemoryallocation" in err_msg
        )
        if is_oom and not low_memory:
            logger.warning(
                f"CUDA Out of Memory while loading fragments. "
                f"Retrying with low_memory=True: {e}"
            )
            _cleanup_memory()
            matrix, cell_metadata, tile_metadata = _read_and_process(True)
        else:
            raise

    # Convert to AnnData
    adata = tile_matrix_to_anndata(matrix, cell_metadata, tile_metadata)
    
    if barcode_prefix:
        adata.obs_names = [f"{barcode_prefix}{b}" for b in adata.obs_names]

    # Save
    adata.write_h5ad(str(output_path))
    total_time = time.perf_counter() - start_time
    logger.info(f"Created {output_path.name}: {adata.shape[0]:,} cells × {adata.shape[1]:,} tiles ({total_time:.1f}s)")

    return adata
