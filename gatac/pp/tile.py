"""
GPU-accelerated tile matrix generation from ATAC fragment data.

Memory strategy: stream Parquet row-groups in batches so the full
fragment file never needs to reside in GPU memory.  Within each batch,
tile insertions are computed for *all* chromosomes at once (one groupby
instead of ~25), then COO triplets are accumulated on the host.  The
final CSR matrix is assembled with SciPy.
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
import pandas as pd
import pyarrow.parquet as pq
import scipy.sparse as sp

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _process_batch_on_gpu(
    batch_df: cudf.DataFrame,
    tile_size: int,
    chrom_offset_df: cudf.DataFrame,
    count_strategy: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorised tile-index computation and aggregation for a fragment
    batch spanning multiple chromosomes.

    Parameters
    ----------
    batch_df : cudf.DataFrame
        Must contain columns: ``cell_idx`` (int32), ``chrom``, ``start``,
        ``end``, ``count``.
    tile_size : int
        Bin size in base pairs.
    chrom_offset_df : cudf.DataFrame
        Two columns: ``chrom``, ``tile_offset`` (int32).
    count_strategy : str
        ``"unique"`` or ``"count"``.  The caller handles ``"binarize"``
        by passing ``"unique"`` here and binarising the final matrix.

    Returns
    -------
    rows, cols, data : np.ndarray
        COO components on the host.
    """
    # Merge chrom → tile_offset (also filters to known chroms)
    batch_df = batch_df.merge(chrom_offset_df, on='chrom', how='inner')

    # Extract CuPy arrays — avoids cuDF arithmetic overhead
    cell_idx = batch_df['cell_idx'].values
    start = batch_df['start'].values
    end = batch_df['end'].values.astype(cp.int32)
    tile_offset = batch_df['tile_offset'].values
    n = len(cell_idx)

    if count_strategy == 'unique':
        count = cp.ones(n, dtype=cp.uint8)
    else:  # 'count'
        count = batch_df['count'].values

    del batch_df

    # Global tile indices for the two insertion sites
    tile_s = (tile_offset + start // tile_size).astype(cp.int32)
    tile_e = (tile_offset + cp.maximum((end - 1) // tile_size, 0)).astype(cp.int32)
    del start, end, tile_offset

    # Fragments spanning two tiles contribute an insertion at each tile
    cross_idx = cp.where(tile_s != tile_e)[0]
    n_cross = len(cross_idx)

    # Build COO arrays in one allocation
    total = n + n_cross
    all_rows = cp.empty(total, dtype=cp.int32)
    all_cols = cp.empty(total, dtype=cp.int32)
    all_data = cp.empty(total, dtype=count.dtype)

    all_rows[:n] = cell_idx
    all_cols[:n] = tile_s
    all_data[:n] = count

    if n_cross > 0:
        all_rows[n:] = cell_idx[cross_idx]
        all_cols[n:] = tile_e[cross_idx]
        all_data[n:] = count[cross_idx]

    del cell_idx, tile_s, tile_e, count, cross_idx

    # Aggregate per (cell, tile) on GPU
    insertions = cudf.DataFrame({
        'cell_idx': cudf.Series(all_rows),
        'tile_idx': cudf.Series(all_cols),
        'count': cudf.Series(all_data),
    })
    del all_rows, all_cols, all_data

    agg = (
        insertions
        .groupby(['cell_idx', 'tile_idx'], sort=False)['count']
        .sum()
        .reset_index()
    )
    del insertions

    # Pull to host
    rows = agg['cell_idx'].values.get().astype(np.int32)
    cols = agg['tile_idx'].values.get().astype(np.int32)
    data = agg['count'].values.get()
    del agg
    cp.get_default_memory_pool().free_all_blocks()

    return rows, cols, data


def _build_tile_metadata_and_offsets(
    included_chroms: List[str],
    chrom_sizes: dict,
    tile_size: int,
) -> Tuple[pd.DataFrame, cudf.DataFrame, int]:
    """Return (tile_metadata_pd, chrom_offset_df_gpu, n_tiles_total).

    tile_metadata is built directly as pandas to avoid an expensive
    cudf→pandas conversion of the 6M-row var table later.
    """
    offsets: dict[str, int] = {}
    chrom_list: list[np.ndarray] = []  # chrom name repeated per tile
    start_list: list[np.ndarray] = []
    end_list: list[np.ndarray] = []
    offset = 0
    for chrom in included_chroms:
        size = chrom_sizes[chrom]
        n_t = (size + tile_size - 1) // tile_size
        offsets[chrom] = offset
        starts = np.arange(0, n_t * tile_size, tile_size, dtype=np.uint32)
        ends = np.minimum(starts + tile_size, size).astype(np.uint32)
        chrom_list.append(np.full(n_t, chrom, dtype=object))
        start_list.append(starts)
        end_list.append(ends)
        offset += n_t

    all_chroms_arr = np.concatenate(chrom_list)
    all_starts = np.concatenate(start_list)
    all_ends = np.concatenate(end_list)

    tile_metadata = pd.DataFrame({
        'chrom': all_chroms_arr,
        'start': all_starts,
        'end': all_ends,
    })
    # Pre-build the var index (avoids doing it later in tile_matrix_to_anndata)
    tile_metadata.index = pd.Index(
        all_chroms_arr.astype(str)
        + ':'
        + all_starts.astype(str)
        + '-'
        + all_ends.astype(str)
    )

    chrom_offset_df = cudf.DataFrame({
        'chrom': list(offsets.keys()),
        'tile_offset': np.array(list(offsets.values()), dtype=np.int32),
    })
    return tile_metadata, chrom_offset_df, offset


def _cleanup_gpu():
    """Free GPU memory pools."""
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()


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

    # For tile content: only included_chroms (excludes chrM, chrY, etc.)
    fragments_for_counting = fragments_df[fragments_df['chrom'].isin(included_chroms)]
    # For n_unique counting: all chroms in chrom_sizes (including chrM/chrY),
    # matching SnapATAC2's min_num_fragments threshold behaviour.
    fragments_for_n_unique = fragments_df[fragments_df['chrom'].isin(all_chroms)]

    # ------------------------------------------------------------------ #
    # 2. Cell filtering                                                    #
    # ------------------------------------------------------------------ #
    if cell_metadata is None:
        logger.debug("Filtering cells by unique fragment count")
        barcode_counts = fragments_for_n_unique.groupby('barcode', observed=True).agg(
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

        # Always recompute n_unique from fragments restricted to included_chroms
        # (metrics n_unique may include non-standard contigs, causing too many
        # cells to pass min_fragments_per_cell compared to filter_fragments).
        # Cast to string to avoid cudf categorical isin mismatches when the two
        # series have different category sets.
        metadata_barcodes = cell_metadata['barcode'].astype('str')
        subset_frags = fragments_for_n_unique[
            fragments_for_n_unique['barcode'].astype('str').isin(metadata_barcodes)
        ]
        barcode_counts = subset_frags.groupby('barcode', observed=True).agg(
            {'count': ['sum', 'size']}
        )
        barcode_counts.columns = ['n_total', 'n_unique']
        barcode_counts = barcode_counts.reset_index()

        valid_bc = barcode_counts[
            barcode_counts['n_unique'] >= min_fragments_per_cell
        ]['barcode'].astype('str')
        cell_metadata = cell_metadata[cell_metadata['barcode'].astype('str').isin(valid_bc)]
        if 'n_unique' in cell_metadata.columns:
            cell_metadata = cell_metadata.drop(columns=['n_unique'])
        # barcode_counts still has the original categorical barcode; merge on str
        barcode_counts['barcode'] = barcode_counts['barcode'].astype('str')
        cell_metadata['barcode'] = cell_metadata['barcode'].astype('str')
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
    # 5. Build tile metadata & chrom-offset lookup                        #
    # ------------------------------------------------------------------ #
    logger.debug("Creating genomic tiles")
    tile_metadata, chrom_offset_df, n_tiles_total = (
        _build_tile_metadata_and_offsets(included_chroms, chrom_sizes, tile_size)
    )
    logger.debug(
        f"Created {n_tiles_total:,} tiles across {len(included_chroms)} chromosomes"
    )

    # ------------------------------------------------------------------ #
    # 6. Vectorised processing: single groupby across all chromosomes     #
    # ------------------------------------------------------------------ #
    logger.debug("Building sparse matrix (vectorised)")
    effective_strategy = 'unique' if count_strategy == 'binarize' else count_strategy
    rows, cols, data = _process_batch_on_gpu(
        fragments_df[['cell_idx', 'chrom', 'start', 'end', 'count']],
        tile_size, chrom_offset_df, effective_strategy,
    )

    # ------------------------------------------------------------------ #
    # 7. Assemble final sparse matrix (GPU COO → CSR for speed)           #
    # ------------------------------------------------------------------ #
    if len(rows) > 0:
        try:
            coo_gpu = cusp.coo_matrix(
                (
                    cp.asarray(data.astype(np.float32)),
                    (cp.asarray(rows), cp.asarray(cols)),
                ),
                shape=(n_cells, n_tiles_total),
            )
            csr_gpu = coo_gpu.tocsr()
            del coo_gpu

            if count_strategy == 'binarize':
                csr_gpu.data[:] = 1.0

            if return_sparse:
                matrix = sp.csr_matrix(
                    (
                        csr_gpu.data.get().astype(np.uint16),
                        csr_gpu.indices.get(),
                        csr_gpu.indptr.get(),
                    ),
                    shape=csr_gpu.shape,
                )
            else:
                matrix = cusp.coo_matrix(csr_gpu).toarray().get()
            del csr_gpu
            _cleanup_gpu()
        except (MemoryError, RuntimeError):
            _cleanup_gpu()
            coo = sp.coo_matrix(
                (data.astype(np.float32), (rows, cols)),
                shape=(n_cells, n_tiles_total),
                dtype=np.float32,
            )
            if count_strategy == 'binarize':
                coo.data[:] = 1.0
            matrix = coo.tocsr() if return_sparse else coo.toarray()
            del coo
    else:
        matrix = (
            sp.csr_matrix((n_cells, n_tiles_total), dtype=np.float32)
            if return_sparse
            else np.zeros((n_cells, n_tiles_total), dtype=np.float32)
        )
    del rows, cols, data

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
        if matrix.dtype != np.uint16:
            matrix_cpu = matrix.astype(np.uint16)
        else:
            matrix_cpu = matrix
        if not isinstance(matrix_cpu, sp.csr_matrix):
            matrix_cpu = matrix_cpu.tocsr()
    else:
        # Dense numpy or cupy array
        if hasattr(matrix, 'get'):
            matrix = matrix.get()
        matrix_cpu = sp.csr_matrix(matrix.astype(np.uint16))

    obs = cell_metadata.to_pandas() if hasattr(cell_metadata, 'to_pandas') else cell_metadata.copy()
    obs.index = obs['barcode'].values

    var = tile_metadata.to_pandas() if hasattr(tile_metadata, 'to_pandas') else tile_metadata.copy()
    if var.index.dtype == object and ':' in str(var.index[0]):
        pass  # index already set (e.g. from _build_tile_metadata_and_offsets)
    else:
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
    row_groups_per_batch: int = 64,
) -> 'sc.AnnData':
    """
    Process ATAC fragments parquet file and generate tile matrix.

    Streams parquet row-groups in batches so the full file never needs to
    reside in GPU memory.  Within each batch, tiles are computed for all
    chromosomes at once (one groupby instead of ~25).

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
        Use smaller batch size for Parquet reading (default: False)
    row_groups_per_batch : int
        Number of Parquet row-groups to read per GPU batch (default: 64).
        Decrease for lower GPU memory usage.

    Returns
    -------
    adata : AnnData
        AnnData object with tile matrix
    """
    from .genome import get_chrom_sizes
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

    start_time = time.perf_counter()
    logger.info(f"Processing {input_parquet.name}")

    if low_memory:
        row_groups_per_batch = min(row_groups_per_batch, 8)

    # ------------------------------------------------------------------ #
    # 1. Chromosome setup                                                  #
    # ------------------------------------------------------------------ #
    all_chroms = sorted(chrom_sizes.keys())
    if exclude_chroms:
        if isinstance(exclude_chroms, str):
            exclude_chroms = [exclude_chroms]
        included_chroms = [c for c in all_chroms if c not in exclude_chroms]
    else:
        included_chroms = all_chroms

    all_chroms_set = set(all_chroms)
    included_chroms_set = set(included_chroms)

    # ------------------------------------------------------------------ #
    # 2. Tile metadata & chrom-offset lookup                               #
    # ------------------------------------------------------------------ #
    tile_metadata, chrom_offset_df, n_tiles_total = (
        _build_tile_metadata_and_offsets(included_chroms, chrom_sizes, tile_size)
    )
    logger.debug(
        f"Created {n_tiles_total:,} tiles across {len(included_chroms)} chromosomes"
    )

    # ------------------------------------------------------------------ #
    # 3. Parquet metadata for streaming                                    #
    # ------------------------------------------------------------------ #
    meta = pq.read_metadata(str(input_parquet))
    n_row_groups = meta.num_row_groups

    def _rg_batches():
        for i in range(0, n_row_groups, row_groups_per_batch):
            yield list(range(i, min(i + row_groups_per_batch, n_row_groups)))

    # ------------------------------------------------------------------ #
    # 4. Count barcodes with DuckDB (fast CPU scan, no GPU memory)         #
    # ------------------------------------------------------------------ #
    import duckdb
    chrom_values = ", ".join(f"'{c}'" for c in all_chroms)
    con = duckdb.connect()

    if metrics is not None:
        if isinstance(metrics, cudf.DataFrame):
            cell_metadata_input = metrics
        else:
            metrics_path = Path(metrics)
            if metrics_path.exists():
                logger.info(f"Loading cell metrics from {metrics_path}")
                cell_metadata_input = cudf.read_csv(str(metrics_path))
            else:
                logger.warning(f"Metrics file {metrics_path} not found.")
                cell_metadata_input = None

        if cell_metadata_input is not None:
            if filter_query:
                cell_metadata_input = cell_metadata_input.query(filter_query)
            cell_meta_extra = cell_metadata_input.to_pandas()
            cell_meta_extra['barcode'] = cell_meta_extra['barcode'].astype(str)
            candidate_barcodes = cell_meta_extra['barcode'].tolist()
            del cell_metadata_input

            # Register candidate barcodes for DuckDB join
            candidate_df = pd.DataFrame({'barcode': candidate_barcodes})
            con.register('candidate_bc', candidate_df)
            barcode_counts = con.execute(f"""
                SELECT f.barcode, COUNT(*) as n_unique
                FROM read_parquet('{str(input_parquet)}') f
                INNER JOIN candidate_bc c ON f.barcode = c.barcode
                WHERE f.chrom IN ({chrom_values})
                GROUP BY f.barcode
                HAVING COUNT(*) >= {int(min_fragments_per_cell)}
            """).fetchdf()
        else:
            cell_meta_extra = None
            barcode_counts = con.execute(f"""
                SELECT barcode, COUNT(*) as n_unique
                FROM read_parquet('{str(input_parquet)}')
                WHERE chrom IN ({chrom_values})
                GROUP BY barcode
                HAVING COUNT(*) >= {int(min_fragments_per_cell)}
            """).fetchdf()
    else:
        cell_meta_extra = None
        barcode_counts = con.execute(f"""
            SELECT barcode, COUNT(*) as n_unique
            FROM read_parquet('{str(input_parquet)}')
            WHERE chrom IN ({chrom_values})
            GROUP BY barcode
            HAVING COUNT(*) >= {int(min_fragments_per_cell)}
        """).fetchdf()

    con.close()

    valid_barcodes = sorted(barcode_counts['barcode'].values)
    n_unique_map = dict(zip(barcode_counts['barcode'], barcode_counts['n_unique']))
    n_cells = len(valid_barcodes)
    if n_cells == 0:
        raise ValueError("No cells passed the min_fragments_per_cell filter.")
    logger.debug(f"Retained {n_cells:,} cells")

    # GPU lookup: barcode → cell_idx
    barcode_idx_df = cudf.DataFrame({
        'barcode': valid_barcodes,
        'cell_idx': np.arange(n_cells, dtype=np.int32),
    })

    # ------------------------------------------------------------------ #
    # 5. Stream row groups: build tile COO                                 #
    # ------------------------------------------------------------------ #
    effective_strategy = (
        'unique' if count_strategy == 'binarize' else count_strategy
    )

    host_rows: list[np.ndarray] = []
    host_cols: list[np.ndarray] = []
    host_data: list[np.ndarray] = []

    frag_columns = ['chrom', 'start', 'end', 'barcode', 'count']

    for rg in _rg_batches():
        batch = cudf.read_parquet(
            str(input_parquet), row_groups=rg, columns=frag_columns,
        )

        # Inner merge: keeps only valid barcodes AND adds cell_idx
        batch = batch.merge(barcode_idx_df, on='barcode', how='inner')
        if len(batch) == 0:
            del batch
            _cleanup_gpu()
            continue

        rows, cols, data = _process_batch_on_gpu(
            batch, tile_size, chrom_offset_df, effective_strategy,
        )
        del batch
        _cleanup_gpu()

        if len(rows) > 0:
            host_rows.append(rows)
            host_cols.append(cols)
            host_data.append(data)

    # ------------------------------------------------------------------ #
    # 6. Assemble sparse matrix (GPU COO → CSR for speed)                  #
    # ------------------------------------------------------------------ #
    if host_rows:
        all_rows = np.concatenate(host_rows)
        all_cols = np.concatenate(host_cols)
        all_data = np.concatenate(host_data).astype(np.float32)
        del host_rows, host_cols, host_data

        # Build COO → CSR on GPU (avoids expensive CPU sort)
        try:
            coo_gpu = cusp.coo_matrix(
                (
                    cp.asarray(all_data),
                    (cp.asarray(all_rows), cp.asarray(all_cols)),
                ),
                shape=(n_cells, n_tiles_total),
            )
            csr_gpu = coo_gpu.tocsr()
            del coo_gpu
            matrix = sp.csr_matrix(
                (
                    csr_gpu.data.get().astype(np.uint16),
                    csr_gpu.indices.get(),
                    csr_gpu.indptr.get(),
                ),
                shape=csr_gpu.shape,
            )
            del csr_gpu
            _cleanup_gpu()
        except (MemoryError, RuntimeError):
            warning_msg = (
                "GPU OOM during sparse matrix assembly. "
                "Falling back to CPU assembly (this may be slow)."
            )
            logger.warning(warning_msg)
            # Fall back to CPU if GPU OOM during assembly
            _cleanup_gpu()
            matrix = sp.coo_matrix(
                (all_data, (all_rows, all_cols)),
                shape=(n_cells, n_tiles_total),
                dtype=np.float32,
            ).tocsr()
        del all_rows, all_cols, all_data
    else:
        matrix = sp.csr_matrix((n_cells, n_tiles_total), dtype=np.float32)

    if count_strategy == 'binarize':
        matrix.data[:] = 1.0

    # ------------------------------------------------------------------ #
    # 7. Build cell metadata & AnnData                                     #
    # ------------------------------------------------------------------ #
    cell_meta_pd = pd.DataFrame({
        'barcode': valid_barcodes,
        'n_unique': [n_unique_map[bc] for bc in valid_barcodes],
    })

    if cell_meta_extra is not None:
        # Merge extra columns from provided metrics
        extra_cols = [
            c for c in cell_meta_extra.columns
            if c not in ('barcode', 'n_unique')
        ]
        if extra_cols:
            cell_meta_pd = cell_meta_pd.merge(
                cell_meta_extra[['barcode'] + extra_cols],
                on='barcode', how='left',
            )

    adata = tile_matrix_to_anndata(matrix, cell_meta_pd, tile_metadata)

    if barcode_prefix:
        adata.obs_names = [f"{barcode_prefix}{b}" for b in adata.obs_names]

    # Save
    adata.write_h5ad(str(output_path))
    total_time = time.perf_counter() - start_time
    logger.info(
        f"Created {output_path.name}: {adata.shape[0]:,} cells × "
        f"{adata.shape[1]:,} tiles ({total_time:.1f}s)"
    )

    return adata
