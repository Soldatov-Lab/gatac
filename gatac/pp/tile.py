"""
GPU-accelerated tile matrix generation from ATAC fragment data.

Memory strategy: stream Parquet row-groups in batches so the full
fragment file never needs to reside in GPU memory.  Within each batch,
tile insertions are computed for *all* chromosomes at once and summed per
(cell, tile); the sums stay on the GPU, where the CSR matrix is built
once at the end and copied to the host.
"""
from __future__ import annotations

import functools
import logging
import time
import gc
from pathlib import Path
from typing import Optional, Tuple, List, Union

import anndata as ad
import cudf
import cupy as cp
import cupyx.scipy.sparse as cusp
import numpy as np
import pandas as pd
import scipy.sparse as sp

from .._readers import read_metrics
from ..io.fragments import FragmentFile
from ._utils import async_memory_pool, cleanup_gpu_memory

logger = logging.getLogger(__name__)

_INTERVAL_FEATURE_PATTERN = r"^(?P<chrom>[^:;]+)[:;](?P<start>\d+)-(?P<end>\d+)$"


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
    count = None if count_strategy == 'unique' else batch_df['count'].values
    rows, cols, data = _tile_counts_gpu(
        batch_df['cell_idx'].values,
        batch_df['tile_offset'].values,
        batch_df['start'].values,
        batch_df['end'].values,
        count,
        tile_size,
    )
    del batch_df
    return rows, cols, data


def _insertions_gpu(
    cell_idx: cp.ndarray,
    tile_offset: cp.ndarray,
    start: cp.ndarray,
    end: cp.ndarray,
    count: Optional[cp.ndarray],
    tile_size: int,
) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    """
    One (cell, tile, count) triplet per Tn5 insertion of a batch of fragments.

    ``tile_offset`` is the global index of the first tile of each fragment's
    chromosome. ``count`` None counts each fragment once (``"unique"``).
    A fragment whose two ends fall in different tiles gives one triplet per
    tile; one within a tile gives a single triplet.
    """
    end = end.astype(cp.int32)
    n = len(cell_idx)
    if count is None:
        count = cp.ones(n, dtype=cp.uint8)

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
    return all_rows, all_cols, all_data


def _tile_counts_gpu(
    cell_idx: cp.ndarray,
    tile_offset: cp.ndarray,
    start: cp.ndarray,
    end: cp.ndarray,
    count: Optional[cp.ndarray],
    tile_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Per (cell, tile) insertion counts for one batch of fragments, as COO
    components on the host. See :func:`_insertions_gpu` for the arguments.
    """
    all_rows, all_cols, all_data = _insertions_gpu(
        cell_idx, tile_offset, start, end, count, tile_size,
    )

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


def _tile_key_counts_gpu(
    cell_idx: cp.ndarray,
    tile_offset: cp.ndarray,
    start: cp.ndarray,
    end: cp.ndarray,
    count: Optional[cp.ndarray],
    tile_size: int,
    n_tiles: int,
) -> Tuple[cp.ndarray, cp.ndarray]:
    """
    Per (cell, tile) insertion counts for one batch of fragments, kept on the
    GPU as sorted int64 keys ``cell * n_tiles + tile`` and int32 counts.
    """
    rows, cols, data = _insertions_gpu(cell_idx, tile_offset, start, end, count, tile_size)
    key = rows.astype(cp.int64) * n_tiles + cols
    del rows, cols
    agg = (
        cudf.DataFrame({"key": key, "count": data.astype(cp.int32)})
        .groupby("key", sort=True)["count"]
        .sum()
    )
    return agg.index.values, agg.values.astype(cp.int32, copy=False)


def _iter_slabs(
    keys: List[cp.ndarray],
    counts: List[cp.ndarray],
    n_cells: int,
    n_tiles: int,
    slab_entries: int = 1 << 25,
):
    """
    Sum the per-batch counts of each (cell, tile) key, a slab of cells at a
    time, and yield ``(r0, r1, rows, cols, data)`` on the GPU: the summed
    entries of rows ``r0:r1``, sorted by row then column, ``rows`` absolute.

    A (cell, tile) pair can appear in several batches, hence the second sum.
    Each slab holds about ``slab_entries`` batch entries, a slice of every
    batch's sorted keys, so that the sum never holds all the batches'
    entries twice over.
    """
    total = sum(len(k) for k in keys)
    n_slabs = max(1, -(-total // slab_entries))
    bounds = np.linspace(0, n_cells, n_slabs + 1).astype(np.int64)
    cuts = [cp.searchsorted(k, cp.asarray(bounds * n_tiles)).get() for k in keys]
    for s in range(n_slabs):
        r0, r1 = int(bounds[s]), int(bounds[s + 1])
        key = cp.concatenate([k[c[s]:c[s + 1]] for k, c in zip(keys, cuts)])
        if len(key) == 0:
            continue
        agg = (
            cudf.DataFrame({
                "key": key,
                "count": cp.concatenate([v[c[s]:c[s + 1]] for v, c in zip(counts, cuts)]),
            })
            .groupby("key", sort=True)["count"]
            .sum()
        )
        del key
        key = agg.index.values
        rows = key // n_tiles
        cols = (key - rows * n_tiles).astype(cp.int32)
        data = agg.values.astype(cp.int32, copy=False)
        del agg, key
        yield r0, r1, rows, cols, data


def _csr_from_key_counts(
    keys: List[cp.ndarray],
    counts: List[cp.ndarray],
    n_cells: int,
    n_tiles: int,
) -> sp.csr_matrix:
    """
    The host CSR matrix of the per-batch (cell, tile) counts: uint16 data,
    sorted column indices, built on the GPU a slab of cells at a time.
    """
    data_parts, index_parts = [], []
    row_nnz = np.zeros(n_cells, dtype=np.int64)
    for r0, r1, rows, cols, data in _iter_slabs(keys, counts, n_cells, n_tiles):
        index_parts.append(cols.get())
        data_parts.append(data.astype(cp.uint16).get())
        row_nnz[r0:r1] = cp.bincount(rows - r0, minlength=r1 - r0).get()
    if not data_parts:
        return sp.csr_matrix((n_cells, n_tiles), dtype=np.uint16)
    nnz = int(row_nnz.sum())
    # one index dtype for both arrays, as scipy picks it
    idx_dtype = np.int32 if max(nnz, n_tiles) <= np.iinfo(np.int32).max else np.int64
    indptr = np.zeros(n_cells + 1, dtype=idx_dtype)
    np.cumsum(row_nnz, out=indptr[1:])
    return sp.csr_matrix(
        (np.concatenate(data_parts), np.concatenate(index_parts).astype(idx_dtype, copy=False), indptr),
        shape=(n_cells, n_tiles),
    )


@functools.lru_cache(maxsize=4)
def _cached_tile_metadata(
    included_chroms: Tuple[str, ...],
    sizes: Tuple[int, ...],
    tile_size: int,
) -> Tuple[pd.DataFrame, cudf.DataFrame, int]:
    """:func:`_build_tile_metadata_and_offsets`, built once per genome and tile size.

    Callers must not modify the returned tables.
    """
    return _build_tile_metadata_and_offsets(
        list(included_chroms), dict(zip(included_chroms, sizes)), tile_size,
    )


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


def _included_chroms(chrom_sizes: dict, exclude_chroms) -> Tuple[List[str], List[str]]:
    """All chromosomes of ``chrom_sizes``, sorted, and those not excluded."""
    all_chroms = sorted(chrom_sizes.keys())
    if not exclude_chroms:
        return all_chroms, all_chroms
    if isinstance(exclude_chroms, str):
        exclude_chroms = [exclude_chroms]
    return all_chroms, [c for c in all_chroms if c not in exclude_chroms]


def _tile_cells(
    fragments: FragmentFile,
    all_chroms: List[str],
    cell_meta_extra: Optional[pd.DataFrame],
    min_fragments_per_cell: int,
) -> pd.DataFrame:
    """
    The cells of one fragment file, sorted by barcode: those with at least
    ``min_fragments_per_cell`` unique fragments on ``all_chroms``, among the
    metrics' barcodes when given, with ``n_unique`` and the metrics' other
    columns.
    """
    barcode_counts = fragments.barcode_counts(chroms=all_chroms)[['barcode', 'n_unique']]
    if cell_meta_extra is not None:
        # Restrict to candidate barcodes
        barcode_counts = barcode_counts[
            barcode_counts['barcode'].isin(set(cell_meta_extra['barcode']))
        ]
    barcode_counts = barcode_counts[barcode_counts['n_unique'] >= int(min_fragments_per_cell)]
    if len(barcode_counts) == 0:
        raise ValueError(
            f"No cells passed the min_fragments_per_cell filter in {fragments.path.name}."
        )
    cells = barcode_counts.sort_values('barcode').reset_index(drop=True)
    if cell_meta_extra is not None:
        # Merge extra columns from provided metrics
        extra_cols = [c for c in cell_meta_extra.columns if c not in ('barcode', 'n_unique')]
        if extra_cols:
            cells = cells.merge(cell_meta_extra[['barcode'] + extra_cols], on='barcode', how='left')
    return cells


def _stream_key_counts(
    fragments: FragmentFile,
    cells: List[str],
    included_chroms: List[str],
    tile_offsets: cp.ndarray,
    n_tiles: int,
    tile_size: int,
    count_strategy: str,
    row_groups_per_batch: int,
) -> Tuple[List[cp.ndarray], List[cp.ndarray]]:
    """
    Per (cell, tile) counts of each batch of row groups, kept on the GPU,
    the cell being the index in ``cells``. See :func:`_tile_key_counts_gpu`.
    """
    effective_strategy = 'unique' if count_strategy == 'binarize' else count_strategy
    frag_columns = ['chrom', 'start', 'end']
    if effective_strategy == 'count':
        frag_columns.append('count')

    keys: list[cp.ndarray] = []
    counts: list[cp.ndarray] = []
    for batch in fragments.iter_batches(
        frag_columns, row_groups_per_batch, chroms=included_chroms, cells=cells,
    ):
        if len(batch) == 0:
            continue
        key, count = _tile_key_counts_gpu(
            batch['cell'].values,
            tile_offsets[batch['chrom'].values],
            batch['start'].values,
            batch['end'].values,
            batch['count'].values if effective_strategy == 'count' else None,
            tile_size,
            n_tiles,
        )
        del batch
        keys.append(key)
        counts.append(count)
    return keys, counts


def _cleanup_gpu():
    """Free GPU memory pools."""
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()


def _load_interval_adata(input_data: ad.AnnData | Path):
    """Load an interval matrix from AnnData or 10x HDF5."""
    import scanpy as sc

    if isinstance(input_data, ad.AnnData):
        return input_data.copy()

    input_path = Path(input_data)
    suffix = input_path.suffix.lower()
    if suffix == '.h5ad':
        return sc.read_h5ad(str(input_path))
    if suffix == '.h5':
        return sc.read_10x_h5(str(input_path), gex_only=False)

    raise ValueError(
        f"Unsupported interval-matrix format: '{input_path.suffix}'. "
        "Expected '.h5ad' or 10x '.h5'."
    )


def _extract_interval_features(var_names: pd.Index) -> pd.DataFrame:
    """Extract genomic intervals from feature names like ``chr1:100-200``."""
    feature_names = pd.Index(var_names).astype(str)
    parsed = feature_names.to_series(
        index=np.arange(len(feature_names)),
        name='feature_name',
    ).str.extract(_INTERVAL_FEATURE_PATTERN)

    parsed = parsed[parsed.notna().all(axis=1)].copy()
    if parsed.empty:
        raise ValueError(
            "No interval-like features found in var_names. "
            "Expected names like 'chr1:100-200'."
        )

    parsed['start'] = parsed['start'].astype(np.int64)
    parsed['end'] = parsed['end'].astype(np.int64)
    parsed = parsed[parsed['end'] > parsed['start']]
    if parsed.empty:
        raise ValueError(
            "No interval-like features with end > start were found in var_names."
        )

    parsed.insert(0, 'feature_idx', parsed.index.to_numpy(dtype=np.int32))
    parsed.insert(1, 'feature_name', feature_names[parsed['feature_idx']].to_numpy())
    return parsed.reset_index(drop=True)


def _load_metrics_dataframe(metrics, filter_query: Optional[str]) -> Optional[pd.DataFrame]:
    """Load cell metrics (see :func:`gatac._readers.read_metrics`) and optionally apply a pandas query."""
    metrics_df = read_metrics(metrics)
    if metrics_df is None:
        return None

    if 'barcode' not in metrics_df.columns:
        raise ValueError("Metrics table must contain a 'barcode' column.")

    metrics_df['barcode'] = metrics_df['barcode'].astype(str)
    if filter_query:
        try:
            metrics_df = metrics_df.query(filter_query)
        except Exception as exc:
            raise ValueError(f"Invalid filter query for metrics: {exc}") from exc

    return metrics_df


def _make_tile_matrix_from_interval_matrix(
    input_data: ad.AnnData | Path,
    chrom_sizes: dict[str, int],
    output_path: Optional[Path] = None,
    tile_size: int = 5000,
    exclude_chroms: Optional[list] = ["chrM", "chrY", "M", "Y"],
    min_fragments_per_cell: int = 100,
    metrics: Optional[str | Path | pd.DataFrame | cudf.DataFrame] = None,
    filter_query: Optional[str] = None,
    count_strategy: str = "unique",
    barcode_prefix: Optional[str] = None,
):
    """Aggregate interval-like matrix features into fixed genomic tiles."""
    adata = _load_interval_adata(input_data)
    interval_features = _extract_interval_features(adata.var_names)
    input_label = input_data.name if isinstance(input_data, Path) else "AnnData object"
    logger.info(
        f"Loaded {input_label}: found {len(interval_features):,} "
        "interval-like features in var_names"
    )

    all_chroms = sorted(chrom_sizes.keys())
    if exclude_chroms is not None:
        if isinstance(exclude_chroms, str):
            exclude_chroms = [exclude_chroms]
        included_chroms = [chrom for chrom in all_chroms if chrom not in exclude_chroms]
    else:
        included_chroms = all_chroms

    interval_features = interval_features[
        interval_features['chrom'].isin(included_chroms)
    ].reset_index(drop=True)
    if interval_features.empty:
        raise ValueError(
            "No interval-like features remained after chromosome filtering."
        )

    feature_indices = interval_features['feature_idx'].to_numpy(dtype=np.int32)
    interval_adata = adata[:, feature_indices]
    matrix = interval_adata.X
    if sp.issparse(matrix):
        matrix = matrix.tocsr()
    else:
        matrix = sp.csr_matrix(np.asarray(matrix))

    obs = interval_adata.obs.copy()
    obs.index = obs.index.astype(str)
    obs['barcode'] = obs.index
    obs['row_idx'] = np.arange(matrix.shape[0], dtype=np.int64)
    obs['n_interval_counts'] = np.asarray(matrix.sum(axis=1)).ravel()

    if 'n_unique' not in obs.columns:
        if 'n_fragment' in obs.columns:
            obs['n_unique'] = np.asarray(obs['n_fragment'])
        else:
            obs['n_unique'] = obs['n_interval_counts']

    metrics_df = _load_metrics_dataframe(metrics, filter_query)
    if metrics_df is not None:
        extra_cols = [column for column in metrics_df.columns if column != 'barcode']
        obs = obs.merge(
            metrics_df[['barcode'] + extra_cols],
            on='barcode',
            how='inner',
            suffixes=('', '_metrics'),
        )
        if obs.empty:
            raise ValueError("No cells remained after applying metrics filtering.")

    count_column = next(
        (
            column
            for column in ('n_unique_metrics', 'n_unique', 'n_fragment', 'n_interval_counts')
            if column in obs.columns
        ),
        None,
    )
    if count_column is None:
        raise ValueError("Unable to determine a per-cell count column for filtering.")

    obs = obs[obs[count_column] >= min_fragments_per_cell].copy()
    if obs.empty:
        raise ValueError("No cells passed the min_fragments_per_cell filter.")

    row_indices = obs.pop('row_idx').to_numpy(dtype=np.int64)
    matrix = matrix[row_indices]
    obs.reset_index(drop=True, inplace=True)

    if count_strategy in ('unique', 'count'):
        logger.info(
            "Interval-matrix input detected: preserving input matrix values "
            "and summing overlapping features per tile."
        )

    tile_metadata, chrom_offset_df, n_tiles_total = _build_tile_metadata_and_offsets(
        included_chroms,
        chrom_sizes,
        tile_size,
    )
    chrom_offsets = dict(
        zip(
            chrom_offset_df['chrom'].to_pandas(),
            chrom_offset_df['tile_offset'].to_pandas(),
        )
    )

    start = interval_features['start'].to_numpy(dtype=np.int64)
    end = interval_features['end'].to_numpy(dtype=np.int64)
    chrom = interval_features['chrom'].to_numpy(dtype=object)
    tile_start = np.array([chrom_offsets[c] for c in chrom], dtype=np.int32)
    tile_start += (start // tile_size).astype(np.int32)
    tile_end = np.array([chrom_offsets[c] for c in chrom], dtype=np.int32)
    tile_end += np.maximum((end - 1) // tile_size, 0).astype(np.int32)

    tiles_per_feature = tile_end - tile_start + 1
    map_rows = np.repeat(
        np.arange(len(interval_features), dtype=np.int32),
        tiles_per_feature,
    )
    map_cols = np.concatenate(
        [
            np.arange(start_idx, end_idx + 1, dtype=np.int32)
            for start_idx, end_idx in zip(tile_start, tile_end)
        ]
    )
    feature_to_tile = sp.csr_matrix(
        (
            np.ones(len(map_rows), dtype=np.uint8),
            (map_rows, map_cols),
        ),
        shape=(len(interval_features), n_tiles_total),
        dtype=np.uint8,
    )

    tile_matrix = matrix @ feature_to_tile
    if not isinstance(tile_matrix, sp.csr_matrix):
        tile_matrix = tile_matrix.tocsr()

    if count_strategy == 'binarize':
        tile_matrix.data[:] = 1

    adata = tile_matrix_to_anndata(tile_matrix, obs, tile_metadata)
    adata.uns['tile_input_kind'] = 'interval_matrix'
    adata.uns['tile_aggregation'] = 'sum_overlapping_features'

    if barcode_prefix:
        adata.obs_names = [f"{barcode_prefix}{barcode}" for barcode in adata.obs_names]

    if output_path is not None:
        adata.write_h5ad(str(output_path))
        logger.info(
            f"Created {output_path.name}: {adata.shape[0]:,} cells × "
            f"{adata.shape[1]:,} tiles from interval matrix"
        )
    else:
        logger.info(
            f"Created interval-matrix tile AnnData: {adata.shape[0]:,} cells × "
            f"{adata.shape[1]:,} tiles"
        )
    return adata


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
    input: str | Path | ad.AnnData,
    chrom_sizes: dict[str, int] | str,
    output_path: Optional[str | Path] = None,
    tile_size: int = 5000,
    min_fragments_per_cell: int = 100,
    exclude_chroms: Optional[list] = ["chrM", "chrY", "M", "Y"],
    metrics: Optional[str | Path | pd.DataFrame | cudf.DataFrame] = None,
    filter_query: Optional[str] = None,
    count_strategy: str = "unique",
    barcode_prefix: Optional[str] = None,
    low_memory: bool = False,
    row_groups_per_batch: int = 64,
) -> ad.AnnData:
    """
    Generate a tile matrix from fragments or interval-like feature matrices.

    For fragment parquet input, row-groups are streamed in batches so the full
    file never needs to reside in GPU memory. Within each batch, tiles are
    computed for all chromosomes at once.

    For `.h5ad` or 10x `.h5` input, features with interval-like names such as
    ``chr1:100-200`` in ``var_names`` are detected and aggregated into fixed
    tiles by overlap.

    Parameters
    ----------
    input : str, Path, or AnnData
        Fragment parquet path, interval-matrix `.h5ad` path, 10x `.h5` path,
        or an in-memory AnnData object containing interval-like features in
        `var_names`.
    chrom_sizes : dict or str
        Dictionary of chromosome names and their sizes, or a genome name (e.g., 'hg38').
    output_path : str or Path, optional
        Path for output .h5ad file. If None, the function returns the AnnData
        object without writing to disk.
    tile_size : int
        Size of genomic bins in base pairs (default: 5000)
    min_fragments_per_cell : int
        Minimum fragments required per barcode (default: 100)
    exclude_chroms : list, optional
        List of chromosomes to exclude. (default: ["chrM", "chrY", "M", "Y"])
    metrics : str, Path, pandas.DataFrame, cudf.DataFrame or polars.DataFrame, optional
        Cell metrics for filtering: the table returned by
        :func:`gatac.pp.compute_metrics`, or a CSV file of it.
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

    Examples
    --------
    >>> import gatac as ga
    >>> # From a GATAC parquet fragment file, using a built-in genome
    >>> adata = ga.pp.make_tile_matrix(
    ...     "pbmc_filtered.parquet",
    ...     chrom_sizes="hg38",
    ...     tile_size=500,
    ...     min_fragments_per_cell=200,
    ...     exclude_chroms=["chrM", "chrY"],
    ... )
    >>> # Or with a custom chromosome-sizes dict
    >>> adata = ga.pp.make_tile_matrix(
    ...     "pbmc_filtered.parquet",
    ...     chrom_sizes={"chr1": 248956422, "chr2": 242193529},
    ... )
    >>> # From a 10x .h5 (or interval-like .h5ad) input
    >>> adata = ga.pp.make_tile_matrix(
    ...     "filtered_peak_bc_matrix.h5",
    ...     chrom_sizes="hg38",
    ...     tile_size=500,
    ... )
    """

    from .genome import get_chrom_sizes
    if isinstance(chrom_sizes, str):
        chrom_sizes = get_chrom_sizes(chrom_sizes)

    with async_memory_pool():
        adata = _make_tile_matrix_one(
            input, output_path, chrom_sizes=chrom_sizes, tile_size=tile_size,
            min_fragments_per_cell=min_fragments_per_cell, exclude_chroms=exclude_chroms,
            metrics=metrics, filter_query=filter_query, count_strategy=count_strategy,
            barcode_prefix=barcode_prefix, low_memory=low_memory,
            row_groups_per_batch=row_groups_per_batch,
        )
    cleanup_gpu_memory()
    return adata


def _make_tile_matrix_one(
    input: str | Path | ad.AnnData,
    output_path: Optional[str | Path],
    chrom_sizes: dict[str, int],
    tile_size: int,
    min_fragments_per_cell: int,
    exclude_chroms: Optional[list],
    metrics,
    filter_query: Optional[str],
    count_strategy: str,
    barcode_prefix: Optional[str],
    low_memory: bool,
    row_groups_per_batch: int,
) -> ad.AnnData:
    """:func:`make_tile_matrix` of one input."""
    input_is_adata = isinstance(input, ad.AnnData)
    input_path = None if input_is_adata else Path(input)
    resolved_output_path = Path(output_path) if output_path is not None else None

    if input_is_adata or input_path.suffix.lower() in {'.h5ad', '.h5'}:
        return _make_tile_matrix_from_interval_matrix(
            input_data=input if input_is_adata else input_path,
            chrom_sizes=chrom_sizes,
            output_path=resolved_output_path,
            tile_size=tile_size,
            exclude_chroms=exclude_chroms,
            min_fragments_per_cell=min_fragments_per_cell,
            metrics=metrics,
            filter_query=filter_query,
            count_strategy=count_strategy,
            barcode_prefix=barcode_prefix,
        )

    start_time = time.perf_counter()
    logger.info(f"Processing {input_path.name}")

    if low_memory:
        row_groups_per_batch = min(row_groups_per_batch, 8)

    all_chroms, included_chroms = _included_chroms(chrom_sizes, exclude_chroms)
    tile_metadata, chrom_offset_df, n_tiles_total = _cached_tile_metadata(
        tuple(included_chroms), tuple(chrom_sizes[c] for c in included_chroms), tile_size,
    )
    fragments = FragmentFile(input_path)
    cell_meta_pd = _tile_cells(
        fragments, all_chroms, _load_metrics_dataframe(metrics, filter_query),
        min_fragments_per_cell,
    )
    n_cells = len(cell_meta_pd)
    logger.debug(f"Retained {n_cells:,} cells")

    # Per (cell, tile) counts of each batch, kept on the GPU
    keys, counts = _stream_key_counts(
        fragments, list(cell_meta_pd['barcode']), included_chroms,
        chrom_offset_df['tile_offset'].values, n_tiles_total, tile_size,
        count_strategy, row_groups_per_batch,
    )

    # The CSR matrix, built on the GPU and copied to the host once
    if keys:
        try:
            matrix = _csr_from_key_counts(keys, counts, n_cells, n_tiles_total)
        except (MemoryError, RuntimeError):
            # cupy's OutOfMemoryError and RMM's bad_alloc are MemoryErrors
            logger.warning(
                "GPU OOM during sparse matrix assembly. "
                "Falling back to CPU assembly (this may be slow)."
            )
            key = np.concatenate([k.get() for k in keys])
            count = np.concatenate([c.get() for c in counts])
            matrix = sp.coo_matrix(
                (count, (key // n_tiles_total, key % n_tiles_total)),
                shape=(n_cells, n_tiles_total),
            ).tocsr().astype(np.uint16)
            del key, count
        del keys, counts
    else:
        matrix = sp.csr_matrix((n_cells, n_tiles_total), dtype=np.uint16)

    if count_strategy == 'binarize':
        matrix.data[:] = 1

    adata = tile_matrix_to_anndata(matrix, cell_meta_pd, tile_metadata)

    if barcode_prefix:
        adata.obs_names = [f"{barcode_prefix}{b}" for b in adata.obs_names]

    # Save
    if resolved_output_path is not None:
        adata.write_h5ad(str(resolved_output_path))
        total_time = time.perf_counter() - start_time
        logger.info(
            f"Created {resolved_output_path.name}: {adata.shape[0]:,} cells × "
            f"{adata.shape[1]:,} tiles ({total_time:.1f}s)"
        )
    else:
        total_time = time.perf_counter() - start_time
        logger.info(
            f"Created tile AnnData in memory: {adata.shape[0]:,} cells × "
            f"{adata.shape[1]:,} tiles ({total_time:.1f}s)"
        )

    return adata
