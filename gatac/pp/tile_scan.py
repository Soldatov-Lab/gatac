"""
A combined tile matrix of several fragment files, without per-sample matrices.

:func:`scan_tiles` reads every fragment file once and returns the cells and
tiles of the combined matrix with per-tile statistics, but no matrix.
:func:`~gatac.pp.select_features` chooses tiles from those statistics, and
:func:`build_tile_matrix` reads the fragment files again to build the matrix
of the chosen tiles only. :func:`~gatac.pp.make_tile_matrix` runs all three
when given a list of files.

A tile matrix of one file takes about a second to compute on the GPU, much
less than writing it to disk and reading it back, so neither pass keeps
per-sample matrices.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List, Optional, Union

import anndata as ad
import cupy as cp
import numpy as np
import pandas as pd
import scipy.sparse as sp

from ..io.fragments import FragmentFile
from ._utils import async_memory_pool, cleanup_gpu_memory
from .tile import (
    _add_tile_stats,
    _cached_tile_metadata,
    _included_chroms,
    _iter_slabs,
    _load_metrics_dataframe,
    _set_tile_stats,
    _stream_key_counts,
    _tile_cells,
)

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]

def _make_unique(names: pd.Index) -> pd.Index:
    """Suffix repeated names ``-1``, ``-2``, ... in order."""
    seen: dict[str, int] = {}
    out = []
    for x in names:
        if x in seen:
            seen[x] += 1
            out.append(f"{x}-{seen[x]}")
        else:
            seen[x] = 0
            out.append(x)
    return pd.Index(out)


def scan_tiles(
    fragments: PathLike | List[PathLike],
    chrom_sizes: dict[str, int] | str,
    tile_size: int = 5000,
    min_fragments_per_cell: int = 100,
    exclude_chroms: Optional[list] = ["chrM", "chrY", "M", "Y"],
    metrics=None,
    filter_query: Optional[str] = None,
    count_strategy: str = "unique",
    barcode_prefix: Optional[str | List[str]] = None,
    row_groups_per_batch: int = 64,
    output_path: Optional[PathLike] = None,
) -> ad.AnnData:
    """
    Cells and tiles of the combined tile matrix of several fragment files,
    with per-tile statistics but no matrix.

    Each file is read once. Its cells are chosen as in
    :func:`~gatac.pp.make_tile_matrix`, and its tile counts are computed on
    the GPU and summed per tile, without keeping the matrix. Choose tiles
    with :func:`~gatac.pp.select_features` (top accessible by default), or
    with your own criterion on ``var``, then build the matrix of the chosen
    tiles with :func:`build_tile_matrix`.

    Parameters
    ----------
    fragments : str, Path, or list
        Fragment parquet file(s).
    chrom_sizes : dict or str
        Chromosome sizes, or a genome name (e.g. ``"hg38"``).
    tile_size, min_fragments_per_cell, exclude_chroms, metrics, filter_query, count_strategy
        As in :func:`~gatac.pp.make_tile_matrix`. ``metrics`` applies to
        every file, matched on the barcodes as stored in the files.
    barcode_prefix : str or list of str, optional
        Prefix of the cell names, one for all files or one per file. Names
        repeated across files are made unique with ``-1``, ``-2``, ...
    row_groups_per_batch : int
        Parquet row groups per GPU batch (default 64).
    output_path : str or Path, optional
        Also write the result to this ``.h5ad``, without X.

    Returns
    -------
    AnnData
        ``n_cells × n_tiles`` with ``X`` None.

        - ``obs``: ``barcode`` (as stored in the fragment file),
          ``source_file`` (the file's name), ``n_unique`` and the metrics'
          other columns. Cells are grouped by file, in the order given.
        - ``var``: ``chrom``, ``start``, ``end``, and the tile's statistics
          over all cells: ``n_cells`` (cells with signal in the tile),
          ``count`` (sum of the cells' counts) and ``count_sq`` (sum of
          their squares), from which means and variances follow. Every
          tile matrix carries them, :func:`~gatac.pp.make_tile_matrix`'s
          too.
        - ``uns["tile_scan"]``: the fragment files and tiling, which
          :func:`build_tile_matrix` reads.

    Examples
    --------
    >>> import gatac as ga
    >>> tiles = ga.pp.scan_tiles(["s1.parquet", "s2.parquet"], chrom_sizes="hg38")
    >>> ga.pp.select_features(tiles, n_features=50_000)
    >>> ga.pp.build_tile_matrix(tiles, output_path="combined.h5ad")

    Selecting on another statistic, e.g. the variance of the counts:

    >>> n = tiles.n_obs
    >>> mean = tiles.var["count"] / n
    >>> var = tiles.var["count_sq"] / n - mean**2
    >>> tiles.var["selected"] = var.rank(ascending=False) <= 50_000
    """
    from .genome import get_chrom_sizes

    if isinstance(chrom_sizes, str):
        chrom_sizes = get_chrom_sizes(chrom_sizes)
    if count_strategy not in ("unique", "count", "binarize"):
        raise ValueError(f"Unknown count_strategy {count_strategy!r}")
    paths = [Path(fragments)] if isinstance(fragments, (str, Path)) else [Path(p) for p in fragments]
    if not paths:
        raise ValueError("No fragment files given")
    names = [p.name for p in paths]
    if len(set(names)) != len(names):
        raise ValueError(
            "Fragment files must have distinct names, which identify each cell's file "
            "(see make_parquet_batch(file_prefix=...))"
        )
    if barcode_prefix is None or isinstance(barcode_prefix, str):
        prefixes = [barcode_prefix or ""] * len(paths)
    else:
        prefixes = list(barcode_prefix)
        if len(prefixes) != len(paths):
            raise ValueError(
                f"Number of barcode prefixes ({len(prefixes)}) must match files ({len(paths)})"
            )

    all_chroms, included_chroms = _included_chroms(chrom_sizes, exclude_chroms)
    tile_metadata, chrom_offset_df, n_tiles = _cached_tile_metadata(
        tuple(included_chroms), tuple(chrom_sizes[c] for c in included_chroms), tile_size,
    )
    tile_offsets = chrom_offset_df["tile_offset"].values
    cell_meta_extra = _load_metrics_dataframe(metrics, filter_query)

    stats: dict = {}
    obs_parts = []
    start_time = time.perf_counter()
    with async_memory_pool():
        for path, name, prefix in zip(paths, names, prefixes):
            fragments_file = FragmentFile(path)
            cells = _tile_cells(fragments_file, all_chroms, cell_meta_extra, min_fragments_per_cell)
            keys, counts = _stream_key_counts(
                fragments_file, list(cells["barcode"]), included_chroms, tile_offsets,
                n_tiles, tile_size, count_strategy, row_groups_per_batch,
            )
            for _, _, _, cols, data in _iter_slabs(keys, counts, len(cells), n_tiles):
                _add_tile_stats(stats, cols, data, n_tiles, count_strategy == "binarize")
            del keys, counts
            cells.index = pd.Index(prefix + cells["barcode"])
            cells.insert(1, "source_file", name)
            obs_parts.append(cells)
            logger.info(f"Scanned {name}: {len(cells):,} cells")
    cleanup_gpu_memory()

    obs = pd.concat(obs_parts)
    if not obs.index.is_unique:
        logger.warning(
            f"Detected {obs.index.duplicated().sum():,} duplicate cell names. Making them unique."
        )
        obs.index = _make_unique(obs.index)
    obs["source_file"] = pd.Categorical(obs["source_file"], categories=names)

    tiles = ad.AnnData(obs=obs, var=tile_metadata.copy())
    _set_tile_stats(tiles, stats, len(obs))
    tiles.uns["tile_scan"] = {
        "fragments": [str(p.resolve()) for p in paths],
        "chroms": list(included_chroms),
        "chrom_sizes": [int(chrom_sizes[c]) for c in included_chroms],
        "tile_size": int(tile_size),
        "count_strategy": count_strategy,
        "n_cells": int(tiles.n_obs),
    }
    logger.info(
        f"Scanned {len(paths)} files: {tiles.n_obs:,} cells × {n_tiles:,} tiles "
        f"({time.perf_counter() - start_time:.1f}s)"
    )
    if output_path is not None:
        tiles.write_h5ad(str(output_path))
    return tiles


def build_tile_matrix(
    tiles: ad.AnnData | PathLike,
    output_path: Optional[PathLike] = None,
    features: Optional[str | np.ndarray] = "selected",
    row_groups_per_batch: int = 64,
) -> ad.AnnData:
    """
    Build the tile matrix of the cells and chosen tiles of a :func:`scan_tiles` result.

    Each fragment file is read again and its tile counts computed on the
    GPU; only the chosen tiles are kept, so the full tile matrix is never
    held. Cells can be dropped from ``tiles.obs`` before the build; they
    must stay grouped by file.

    Parameters
    ----------
    tiles : AnnData or str or Path
        The result of :func:`scan_tiles`, or its ``.h5ad``.
    output_path : str or Path, optional
        Write the matrix to this ``.h5ad`` as it is built, one file at a
        time, instead of returning it in memory.
    features : str, array of bool, or None
        Tiles to keep: a boolean column of ``var`` (default ``"selected"``,
        set by :func:`~gatac.pp.select_features`), a boolean mask, or None
        for all tiles.
    row_groups_per_batch : int
        Parquet row groups per GPU batch (default 64).

    Returns
    -------
    AnnData
        The ``cells × chosen tiles`` AnnData with uint16 counts, ``obs`` and
        the chosen rows of ``var``. When ``output_path`` is given, it is
        opened from the file in backed mode, so the matrix is not loaded.
    """
    import h5py

    if not isinstance(tiles, ad.AnnData):
        tiles = ad.read_h5ad(tiles)
    if "tile_scan" not in tiles.uns:
        raise ValueError("`tiles` is not a scan_tiles result (no uns['tile_scan'])")
    scan = tiles.uns["tile_scan"]
    tile_size = int(scan["tile_size"])
    chroms = list(scan["chroms"])
    tile_metadata, chrom_offset_df, n_tiles = _cached_tile_metadata(
        tuple(chroms), tuple(int(s) for s in scan["chrom_sizes"]), tile_size,
    )
    tile_offsets = chrom_offset_df["tile_offset"].values

    if features is None:
        mask = np.ones(tiles.n_vars, dtype=bool)
    elif isinstance(features, str):
        if features not in tiles.var.columns:
            raise KeyError(
                f"Column '{features}' not found in var. Call `pp.select_features` first "
                "or set `features=None`."
            )
        mask = tiles.var[features].to_numpy().astype(bool)
    else:
        mask = np.asarray(features, dtype=bool)
    var = tiles.var.loc[mask].copy()
    # Column of each kept tile in the full tiling; var may have been subset
    tile_ids = tile_metadata.index.get_indexer(var.index)
    if (tile_ids < 0).any():
        raise ValueError("Some tiles of `tiles.var` are not in the tiling of uns['tile_scan']")
    order = np.argsort(tile_ids, kind="stable")
    if (order != np.arange(len(order))).any():
        raise ValueError("`tiles.var` must keep the tiles in genome order")
    remap = cp.full(n_tiles, -1, dtype=cp.int32)
    remap[cp.asarray(tile_ids)] = cp.arange(len(tile_ids), dtype=cp.int32)

    obs = tiles.obs
    source = obs["source_file"].astype(str).to_numpy()
    files = {Path(p).name: p for p in scan["fragments"]}
    # cells of each file, in obs order; files must be contiguous
    runs = np.flatnonzero(np.r_[True, source[1:] != source[:-1]])
    if len(set(source[runs])) != len(runs):
        raise ValueError("`tiles.obs` must keep the cells of each file together")
    bounds = list(zip(runs, np.r_[runs[1:], len(source)]))
    n_cells, n_kept = tiles.n_obs, len(var)
    count_strategy = scan["count_strategy"]

    def file_parts():
        """Per file: (row0, row_nnz, column ids, data) of its kept entries, on the host."""
        for r0, r1 in bounds:
            name = source[r0]
            fragments_file = FragmentFile(files[name])
            barcodes = list(obs["barcode"].iloc[r0:r1])
            keys, counts = _stream_key_counts(
                fragments_file, barcodes, chroms, tile_offsets, n_tiles, tile_size,
                count_strategy, row_groups_per_batch,
            )
            row_nnz = np.zeros(r1 - r0, dtype=np.int64)
            cols_parts, data_parts = [], []
            for s0, s1, rows, cols, data in _iter_slabs(keys, counts, r1 - r0, n_tiles):
                new_col = remap[cols]
                keep = new_col >= 0
                row_nnz[s0:s1] = cp.bincount(rows[keep] - s0, minlength=s1 - s0).get()
                cols_parts.append(new_col[keep].get())
                kept = data[keep]
                data_parts.append(
                    np.ones(len(kept), dtype=np.uint16) if count_strategy == "binarize"
                    else kept.astype(cp.uint16).get()
                )
                del new_col, keep, kept
            del keys, counts
            logger.info(f"Built {name}: {r1 - r0:,} cells")
            yield (
                r0, row_nnz,
                np.concatenate(cols_parts) if cols_parts else np.zeros(0, np.int32),
                np.concatenate(data_parts) if data_parts else np.zeros(0, np.uint16),
            )

    # the statistics of var stay valid while every scanned cell is kept
    uns = {"tile_scan": dict(scan), "tile_stats": {"n_obs": int(scan["n_cells"])}}
    start_time = time.perf_counter()
    if output_path is None:
        with async_memory_pool():
            parts = list(file_parts())
        cleanup_gpu_memory()
        row_nnz = np.concatenate([p[1] for p in parts])
        nnz = int(row_nnz.sum())
        idx_dtype = np.int32 if max(nnz, n_kept) <= np.iinfo(np.int32).max else np.int64
        indptr = np.zeros(n_cells + 1, dtype=idx_dtype)
        np.cumsum(row_nnz, out=indptr[1:])
        X = sp.csr_matrix(
            (np.concatenate([p[3] for p in parts]),
             np.concatenate([p[2] for p in parts]).astype(idx_dtype, copy=False), indptr),
            shape=(n_cells, n_kept),
        )
        del parts
        out = ad.AnnData(X=X, obs=obs.copy(), var=var, uns=uns)
        logger.info(
            f"Built {n_cells:,} cells × {n_kept:,} tiles "
            f"({time.perf_counter() - start_time:.1f}s)"
        )
        return out

    # Written as it is built, into datasets sized from the tile statistics:
    # their cell counts are the entry count when every scanned cell is kept,
    # an upper bound otherwise, and the datasets are then cut to size.
    output_path = Path(output_path)
    ad.AnnData(obs=obs, var=var, uns=uns).write_h5ad(str(output_path))
    if "n_cells" not in var.columns:
        raise ValueError("`tiles.var` has lost its 'n_cells' column from scan_tiles")
    upper = int(var["n_cells"].sum())
    exact = n_cells == int(scan["n_cells"])
    # one index dtype for both arrays, as scipy picks it
    idx_dtype = np.int32 if max(upper, n_kept) <= np.iinfo(np.int32).max else np.int64
    shrink = {} if exact else {"maxshape": (None,), "chunks": (1 << 22,)}
    with async_memory_pool(), h5py.File(output_path, "a") as h5:
        group = h5.create_group("X")
        group.attrs.update({"encoding-type": "csr_matrix", "encoding-version": "0.1.0",
                            "shape": np.array([n_cells, n_kept])})
        out_data = group.create_dataset("data", shape=(upper,), dtype=np.uint16, **shrink)
        out_indices = group.create_dataset("indices", shape=(upper,), dtype=idx_dtype, **shrink)
        indptr = np.zeros(n_cells + 1, dtype=idx_dtype)
        pos = 0
        for r0, row_nnz, cols, data in file_parts():
            k = len(cols)
            out_data[pos:pos + k] = data
            out_indices[pos:pos + k] = cols
            indptr[r0 + 1:r0 + 1 + len(row_nnz)] = pos + np.cumsum(row_nnz)
            pos += k
        if exact and pos != upper:
            raise RuntimeError(f"Wrote {pos:,} entries, expected {upper:,}")
        if not exact:
            out_data.resize((pos,))
            out_indices.resize((pos,))
        group.create_dataset("indptr", data=indptr)
    cleanup_gpu_memory()
    logger.info(
        f"Wrote {output_path.name}: {n_cells:,} cells × {n_kept:,} tiles "
        f"({time.perf_counter() - start_time:.1f}s)"
    )
    return ad.read_h5ad(output_path, backed="r")

