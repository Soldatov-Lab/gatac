"""
One matrix of several fragment files: the cells of each file in turn, named
and tagged with their file as :func:`~gatac.pp.make_tile_matrix` does.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

logger = logging.getLogger(__name__)


def _file_names(paths: Sequence[Path]) -> List[str]:
    """The files' names, which must be distinct: they identify each cell's file."""
    if not paths:
        raise ValueError("No fragment files given")
    names = [p.name for p in paths]
    if len(set(names)) != len(names):
        raise ValueError(
            "Fragment files must have distinct names, which identify each cell's file: "
            'convert them with gatac convert --file-prefix "{sample}_" '
            '(make_parquet_batch(file_prefix="{sample}_")), or rename them'
        )
    return names


def _shared_names_error(cells: pd.Index, files: np.ndarray, names: Sequence[str]) -> ValueError:
    """The error for cell names found in several files."""
    dup = cells.duplicated(keep=False)
    shared = list(dict.fromkeys(np.asarray(names, dtype=object)[files[dup]]))
    return ValueError(
        f"{cells[dup].nunique():,} cell names are in more than one of the files "
        f"{shared}, e.g. {list(cells[dup].unique()[:3])}. Give each sample its own barcode "
        "prefix, which only rewrites the file's footer: `gatac rename FILE --barcode-prefix "
        "PREFIX` or `gatac.io.set_barcode_prefix(FILE, PREFIX)`"
    )


def _check_distinct_cells(paths: Sequence[Path], names: Sequence[str]) -> None:
    """Raise, before any work, if fragment files share barcode names."""
    from ..io.fragments import FragmentFile

    barcodes = [FragmentFile(p).barcodes for p in paths]
    cells = pd.Index(np.concatenate(barcodes))
    if not cells.is_unique:
        files = np.repeat(np.arange(len(paths)), [len(b) for b in barcodes])
        raise _shared_names_error(cells, files, names)


def _combined_obs(obs_parts: List[pd.DataFrame], names: Sequence[str]) -> pd.DataFrame:
    """The files' obs in order, with ``source_file`` categorical; cell names must be distinct."""
    obs = pd.concat(obs_parts)
    if not obs.index.is_unique:
        files = np.repeat(np.arange(len(obs_parts)), [len(o) for o in obs_parts])
        raise _shared_names_error(obs.index, files, names)
    obs["source_file"] = pd.Categorical(obs["source_file"], categories=list(names))
    return obs


def _widen(group, key: str, dtype) -> object:
    """Replace the 1-D dataset ``group[key]`` by a copy of it in ``dtype``."""
    old = group[key]
    new = group.create_dataset(key + "_wide", shape=old.shape, dtype=dtype,
                               maxshape=(None,), chunks=old.chunks)
    step = 1 << 24
    for i in range(0, old.shape[0], step):
        new[i:i + step] = old[i:i + step]
    del group[key]
    group.move(key + "_wide", key)
    return group[key]


def combine_file_matrices(
    parts: Iterable[ad.AnnData],
    names: Sequence[str],
    output_path: Optional[str | Path] = None,
) -> ad.AnnData:
    """
    One AnnData of the per-file matrices ``parts``, cells of the same features.

    The cells keep their order, file after file; ``obs["source_file"]`` is
    the cell's file among ``names``, cell names must be distinct across
    files, and ``X`` takes the widest dtype of the parts. ``parts`` is consumed once: with ``output_path``, each part's
    rows are written to the ``.h5ad`` as it comes, so only one part is held,
    and the result is opened from the file in backed mode.
    """
    obs_parts: List[pd.DataFrame] = []
    var: Optional[pd.DataFrame] = None

    def rows(part: ad.AnnData, name: str) -> sp.csr_matrix:
        nonlocal var
        if var is None:
            var = part.var
        elif not part.var_names.equals(var.index):
            raise ValueError(f"{name} has other features than {names[0]}")
        obs = part.obs.copy()
        obs.insert(min(1, obs.shape[1]), "source_file", name)
        obs_parts.append(obs)
        return sp.csr_matrix(part.X)

    if output_path is None:
        X = [rows(part, name) for part, name in zip(parts, names)]
        X = sp.vstack(X, format="csr", dtype=np.result_type(*(x.dtype for x in X)))
        return ad.AnnData(X=X, obs=_combined_obs(obs_parts, names), var=var)

    import h5py

    output_path = Path(output_path)
    indptr = [np.zeros(1, dtype=np.int64)]
    n_rows = 0
    with h5py.File(output_path, "w") as h5:
        group = h5.create_group("X")
        data = indices = None
        for part, name in zip(parts, names):
            x = rows(part, name)
            del part
            if data is None:
                grow = {"maxshape": (None,), "chunks": (1 << 20,)}
                data = group.create_dataset("data", shape=(0,), dtype=x.dtype, **grow)
                idx = np.int32 if len(var) <= np.iinfo(np.int32).max else np.int64
                indices = group.create_dataset("indices", shape=(0,), dtype=idx, **grow)
            elif np.result_type(data.dtype, x.dtype) != data.dtype:
                data = _widen(group, "data", np.result_type(data.dtype, x.dtype))
            pos = int(indptr[-1][-1])
            data.resize((pos + x.nnz,))
            data[pos:] = x.data
            indices.resize((pos + x.nnz,))
            indices[pos:] = x.indices
            indptr.append(pos + x.indptr[1:].astype(np.int64))
            n_rows += x.shape[0]
            del x
        indptr = np.concatenate(indptr)
        if indptr[-1] <= np.iinfo(np.int32).max:
            indptr = indptr.astype(np.int32)
        group.create_dataset("indptr", data=indptr)
        group.attrs.update({"encoding-type": "csr_matrix", "encoding-version": "0.1.0",
                            "shape": np.array([n_rows, len(var)])})

    # obs, var and the file's AnnData attributes, from an AnnData without X
    # written alongside, so that X is never rewritten
    meta = output_path.with_name(output_path.name + ".obs.h5ad")
    try:
        ad.AnnData(obs=_combined_obs(obs_parts, names), var=var).write_h5ad(str(meta))
        with h5py.File(meta, "r") as src, h5py.File(output_path, "a") as dst:
            for key in src:
                if key != "X":
                    src.copy(src[key], dst, key)
            dst.attrs.update(src.attrs)
    finally:
        meta.unlink(missing_ok=True)
    return ad.read_h5ad(output_path, backed="r")
