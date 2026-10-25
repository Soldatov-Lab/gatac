"""
``gatac.pp`` feature selection and the multi-sample helpers.

The fragment-level builders (tile, gene and metric matrices, filtering,
doublets) are covered in ``tests/regression/test_fragment_files.py`` and the
reproducibility suite; this file covers what operates on built matrices.
"""
from __future__ import annotations

import numpy as np
import pytest
import scanpy as sc

import gatac as ga


@pytest.fixture
def split_h5ad(tiles, tmp_path):
    """The tile matrix written as two h5ad files, alternate cells in each, and the matrix in that cell order."""
    paths = [tmp_path / "A_tile.h5ad", tmp_path / "B_tile.h5ad"]
    tiles[0::2].write_h5ad(paths[0])
    tiles[1::2].write_h5ad(paths[1])
    order = np.r_[np.arange(0, tiles.n_obs, 2), np.arange(1, tiles.n_obs, 2)]
    return paths, tiles[order].copy()


def _binarized_tile_matrix(tiles, world):
    from pyfaidx import Fasta
    sizes = {c: len(s) for c, s in Fasta(str(world["fasta"])).items()}
    a = ga.pp.make_tile_matrix(world["parquet"], chrom_sizes=sizes, tile_size=5_000,
                               min_fragments_per_cell=100, count_strategy="binarize")
    return a[tiles.obs_names].copy()


@pytest.mark.parametrize("source", ["bool", "make_tile_matrix"])
def test_select_features_ignores_quantiles_on_binary_input(tiles, world, source):
    """Binary matrices: plain top-N by accessibility, quantile filters "ignored for binary matrices"."""
    a = _binarized_tile_matrix(tiles, world) if source == "make_tile_matrix" else tiles
    if source == "bool":
        a.X = a.X.astype(bool)
    n = a.n_vars // 3
    ga.pp.select_features(a, n_features=n, filter_lower_quantile=0.1, filter_upper_quantile=0.1)
    sel = a.var["selected"].to_numpy()
    count = a.var["accessibility_count"].to_numpy()
    assert sel.sum() == n
    assert count[sel].min() >= count[~sel].max()


def test_combine_concatenates_files_and_rejects_duplicate_barcodes(split_h5ad, tmp_path):
    """combine stacks the files' cells in order with ``source_file``, and refuses colliding barcodes."""
    paths, ordered = split_h5ad
    out = tmp_path / "combined.h5ad"
    ga.pp.combine(paths, out)
    got = sc.read_h5ad(out)
    assert list(got.obs_names) == list(ordered.obs_names)
    assert list(got.var_names) == list(ordered.var_names)
    assert (got.X != ordered.X).nnz == 0
    n_a = (ordered.n_obs + 1) // 2
    assert got.obs["source_file"].tolist() == ["A_tile.h5ad"] * n_a + ["B_tile.h5ad"] * (ordered.n_obs - n_a)
    with pytest.raises(ValueError, match="duplicate cell barcode"):
        ga.pp.combine([paths[0], paths[0]], tmp_path / "dup.h5ad")
