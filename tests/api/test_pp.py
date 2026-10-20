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


@pytest.mark.parametrize("source", ["bool", pytest.param("make_tile_matrix", marks=pytest.mark.xfail(
    strict=True, reason=(
        "select_features detects binary input by bool dtype only (gatac/pp/features.py:38); "
        "make_tile_matrix(count_strategy='binarize') stores 0/1 in a numeric dtype, so GATAC's own "
        "binary tile matrices take the count path and get quantile-trimmed.")))])
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


@pytest.mark.xfail(strict=True, raises=ValueError, reason=(
    "Every multi-file call crashes: with the locked anndata 0.10.9, "
    "adata_meta[:, idx].to_memory().X of a backed file is None (gatac/pp/features.py:437), "
    "and sp.csr_matrix(None) raises 'does not support dtype object'."))
def test_select_features_multi_matches_in_memory_selection(split_h5ad, tmp_path):
    """Streaming over files selects what select_features selects on all cells, and keeps every cell's row."""
    paths, ordered = split_h5ad
    out = tmp_path / "combined.h5ad"
    ga.pp.select_features_multi(paths, out, n_features=300)
    got = sc.read_h5ad(out)

    ga.pp.select_features(ordered, n_features=300)
    ref = ordered[:, ordered.var["selected"].to_numpy()]
    assert list(got.var_names) == list(ref.var_names)
    np.testing.assert_array_equal(got.var["accessibility_count"], ref.var["accessibility_count"])
    assert list(got.obs_names) == list(ref.obs_names)
    n_a = (ordered.n_obs + 1) // 2
    assert got.obs["source_file"].tolist() == ["A_tile.h5ad"] * n_a + ["B_tile.h5ad"] * (ordered.n_obs - n_a)
    assert got.X.dtype == bool
    assert (got.X != (ref.X > 0)).nnz == 0


@pytest.mark.xfail(strict=True, reason=(
    "With one input, select_features_multi falls back to select_features (gatac/pp/features.py:323): "
    "the output keeps every feature and ignores binarize, unlike any multi-file call."))
def test_select_features_multi_output_does_not_depend_on_number_of_inputs(split_h5ad, tmp_path):
    """One file in gives the same kind of output as several: only selected features, binarized."""
    paths, _ = split_h5ad
    out = tmp_path / "one.h5ad"
    ga.pp.select_features_multi(paths[:1], out, n_features=300)
    got = sc.read_h5ad(out)
    assert got.n_vars == 300 and got.X.dtype == bool


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
