"""`sample_bg_peaks(method="chromvar", random_state=...)` reproduces a background."""
import warnings

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

import gatac as ga


def _peaks(n_cells=300, n_peaks=4000):
    X = sp.random(n_cells, n_peaks, density=0.05, format="csr", random_state=0,
                  dtype=np.float32)
    X.data[:] = 1
    X[0, :] = 1                                   # every peak has reads
    adata = ad.AnnData(sp.csr_matrix(X))
    adata.var["gc_content"] = np.random.default_rng(1).uniform(0.3, 0.7, n_peaks)
    return adata


def _background(adata, **kwargs):
    adata = adata.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # empty bias bins
        ga.tl.sample_bg_peaks(adata, method="chromvar", n_iterations=20, **kwargs)
    return adata.varm["bg_peaks"]


def test_same_seed_same_background():
    pytest.importorskip("cuml")
    adata = _peaks()
    assert np.array_equal(_background(adata, random_state=7),
                          _background(adata, random_state=7))


def test_different_seed_different_background():
    pytest.importorskip("cuml")
    adata = _peaks()
    assert not np.array_equal(_background(adata, random_state=7),
                              _background(adata, random_state=8))


def test_default_still_follows_global_state():
    pytest.importorskip("cuml")
    adata = _peaks()
    np.random.seed(3)
    first = _background(adata)
    np.random.seed(3)
    assert np.array_equal(first, _background(adata))
