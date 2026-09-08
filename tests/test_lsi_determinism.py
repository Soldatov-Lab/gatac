"""
G6: LSI is a bitwise-reproducible function of its input and ``random_state``.

Self-contained — synthetic data, no ArchR oracle — so it can run anywhere a GPU
is available and guard the property that the reproducibility suite's fixtures
are too slow to re-check on every change.

What is asserted is *identity*, not a tolerance: two runs at one
``random_state`` must return the same bytes, the same feature set and the same
cluster labels, in one process and across a fresh one. Before the deterministic
solver path this failed badly — repeated runs of ``iterative_lsi`` on the PBMC
oracle agreed on only 74-95 % of their selected features and reached a
cluster ARI of 0.75-0.86 with themselves.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap

import numpy as np
import pytest
import scipy.sparse as sp

pytest.importorskip("cupy")


def _synthetic(n_cells: int = 900, n_features: int = 4000, n_groups: int = 4,
               seed: int = 0) -> "sp.csr_matrix":
    """A binary cells x features matrix with real group structure."""
    rng = np.random.default_rng(seed)
    group = rng.integers(0, n_groups, n_cells)
    block = n_features // n_groups
    rows, cols = [], []
    for c in range(n_cells):
        # a shared background plus a block enriched in this cell's group
        bg = rng.choice(n_features, size=rng.integers(150, 300), replace=False)
        lo = group[c] * block
        fg = lo + rng.choice(block, size=rng.integers(120, 200), replace=False)
        idx = np.unique(np.concatenate([bg, fg]))
        rows.append(np.full(idx.size, c))
        cols.append(idx)
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    X = sp.csr_matrix((np.ones(rows.size, np.float32), (rows, cols)),
                      shape=(n_cells, n_features))
    return X


def _adata(X):
    import anndata as ad
    a = ad.AnnData(X.copy())
    a.obs["n_unique"] = np.asarray(X.sum(axis=1)).ravel().astype(float)
    a.var["selected"] = True
    return a


def test_spmv_matches_cusparse_and_repeats_bitwise():
    """The fixed-order product agrees with cuSPARSE and, unlike it, repeats."""
    import cupy as cp
    import cupyx.scipy.sparse as cusp

    from gatac.tl.lsi import _spmm, _spmv

    X = cusp.csr_matrix(_synthetic(400, 1500).astype(np.float32))
    cp.random.seed(0)
    v = cp.random.random(X.shape[1], dtype=cp.float32)
    B = cp.random.random((X.shape[1], 4), dtype=cp.float32)

    ref = X @ v
    out = _spmv(X, v, cp.empty(X.shape[0], dtype=cp.float32))
    assert float(cp.abs(out - ref).max() / cp.abs(ref).max()) < 1e-5
    for _ in range(4):
        assert bool((_spmv(X, v, cp.empty(X.shape[0], dtype=cp.float32))
                     == out).all())

    refm = X @ B
    outm = _spmm(X, B)
    assert float(cp.abs(outm - refm).max() / cp.abs(refm).max()) < 1e-5
    for _ in range(4):
        assert bool((_spmm(X, B) == outm).all())


def test_lsi_is_bitwise_reproducible():
    import cupy as cp

    import gatac as ga

    X = _synthetic()
    runs = []
    for i in range(3):
        if i:                      # unrelated draws must not move the result
            np.random.default_rng(i).random(101)
            cp.random.seed(i + 17)
            cp.random.random(53)
        a = _adata(X)
        ga.tl.lsi(a, 10, random_state=3, outlier_quantiles=(0.02, 0.98))
        runs.append((a.obsm["X_lsi"], a.uns["lsi"]["singular_values"]))

    for emb, sv in runs[1:]:
        assert np.array_equal(emb, runs[0][0])
        assert np.array_equal(sv, runs[0][1])

    a = _adata(X)
    ga.tl.lsi(a, 10, random_state=4, outlier_quantiles=(0.02, 0.98))
    # a different seed must actually be a different draw, or the seed is a lie
    assert not np.array_equal(a.obsm["X_lsi"], runs[0][0])


def test_iterative_lsi_is_bitwise_reproducible():
    import gatac as ga

    X = _synthetic()
    out = []
    for _ in range(2):
        a = _adata(X)
        ga.tl.iterative_lsi(
            a, 10, iterations=2, n_features=800, total_features=3000,
            cluster_params={"k": 15, "max_clusters": 4}, random_state=1)
        out.append((
            a.obsm["X_iterative_lsi"],
            np.where(a.var["selected_iterative_lsi"].to_numpy())[0],
            a.obs["iterative_lsi_clusters_iter1"].to_numpy().astype(str),
        ))
    assert np.array_equal(out[0][0], out[1][0]), "embedding differs"
    assert np.array_equal(out[0][1], out[1][1]), "selected features differ"
    assert np.array_equal(out[0][2], out[1][2]), "cluster labels differ"


def test_lsi_reproducible_across_processes():
    """A fresh interpreter must produce the same bytes."""
    script = textwrap.dedent("""
        import sys, numpy as np
        sys.path.insert(0, {tests!r})
        from test_lsi_determinism import _adata, _synthetic
        import gatac as ga
        a = _adata(_synthetic())
        ga.tl.lsi(a, 10, random_state=3, outlier_quantiles=(0.02, 0.98))
        sys.stdout.buffer.write(a.obsm["X_lsi"].tobytes())
    """).format(tests=__file__.rsplit("/", 1)[0])
    outs = [subprocess.run([sys.executable, "-c", script], check=True,
                           capture_output=True).stdout for _ in range(2)]
    assert outs[0] == outs[1]

    import gatac as ga
    a = _adata(_synthetic())
    ga.tl.lsi(a, 10, random_state=3, outlier_quantiles=(0.02, 0.98))
    assert a.obsm["X_lsi"].tobytes() == outs[0]


def test_deterministic_false_still_runs_and_agrees_numerically():
    """The escape hatch must stay usable, and stay the same algorithm."""
    import gatac as ga

    X = _synthetic()
    a = _adata(X)
    ga.tl.lsi(a, 10, random_state=3, outlier_quantiles=(0.02, 0.98))
    b = _adata(X)
    ga.tl.lsi(b, 10, random_state=3, outlier_quantiles=(0.02, 0.98),
              deterministic=False)
    r = np.array([abs(np.corrcoef(a.obsm["X_lsi"][:, i], b.obsm["X_lsi"][:, i])[0, 1])
                  for i in range(10)])
    assert r.min() > 0.999, r
    sa, sb = a.uns["lsi"]["singular_values"], b.uns["lsi"]["singular_values"]
    assert np.abs(sa - sb).max() / sa.max() < 1e-5
