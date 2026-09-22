"""
The batched GPU logistic fitter behind ``motif_enrichment_regression`` agrees
with an independent maximum-likelihood fit.

Self-contained — synthetic design, no genome or motif scan — so it guards the
fitter's numerics (the Hessian is assembled from precomputed column products,
not an einsum) without the cost of a real scan. statsmodels is the reference:
coefficient and standard error of the score column must match per motif.
"""
from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
sm = pytest.importorskip("statsmodels.api")

from gatac.tl.motif import _design_products, _fit_logistic_batch  # noqa: E402


def _synthetic(n: int = 4000, n_cov: int = 5, n_motifs: int = 6, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones(n), rng.standard_normal((n, 1 + n_cov))])
    true_beta = rng.normal(0, 0.5, size=(n_motifs, X.shape[1]))
    true_beta[:, 0] = rng.uniform(-2.5, -0.5, size=n_motifs)
    prob = 1.0 / (1.0 + np.exp(-(true_beta @ X.T)))
    return X, (rng.random(prob.shape) < prob).astype(np.float64)


def test_design_products_give_weighted_gram():
    X, _ = _synthetic()
    w = np.random.default_rng(1).random((3, X.shape[0]))
    Z, rows, cols = _design_products(cp.asarray(X))
    upper = cp.asnumpy(cp.asarray(w) @ Z)
    expected = np.einsum("np,mn,nq->mpq", X, w, X)
    np.testing.assert_allclose(upper, expected[:, rows, cols], rtol=1e-12)


@pytest.mark.parametrize("pass_products", [False, True])
def test_fit_matches_statsmodels(pass_products):
    X, Y = _synthetic()
    Xg = cp.asarray(X)
    products = _design_products(Xg) if pass_products else None
    coef, se, converged = _fit_logistic_batch(
        Xg, cp.asarray(Y), 1, max_iter=50, tol=1e-8, ridge=1e-8,
        products=products)

    assert converged.all()
    for m in range(Y.shape[0]):
        ref = sm.Logit(Y[m], X).fit(disp=0, method="newton", tol=1e-12)
        np.testing.assert_allclose(coef[m], ref.params[1], rtol=1e-6)
        np.testing.assert_allclose(se[m], ref.bse[1], rtol=1e-6)


def test_separated_fit_is_flagged():
    X, Y = _synthetic(n_motifs=2)
    Y[1] = (X[:, 1] > 0.5).astype(np.float64)  # perfectly separated by score
    _, _, converged = _fit_logistic_batch(
        cp.asarray(X), cp.asarray(Y), 1, max_iter=50, tol=1e-8, ridge=1e-8)
    assert converged[0] and not converged[1]
