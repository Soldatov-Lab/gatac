"""
What ``gatac.dr`` promises its callers, checked on the synthetic world of ``tests/conftest.py``.

LSI determinism lives in ``tests/regression/test_lsi_determinism.py`` and
ArchR / SnapATAC2 parity in the reproducibility suite; neither is repeated here.
"""
from __future__ import annotations

import anndata as ad
import cupyx.scipy.sparse as cusp
import numpy as np
import pytest
from sklearn.metrics import silhouette_score

import gatac as ga

N_COMPS = 8


@pytest.fixture
def selected(tiles):
    ga.pp.select_features(tiles, n_features=2_000)
    return tiles


def _true_labels(a):
    return a.obs["group"].cat.codes.to_numpy()


def _lsi(a, **kw):
    return ga.dr.lsi(a, N_COMPS, **kw)


def _spectral(a, **kw):
    return ga.dr.spectral(a, N_COMPS, **kw)


EMBEDDINGS = {"lsi": (_lsi, "X_lsi"), "spectral": (_spectral, "X_spectral")}


def _abs_corr(a, b):
    """Per-component |Pearson r|: equality up to the sign an eigensolver may pick."""
    return np.array([abs(np.corrcoef(a[:, i], b[:, i])[0, 1]) for i in range(a.shape[1])])


@pytest.mark.parametrize("name", ["lsi", "spectral", "iterative_lsi"])
def test_embedding_separates_the_groups(selected, name):
    """The point of an embedding: groups with disjoint accessible hotspots land apart."""
    if name == "iterative_lsi":
        ga.dr.iterative_lsi(selected, N_COMPS, total_features=selected.n_vars, n_features=300)
        key = "X_iterative_lsi"
    else:
        fn, key = EMBEDDINGS[name]
        fn(selected)
    assert silhouette_score(selected.obsm[key], selected.obs["group"]) > 0.3


@pytest.mark.xfail(strict=True, reason=(
    "spectral: cusla.eigsh runs on cuSPARSE products that are not run-to-run "
    "deterministic, and the eigenvector signs are never fixed, so the same "
    "random_state flips component signs at random (spectral.py:_spectral_mf)"))
def test_spectral_same_seed_same_result(selected):
    """``random_state`` is documented as the 'seed for reproducibility'."""
    out = [ga.dr.spectral(selected.copy(), N_COMPS, random_state=3, inplace=False)[1] for _ in range(3)]
    for o in out[1:]:
        np.testing.assert_array_equal(o, out[0])


@pytest.mark.parametrize("name", list(EMBEDDINGS))
def test_chunked_matches_resident(selected, name):
    """``chunk_size`` streams the same computation; for LSI the recorded mode says it is not reproducible."""
    fn, key = EMBEDDINGS[name]
    ref = selected.copy()
    fn(ref)
    fn(selected, chunk_size=50)
    assert selected.obsm[key].shape == ref.obsm[key].shape
    # The solver stops at tol=1e-5 in both paths, so they agree to solver
    # precision, not bitwise; a wrong chunk boundary would show up far above it.
    assert _abs_corr(selected.obsm[key], ref.obsm[key]).min() > 0.9999
    if name == "lsi":
        assert ref.uns["lsi"]["params"]["deterministic"]
        assert not selected.uns["lsi"]["params"]["deterministic"]


@pytest.mark.parametrize("name", list(EMBEDDINGS))
def test_cell_without_signal_raises(selected, name):
    """A cell with nothing in the selected features raises instead of becoming a degenerate row."""
    fn, _ = EMBEDDINGS[name]
    X = selected.X.tolil()
    X[0, np.flatnonzero(selected.var["selected"])] = 0
    selected.X = X.tocsr()
    with pytest.raises(ValueError, match="cell"):
        fn(selected)


@pytest.mark.parametrize("name", [
    "lsi",
    pytest.param("spectral", marks=pytest.mark.xfail(strict=True, reason=(
        "spectral: _to_gpu_csr returns a float32 cupy CSR as-is (_gpu.py:104) and "
        "_normalize_rows_inplace / _scale_rows_kernel then rewrite it, i.e. adata.X "
        "(spectral.py:582-592)"))),
])
def test_gpu_input_gives_the_same_result_and_is_left_alone(tiles, name):
    """An AnnData already on the GPU (e.g. via rapids_singlecell) embeds like the host one and keeps its X."""
    fn, key = EMBEDDINGS[name]
    host = tiles.copy()
    fn(host, features=None)
    tiles.X = cusp.csr_matrix(tiles.X.astype(np.float32))
    before = tiles.X.copy()
    fn(tiles, features=None)
    assert float(abs(tiles.X - before).max()) == 0, "adata.X was modified"
    assert _abs_corr(tiles.obsm[key], host.obsm[key]).min() > 0.9999


@pytest.mark.xfail(strict=True, reason=(
    "lsi: with scale_by_sv=False the fitted cells get U (lsi.py:969) but held-out "
    "depth outliers are reinserted by _project, which returns X @ V = U·S "
    "(lsi.py:999), so their rows are scaled by the singular values"))
def test_scale_by_sv_only_rescales_columns(selected):
    """``scale_by_sv`` returns ``V @ diag(d)`` instead of ``V`` — for every cell, held-out ones included."""
    kw = dict(outlier_quantiles=(0.02, 0.98), depth_cor_cutoff=None)
    scaled, unscaled = selected.copy(), selected.copy()
    ga.dr.lsi(scaled, N_COMPS, **kw)
    ga.dr.lsi(unscaled, N_COMPS, scale_by_sv=False, **kw)
    assert scaled.uns["lsi"]["n_held_out"] > 0
    s = scaled.uns["lsi"]["singular_values"]
    np.testing.assert_allclose(unscaled.obsm["X_lsi"] * s, scaled.obsm["X_lsi"], rtol=1e-4, atol=1e-5)


def test_depth_filter_drops_exactly_the_correlated_components(selected):
    """``depth_cor_cutoff`` removes the components over the cutoff, and only those, from the full embedding."""
    full, cut = selected.copy(), selected.copy()
    ga.dr.lsi(full, N_COMPS, depth_cor_cutoff=None)
    ga.dr.lsi(cut, N_COMPS, depth_cor_cutoff=0.75)
    r = cut.uns["lsi"]["depth_cor"]
    np.testing.assert_array_equal(r, full.uns["lsi"]["depth_cor"])
    np.testing.assert_array_equal(cut.uns["lsi"]["dims_dropped"], np.flatnonzero(r > 0.75))
    assert cut.uns["lsi"]["dims_dropped"].size > 0  # method 2's depth component, at least
    np.testing.assert_array_equal(cut.obsm["X_lsi"], full.obsm["X_lsi"][:, r <= 0.75])


def test_stored_models_survive_h5ad_and_project_cells_back(selected, tmp_path):
    """
    ``store_model`` + ``model_from_adata`` + ``project_lsi``: after a write/read
    round trip, both models on one object (their keys must not collide) put the
    cells back where the fit did — held-out cells exactly, fitted cells to
    solver precision.
    """
    labels = _true_labels(selected)
    ga.dr.lsi(selected, N_COMPS, outlier_quantiles=(0.02, 0.98), store_model=True)
    ga.dr.iterative_lsi(selected, N_COMPS, total_features=selected.n_vars, n_features=300, store_model=True,
                        cluster_fn=lambda z: labels)
    selected.write_h5ad(tmp_path / "a.h5ad")
    back = ad.read_h5ad(tmp_path / "a.h5ad")

    for key, obsm in [("lsi", "X_lsi"), ("iterative_lsi", "X_iterative_lsi")]:
        uns = back.uns[key]
        model = ga.dr.model_from_adata(back, key)
        mask = np.asarray(uns["model"]["feature_mask"], bool)
        proj = ga.dr.project_lsi(model, back.X[:, mask])
        kept = np.setdiff1d(np.arange(proj.shape[1]), uns["dims_dropped"])
        proj, emb = proj[:, kept], back.obsm[obsm]
        assert uns["n_held_out"] > 0
        # Held-out cells were placed by this very projection, from the same loadings
        held = _held_out(back, key)
        np.testing.assert_array_equal(proj[held], emb[held], err_msg=key)
        # Fitted cells: U·S from the solver vs X·V — equal up to tol=1e-5
        np.testing.assert_allclose(proj[~held], emb[~held], rtol=1e-3,
                                   atol=1e-3 * np.abs(emb).max(), err_msg=key)


def _held_out(a, key):
    """The depth-outlier cells ``compute_lsi`` held out: depth over the fitted features at the stored quantiles."""
    mask = np.asarray(a.uns[key]["model"]["feature_mask"], bool)
    sub = a.X[:, mask]
    depth = np.diff(sub.tocsr().indptr) if a.uns[key]["params"]["binarize"] else np.asarray(sub.sum(1)).ravel()
    lo, hi = np.quantile(depth[depth > 0], sorted(a.uns[key]["params"]["outlier_quantiles"]))
    held = (depth <= lo) | (depth >= hi)
    assert held.sum() == a.uns[key]["n_held_out"]
    return held


def test_cluster_fn_drives_feature_selection(selected, world):
    """
    ``cluster_fn`` replaces the clusterer: it gets the depth-filtered embedding,
    its labels are recorded, and the re-selected features are the ones that vary
    between *those* labels. Labelled g0-vs-rest, g0's hotspots are absent from
    one pseudo-bulk and diluted in the other, so they must make up most of the
    selection — accessibility alone picks the three groups' hotspots evenly
    (a third each).
    """
    labels, seen = (selected.obs["group"] == "g0").to_numpy().astype(int), []

    def oracle(z):
        seen.append(z.shape)
        return labels

    ga.dr.iterative_lsi(selected, N_COMPS, total_features=selected.n_vars, iterations=3,
                        n_features=150, cluster_fn=oracle)
    assert len(seen) == 2 and all(s[0] == selected.n_obs and s[1] >= 2 for s in seen)
    for it in (1, 2):
        np.testing.assert_array_equal(
            selected.obs[f"iterative_lsi_clusters_iter{it}"].astype(int).to_numpy(), labels)
    assert selected.uns["iterative_lsi"]["params"]["cluster_fn_supplied"]

    tile = selected.var_names.to_series().str.extract(r"(\w+):(\d+)-(\d+)").astype({1: int, 2: int})
    owner = np.full(selected.n_vars, -1)
    for h in world["hotspots"].itertuples():  # fragments reach ~350 bp either side of a centre
        near = (tile[0] == h.chrom) & (tile[1] < h.centre + 350) & (tile[2] > h.centre - 350)
        owner[near.to_numpy()] = h.group
    chosen = selected.var["selected_iterative_lsi"].to_numpy()
    assert chosen.sum() == 150
    assert (owner[chosen] == 0).mean() > 0.5, np.bincount(owner[chosen] + 1)


def test_cluster_params_reach_the_clusterer_and_are_recorded(selected, monkeypatch):
    """Every ``snn_cluster`` keyword is reachable through ``cluster_params`` and recorded under ``cluster_``; unknown keys raise."""
    import gatac.dr._clustering as clustering

    got = {}

    def spy(z, **kw):
        got.update(kw)
        return _true_labels(selected)

    monkeypatch.setattr(clustering, "snn_cluster", spy)
    override = {"resolution": 0.7, "quantize": 0, "max_resolution_retries": 1}
    ga.dr.iterative_lsi(selected, N_COMPS, total_features=selected.n_vars, n_features=300, cluster_params=override, random_state=5)
    assert {k: got[k] for k in override} == override and got["random_state"] == 5
    params = selected.uns["iterative_lsi"]["params"]
    assert {k: params[f"cluster_{k}"] for k in override} == override

    with pytest.raises(ValueError, match="Unknown cluster_params"):
        ga.dr.iterative_lsi(selected, N_COMPS, total_features=selected.n_vars, cluster_params={"resolutoin": 1.0})
