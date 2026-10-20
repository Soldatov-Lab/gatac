"""
``gatac.motif`` against the planted truth of the synthetic world.

Motif ``M{g}`` sits at the centre of every group-``g`` hotspot and ``DECOY``
nowhere, so each function can be judged on whether it recovers that, not only
on whether it runs. Parity with chromVAR, motifmatchr and GSEApy is the
reproducibility suite's job; the logistic fitter's numerics are guarded in
``tests/regression/test_motif_regression.py``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import anndata as ad

import gatac as ga

N_GROUPS = 3          # as in tests/conftest.py
PLANTED = [f"M{g}" for g in range(N_GROUPS)]


def _region_group(var_names, hotspots) -> np.ndarray:
    """The group whose hotspot centre lies in each ``chr:start-end`` region, -1 if none."""
    out = np.full(len(var_names), -1)
    for i, r in enumerate(var_names):
        chrom, se = r.split(":")
        s, e = map(int, se.split("-"))
        h = hotspots[(hotspots.chrom == chrom) & (hotspots.centre >= s) & (hotspots.centre < e)]
        if len(h):
            out[i] = h.group.iloc[0]
    return out


def _log2fc(peaks, g) -> np.ndarray:
    X = peaks.X.tocsr() if sp.issparse(peaks.X) else sp.csr_matrix(peaks.X)
    inside = (peaks.obs["group"] == f"g{g}").to_numpy()
    a = np.asarray(X[inside].mean(0)).ravel()
    b = np.asarray(X[~inside].mean(0)).ravel()
    return np.log2((a + 1e-3) / (b + 1e-3))


@pytest.fixture(scope="module")
def motifs(world):
    return ga.motif.read(world["meme"])


@pytest.fixture(scope="module")
def truth(_peaks, world):
    return _region_group(_peaks.var_names, world["hotspots"])


def test_scan_finds_planted_motifs_only_where_planted(peaks, motifs, world, truth):
    """``scan`` marks a peak for a motif exactly when that motif sits in the peak's sequence."""
    pvalue = 5e-5
    ga.motif.scan(peaks, motifs, world["fasta"], pvalue=pvalue)
    hits = peaks.varm["motif_match"].toarray()
    names = list(peaks.uns["motif_name"])
    # Where a motif is absent, a peak is still hit by chance at the scan's
    # p-value threshold: both strands of every window.
    width = int(np.diff([int(x) for x in peaks.var_names[0].split(":")[1].split("-")])[0])
    chance = 1 - (1 - pvalue) ** (2 * (width - len(motifs[0]) + 1))
    for g in range(N_GROUPS):
        col = hits[:, names.index(f"M{g}")]
        assert col[truth == g].all(), f"M{g} missed in some group-{g} peaks"
        assert col[truth != g].mean() < 3 * chance, f"M{g} called in other groups' peaks"
    assert hits[:, names.index("DECOY")].mean() < 3 * chance


def test_scan_coordinate_system_shifts_extraction_by_one_base(motifs, world):
    """``"1-based"`` reads ``genome[start-1:end]``, ``"0-based"`` reads ``genome[start:end]``."""
    hot = world["hotspots"]
    hot = hot[hot.group == 0].head(30)
    # In 0-based terms each region starts one base after the planted motif, so
    # only the 1-based reading, which starts one base earlier, contains it whole.
    m0 = [m for m in motifs if m.name == "M0"]
    L = len(m0[0])
    s = hot.centre - L // 2
    names = [f"{c}:{a + 1}-{a + L + 20}" for c, a in zip(hot.chrom, s)]
    found = {}
    for cs in ("0-based", "1-based"):
        a = ad.AnnData(sp.csr_matrix((1, len(names)), dtype=np.float32),
                       var=pd.DataFrame(index=names))
        # A strict threshold: M0 is the near-palindromic AP-1 site, and at the
        # default p-value a window half-overlapping it on the other strand scores in.
        ga.motif.scan(a, m0, world["fasta"], bg="even", pvalue=1e-6, coordinate_system=cs)
        found[cs] = a.varm["motif_match"].toarray()[:, 0]
    assert found["1-based"].all()
    assert not found["0-based"].any()


def test_enrichment_ranks_each_groups_motif_first(motifs, world, _peaks, truth):
    """Group-``g`` regions tested against the union of all groups return ``M{g}`` enriched and nothing else."""
    regions = {f"g{g}": list(_peaks.var_names[truth == g]) for g in range(N_GROUPS)}
    res = ga.motif.enrichment(motifs, regions, world["fasta"])
    for g in range(N_GROUPS):
        df = res[f"g{g}"].set_index("name")
        assert df.loc[f"M{g}", "adjusted p-value"] < 1e-10
        assert df.loc[f"M{g}", "log2(fold change)"] > 1
        others = [m for m in PLANTED if m != f"M{g}"]
        assert (df.loc[others, "log2(fold change)"] < 0).all()
        assert not df.loc["DECOY", "adjusted p-value"] < 0.05


@pytest.fixture(scope="module")
def regression_inputs(_peaks):
    return list(_peaks.var_names), _log2fc(_peaks, 1)


def test_enrichment_regression_recovers_score_motif(motifs, world, regression_inputs):
    """A score high in group-1 peaks gives ``M1`` a positive, significant coefficient and the others negative."""
    regions, score = regression_inputs
    res = ga.motif.enrichment_regression(motifs, regions, score, world["fasta"]).set_index("name")
    assert res.loc["M1", "coefficient"] > 0 and res.loc["M1", "adjusted p-value"] < 1e-3
    assert (res.loc[["M0", "M2"], "coefficient"] < 0).all()


def test_enrichment_regression_precomputed_hits_equal_fresh_scan(motifs, world, regression_inputs):
    """A ``presence_matrix`` over the same regions and motifs reproduces the scanning call exactly."""
    regions, score = regression_inputs
    fresh = ga.motif.enrichment_regression(motifs, regions, score, world["fasta"])
    hits = ga.motif.presence_matrix(motifs, regions, world["fasta"])
    cached = ga.motif.enrichment_regression(motifs, regions, score, world["fasta"],
                                            precomputed_hits=hits)
    pd.testing.assert_frame_equal(fresh, cached)


@pytest.mark.parametrize("mismatch", ["pvalue", "region_order"])
def test_enrichment_regression_refuses_mismatched_cache(motifs, world, regression_inputs, mismatch):
    """A cache built for another threshold or region order raises instead of giving wrong coefficients."""
    regions, score = regression_inputs
    if mismatch == "pvalue":
        hits = ga.motif.presence_matrix(motifs, regions, world["fasta"], pvalue=1e-4)
    else:
        hits = ga.motif.presence_matrix(motifs, regions[::-1], world["fasta"])
    with pytest.raises(ValueError, match="does not match"):
        ga.motif.enrichment_regression(motifs, regions, score, world["fasta"], precomputed_hits=hits)


def test_enrichment_regression_drops_nonfinite_scores(motifs, world, regression_inputs):
    """Non-finite scores are dropped with their regions: same result as never passing those regions."""
    regions, score = regression_inputs
    score = score.copy()
    bad = np.arange(0, len(score), 7)
    score[bad[::2]] = np.nan
    score[bad[1::2]] = np.inf
    keep = np.isfinite(score)
    got = ga.motif.enrichment_regression(motifs, regions, score, world["fasta"])
    ref = ga.motif.enrichment_regression(motifs, [r for r, k in zip(regions, keep) if k],
                                         score[keep], world["fasta"])
    pd.testing.assert_frame_equal(got, ref)


@pytest.fixture(scope="module")
def gc_world(tmp_path_factory):
    """One contig of 500 bp segments whose GC fraction ranges 0.25-0.75."""
    rng = np.random.default_rng(5)
    gc = rng.uniform(0.25, 0.75, 1_000)
    seq = "".join("".join(rng.choice(list("GCAT"), 500, p=[f / 2, f / 2, (1 - f) / 2, (1 - f) / 2]))
                  for f in gc)
    fasta = tmp_path_factory.mktemp("gc") / "gc.fa"
    fasta.write_text(">chrA\n" + "\n".join(seq[i:i + 60] for i in range(0, len(seq), 60)) + "\n")
    regions = [f"chrA:{i * 500}-{(i + 1) * 500}" for i in range(1_000)]
    actual = np.array([(seq[i * 500:(i + 1) * 500].count("G") + seq[i * 500:(i + 1) * 500].count("C")) / 500
                       for i in range(1_000)])
    return {"fasta": fasta, "regions": regions, "gc": actual, "seq": seq}


def test_gc_matched_background_matches_target_gc(gc_world):
    """Sampled background has the targets' GC distribution, unique without replacement, fixed by the seed."""
    gc = gc_world["gc"]
    regions = np.array(gc_world["regions"])
    # Targets are removed from the pool, so leave GC-rich candidates in it.
    rich = np.flatnonzero(gc > 0.6)
    target = regions[rich[::2]]
    kw = dict(background_pool=list(regions), replace=False, random_state=0)
    bg = ga.motif.sample_gc_matched_background(list(target), gc_world["fasta"], **kw)
    assert len(bg) == len(target) and len(set(bg)) == len(bg)
    lookup = dict(zip(regions, gc))
    got = np.array([lookup[r] for r in bg])
    want = gc[rich[::2]]
    assert abs(np.median(got) - np.median(want)) < 0.02
    assert abs(np.median(gc) - np.median(want)) > 0.1    # the pool alone would not match
    assert bg == ga.motif.sample_gc_matched_background(list(target), gc_world["fasta"], **kw)


def test_compute_peak_bias_gc_is_exact_for_unequal_widths(gc_world):
    """``gc_content`` is the G+C fraction of each region's own sequence, whatever its width."""
    rng = np.random.default_rng(0)
    starts = rng.integers(0, 490_000, 300)
    ends = starts + rng.integers(20, 5_000, 300)
    names = [f"chrA:{s}-{e}" for s, e in zip(starts, ends)]
    a = ad.AnnData(sp.csr_matrix((1, len(names)), dtype=np.float32), var=pd.DataFrame(index=names))
    ga.motif.compute_peak_bias(a, gc_world["fasta"])
    seq = gc_world["seq"]
    want = [(seq[s:e].count("G") + seq[s:e].count("C")) / (e - s) for s, e in zip(starts, ends)]
    np.testing.assert_allclose(a.var["gc_content"].to_numpy(), want, rtol=1e-6)


def test_knn_background_peaks_share_the_bias_of_their_peak(peaks, world):
    """Each peak's k-NN background is close to it in GC and accessibility, far closer than random peaks."""
    ga.motif.sample_bg_peaks(peaks, method="knn", genome_fasta=world["fasta"])
    bg = np.asarray(peaks.varm["bg_peaks"])
    assert bg.shape[0] == peaks.n_vars
    for col in ("gc_content", "reads_per_peak"):
        v = peaks.var[col].to_numpy(float)
        if col == "reads_per_peak":
            v = np.log1p(v)
        to_bg = np.median(np.abs(v[bg] - v[:, None]))
        to_any = np.median(np.abs(v[:, None] - v[None, :]))
        assert to_bg < 0.5 * to_any, col


@pytest.fixture(scope="module")
def deviations(_peaks, world):
    a = _peaks.copy()
    ga.motif.chromvar(a, world["fasta"], world["meme"], method="knn")
    return a


def test_chromvar_deviation_is_highest_in_the_motifs_group(deviations):
    """Cells of group ``g`` have the highest ``M{g}`` deviation of all groups, by a wide margin."""
    dev = deviations.obsm["chromvar"]
    group = deviations.obs["group"].to_numpy()
    for g in range(N_GROUPS):
        means = {h: dev.loc[group == f"g{h}", f"M{g}"].mean() for h in range(N_GROUPS)}
        assert max(means, key=means.get) == g, means
        assert means[g] - max(v for h, v in means.items() if h != g) > 1, means


def test_compute_deviations_independent_of_batching(deviations):
    """``batch_size`` and ``motif_batch_size`` trade memory for speed and do not change the scores."""
    a = deviations.copy()
    ga.motif.compute_deviations(a, batch_size=37, motif_batch_size=1, key_added="small")
    np.testing.assert_allclose(a.obsm["small"].to_numpy(), deviations.obsm["chromvar"].to_numpy(),
                               rtol=1e-5, atol=1e-6)


def test_gsea_puts_the_groups_motif_on_top(deviations):
    """Ranking peaks by group-``g`` log2FC makes ``M{g}`` the top, significant motif; same seed, same table."""
    rankings = {f"g{g}": pd.DataFrame({"log2fc": _log2fc(deviations, g)}, index=deviations.var_names)
                for g in range(N_GROUPS)}
    res = ga.motif.gsea(deviations, rankings, permutation_num=200, min_size=5)
    for g in range(N_GROUPS):
        df = res[f"g{g}"]
        assert df.iloc[0]["motif"] == f"M{g}" and df.iloc[0]["fdr"] < 0.05
        others = df.set_index("motif").loc[[m for m in PLANTED if m != f"M{g}"], "NES"]
        assert (others < 0).all()
    again = ga.motif.gsea(deviations, rankings["g0"], permutation_num=200, min_size=5)
    pd.testing.assert_frame_equal(res["g0"].reset_index(drop=True), again.reset_index(drop=True))
