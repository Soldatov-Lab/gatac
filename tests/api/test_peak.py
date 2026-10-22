"""
``gatac.peak``: calling, merging and counting peaks, and finding marker peaks.

Checked against the planted truth of the shared synthetic dataset (each
group's fragments pile up on that group's hotspots) and, for counting, against
a brute-force per-peak count. Parity with MACS/SnapATAC2 on real data is the
reproducibility suite's job.
"""
from __future__ import annotations

import gzip

import numpy as np
import pandas as pd
import pytest

import gatac as ga


def _coords(names) -> pd.DataFrame:
    df = pd.Series(names).str.extract(r"^(\w+):(\d+)-(\d+)$")
    df.columns = ["chrom", "start", "end"]
    return df.astype({"start": np.int64, "end": np.int64})


def _hotspot_at(coords: pd.DataFrame, hot: pd.DataFrame) -> np.ndarray:
    """Row of ``hot`` whose centre lies in each interval, -1 if none."""
    out = np.full(len(coords), -1)
    for i, r in enumerate(coords.itertuples()):
        h = np.flatnonzero((hot.chrom == r.chrom) & (hot.centre >= r.start) & (hot.centre < r.end))
        if len(h):
            out[i] = h[0]
    return out


def _hotspot_group(coords: pd.DataFrame, hot: pd.DataFrame) -> np.ndarray:
    """Group of the hotspot inside each interval, -1 if none."""
    at = _hotspot_at(coords, hot)
    return np.where(at >= 0, hot.group.to_numpy()[at], -1)


def _brute_force_counts(frags: pd.DataFrame, cells, peaks: pd.DataFrame, strategy: str) -> np.ndarray:
    """Cell × peak counts, one peak at a time, so overlapping peaks are each counted in full."""
    row = pd.Index(cells).get_indexer(frags.barcode)
    frags = frags[row >= 0].assign(row=row[row >= 0])
    longest = int((frags.end - frags.start).max())
    by_chrom = {c: f.sort_values("start") for c, f in frags.groupby("chrom")}
    out = np.zeros((len(cells), len(peaks)), np.int64)
    for j, p in enumerate(peaks.itertuples()):
        f = by_chrom[p.chrom]
        lo, hi = np.searchsorted(f.start.to_numpy(), [p.start - longest, p.end])
        f = f.iloc[lo:hi]
        in1 = (f.start >= p.start) & (f.start < p.end)
        in2 = (f.end - 1 >= p.start) & (f.end - 1 < p.end)
        n = {"insertion": in1.astype(int) + in2.astype(int),
             "paired-insertion": (in1 | in2).astype(int),
             "fragment": ((f.start < p.end) & (f.end > p.start)).astype(int)}[strategy]
        np.add.at(out[:, j], f.row.to_numpy(), n.to_numpy())
    return out


@pytest.fixture(scope="module")
def chrom_sizes(world):
    from pyfaidx import Fasta
    return {c: len(s) for c, s in Fasta(str(world["fasta"])).items()}


@pytest.fixture(scope="module")
def split_parquets(world, tmp_path_factory):
    """The same fragments in two files, S1.parquet and S2.parquet, every group spread over both."""
    d = tmp_path_factory.mktemp("split")
    df = world["fragments"]
    barcodes = pd.Index(df.barcode.unique())
    odd = barcodes.str.extract(r"c(\d+)-")[0].astype(int).to_numpy() % 2 == 1
    source = pd.Series(np.where(odd, "S2_tile.h5ad", "S1_tile.h5ad"), index=barcodes)
    for name in ("S1", "S2"):
        with gzip.open(d / f"{name}.tsv.gz", "wt") as fh:
            df[source.loc[df.barcode].to_numpy() == f"{name}_tile.h5ad"].to_csv(
                fh, sep="\t", header=False, index=False)
        ga.io.make_parquet(d / f"{name}.tsv.gz", d / f"{name}.parquet")
    return d, source


# --------------------------------------------------------------------------- call


def test_call_finds_each_groups_own_hotspots(tiles, world, chrom_sizes):
    """Peaks are called per group: g's peaks sit on g's hotspots, cover all of them, and summit near the centre."""
    hot = world["hotspots"]
    out = ga.peak.call(tiles, "group", world["parquet"], genome=chrom_sizes, inplace=False, verbose=False)
    assert "gmacs" not in tiles.uns
    assert sorted(out) == ["g0", "g1", "g2"]
    for g, df in out.items():
        at = _hotspot_at(df[["chrom", "start", "end"]], hot)
        assert (at >= 0).all() and (hot.group.to_numpy()[at] == int(g[1])).all(), f"{g}: peaks off its hotspots"
        assert set(at) == set(np.flatnonzero(hot.group == int(g[1]))), f"{g}: hotspots without a peak"
        summit = df.start.to_numpy() + df.peak.to_numpy()
        assert np.abs(summit - hot.centre.to_numpy()[at]).max() < 250


def test_call_is_independent_of_chromosome_batches(tiles, world, chrom_sizes, monkeypatch):
    """Calling one chromosome per batch (a tiny GPU budget) gives the peaks of a single batch."""
    from gatac.peak import _macs
    one = ga.peak.call(tiles, "group", world["parquet"], genome=chrom_sizes, inplace=False, verbose=False)
    callpeak = _macs.callpeak
    monkeypatch.setattr(_macs, "callpeak", lambda f, p, budget_bytes=None: callpeak(f, p, budget_bytes=1))
    many = ga.peak.call(tiles, "group", world["parquet"], genome=chrom_sizes, inplace=False, verbose=False)
    for g in one:
        pd.testing.assert_frame_equal(one[g], many[g])


def test_call_is_independent_of_read_passes(tiles, world, split_parquets, chrom_sizes):
    """Reading each group in its own pass over the files (no host memory to share) gives the same peaks, in group order."""
    d, source = split_parquets
    tiles.obs["source_file"] = source.loc[tiles.obs_names].to_numpy()
    one = ga.peak.call(tiles, "group", d, genome=chrom_sizes, inplace=False, verbose=False)
    many = ga.peak.call(tiles, "group", d, genome=chrom_sizes, inplace=False, verbose=False, host_memory_gb=0)
    assert list(many) == list(one)
    for g in one:
        pd.testing.assert_frame_equal(one[g], many[g])


def test_group_passes_fit_the_cap():
    """Groups are packed into passes of at most ``cap`` fragments; a group above it gets its own pass."""
    from gatac.peak.peaks import _group_passes
    n = np.array([5, 1, 7, 3, 12, 2])
    passes = _group_passes(n, 8)
    assert sorted(i for p in passes for i in p) == list(range(n.size))
    assert [12] in [list(n[p]) for p in passes]
    assert all(n[p].sum() <= 8 for p in passes if len(p) > 1)
    assert len(passes) == 4


def test_call_without_summits_gives_one_row_per_peak(tiles, world, chrom_sizes):
    """``call_summits=False``: the same peaks, each once, with its summit inside it."""
    sub = ga.peak.call(tiles, "group", world["parquet"], genome=chrom_sizes, inplace=False, verbose=False)
    single = ga.peak.call(tiles, "group", world["parquet"], genome=chrom_sizes, inplace=False, verbose=False,
                          call_summits=False)
    for g, df in single.items():
        key = ["chrom", "start", "end"]
        assert not df.duplicated(key).any()
        assert set(map(tuple, df[key].to_numpy())) == set(map(tuple, sub[g][key].to_numpy()))
        assert ((df.peak >= 0) & (df.peak < df.end - df.start)).all()


def test_call_from_directory_matches_single_file(tiles, world, split_parquets, chrom_sizes):
    """Directory mode reads each cell from its ``source_file`` and gives the peaks of the single combined file."""
    d, source = split_parquets
    tiles.obs["source_file"] = source.loc[tiles.obs_names].to_numpy()
    one = ga.peak.call(tiles, "group", world["parquet"], genome=chrom_sizes, inplace=False, verbose=False)
    many = ga.peak.call(tiles, "group", d, genome=chrom_sizes, inplace=False, verbose=False)
    for g in one:
        pd.testing.assert_frame_equal(one[g].reset_index(drop=True), many[g].reset_index(drop=True))


# -------------------------------------------------------------------------- merge


def test_merge_keeps_most_significant_of_each_overlap_chain(chrom_sizes):
    """Fixed width 2*half_width+1 around the summit; iteratively keep the best and drop what overlaps it."""
    # a—b and b—c overlap at half_width=50, a and c do not; b is weakest, so a and c survive.
    # d overlaps nothing but sits on another group; e duplicates c's summit with a lower score.
    g1 = pd.DataFrame({"chrom": ["chr1", "chr1"], "start": [1_000, 1_160], "end": [1_100, 1_260],
                       "peak": [50, 50], "p_value": [9.0, 8.0]})          # a: summit 1050, c: 1210
    g2 = pd.DataFrame({"chrom": ["chr1", "chr2", "chr1"], "start": [1_080, 5_000, 1_200], "end": [1_180, 5_100, 1_220],
                       "peak": [50, 50, 10], "p_value": [3.0, 1.0, 2.0]})  # b: 1130, d: chr2 5050, e: 1210
    out = ga.peak.merge({"g1": g1, "g2": g2}, chrom_sizes=chrom_sizes, half_width=50)
    got = sorted(zip(out.chrom, out.start, out.end, out.p_value))
    assert got == [("chr1", 1_000, 1_101, 9.0), ("chr1", 1_160, 1_261, 8.0), ("chr2", 5_000, 5_101, 1.0)]


def test_merge_of_called_peaks_is_non_overlapping_and_covers_every_call(peaks, chrom_sizes):
    """No two merged peaks overlap, and every called peak was dropped only for an overlapping, more significant one."""
    out = ga.peak.merge(peaks.uns["gmacs"], chrom_sizes=chrom_sizes)
    out = out.sort_values(["chrom", "start"])
    for _, m in out.groupby("chrom"):
        assert (m.start.to_numpy()[1:] >= m.end.to_numpy()[:-1]).all()
    for df in peaks.uns["gmacs"].values():
        for r in df.itertuples():
            s = r.start + r.peak
            cover = out[(out.chrom == r.chrom) & (out.start < s + 251) & (out.end > s - 250)]
            assert len(cover) and cover.p_value.max() >= r.p_value


# -------------------------------------------------------------------- make_matrix


@pytest.mark.parametrize("strategy", ["paired-insertion", "insertion", "fragment"])
def test_make_matrix_counts_as_documented(peaks, world, strategy, chrom_sizes):
    """Each counting strategy counts exactly what its docstring says, per cell and peak."""
    merged = ga.peak.merge(peaks.uns["gmacs"], chrom_sizes=chrom_sizes)
    peaks.uns["peaks"] = merged
    got = ga.peak.make_matrix(peaks, world["parquet"], genome=chrom_sizes, counting_strategy=strategy,
                              verbose=False)
    expected = _brute_force_counts(world["fragments"], got.obs_names, _coords(got.var_names), strategy)
    np.testing.assert_array_equal(got.X.toarray(), expected)


def test_make_matrix_counts_overlapping_peaks_from_file(peaks, world, tmp_path, chrom_sizes):
    """A BED of overlapping peaks (e.g. from another caller) gets every peak counted in full."""
    merged = _coords(peaks.var_names)
    shifted = merged.assign(start=merged.start + 200, end=merged.end + 200)
    bed = tmp_path / "overlapping.bed"
    pd.concat([merged, shifted]).to_csv(bed, sep="\t", header=False, index=False)
    got = ga.peak.make_matrix(peaks, world["parquet"], peak_file=bed, use_rep=None, genome=chrom_sizes,
                              verbose=False)
    expected = _brute_force_counts(world["fragments"], got.obs_names, _coords(got.var_names), "paired-insertion")
    np.testing.assert_array_equal(got.X.toarray(), expected)


def test_make_matrix_peak_file_takes_precedence(peaks, world, tmp_path, chrom_sizes):
    """``peak_file``: "If provided, peak information will be read from this file instead of use_rep"."""
    bed = tmp_path / "peaks.bed"
    _coords(peaks.var_names).to_csv(bed, sep="\t", header=False, index=False)
    got = ga.peak.make_matrix(peaks, world["parquet"], peak_file=bed, genome=chrom_sizes, verbose=False)
    assert list(got.var_names) == list(peaks.var_names)


@pytest.mark.parametrize("mode", ["directory", "list"])
def test_make_matrix_from_several_files_matches_single_file(peaks, split_parquets, mode, chrom_sizes):
    """Cells spread over several fragment files give the same matrix as one combined file."""
    d, source = split_parquets
    peaks.uns["peaks"] = _coords(peaks.var_names)
    peaks.obs["source_file"] = source.loc[peaks.obs_names].to_numpy()
    path = d if mode == "directory" else [d / "S1.parquet", d / "S2.parquet"]
    got = ga.peak.make_matrix(peaks, path, genome=chrom_sizes, verbose=False)
    assert list(got.var_names) == list(peaks.var_names)
    assert (got.X != peaks.X).nnz == 0


# ------------------------------------------------------------------------ markers


@pytest.mark.parametrize("reference", ["rest", "g1"])
def test_markers_enriched_peaks_are_the_groups_hotspots(peaks, world, reference):
    """Significantly enriched peaks of g0 are exactly g0's hotspots; against one group, only that group's are depleted."""
    owner = _hotspot_group(_coords(peaks.var_names), world["hotspots"])
    res = ga.peak.markers(peaks, "group", groups=["g0"], reference=reference)["g0"]
    # ArchR's binomial test treats the background rate as known, so background noise between two
    # 80-cell groups reaches FDR ~1e-5; planted hotspots sit near 1e-50.
    sig = res[res.fdr < 1e-10]
    up = owner[peaks.var_names.get_indexer(sig.feature[sig.log2_fc > 0])]
    down = owner[peaks.var_names.get_indexer(sig.feature[sig.log2_fc < 0])]
    assert (up == 0).all() and len(up) == (owner == 0).sum()
    assert set(down) == ({1, 2} if reference == "rest" else {1})
