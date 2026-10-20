"""GPU site bookkeeping of tobias.bindetect against its sequential references."""
import cupy as cp
import numpy as np
import scipy.stats

from gatac.tobias.bindetect import (
    _count_kernel, _count_overlaps, _launch, _resolve_kernel, _resolve_overlaps, _sample_changes,
    _sample_changes_exact_null, _segments, _stable_order, _strkey_kernel,
)


def _random_sites(seed, n=200_000, n_mot=40):
    rng = np.random.default_rng(seed)
    motif = np.sort(rng.integers(0, n_mot, n))
    chunk = rng.integers(0, 5, n)
    chrom = rng.integers(0, 3, n)
    start = rng.integers(0, 2_000_000, n)
    o = np.lexsort((start, chrom, chunk, motif))
    motif, chunk, chrom, start = motif[o], chunk[o], chrom[o], start[o]
    end = start + rng.integers(4, 25, n)
    score = np.round(rng.normal(8, 2, n), 2)          # ties on purpose
    return motif, chunk, chrom.astype(np.int32), start, end, score


def test_resolve_kernel_matches_reference():
    motif, chunk, chrom, start, end, score = _random_sites(0)
    gstart = np.concatenate([[0], np.flatnonzero((np.diff(motif) != 0) | (np.diff(chunk) != 0)) + 1, [len(motif)]])
    ref = np.ones(len(motif), np.bool_)
    _resolve_overlaps(gstart, chrom, start, end, score, ref)
    keep = cp.ones(len(motif), cp.uint8)
    _launch(_resolve_kernel, len(gstart) - 1, (cp.asarray(gstart), cp.int64(len(gstart) - 1), cp.asarray(chrom),
                                               cp.asarray(start), cp.asarray(end), cp.asarray(score), keep))
    assert ref.sum() < len(ref)
    assert np.array_equal(keep.get().astype(bool), ref)


def test_count_kernel_matches_reference():
    n_mot = 40
    motif, chunk, chrom, start, end, _ = _random_sites(1, n_mot=n_mot)
    o = np.lexsort((motif, end, start, chrom, chunk))           # loc_sort per chunk, name last
    motif, chunk, chrom, start, end = motif[o], chunk[o], chrom[o], start[o], end[o]
    gstart = np.concatenate([[0], np.flatnonzero(np.diff(chunk) != 0) + 1, [len(chunk)]])
    ov_ref = np.zeros((n_mot, n_mot), np.int64)
    tot_ref = np.zeros(n_mot, np.int64)
    _count_overlaps(gstart, chrom, start, end, motif, n_mot, ov_ref, tot_ref)

    join = np.zeros(len(start), bool)
    join[1:] = (chunk[1:] == chunk[:-1]) & (chrom[1:] == chrom[:-1]) & (motif[1:] == motif[:-1]) & (start[1:] < end[:-1])
    first = np.flatnonzero(~join)
    last = np.append(first[1:], len(start)) - 1
    ov = cp.zeros(n_mot * n_mot, cp.uint64)
    tot = cp.zeros(n_mot, cp.uint64)
    _launch(_count_kernel, len(first), (cp.int64(len(first)), cp.asarray(chunk[first].astype(np.int32)),
                                        cp.asarray(chrom[first]), cp.asarray(start[first]), cp.asarray(end[last]),
                                        cp.asarray(motif[first].astype(np.int32)), cp.int64(n_mot), ov, tot))
    assert ov_ref.sum() > 0
    assert np.array_equal(ov.get().view(np.int64).reshape(n_mot, n_mot), ov_ref)
    assert np.array_equal(tot.get().view(np.int64), tot_ref)


def test_string_order_key():
    rng = np.random.default_rng(2)
    s = np.concatenate([rng.integers(0, 10**rng.integers(1, 10, 5000)), [0, 9, 10, 99, 100, 999999999]])
    e = s + rng.integers(1, 30, len(s))
    hi, lo = _strkey_kernel(cp.asarray(s), cp.asarray(e))
    got = _stable_order(hi, lo).get()
    want = sorted(range(len(s)), key=lambda i: f"{s[i]}_{e[i]}")
    assert [f"{s[i]}_{e[i]}" for i in got] == [f"{s[i]}_{e[i]}" for i in want]


def test_segments():
    key = cp.asarray([3, 3, 5, 5, 5, 9])
    assert _segments(key).get().tolist() == [0, 2, 5, 6]


def test_exact_null_matches_resampled_null():
    # 2 draws per replicate (mean ~ normal, n*var ~ chi2) vs n normal draws per replicate
    rng = np.random.default_rng(3)
    for n in (1, 2, 30, 500):
        exact = _sample_changes_exact_null(n, 0.1, 0.8, seed=4, size=20_000)
        x = rng.normal(0.1, 0.8, (20_000, n))
        direct = (x.mean(axis=1) - 0.1) / ((x.std(axis=1) + 0.8) / 2.0)
        assert scipy.stats.ks_2samp(exact, direct).pvalue > 1e-3
    # the TOBIAS-compatible sampler is the direct procedure on NumPy's legacy stream
    rs = np.random.RandomState(30)
    x = (rs.standard_normal(100 * 30) * 0.8 + 0.1).reshape(100, 30)
    assert np.array_equal(_sample_changes(30, 0.1, 0.8), (x.mean(axis=1) - 0.1) / ((x.std(axis=1) + 0.8) / 2.0))
