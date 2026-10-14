"""GPU motif scan of tl.bindetect against MOODS itself (the scanner TOBIAS uses)."""
import numpy as np
import pytest

from gatac.tl.bindetect import _moods_order, _scan, _short_run_hits, _thresholds
from gatac.tl.footprint import _merge

MOODS = pytest.importorskip("MOODS.scan")


def _random_case(seed):
    rng = np.random.default_rng(seed)
    # ACGT with N runs of 1-12 bp: many runs shorter than the 7-bp MOODS window
    seq = rng.integers(0, 4, 30_000).astype(np.uint8)
    for s in rng.integers(0, len(seq) - 12, 1500):
        seq[s:s + rng.integers(1, 13)] = 4
    bg = np.array([0.3, 0.2, 0.2, 0.3])
    mats = []
    for length in [3, 4, 5, 5, 6, 7, 8, 11, 15, 24]:
        p = rng.dirichlet(np.full(4, 0.4), size=length).T
        mats.append(np.log((p + 0.01 * bg[:, None]) / (1 + 0.01)) - np.log(bg[:, None]))
    return seq, bg, mats


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_scan_matches_moods(seed):
    seq, bg, mats = _random_case(seed)
    n = len(mats)
    thr = _thresholds(mats, bg, 1e-3)
    rev = [m[::-1, ::-1].copy() for m in mats]
    allm = mats + rev
    orders = [_moods_order(m, bg) for m in mats] + [_moods_order(m, bg[::-1]) for m in rev]
    t2 = np.concatenate([thr, thr])
    hm, hp, hs = _scan(seq, allm, orders, t2)
    sm, sp, ss = _short_run_hits(seq, allm, orders, t2, n)
    got = set(zip(np.concatenate([hm, sm]).tolist(), np.concatenate([hp, sp]).tolist(),
                  np.concatenate([hs, ss]).tolist()))

    text = "".join("ACGTN"[c] for c in seq)
    want = set()
    for offset, ms, b in ((0, mats, bg), (n, rev, bg[::-1])):      # one scanner per strand, as TOBIAS
        sc = MOODS.Scanner(7)
        sc.set_motifs([m.tolist() for m in ms], b.tolist(), thr.tolist())
        for k, hits in enumerate(sc.scan(text)):
            want |= {(offset + k, h.pos, h.score) for h in hits}
    assert len(want) > 500
    assert got == want


def test_merge_quirk():
    iv = np.array([[0, 100], [10, 50], [40, 60], [200, 300]])
    # TOBIAS: the fused end is the current end, so [10, 50] shortens [0, 100]
    assert _merge(iv, tobias_compat=True).tolist() == [[0, 60], [200, 300]]
    assert _merge(iv, tobias_compat=False).tolist() == [[0, 100], [200, 300]]
