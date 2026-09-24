"""AMULET per-cell Poisson scoring edge cases."""
import numpy as np

from gatac.pp.amulet import _get_doublets


def test_zero_overlap_cells_are_never_doublets_when_overlaps_are_sparse():
    # Shallow sample: one overlap among 3,602 cells (CAC_KG607Li). The mean
    # collapses to ~2.8e-4 and, without the guard, every zero-overlap cell got
    # p ~= 2.8e-4 < 0.01 and the whole sample was called doublets.
    colsum = np.zeros(3602, dtype=np.int64)
    colsum[0] = 1
    res = _get_doublets(colsum, np.array([f"c{i}" for i in range(len(colsum))]))
    assert (res.loc[colsum == 0, "p_value"] == 1.0).all()
    # Only the cell that has an overlap can still be called.
    assert set(res.index[res["q_value"] < 0.01]) <= {0}


def test_typical_depth_is_unchanged_apart_from_zero_cells():
    rng = np.random.default_rng(0)
    colsum = rng.poisson(8, size=5000)
    colsum[:20] = 60  # planted multiplets
    colsum[20:25] = 0
    res = _get_doublets(colsum, np.array([f"c{i}" for i in range(len(colsum))]))
    called = set(res.index[res["q_value"] < 0.01])
    assert set(range(20)) <= called
    assert not called & set(range(20, 25))
