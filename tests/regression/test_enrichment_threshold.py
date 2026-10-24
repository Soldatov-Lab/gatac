"""`motif.enrichment` turns its p-value into a score threshold as SnapATAC2 does."""
import numpy as np

from gatac.motif import DNAMotif
from gatac.motif.motif import _compute_score_threshold_jit, _snapatac2_score_threshold_jit

EVEN = np.full(4, 0.25)

# The motif and expected threshold of SnapATAC2 2.9's own ScoreCDF unit test
# (snapatac2-core/src/motif.rs, `it_works`)
ASCCAGGCKGG = np.array([
    [0.768791, 0.07577, 0.120456, 0.034983],
    [0.057276, 0.532522375, 0.282802875, 0.12739862500000002],
    [0.048894125, 0.796682375, 0.09644475, 0.05797825],
    [0.044861750000000006, 0.6375725, 0.263928, 0.05363775],
    [0.61916725, 0.104690625, 0.17939824999999998, 0.096743875],
    [0.086404375, 0.0745045, 0.807213125, 0.031878125],
    [0.08837187499999999, 0.07052475, 0.828965875, 0.012137499999999999],
    [0.039856875, 0.80558425, 0.08105799999999999, 0.07350025],
    [0.079739875, 0.110598625, 0.40112975, 0.40853125],
    [0.03881925, 0.12873025, 0.781449, 0.05100175],
    [0.136913, 0.15827100000000002, 0.5818685, 0.1229465],
])


def test_threshold_matches_snapatac2_unit_test():
    thr = _snapatac2_score_threshold_jit(np.log(ASCCAGGCKGG / EVEN), EVEN, 1e-4)
    assert abs(thr - 7.009906220318511) < 1e-3


def test_coarse_motif_admits_one_mismatch():
    # Nine near-certain Ts: P(consensus) = 0.25**9 < 1e-5 < P(<= 1 mismatch),
    # and SnapATAC2's interpolated threshold falls just below the 1-mismatch score
    motif = DNAMotif("polyT", np.tile([0.0013, 0.0013, 0.0013, 0.9961], (9, 1)))
    lo = motif.to_log_odds(mode="snapatac2")
    best = lo[:, 3].sum()
    one_off = best - lo[0, 3] + lo[0, 0]
    thr = _snapatac2_score_threshold_jit(lo, EVEN, 1e-5)
    assert one_off >= thr
    assert one_off - lo[1, 3] + lo[1, 0] < thr
    # the right-tail threshold used for chromVAR keeps the consensus only
    assert _compute_score_threshold_jit(lo, EVEN, 1e-5, 1e-4) > one_off


def test_snapatac2_pseudocount_replaces_zeros():
    motif = DNAMotif("m", np.array([[1.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0]]))
    prob = np.exp(motif.to_log_odds(mode="snapatac2")) * 0.25
    np.testing.assert_allclose(prob[0], np.array([1, 1e-4, 1e-4, 1e-4]) / (1 + 3e-4))
    np.testing.assert_allclose(prob[1], np.array([0.5, 0.5, 1e-4, 1e-4]) / (1 + 2e-4))


def test_snapatac2_pseudocount_finds_zeros_in_rows_summing_under_one():
    # MEME rows are rounded, so a zero can sit in a row summing to 0.999; the
    # 1e-10 floor then grows a little on renormalisation and must still count
    motif = DNAMotif("m", np.array([[0.333, 0.333, 0.333, 0.0]]))
    assert motif.pwm[0, 3] > 1e-10
    prob = np.exp(motif.to_log_odds(mode="snapatac2")) * 0.25
    assert abs(prob[0, 3] - 1e-4) < 1e-6
