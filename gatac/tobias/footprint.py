"""
GPU footprinting: TOBIAS ATACorrect (Tn5 bias correction) and ScoreBigwig (footprint scores).

Reimplements TOBIAS 0.17.5 ``ATACorrect`` + ``ScoreBigwig --score footprint`` on
pseudobulks built straight from a GATAC fragments parquet, with no BAM or bigwig
round trip. Semantics follow TOBIAS line by line; only the execution model
differs: whole-chromosome arrays on the device instead of per-region Python
loops in a multiprocessing pool.

* Dinucleotide weight matrix (DWM) bias model: k-mer counts as one GEMM.
* Bias scores of every 25-mer as one GEMM.
* The ~10⁶ per-window ``scipy.optimize.curve_fit(relu, ...)`` calls run as one
  CUDA kernel compiled from SciPy's MINPACK C code (``_minpack_relu.cu``) with
  ``--fmad=false``, so every fit rounds like the x86-64 build.
* Rolling sums keep TOBIAS's float32 accumulator; footprint scores keep its
  float32 arithmetic.

Given the same bias scores, every track matches TOBIAS bit for bit. The DWM bias
score itself differs in its last bits (GEMM summation order), which is ~2x below
TOBIAS's own sensitivity to a 1e-13 perturbation of that score.

``tobias_compat=True`` (default) also reproduces TOBIAS behaviours that are
arguably bugs:

* the bias model depends on the read length of the BAM TOBIAS would have read
  (``read_length``, default 50 bp as for 10x); with ``tobias_compat=False`` each
  cut is counted once;
* ``RegionList.merge`` sets a fused interval's end to the *current* end, so an
  interval contained in the previous one shortens the merge.
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Optional, Sequence, Union

import cupy as cp
import numpy as np
import pandas as pd

from .._cupy_compat import bincount
from .._readers import read_bed
from ..io.fragments import FragmentFile

if TYPE_CHECKING:
    from anndata import AnnData

logger = logging.getLogger(__name__)

__all__ = ["footprint", "FootprintTracks", "TrackLayout"]

K_FLANK = 12                  # ATACorrect --k_flank
L_KMER = 2 * K_FLANK + 1
BG_SHIFT = 100                # --bg_shift
WINDOW = 100                  # --window
EXTEND = 100                  # --extend (ATACorrect and ScoreBigwig)
SPLIT = 50_000                # RegionList.apply_method(OneRegion.split_region, 50000)
FP_FLANK_MIN, FP_FLANK_MAX = 10, 30
FP_MIN, FP_MAX = 20, 50
TRACKS = ("uncorrected", "bias", "expected", "corrected")

# TOBIAS nucleotide code: A T C G N -> 0 1 2 3 4; complement swaps 0<->1, 2<->3
_LUT = np.full(256, 4, dtype=np.uint8)
for _ch, _v in zip(b"ATCGatcg", [0, 1, 2, 3, 0, 1, 2, 3]):
    _LUT[_ch] = _v
_COMP = np.array([1, 0, 3, 2, 4], dtype=np.uint8)


# =============================================================================
# Genome access
# =============================================================================


def _fai(fasta: Union[str, Path]) -> dict[str, tuple[int, int, int, int]]:
    """``{chrom: (length, byte offset, bases per line, bytes per line)}``; builds the .fai if missing."""
    fasta = Path(fasta)
    if str(fasta).endswith(".gz"):
        raise ValueError("footprinting needs an uncompressed, indexable FASTA")
    fai = Path(str(fasta) + ".fai")
    if not fai.exists():
        from pyfaidx import Fasta

        Fasta(str(fasta))
    out = {}
    for line in open(fai):
        f = line.split("\t")
        out[f[0]] = (int(f[1]), int(f[2]), int(f[3]), int(f[4]))
    return out


def _chrom_bytes(fasta: Union[str, Path], fai: dict, chrom: str) -> np.ndarray:
    """Raw sequence bytes of one chromosome (newlines removed)."""
    length, off, lb, lw = fai[chrom]
    n_bytes = (length // lb) * lw + length % lb
    blk = np.asarray(np.memmap(fasta, dtype=np.uint8, mode="r", offset=off, shape=(n_bytes,)))
    if lw == lb:
        return blk.copy()
    return blk[(blk != 10) & (blk != 13)][:length]


def _region_bytes(fasta, fai: dict, chrom: str, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Concatenated raw bytes of ``[start, end)`` intervals on one chromosome."""
    _, off, lb, lw = fai[chrom]
    lens = ends - starts
    within = np.arange(lens.sum()) - np.repeat(np.cumsum(lens) - lens, lens)
    gpos = np.repeat(starts, lens) + within
    mm = np.memmap(fasta, dtype=np.uint8, mode="r")
    return np.asarray(mm[off + (gpos // lb) * lw + gpos % lb])


# =============================================================================
# Region bookkeeping (host side, cheap)
# =============================================================================


def _merge(iv: np.ndarray, tobias_compat: bool = True) -> np.ndarray:
    """``RegionList.merge``: sort, fuse when ``start < previous end``.

    TOBIAS sets the fused end to the *current* end, so an interval contained in
    the previous one shortens the merge; ``tobias_compat=False`` keeps the max.
    """
    if len(iv) == 0:
        return np.zeros((0, 2), np.int64)
    iv = np.asarray(iv, dtype=np.int64)
    iv = iv[np.lexsort((iv[:, 1], iv[:, 0]))]
    if not tobias_compat:
        run_end = np.maximum.accumulate(iv[:, 1])
        new = np.ones(len(iv), bool)
        new[1:] = iv[1:, 0] >= run_end[:-1]
        first = np.flatnonzero(new)
        last = np.append(first[1:], len(iv)) - 1
        return np.stack([iv[first, 0], run_end[last]], axis=1)
    out = [list(iv[0])]
    for s, e in iv[1:]:
        if s < out[-1][1]:
            out[-1][1] = e
        else:
            out.append([s, e])
    return np.array(out, dtype=np.int64)


def _subtract(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Interval subtraction ``a - b`` (both sorted, ``b`` disjoint)."""
    out = []
    for s, e in a:
        cur = s
        lo = np.searchsorted(b[:, 1], s, side="right")
        for bs, be in b[lo:]:
            if bs >= e:
                break
            if bs > cur:
                out.append([cur, bs])
            cur = max(cur, be)
        if cur < e:
            out.append([cur, e])
    return np.array(out, dtype=np.int64).reshape(-1, 2)


def _split(iv: np.ndarray, max_len: int = SPLIT) -> np.ndarray:
    if len(iv) == 0:
        return np.zeros((0, 2), np.int64)
    n = -(-(iv[:, 1] - iv[:, 0]) // max_len)
    rid = np.repeat(np.arange(len(iv)), n)
    k = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    starts = iv[rid, 0] + k * max_len
    ends = np.minimum(starts + max_len, iv[rid, 1])
    return np.stack([starts, ends], axis=1)


def _atacorrect_regions(peaks: np.ndarray, length: int, tobias_compat: bool) -> dict[str, np.ndarray]:
    """run_atacorrect's region bookkeeping for one chromosome: split peaks and
    non-peaks (bias input) and the split, unextended output regions."""
    flank_extend = K_FLANK + WINDOW // 2
    pk = np.clip(_merge(peaks, tobias_compat), 0, length)
    pk = pk[pk[:, 1] > pk[:, 0]]
    nonpk = _subtract(np.array([[0, length]]), pk)
    out = pk.copy()
    out[:, 0] -= EXTEND + flank_extend
    out[:, 1] += EXTEND + flank_extend
    out = np.clip(_merge(out, tobias_compat), 0, length)
    out[:, 0] += flank_extend
    out[:, 1] -= flank_extend
    out = out[out[:, 1] > out[:, 0]]
    return dict(peaks=_split(pk), nonpeaks=_split(nonpk), output=_split(out))


def _score_regions(peaks: np.ndarray, length: int, tobias_compat: bool) -> np.ndarray:
    """ScoreBigwig regions: merged peaks ± ``EXTEND``, merged, kept ``FP_FLANK_MAX`` from the edges."""
    reg = _merge(peaks, tobias_compat)
    reg[:, 0] -= EXTEND
    reg[:, 1] += EXTEND
    reg = np.clip(_merge(reg, tobias_compat), 0, length)
    reg[:, 0] = np.maximum(reg[:, 0] - FP_FLANK_MAX, 0) + FP_FLANK_MAX
    reg[:, 1] = np.minimum(reg[:, 1] + FP_FLANK_MAX, length) - FP_FLANK_MAX
    return reg[reg[:, 1] > reg[:, 0]]


def _region_members(pos: np.ndarray, lo: np.ndarray, hi: np.ndarray):
    """All (region, position) pairs with ``lo[r] <= p < hi[r]``; ``pos`` sorted, regions may overlap."""
    a = np.searchsorted(pos, lo, side="left")
    b = np.searchsorted(pos, np.maximum(hi, lo), side="left")
    n = b - a
    rid = np.repeat(np.arange(len(lo)), n)
    offs = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    return rid, pos[np.repeat(a, n) + offs]


def _count_in(pos: np.ndarray, iv: np.ndarray) -> int:
    return int((np.searchsorted(pos, iv[:, 1]) - np.searchsorted(pos, iv[:, 0])).sum())


# =============================================================================
# Cut sites
# =============================================================================


def _load_cuts(parquet_path, barcode_groups: dict[str, int], chrom_sizes: dict[str, int],
               read_length: Optional[int], deduplicate: bool) -> dict[int, dict[str, tuple]]:
    """Per group and chromosome: sorted forward and reverse Tn5 cut positions (0-based).

    Forward cut = ``start``, reverse cut = ``end - 1``. With ``read_length`` set,
    fragments whose reads would not fit on the chromosome are dropped (as a BAM
    of ``read_length``-bp mates would). ``deduplicate`` keeps one copy of each
    (chrom, start, end) per group, like a duplicate-free pseudobulk BAM.
    """
    chroms = list(chrom_sizes)
    n_chroms = len(chroms)
    group_of = cp.asarray(np.array(list(barcode_groups.values()), np.int32))
    # Each batch is split on the device by (group, chromosome) and only start and end come to the
    # host, as views per part. No polars here: its allocator kept ~4x the table resident after
    # the frames were freed.
    pieces: dict[tuple[int, int], list] = {}
    for b in FragmentFile(parquet_path).iter_batches(
        ["chrom", "start", "end"], chroms=chroms, cells=list(barcode_groups),
    ):
        key = group_of[b["cell"].values].astype(cp.int64) * n_chroms + b["chrom"].values.astype(cp.int64)
        o = cp.argsort(key)
        key = key[o]
        st, en = b["start"].values[o].get(), b["end"].values[o].get()
        del b, o
        ks, first = cp.unique(key, return_index=True)
        bounds = np.append(first.get(), len(key))
        for k, lo, hi in zip(ks.get().tolist(), bounds[:-1], bounds[1:]):
            pieces.setdefault(divmod(k, n_chroms), []).append((st[lo:hi], en[lo:hi]))
        del key, ks, first
    cuts: dict[int, dict[str, tuple]] = {g: {} for g in set(barcode_groups.values())}
    for (g, ci) in sorted(pieces):
        parts = pieces.pop((g, ci))
        c = chroms[ci]
        length = chrom_sizes[c]
        s = np.concatenate([x for x, _ in parts])
        e = np.concatenate([y for _, y in parts])
        del parts
        if deduplicate:
            # one copy of each (start, end) of the group on this chromosome, by a sort of the
            # packed uint64 keys (a hash-based unique took ~10x the table's memory)
            key = np.unique((s.astype(np.uint64) << np.uint64(32)) | e.astype(np.uint64))
            s, e = (key >> np.uint64(32)).astype(np.int64), (key & np.uint64(0xFFFFFFFF)).astype(np.int64)
            del key
        else:
            s, e = s.astype(np.int64), e.astype(np.int64)
        if read_length is None:
            ok = (s >= 0) & (e <= length)
        else:
            ok = (s - 4 >= 0) & (e + 4 - read_length >= 0) & (e + 4 <= length) & (s - 4 + read_length <= length)
        cuts[g][c] = (np.sort(s[ok]), np.sort(e[ok] - 1))
    empty = np.zeros(0, np.int64)
    for g in cuts:
        for c in chrom_sizes:
            cuts[g].setdefault(c, (empty, empty))
    return cuts


# =============================================================================
# Bias model: DWM counts and scores as GEMMs
# =============================================================================


def _kmers(seq_d: cp.ndarray, centers: cp.ndarray, reverse: bool) -> cp.ndarray:
    k = seq_d[centers[:, None] + cp.arange(-K_FLANK, K_FLANK + 1)[None, :]]
    if reverse:
        k = cp.asarray(_COMP)[k[:, ::-1]]
    return k


def _onehot(kk: cp.ndarray) -> cp.ndarray:
    X = cp.zeros((len(kk), L_KMER * 5), dtype=cp.float64)
    X[cp.arange(len(kk))[:, None], cp.arange(L_KMER)[None, :] * 5 + kk] = 1.0
    return X


def _dwm_counts(kmers: cp.ndarray, w: cp.ndarray, chunk: int = 2_000_000) -> np.ndarray:
    """``counts[Sm, Sn, m, n] = Σ_k w_k [kmer_k[m] == Sm][kmer_k[n] == Sn]``; k-mers with N skipped."""
    keep = ~(kmers == 4).any(axis=1)
    kmers, w = kmers[keep], w[keep].astype(cp.float64)
    G = cp.zeros((L_KMER * 5, L_KMER * 5), dtype=cp.float64)
    for i in range(0, len(kmers), chunk):
        X = _onehot(kmers[i:i + chunk].astype(cp.int64))
        G += (X * w[i:i + chunk, None]).T @ X
    return cp.asnumpy(G.reshape(L_KMER, 5, L_KMER, 5).transpose(1, 3, 0, 2))


def _bias_counts(cuts_fr, regions: dict, seq_d: cp.ndarray, length: int, read_length: Optional[int]):
    """bias_estimation() on one chromosome: unique cut positions per strand and region,
    weight ``min(count, 10)``, foreground k-mer at the cut, background k-mer ``BG_SHIFT``
    upstream (strand-aware).

    TOBIAS fetches reads overlapping each non-peak chunk, then (because extend_reg
    mutates the region) keeps cuts inside the chunk extended by ``K_FLANK + BG_SHIFT``.
    With ``read_length`` set this BAM-level behaviour is reproduced exactly;
    ``None`` counts each cut inside the chunk once.
    """
    shape = (5, 5, L_KMER, L_KMER)
    counts = {s: np.zeros(shape) for s in ("forward", "reverse")}
    bg = {s: np.zeros(shape) for s in ("forward", "reverse")}
    iv = regions["nonpeaks"]
    if len(iv) == 0:
        return counts, bg
    ext = K_FLANK + BG_SHIFT
    ext_s = np.maximum(0, iv[:, 0] - ext)
    ext_e = np.minimum(length, iv[:, 1] + ext)
    for strand, pos in zip(("forward", "reverse"), cuts_fr):
        if read_length is None:
            lo, hi = iv[:, 0], iv[:, 1]
        else:
            if strand == "forward":      # alignment [p-4, p-4+RL) overlaps [s, e)
                lo, hi = iv[:, 0] + 4 - read_length + 1, iv[:, 1] + 4
            else:                        # alignment [p+5-RL, p+5) overlaps [s, e)
                lo, hi = iv[:, 0] - 5 + 1, iv[:, 1] + read_length - 5
            # cutsite (1-based) > ext_start and < ext_end
            lo, hi = np.maximum(lo, ext_s), np.minimum(hi, ext_e - 1)
        rid, p = _region_members(pos, lo, hi)
        if len(p) == 0:
            continue
        new = np.ones(len(p), bool)
        new[1:] = (rid[1:] != rid[:-1]) | (p[1:] != p[:-1])
        starts = np.flatnonzero(new)
        n = np.diff(np.append(starts, len(p)))
        u, ridx = p[starts], rid[starts]
        w = np.minimum(n, 10).astype(np.float64)
        shift = -BG_SHIFT if strand == "forward" else BG_SHIFT
        for target, centers in ((counts, u), (bg, u + shift)):
            # get_kmer bounds: cutsite > seq_start + k + 1 and cutsite <= seq_end - k (1-based)
            valid = (centers + 1 > ext_s[ridx] + K_FLANK + 1) & (centers + 1 <= ext_e[ridx] - K_FLANK)
            km = _kmers(seq_d, cp.asarray(centers[valid]), strand == "reverse")
            target[strand] += _dwm_counts(km, cp.asarray(w[valid]))
    return counts, bg


def _prepare_dwm(counts: np.ndarray, bg_counts: np.ndarray) -> dict[str, np.ndarray]:
    """DiNucleotideMatrix.prepare_mat() for the bias and background models."""
    def one(cnt):
        cm = np.sum(cnt, axis=(0, 2)) / float(L_KMER)
        no = np.mean(np.sum(cm[:4, :], axis=0))
        pwm = np.true_divide(cm + 1, no + 4)
        pseudo = max([16, no])
        dwm = (cnt + pseudo * pwm[None, :, None, :] * pwm[:, None, :, None]) / float(no + pseudo)
        return np.log2(pwm), np.log2(dwm), pwm

    bias_pwm_log, bias_dwm_log, bias_pwm = one(counts)
    bg_pwm_log, bg_dwm_log, bg_pwm = one(bg_counts)
    return dict(counts=counts, bg_counts=bg_counts, bias_pwm_log=bias_pwm_log, bias_dwm_log=bias_dwm_log,
                bg_pwm_log=bg_pwm_log, bg_dwm_log=bg_dwm_log, pssm=np.log2(bias_pwm / bg_pwm))


def _score_tables(pwm_log: np.ndarray, dwm_log: np.ndarray):
    # T[(m, Sm), (n, a)] = DWM[Sm, a, m, n] - PWM[a, n] for m != n, else 0
    T = dwm_log[:, :4, :, :] - pwm_log[None, :4, None, :]       # (Sm, a, m, n)
    T = T.transpose(2, 0, 3, 1).copy()                          # (m, Sm, n, a)
    for m in range(L_KMER):
        T[m, :, m, :] = 0.0
    return cp.asarray(T.reshape(L_KMER * 5, L_KMER * 4)), cp.asarray(pwm_log[:4, :].T.copy())


def _dwm_score_kmers(kmers: cp.ndarray, model: dict, chunk: int = 1_000_000) -> cp.ndarray:
    """DiNucleotideMatrix.score_sequence at each k-mer centre (k-mers without N)."""
    out = cp.empty(len(kmers), dtype=cp.float64)
    tabs = [_score_tables(model["bias_pwm_log"], model["bias_dwm_log"]),
            _score_tables(model["bg_pwm_log"], model["bg_dwm_log"])]
    ar = cp.arange(L_KMER)[None, :]
    for i in range(0, len(kmers), chunk):
        kk = kmers[i:i + chunk].astype(cp.int64)
        X = _onehot(kk)
        k3 = cp.minimum(kk, 3)
        tot = []
        for T, P in tabs:
            R = (X @ T).reshape(len(kk), L_KMER, 4)                   # Σ_{m≠n} DWM - PWM, per (n, a)
            Cn = cp.exp2(R + P[None]).sum(axis=2)
            cond = cp.take_along_axis(R, k3[:, :, None], axis=2)[:, :, 0]
            tot.append((cond + P[ar, k3] - cp.log2(Cn)).sum(axis=1))
        out[i:i + chunk] = tot[0] - tot[1]
    return out


# =============================================================================
# CUDA kernels
# =============================================================================


@functools.cache
def _relu_kernel():
    src = (Path(__file__).parent / "_minpack_relu.cu").read_text()
    return cp.RawModule(code=src, options=("--fmad=false", "-std=c++14")).get_function("relu_windows")


# fast_rolling_math(arr, w, "sum"): float32 accumulator, value at i + floor(w/2), NaN elsewhere
_roll_sum = cp.RawKernel(r"""
extern "C" __global__ void roll_sum(const double* a, const long long* offs, const int* rid,
                                    const long long n, const int w, double* out) {
    long long t = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= n) return;
    int r = rid[t];
    long long o = offs[r], len = offs[r + 1] - o, q = t - o;
    long long i = q - w / 2;
    if (i < 0 || i > len - w) { out[t] = nan(""); return; }
    float v = 0.0f;
    for (int j = 0; j < w; j++) v = (float)((double)v + a[o + i + j]);
    out[t] = (double)v;
}
""", "roll_sum", options=("--fmad=false",))

# tobias_footprint_array, gather form. Stage A: best score per (footprint start, footprint width)
_fp_stage_a = cp.RawKernel(r"""
extern "C" __global__ void fp_a(const float* a, const long long* offs, const int* rid, const long long n,
                                const int flank_min, const int flank_max, const int fp_min, const int fp_max,
                                float* best) {
    long long s = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (s >= n) return;
    int r = rid[s];
    long long o = offs[r], len = offs[r + 1] - o, ls = s - o;
    long long imax = len - 2 * flank_max - fp_max;       // i in range(imax)
    int nfp = fp_max - fp_min + 1;
    for (int fpw = fp_min; fpw <= fp_max; fpw++) {
        float bmax = 0.0f;                                // scores start at 0; only larger values win
        for (int fw = flank_min; fw <= flank_max; fw++) {
            long long i = ls - fw;
            if (i < 0 || i >= imax) continue;
            float left = 0.0f, fps = 0.0f, right = 0.0f, val;
            for (int j = 0; j < fw; j++) { val = a[o + i + j]; if (val > 0.0f) left += val; }
            for (int j = 0; j < fpw; j++) { val = a[o + i + fw + j]; if (val < 0.0f) fps += val; }
            for (int j = 0; j < fw; j++) { val = a[o + i + fw + fpw + j]; if (val > 0.0f) right += val; }
            float fp_mean = (float)((double)fps / (1.0 * fpw));
            float flank_mean = (float)((double)(right + left) / (2.0 * fw));
            float sc = flank_mean - fp_mean;
            if (sc > bmax) bmax = sc;
        }
        best[s * nfp + (fpw - fp_min)] = bmax;
    }
}
""", "fp_a", options=("--fmad=false",))

# Stage B: each position takes the max over footprints covering it
_fp_stage_b = cp.RawKernel(r"""
extern "C" __global__ void fp_b(const float* best, const long long* offs, const int* rid, const long long n,
                                const int fp_min, const int fp_max, float* out) {
    long long p = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (p >= n) return;
    long long o = offs[rid[p]];
    int nfp = fp_max - fp_min + 1;
    float m = 0.0f;
    for (int fpw = fp_min; fpw <= fp_max; fpw++) {
        for (int k = 0; k < fpw; k++) {
            long long s = p - k;
            if (s < o) break;
            float v = best[s * nfp + (fpw - fp_min)];
            if (v > m) m = v;
        }
    }
    out[p] = m;
}
""", "fp_b", options=("--fmad=false",))


def _launch(kernel, n, args):
    kernel(((int(n) + 255) // 256,), (256,), args)


class _DeviceLayout:
    """Concatenated device layout of ``[start, end)`` regions on one chromosome."""

    def __init__(self, ext: np.ndarray):
        lens = (ext[:, 1] - ext[:, 0]).astype(np.int64)
        self.lens_h = lens
        self.offs_h = np.concatenate([[0], np.cumsum(lens)])
        self.offs = cp.asarray(self.offs_h)
        self.n = int(self.offs_h[-1])
        self.rid = cp.asarray(np.repeat(np.arange(len(ext), dtype=np.int32), lens))
        self.local = cp.arange(self.n, dtype=cp.int64) - self.offs[self.rid]
        self.gpos = cp.asarray(ext[:, 0])[self.rid] + self.local
        self.lens = cp.asarray(lens)

    def roll(self, a: cp.ndarray, w: int = WINDOW) -> cp.ndarray:
        out = cp.empty(self.n, dtype=cp.float64)
        _launch(_roll_sum, self.n, (cp.ascontiguousarray(a, dtype=cp.float64), self.offs, self.rid,
                                   cp.int64(self.n), cp.int32(w), out))
        return out


# =============================================================================
# Correction and footprint scores
# =============================================================================


def _correct_chrom(out_regions: np.ndarray, cuts_fr, seq_d: cp.ndarray, model: dict, cf: float,
                   keep: Iterable[str] = TRACKS, bias_override: Optional[dict] = None):
    """bias_correction() for every output region of one chromosome at once.

    Returns the tracks over the concatenated ``out_regions`` (in order) and the
    per-strand window fit status counts (no signal, fit, RuntimeError fallback,
    OptimizeWarning fallback). ``bias_override`` injects per-strand bias scores
    over the extended layout (testing hook).
    """
    f_extend = K_FLANK + WINDOW // 2
    ext = out_regions.copy()
    ext[:, 0] -= f_extend
    ext[:, 1] += f_extend
    lay = _DeviceLayout(ext)
    keep = tuple(keep)
    tracks = {k: cp.zeros(lay.n) for k in keep}
    stats = {}

    # windows: zip(range(k, reg_end - w, 10), range(k + w, reg_end, 10)), reg_end = reg_len - k
    reg_end = lay.lens_h - K_FLANK
    n_w = np.maximum(0, -(-(reg_end - WINDOW - K_FLANK) // 10))
    wr = np.repeat(np.arange(len(ext)), n_w)
    wj = np.arange(n_w.sum()) - np.repeat(np.cumsum(n_w) - n_w, n_w)
    wstart = cp.asarray(lay.offs_h[wr] + K_FLANK + 10 * wj)
    wslot = cp.asarray(wj % 10)
    n_win = int(len(wj))
    relu = _relu_kernel()

    for strand, pos in zip(("forward", "reverse"), cuts_fr):
        # uncorrected: cutsite (1-based) in (start, end) of the extended region
        rid, p = _region_members(pos, ext[:, 0], ext[:, 1] - 1)
        idx = cp.asarray(lay.offs_h[rid] + (p - ext[rid, 0]))
        # cp.add.at, not cp.bincount: CuPy 14.0.1's CUB histogram (bincount's default path without
        # weights) hits an illegal memory access for large outputs (e.g. 617k cuts into 11.2M
        # positions on chr1); counts are exact in float64
        unc = cp.zeros(lay.n, dtype=cp.float64)
        cp.add.at(unc, idx, 1.0)

        # DWM bias score at every position whose 25-mer lies inside the region and has no N
        if bias_override is not None:
            bias = cp.asarray(bias_override[strand], dtype=cp.float64)
        else:
            valid = (lay.local >= K_FLANK) & (lay.local < lay.lens[lay.rid] - K_FLANK)
            vi = cp.flatnonzero(valid)
            km = _kmers(seq_d, lay.gpos[vi], strand == "reverse")
            noN = ~(km == 4).any(axis=1)
            bias = cp.zeros(lay.n)
            bias[vi[noN]] = _dwm_score_kmers(km[noN], model[strand])
            del km, vi, valid, noN

        # per-window relu fit (batched MINPACK) -> normalised predictions, averaged over the
        # 10 overlapping windows (TOBIAS's nan assignment is a no-op, so it is a mean with zeros)
        pred = cp.empty((n_win, WINDOW))
        popt = cp.empty((n_win, 2))
        status = cp.empty(n_win, dtype=cp.int32)
        _launch(relu, n_win, (bias, unc, wstart, cp.int32(n_win), cp.int32(WINDOW), pred, popt, status))
        stats[strand] = cp.asnumpy(bincount(status, minlength=4))
        slots = cp.zeros((lay.n, 10))
        slots[wstart[:, None] + cp.arange(WINDOW)[None, :], wslot[:, None]] = pred
        del pred, popt, status
        acc = slots[:, 0].copy()
        for r in range(1, 10):
            acc = acc + slots[:, r]
        del slots
        bias_pred = acc / 10

        signal_sum = cp.nan_to_num(lay.roll(unc), nan=0.0)
        bias_sum = lay.roll(bias_pred)
        nulls = (cp.abs(bias_sum) <= 1e-8) | cp.isnan(bias_sum)
        bias_sum[nulls] = 1
        probas = bias_pred / bias_sum
        probas[nulls] = 0
        expected = signal_sum * probas

        unc = unc * cf
        expected = expected * cf
        corr = unc - expected
        unc_sum = cp.nan_to_num(lay.roll(unc), nan=0.0)
        corr_sum = cp.nan_to_num(lay.roll(cp.abs(corr)), nan=0.0)
        cpos = corr.copy()
        cpos[cpos < 0] = 0
        pos_sum = cp.nan_to_num(lay.roll(cpos), nan=0.0)
        neg_sum = corr_sum - pos_sum
        zero = pos_sum == 0
        pos_sum[zero] = cp.nan
        scale = (unc_sum - neg_sum) / pos_sum
        scale[zero] = 1
        scale[scale < 1] = 1
        pb = corr > 0
        corr[pb] *= scale[pb]

        for k, v in (("uncorrected", unc), ("bias", bias), ("expected", expected), ("corrected", corr)):
            if k in tracks:
                tracks[k] += v

    inner = (lay.local >= f_extend) & (lay.local < lay.lens[lay.rid] - f_extend)
    return {k: v[inner] for k, v in tracks.items()}, stats


def _footprints_chrom(regions: np.ndarray, dense_signal: cp.ndarray) -> cp.ndarray:
    """ScoreBigwig ``--score footprint`` over ``regions`` (concatenated, float32);
    ``dense_signal`` is the float32 corrected track of the whole chromosome."""
    ext = regions.copy()
    ext[:, 0] -= FP_FLANK_MAX
    ext[:, 1] += FP_FLANK_MAX
    lay = _DeviceLayout(ext)
    a = cp.ascontiguousarray(dense_signal[lay.gpos].astype(cp.float32))
    nfp = FP_MAX - FP_MIN + 1
    best = cp.empty(lay.n * nfp, dtype=cp.float32)
    _launch(_fp_stage_a, lay.n, (a, lay.offs, lay.rid, cp.int64(lay.n), cp.int32(FP_FLANK_MIN),
                                 cp.int32(FP_FLANK_MAX), cp.int32(FP_MIN), cp.int32(FP_MAX), best))
    out = cp.empty(lay.n, dtype=cp.float32)
    _launch(_fp_stage_b, lay.n, (best, lay.offs, lay.rid, cp.int64(lay.n), cp.int32(FP_MIN), cp.int32(FP_MAX), out))
    del best
    inner = (lay.local >= FP_FLANK_MAX) & (lay.local < lay.lens[lay.rid] - FP_FLANK_MAX)
    return out[inner]


# =============================================================================
# Public containers
# =============================================================================


@dataclass
class TrackLayout:
    """Disjoint, sorted ``[start, end)`` intervals per chromosome, concatenated in chromosome order.

    A track over this layout is one 1-D array; positions outside the intervals have no value.
    """

    chroms: list[str]
    intervals: dict[str, np.ndarray]
    offsets: dict[str, int] = field(init=False)
    n: int = field(init=False)

    def __post_init__(self):
        self.offsets, o = {}, 0
        for c in self.chroms:
            self.offsets[c] = o
            iv = self.intervals[c]
            o += int((iv[:, 1] - iv[:, 0]).sum())
        self.n = o

    def chrom_slice(self, chrom: str) -> slice:
        iv = self.intervals[chrom]
        o = self.offsets[chrom]
        return slice(o, o + int((iv[:, 1] - iv[:, 0]).sum()))

    def positions(self, chrom: str) -> np.ndarray:
        iv = self.intervals[chrom]
        lens = iv[:, 1] - iv[:, 0]
        return np.repeat(iv[:, 0], lens) + (np.arange(lens.sum()) - np.repeat(np.cumsum(lens) - lens, lens))

    def index(self, chrom: str, pos: np.ndarray) -> np.ndarray:
        """Index into the concatenated track of each position, ``-1`` where not covered."""
        iv = self.intervals[chrom]
        pos = np.asarray(pos, dtype=np.int64)
        lens = iv[:, 1] - iv[:, 0]
        cum = np.concatenate([[0], np.cumsum(lens)])
        r = np.searchsorted(iv[:, 0], pos, side="right") - 1
        ok = r >= 0
        ok[ok] &= pos[ok] < iv[r[ok], 1]
        out = np.full(len(pos), -1, dtype=np.int64)
        out[ok] = self.offsets[chrom] + cum[r[ok]] + (pos[ok] - iv[r[ok], 0])
        return out


@dataclass
class FootprintTracks:
    """Bias-corrected Tn5 signal and footprint scores per pseudobulk.

    Attributes
    ----------
    groups : list of str
        Pseudobulk names, in the order given.
    chrom_sizes : dict
        Chromosome lengths of the processed chromosomes.
    peaks : dict
        Per chromosome, the merged input peaks (``int64`` array ``(n, 2)``).
    layouts : dict
        ``"corrected"`` (and other ATACorrect tracks): ATACorrect output regions
        (peaks ± 100 bp, merged); ``"footprint"``: ScoreBigwig regions.
    values : dict
        ``values[track][group]`` is a ``float32`` array over ``layouts[track]``
        (``layouts["corrected"]`` for every ATACorrect track).
    stats : pandas.DataFrame
        Per group: cells, cut sites, reads in peaks, FRiP, correction factor.
    bias_models : dict
        Per group and strand, the DWM model: k-mer ``counts`` and ``bg_counts`` and
        the log2 PWM/DWM of the bias and background models.
    params : dict
        Parameters of the run.
    """

    groups: list[str]
    chrom_sizes: dict[str, int]
    peaks: dict[str, np.ndarray]
    layouts: dict[str, TrackLayout]
    values: dict[str, dict[str, np.ndarray]]
    stats: pd.DataFrame
    bias_models: dict
    params: dict

    def __repr__(self) -> str:
        tracks = ", ".join(self.values)
        return (f"FootprintTracks({len(self.groups)} groups × {len(self.chrom_sizes)} chromosomes; "
                f"tracks: {tracks}; {self.layouts['footprint'].n:,} footprint positions)")

    def layout(self, track: str) -> TrackLayout:
        return self.layouts["footprint" if track == "footprint" else "corrected"]

    def get(self, group: str, chrom: str, start: Optional[int] = None, end: Optional[int] = None,
            track: str = "footprint") -> pd.DataFrame:
        """Values of one track over a chromosome or ``[start, end)``, as ``position``/``value`` columns."""
        lay = self.layout(track)
        pos = lay.positions(chrom)
        vals = self.values[track][group][lay.chrom_slice(chrom)]
        if start is not None or end is not None:
            m = (pos >= (start or 0)) & (pos < (end if end is not None else np.inf))
            pos, vals = pos[m], vals[m]
        return pd.DataFrame({"position": pos, "value": vals})

    def dense(self, group: str, chrom: str, start: int, end: int, track: str = "footprint") -> np.ndarray:
        """Values over ``[start, end)`` with uncovered positions as 0 (how a bigwig gap reads)."""
        lay = self.layout(track)
        idx = lay.index(chrom, np.arange(start, end))
        out = np.zeros(end - start, dtype=np.float32)
        out[idx >= 0] = self.values[track][group][idx[idx >= 0]]
        return out

    def to_bigwig(self, path: Union[str, Path], group: str, track: str = "footprint") -> None:
        """Write one track to a bigwig (requires ``pyBigWig``)."""
        try:
            import pyBigWig
        except ImportError as e:
            raise ImportError("to_bigwig needs pyBigWig (pip install pyBigWig)") from e
        lay = self.layout(track)
        bw = pyBigWig.open(str(path), "w")
        bw.addHeader([(c, int(self.chrom_sizes[c])) for c in lay.chroms])
        for c in lay.chroms:
            pos = lay.positions(c)
            if len(pos) == 0:
                continue
            vals = self.values[track][group][lay.chrom_slice(c)].astype(np.float64)
            bw.addEntries(c, pos.astype(np.int64), values=vals, span=1)
        bw.close()


# =============================================================================
# Public entry point
# =============================================================================


def _peaks_from_input(peaks, adata) -> pd.DataFrame:
    if peaks is None:
        names = pd.Series(adata.var_names.astype(str))
    elif isinstance(peaks, (str, Path)):
        return read_bed(peaks)
    elif isinstance(peaks, pd.DataFrame):
        return peaks[["chrom", "start", "end"]].copy()
    else:
        names = pd.Series(list(peaks), dtype=str)
    parts = names.str.extract(r"^(?P<chrom>.+)[:_](?P<start>\d+)[-_](?P<end>\d+)$")
    if parts.isna().any().any():
        bad = names[parts.isna().any(axis=1)].iloc[0]
        raise ValueError(f"cannot parse peak name {bad!r}; expected 'chrom:start-end'")
    return parts.astype({"start": np.int64, "end": np.int64})


def footprint(
    adata: "AnnData",
    groupby: str,
    parquet_path: Union[str, Path],
    genome_fasta: Union[str, Path],
    *,
    groups: Optional[Sequence[str]] = None,
    peaks: Union[None, str, Path, pd.DataFrame, Sequence[str]] = None,
    read_length: Optional[int] = None,
    tobias_compat: bool = True,
    deduplicate: bool = True,
    keep_tracks: Sequence[str] = ("corrected",),
    chromosomes: Optional[Sequence[str]] = None,
    verbose: bool = True,
) -> FootprintTracks:
    """
    Tn5 bias correction and footprint scores per pseudobulk (GPU TOBIAS).

    Reproduces TOBIAS 0.17.5 :cite:p:`Bentsen2020` ``ATACorrect`` followed by
    ``ScoreBigwig --score footprint`` with default parameters, for each group
    of cells, reading cut sites directly from the fragments parquet. Feed the result to
    :func:`gatac.tobias.bindetect` for differential TF binding.

    Parameters
    ----------
    adata : AnnData
        Cells × features. ``adata.obs[groupby]`` defines the pseudobulks; if
        ``peaks`` is None, ``adata.var_names`` (``chrom:start-end``) are the peaks.
    groupby : str
        Column of ``adata.obs`` with the group of each cell.
    parquet_path : str or Path
        GATAC fragments parquet (``chrom``, ``start``, ``end``, ``barcode``);
        barcodes must match ``adata.obs_names``.
    genome_fasta : str or Path
        Uncompressed genome FASTA (indexed, or writable to build the ``.fai``).
    groups : sequence of str, optional
        Groups to process (default: all, in category order).
    peaks : None, BED path, DataFrame or sequence of ``chrom:start-end``
        Peak set used for the bias model (peaks are excluded from it), the
        correction regions and the footprint regions. Use the union peak set
        of all groups, as TOBIAS recommends.
    read_length : int, optional
        Read length of the BAM TOBIAS would read. ATACorrect's bias model
        depends on it through its read-overlap fetch. Default: 50 (10x scATAC)
        when ``tobias_compat=True``; otherwise every cut site in a non-peak
        region is counted once, independent of read length.
    tobias_compat : bool, default True
        Reproduce TOBIAS quirks exactly (read-length-dependent bias model,
        merge-end quirk). False uses the corrected behaviours.
    deduplicate : bool, default True
        Keep one copy of identical fragments within a pseudobulk.
    keep_tracks : sequence of str
        ATACorrect tracks to keep besides the footprint scores; any of
        ``"uncorrected"``, ``"bias"``, ``"expected"``, ``"corrected"``
        (default ``("corrected",)``).
    chromosomes : sequence of str, optional
        Restrict to these chromosomes (default: chromosomes with peaks that are
        in the FASTA).
    verbose : bool, default True
        Show progress bars.

    Returns
    -------
    FootprintTracks

    Examples
    --------
    >>> tracks = ga.tobias.footprint(
    ...     adata, "cell_type", "fragments.parquet", "GRCh38.fa",
    ...     groups=["CD14 Mono", "CD4 Memory"],
    ... )
    >>> res = ga.tobias.bindetect(tracks, "cisBP_human.meme", "GRCh38.fa")
    """
    import time

    from tqdm.auto import tqdm

    keep_tracks = tuple(keep_tracks)
    bad = set(keep_tracks) - set(TRACKS)
    if bad:
        raise ValueError(f"unknown tracks {sorted(bad)}; choose from {TRACKS}")
    if groupby not in adata.obs.columns:
        raise ValueError(f"'{groupby}' not found in adata.obs")
    col = adata.obs[groupby]
    if groups is None:
        groups = list(col.cat.categories) if hasattr(col, "cat") else sorted(col.dropna().unique())
        groups = [g for g in groups if (col == g).any()]
    groups = [str(g) for g in groups]
    col = col.astype(str)
    barcode_groups = {}
    for gi, g in enumerate(groups):
        bcs = adata.obs_names[(col == g).to_numpy()]
        if len(bcs) == 0:
            raise ValueError(f"group {g!r} has no cells")
        barcode_groups.update(dict.fromkeys(bcs.astype(str), gi))

    fasta = Path(genome_fasta)
    fai = _fai(fasta)
    pk = _peaks_from_input(peaks, adata)
    pk = pk[pk["chrom"].isin(fai)]
    if chromosomes is not None:
        pk = pk[pk["chrom"].isin(set(chromosomes))]
    chroms = [c for c in fai if c in set(pk["chrom"])]
    if not chroms:
        raise ValueError("no peaks on chromosomes of the FASTA")
    sizes = {c: fai[c][0] for c in chroms}
    by_chrom = {c: sub[["start", "end"]].to_numpy(np.int64) for c, sub in pk.groupby("chrom", sort=False)}
    rl = read_length if read_length is not None else (50 if tobias_compat else None)

    T = {}
    t0 = time.perf_counter()
    cuts = _load_cuts(parquet_path, barcode_groups, sizes, rl, deduplicate)
    regions = {c: _atacorrect_regions(by_chrom[c], sizes[c], tobias_compat) for c in chroms}
    T["load"] = time.perf_counter() - t0

    # ---- pass 1: read counts and DWM k-mer counts, genome-wide per group ----
    t0 = time.perf_counter()
    n_g = len(groups)
    shape = (5, 5, L_KMER, L_KMER)
    counts = [{s: np.zeros(shape) for s in ("forward", "reverse")} for _ in range(n_g)]
    bgc = [{s: np.zeros(shape) for s in ("forward", "reverse")} for _ in range(n_g)]
    reads_peaks = np.zeros(n_g, np.int64)
    reads_nonpeaks = np.zeros(n_g, np.int64)
    for c in tqdm(chroms, desc="Bias model", disable=not verbose):
        seq_d = cp.asarray(_LUT[_chrom_bytes(fasta, fai, c)])
        for g in range(n_g):
            for strand in cuts[g][c]:
                reads_peaks[g] += _count_in(strand, regions[c]["peaks"])
                reads_nonpeaks[g] += _count_in(strand, regions[c]["nonpeaks"])
            cnt, bg = _bias_counts(cuts[g][c], regions[c], seq_d, sizes[c], rl)
            for s in cnt:
                counts[g][s] += cnt[s]
                bgc[g][s] += bg[s]
        del seq_d
    reads_total = reads_peaks + reads_nonpeaks
    if (reads_total == 0).any():
        raise ValueError(f"no cut sites for groups {[groups[g] for g in np.flatnonzero(reads_total == 0)]}")
    frip = reads_peaks / reads_total
    cf = (10000000 / reads_total) * (1 / frip)
    models = [{s: _prepare_dwm(counts[g][s], bgc[g][s]) for s in ("forward", "reverse")} for g in range(n_g)]
    T["bias_model"] = time.perf_counter() - t0

    # ---- pass 2: correction and footprints per chromosome ----
    t0 = time.perf_counter()
    corr_lay = TrackLayout(chroms, {c: regions[c]["output"] for c in chroms})
    fp_lay = TrackLayout(chroms, {c: _score_regions(by_chrom[c], sizes[c], tobias_compat) for c in chroms})
    values = {k: {g: np.zeros(corr_lay.n, np.float32) for g in groups} for k in keep_tracks}
    values["footprint"] = {g: np.zeros(fp_lay.n, np.float32) for g in groups}
    fit_stats = np.zeros((n_g, 4), np.int64)
    want = tuple(set(keep_tracks) | {"corrected"})
    for c in tqdm(chroms, desc="Correct + footprint", disable=not verbose):
        seq_d = cp.asarray(_LUT[_chrom_bytes(fasta, fai, c)])
        out_reg = regions[c]["output"]
        sl_c, sl_f = corr_lay.chrom_slice(c), fp_lay.chrom_slice(c)
        gpos = cp.asarray(corr_lay.positions(c))
        for g, name in enumerate(groups):
            if len(out_reg) == 0:
                continue
            tr, st = _correct_chrom(out_reg, cuts[g][c], seq_d, models[g], float(cf[g]), keep=want)
            for s in st:
                fit_stats[g] += st[s]
            for k in keep_tracks:
                tr[k].astype(cp.float32).get(out=values[k][name][sl_c])    # no host temporary
            dense = cp.zeros(sizes[c], dtype=cp.float32)
            dense[gpos] = tr["corrected"].astype(cp.float32)
            del tr
            if len(fp_lay.intervals[c]):
                _footprints_chrom(fp_lay.intervals[c], dense).get(out=values["footprint"][name][sl_f])
            del dense
        del seq_d, gpos
        cp.get_default_memory_pool().free_all_blocks()
    T["correct_footprint"] = time.perf_counter() - t0
    logger.info("footprint timings: " + ", ".join(f"{k} {v:.1f}s" for k, v in T.items()))

    stats = pd.DataFrame({
        "n_cells": [int((col == g).sum()) for g in groups],
        "reads_total": reads_total, "reads_peaks": reads_peaks, "frip": frip,
        "correction_factor": cf,
        "windows_fit": fit_stats[:, 1], "windows_no_signal": fit_stats[:, 0],
        "windows_fallback": fit_stats[:, 2] + fit_stats[:, 3],
    }, index=pd.Index(groups, name="group"))
    return FootprintTracks(
        groups=groups,
        chrom_sizes=sizes,
        peaks={c: _merge(by_chrom[c], tobias_compat) for c in chroms},
        layouts={"corrected": corr_lay, "footprint": fp_lay},
        values=values,
        stats=stats,
        bias_models={g: models[i] for i, g in enumerate(groups)},
        params=dict(groupby=groupby, read_length=rl, tobias_compat=tobias_compat,
                    deduplicate=deduplicate, genome_fasta=str(fasta), timings=T),
    )
