"""
MACS3 ``callpeak`` (narrow peaks, no control) on run-length pileups on the GPU.

Reproduces the peaks SnapATAC2's ``snap.tl.macs3`` gets from MACS3 for one
group of cells. Each fragment gives two Tn5 insertions, read as single-end
tags: ``start`` on the + strand and ``end - 1`` on the - strand. The treatment
is the pileup of tags shifted by ``shift`` and extended to ``extsize``. The
local lambda is the pileup in a ``llocal`` window, scaled by
``extsize / llocal`` and floored at the genome background. Each run gets a
Poisson p-score, q-scores come from a Benjamini-Hochberg table weighted by run
length, runs above the q cutoff are joined across gaps of at most
``max_gap``, and summits are found by MACS3's Savitzky-Golay sub-peak caller.

Pileups are kept as runs between breakpoints, with the float32 values MACS3
stores, and the p-score of each distinct (pileup, lambda) pair is computed
once on the host with MACS3's own series and rounding. Peaks, summits and
scores are therefore those MACS3 writes, not approximations of them.

Portions of this module are translated from MACS3
(https://github.com/macs3-project/MACS), distributed under the BSD 3-Clause
License:

    Copyright (c) 2025, macs3 project team All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

    1. Redistributions of source code must retain the above copyright notice,
       this list of conditions and the following disclaimer.

    2. Redistributions in binary form must reproduce the above copyright
       notice, this list of conditions and the following disclaimer in the
       documentation and/or other materials provided with the distribution.

    3. Neither the name of the copyright holder nor the names of its
       contributors may be used to endorse or promote products derived from
       this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
    IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
    THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
    PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
    CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
    EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
    PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
    PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
    LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
    NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from functools import lru_cache

import cupy as cp
import numpy as np
import pandas as pd
from numba import njit, prange

logger = logging.getLogger(__name__)

PEAK_COLUMNS = ["chrom", "start", "end", "name", "score", "strand",
                "signal_value", "p_value", "q_value", "peak"]

# Positions and chromosomes share one int64 key, chromosome in the high bits,
# so a whole batch of chromosomes sorts and scans as one array.
_POS_BITS = 32
_POS_MASK = (1 << _POS_BITS) - 1

# GPU memory a batch takes per fragment, measured at the peak of _segments
# (the sort of the 4 events per tag of the control track) with headroom.
_BYTES_PER_FRAGMENT = 450


@dataclass(frozen=True)
class Params:
    """MACS3 options, named as in ``macs3 callpeak``."""
    gsize: int
    log_qvalue: float          # -log10 of the q-value cutoff
    extsize: int = 200
    shift: int = -100
    llocal: int = 10000
    max_gap: int = 30
    min_length: int = 200
    fe_cutoff: float = 1.0
    call_summits: bool = True
    macs3_version: str = "3.0"  # "3.0" (SnapATAC2 <= 2.9) or "3.1"


# =============================================================================
# Run-length pileups
# =============================================================================


def _track(pos, cid, minus, five, three):
    """
    Pileup of the tags extended as MACS3's ``pileup_from_PN_shifted``.

    A + strand tag at ``p`` covers ``[p - five, p + three)``, a - strand tag
    ``[p - three, p + five)``, both clipped at 0. Returns the breakpoints
    (keys of the positions where the pileup changes, the last being the end
    of the last interval) and the pileup of the run ending at each.
    """
    start = cp.maximum(pos - cp.where(minus, three, five), 0)
    end = cp.maximum(pos + cp.where(minus, five, three), 0)
    base = cid << _POS_BITS
    # Events as (key << 1) | is_start, sorted together
    ev = cp.concatenate((((base | start) << 1) | 1, (base | end) << 1))
    del start, end, base
    ev.sort()
    cov = cp.cumsum((ev & 1).astype(cp.int32) * 2 - 1, dtype=cp.int32)
    ev >>= 1
    last = cp.empty(ev.size, dtype=bool)
    last[-1] = True
    cp.not_equal(ev[1:], ev[:-1], out=last[:-1])
    key, after = ev[last], cov[last]
    del ev, cov, last
    # MACS3 emits a breakpoint only where the pileup changes: starts and ends
    # at the same position cancel out.
    before = cp.concatenate((cp.zeros(1, cp.int32), after[:-1]))
    changed = after != before
    key, after = key[changed], after[changed]
    value = cp.concatenate((cp.zeros(1, cp.int32), after[:-1]))
    return key, value


def _segments(pos, cid, minus, n_chrom, p: Params, lambda_bg, ctrl_scale):
    """
    Pair the treatment and control tracks of a batch of chromosomes as
    MACS3's ``__chrom_pair_treat_ctrl``.

    Returns the end key of each run, the start key, the treatment pileup
    (int32) and the local lambda (float32) on it. Each chromosome stops at the
    end of whichever track ends first, as in MACS3.
    """
    t_key, t_val = _track(pos, cid, minus, -p.shift, p.shift + p.extsize)
    # MACS3 piles the control up around the unshifted tags (no end_shift)
    c_key, c_val = _track(pos, cid, minus, p.llocal // 2, p.llocal - p.llocal // 2)
    key = cp.unique(cp.concatenate((t_key, c_key)))
    bounds = cp.arange(1, n_chrom + 1, dtype=cp.int64) << _POS_BITS
    limit = cp.minimum(t_key[cp.searchsorted(t_key, bounds) - 1],
                       c_key[cp.searchsorted(c_key, bounds) - 1])
    key = key[key <= limit[key >> _POS_BITS]]
    treat = t_val[cp.searchsorted(t_key, key)]
    ctrl = c_val[cp.searchsorted(c_key, key)].astype(cp.float32) * cp.float32(ctrl_scale)
    ctrl = cp.maximum(ctrl, cp.float32(lambda_bg))
    del t_key, t_val, c_key, c_val
    prev = cp.concatenate((cp.full(1, -1, cp.int64), key[:-1]))
    chrom_base = (key >> _POS_BITS) << _POS_BITS
    start = cp.where((prev >> _POS_BITS) == (key >> _POS_BITS), prev, chrom_base)
    return key, start, treat, ctrl


# =============================================================================
# Scores
# =============================================================================


@njit(cache=True)
def _log_sums(m_max):
    """log(1) + ... + log(m), summed in ascending order as MACS3's table."""
    sums = np.empty(m_max + 1, np.float64)
    sums[0] = 0.0
    for i in range(1, m_max + 1):
        sums[i] = sums[i - 1] + math.log(i)
    return sums


@njit(parallel=True, cache=True)
def _log10_poisson_upper(k, lam, sums):
    """
    log10 P(X > k), X ~ Poisson(lam), before rounding: MACS3's
    ``log10_poisson_cdf_Q_large_lambda`` operation for operation.
    """
    out = np.empty(k.size, np.float64)
    ln10 = math.log(10.0)
    for i in prange(k.size):
        lbd = np.float64(lam[i])
        ln_lbd = math.log(lbd)
        m = np.int64(k[i]) + 1
        logx = m * ln_lbd - sums[m]
        residue = logx
        while True:
            m += 1
            logy = logx + ln_lbd - math.log(m)
            pre = residue
            if pre > logy:
                residue = pre + math.log1p(math.exp(logy - pre))
            else:
                residue = logy + math.log1p(math.exp(pre - logy))
            if abs(pre - residue) < 1e-5:
                break
            logx = logy
        out[i] = (residue - lbd) / ln10
    return out


def _round5(x: np.ndarray) -> np.ndarray:
    """Python's ``round(x, 5)``, which MACS3 applies to every p-score."""
    scaled = x * 1e5
    r = np.rint(scaled)
    out = r / 1e5
    # rint of the rounded product is the exact rounding unless the product
    # lies within its rounding error of a half-way point.
    near = np.abs(np.abs(scaled - r) - 0.5) < 1e-6
    for i in np.flatnonzero(near):
        out[i] = round(float(x[i]), 5)
    return np.where(r == 0, np.copysign(0.0, x), out)


@njit(cache=True)
def _libm_log(x):
    out = np.empty(x.size, np.float64)
    for i in range(x.size):
        out[i] = math.log(x[i])
    return out


# The series of _log10_poisson_upper on the GPU, one pair per thread. Logs of
# integers and of lambdas come from the host's libm, and --fmad=false keeps
# each multiply and add rounded on its own, as in MACS3's C. Only exp and
# log1p are the GPU's own: a pair is ``unsure`` when their last bits could
# change when the loop stops, and the caller redoes it on the host.
_POISSON_KERNEL = cp.ElementwiseKernel(
    "int64 k, float64 lbd, float64 ln_lbd, raw float64 sums, raw float64 logs, int64 n_logs, float64 ln10",
    "float64 out, bool unsure",
    """
    long long m = k + 1;
    double logx = (double)m * ln_lbd - sums[m];
    double residue = logx;
    unsure = false;
    while (true) {
        m += 1;
        if (m >= n_logs) { unsure = true; break; }
        double logy = logx + ln_lbd - logs[m];
        double pre = residue;
        if (pre > logy) {
            residue = pre + log1p(exp(logy - pre));
        } else {
            residue = logy + log1p(exp(pre - logy));
        }
        double d = fabs(pre - residue);
        if (fabs(d - 1e-5) < 1e-10) unsure = true;
        if (d < 1e-5) break;
        logx = logy;
    }
    out = (residue - lbd) / ln10;
    """,
    "macs3_log10_poisson_upper",
    options=("--fmad=false",),
)


def _log10_poisson_upper_gpu(k: np.ndarray, lam: np.ndarray, sums: np.ndarray):
    lam_values, lam_index = np.unique(lam, return_inverse=True)
    ln_lam = _libm_log(lam_values.astype(np.float64))[lam_index]
    logs = np.zeros(sums.size, np.float64)
    logs[1:] = _libm_log(np.arange(1, sums.size, dtype=np.float64))
    raw, unsure = _POISSON_KERNEL(
        cp.asarray(k), cp.asarray(lam.astype(np.float64)), cp.asarray(ln_lam),
        cp.asarray(sums), cp.asarray(logs), sums.size, math.log(10.0))
    return raw.get(), unsure.get()


def pscores(treat: np.ndarray, ctrl: np.ndarray) -> np.ndarray:
    """MACS3 p-scores (-log10 p, float32) of pileups ``treat`` under lambdas ``ctrl``."""
    if treat.size == 0:
        return np.empty(0, np.float32)
    treat = treat.astype(np.int64)
    ctrl = ctrl.astype(np.float32)
    # The series runs past k by a few times sqrt(lambda) at most
    n_logs = int(max(treat.max(), 2 * float(ctrl.max()))) + 2000
    sums = _log_sums(n_logs)
    raw, unsure = _log10_poisson_upper_gpu(treat, ctrl, sums)
    # Pairs whose rounding to 5 decimals the GPU's last bits could change
    frac = np.abs(raw * 1e5 - np.rint(raw * 1e5))
    redo = np.flatnonzero(unsure | (np.abs(frac - 0.5) < 1e-4))
    if redo.size:
        host_sums = _log_sums(int(treat[redo].max()) + 1)
        raw[redo] = _log10_poisson_upper(treat[redo], ctrl[redo], host_sums)
    return (-1.0 * _round5(raw)).astype(np.float32)


@njit(cache=True)
def _bh(pscore_desc, lengths):
    """MACS3's ``__cal_pvalue_qvalue_table``: q-score of each p-score, descending."""
    n = pscore_desc.size
    q_out = np.zeros(n, np.float32)
    total = lengths.sum()
    f = np.float32(-math.log10(np.float64(total))) if total > 0 else np.float32(0.0)
    pre_q = np.float32(2147483647.0)
    k = np.int64(1)
    for i in range(n):
        q = np.float32(np.float64(pscore_desc[i]) + (math.log10(np.float64(k)) + np.float64(f)))
        if q > pre_q:
            q = pre_q
        if q <= 0:
            break
        q_out[i] = q
        pre_q = q
        k += lengths[i]
    return q_out


def qscores(pscore: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """q-score of each p-score, from the genome-wide length of each p-score."""
    # -0.0 and 0.0 are one p-score
    values, inverse = np.unique(pscore + np.float32(0.0), return_inverse=True)
    summed = np.bincount(inverse, weights=lengths, minlength=values.size).astype(np.int64)
    q_desc = _bh(values[::-1].copy(), summed[::-1].copy())
    return q_desc[::-1][inverse]


# =============================================================================
# Summits
# =============================================================================


@njit(parallel=True, cache=True)
def _fill_regions(a_start, a_end, a_treat, g0, g1, offset, signal, index, sign, h, version31):
    """
    Per region: the treatment pileup over the region padded by 10 bp
    (``peakdata`` in MACS3's ``__close_peak_with_subpeaks``), the run each base
    belongs to, and the sign of the Savitzky-Golay first derivative.

    MACS3 3.0 writes the pileup ``start_boundary`` bases to the right (its
    last bases fall off the end) and maps gaps between runs to the first run;
    3.1 places it at its bases and maps gaps to -1.

    The derivative is ``sum_i m[i] * x[j + i]`` with ``m[i] = (i - h) / sum(k^2)``
    up to rounding. Its numerator over integer pileups is an integer, computed
    exactly here: when it is not zero its sign is numpy's. Where it is zero
    numpy's value is rounding noise, whose sign MACS3 uses all the same:
    those bases get 2, for the caller to take numpy's sign.
    """
    W = 2 * h + 1
    for r in prange(g0.size):
        ps = a_start[g0[r]]
        pe = a_end[g1[r] - 1]
        start = max(ps - 10, 0)
        L = pe + 10 - start
        o = offset[r]
        shift = 0 if version31 else ps - start
        for j in range(L):
            signal[o + j] = 0
            index[o + j] = -1 if version31 else 0
        for i in range(g0[r], g1[r]):
            for j in range(a_start[i] - start + shift, min(a_end[i] - start + shift, L)):
                signal[o + j] = a_treat[i]
                index[o + j] = i - g0[r]
        s0 = np.int64(signal[o])
        sl = np.int64(signal[o + L - 1])
        # padded[q], q in [0, L + 2h): mirrored ends as savitzky_golay_order2_deriv1
        z = np.int64(0)
        s = np.int64(0)
        for i in range(W):
            v = _padded(signal, o, L, h, s0, sl, i)
            z += (i - h) * v
            s += v
        for j in range(L):
            if z > 0:
                sign[o + j] = 1
            elif z < 0:
                sign[o + j] = -1
            else:
                sign[o + j] = 2
            if j + 1 < L:
                first = _padded(signal, o, L, h, s0, sl, j)
                nxt = _padded(signal, o, L, h, s0, sl, j + W)
                s += nxt - first
                z += h * first + (h + 1) * nxt - s


@njit(cache=True, inline="always")
def _padded(signal, o, L, h, s0, sl, q):
    if q < h:
        return s0 - abs(np.int64(signal[o + h - q]) - s0)
    if q < h + L:
        return np.int64(signal[o + q - h])
    return sl + abs(np.int64(signal[o + L - 2 - (q - h - L)]) - sl)


@lru_cache(maxsize=None)
def _sg_kernel(window: int) -> np.ndarray:
    """First-derivative Savitzky-Golay coefficients, as MACS3 computes them."""
    half = (window - 1) // 2
    b = np.array([[1, k, k ** 2] for k in range(-half, half + 1)], dtype="i8")
    return np.linalg.pinv(b)[1]


def _numpy_signs(signal: np.ndarray, js: np.ndarray, window: int) -> np.ndarray:
    """
    Sign of MACS3's smoothed derivative of the region ``signal`` at bases
    ``js``, computed by numpy exactly as in ``savitzky_golay_order2_deriv1``.

    Each base is one ``np.convolve`` over its own window, which runs the same
    dot product as the whole-region call (one base further keeps the
    orientation of that call).
    """
    half = (window - 1) // 2
    m = _sg_kernel(window)
    first = signal[0] - np.abs(signal[1:half + 1][::-1] - signal[0])
    last = signal[-1] + np.abs(signal[-half - 1:-1][::-1] - signal[-1])
    padded = np.concatenate((first, signal, last)).astype("f8")
    out = np.empty(js.size, np.int8)
    for i, j in enumerate(js):
        if j + window < padded.size:
            v = np.convolve(m[::-1], padded[j:j + window + 1], mode="valid").round(16)[0]
        else:
            v = np.convolve(m[::-1], padded[j - 1:j + window], mode="valid").round(16)[1]
        out[i] = np.sign(v)
    return out


@njit(cache=True)
def _distinct_at_least(x, n):
    y = np.sort(x)
    count = 1 if y.size else 0
    for i in range(1, y.size):
        if y[i] != y[i - 1]:
            count += 1
            if count >= n:
                return True
    return count >= n


@njit(cache=True)
def _valid_peak(signal, lo, hi, threshold, second, maximum, version31):
    """MACS3's ``is_valid_peak`` on ``signal[lo:hi] - threshold - second``."""
    n = hi - lo
    x = np.empty(n, np.float32)
    for i in range(n):
        x[i] = (signal[lo + i] - threshold) - second
    if version31:
        left = maximum
        while left > 0 and x[left - 1] >= 0:
            left -= 1
        right = maximum + 1
        while right < n and x[right] >= 0:
            right += 1
        right = min(right, n)
    else:
        # MACS3 3.0's hard_clip keeps everything left of the maximum
        left = 0
        right = n
        for i in range(maximum, n):
            if x[i] < 0:
                right = i
                break
    if right - left < 50:
        return False
    return _distinct_at_least(x[left:right], 6)


@njit(cache=True)
def _f32_sqrt(x):
    return np.float32(math.sqrt(np.float64(x)))


@njit(cache=True)
def _peaky(signal, o, L, maxima, n_max, version31, out):
    """MACS3's ``enforce_peakyness`` on the region ``signal[o:o+L]``; writes the kept maxima to ``out``."""
    if n_max < 2:
        for i in range(n_max):
            out[i] = maxima[i]
        return n_max
    n = n_max - 1
    minima = np.empty(n, np.int64)
    for i in range(n):
        best = maxima[i]
        for j in range(maxima[i] + 1, maxima[i + 1]):
            if signal[o + j] < signal[o + best]:
                best = j
        minima[i] = best
    zero = np.float32(0.0)
    j = 0
    thr = signal[o + minima[0]]
    thr = np.float32(thr + _f32_sqrt(thr))
    # MACS3 3.0 subtracts the square root a second time for the first peak
    second = zero if version31 else _f32_sqrt(thr)
    if _valid_peak(signal, o, o + minima[0], thr, second, maxima[0], version31):
        out[j] = maxima[0]
        j += 1
    for i in range(n - 1):
        thr = max(signal[o + minima[i]], signal[o + minima[i + 1]])
        thr = np.float32(thr + _f32_sqrt(thr))
        if _valid_peak(signal, o + minima[i], o + minima[i + 1], thr, zero,
                       maxima[i + 1] - minima[i], version31):
            out[j] = maxima[i + 1]
            j += 1
    thr = signal[o + minima[n - 1]]
    thr = np.float32(thr + _f32_sqrt(thr))
    if _valid_peak(signal, o + minima[n - 1], o + L, thr, zero,
                   maxima[n_max - 1] - minima[n - 1], version31):
        out[j] = maxima[n_max - 1]
        j += 1
    return j


@njit(cache=True)
def _single_summit(a_start, a_end, a_treat, lo, hi):
    """
    MACS3's summit without sub-peaks: the middle one of the runs with the
    highest pileup, at the middle of that run. Returns (run, position).
    """
    top = np.float32(0.0)
    n_tie = 0
    for i in range(lo, hi):
        t = a_treat[i]
        if top == 0 or top < t:
            top = t
            n_tie = 1
        elif top == t:
            n_tie += 1
    mid = (n_tie + 1) // 2 - 1
    seen = -1
    for i in range(lo, hi):
        if a_treat[i] == top:
            seen += 1
            if seen == mid:
                return i, (a_start[i] + a_end[i]) // 2
    return lo, (a_start[lo] + a_end[lo]) // 2


@njit(parallel=True, cache=True)
def _mark_summits(a_start, a_end, a_treat, g0, g1, offset, signal, index, sign, mark, version31):
    """
    MACS3's ``__close_peak_with_subpeaks``: local maxima of the smoothed
    pileup inside the peak, kept by ``enforce_peakyness`` (and, in 3.1, when
    they fall on a run above the cutoff); the single summit when none is
    left. ``mark`` gets, at each summit's offset from the region start, the
    summit's run (from the region's first) plus one.
    """
    for r in prange(g0.size):
        ps = a_start[g0[r]]
        pe = a_end[g1[r] - 1]
        start = max(ps - 10, 0)
        L = pe + 10 - start
        o = offset[r]
        lo = ps - start
        # 3.1 keeps maxima in [lo, lo + length), 3.0 in [lo, lo + length]
        hi = lo + (pe - ps) + (0 if version31 else 1)
        maxima = np.empty(L, np.int64)
        n_max = 0
        for j in range(L - 1):
            if sign[o + j + 1] - sign[o + j] <= -1 and lo <= j < hi:
                maxima[n_max] = j
                n_max += 1
        found = 0
        if n_max > 0:
            kept = np.empty(n_max, np.int64)
            n_kept = _peaky(signal, o, L, maxima, n_max, version31, kept)
            for i in range(n_kept):
                j = kept[i]
                run = index[o + j]
                if version31:
                    if run >= 0:
                        mark[o + j] = run + 1
                        found += 1
                else:
                    # 3.0 reports the summit start_boundary bases to the left
                    mark[o + j - lo] = run + 1
                    found += 1
        if found == 0:
            run, pos = _single_summit(a_start, a_end, a_treat, g0[r], g1[r])
            mark[o + pos - start] = run - g0[r] + 1


@njit(parallel=True, cache=True)
def _single_summits(a_start, a_end, a_treat, g0, g1):
    run = np.empty(g0.size, np.int64)
    pos = np.empty(g0.size, np.int64)
    for r in prange(g0.size):
        run[r], pos[r] = _single_summit(a_start, a_end, a_treat, g0[r], g1[r])
    return run, pos


def _resolve_zero_slopes(signal, sign, offset, lo, length, window):
    """
    Replace the 2s of ``_fill_regions`` by numpy's sign where a summit can
    depend on it (bases ``lo`` to ``lo + length + 1`` of the region), by 0
    elsewhere.
    """
    zero = np.flatnonzero(sign == 2)
    sign[zero] = 0
    if zero.size == 0:
        return
    region = np.searchsorted(offset, zero, side="right") - 1
    j = zero - offset[region]
    used = (j >= lo[region]) & (j <= lo[region] + length[region] + 1)
    zero, region, j = zero[used], region[used], j[used]
    if zero.size == 0:
        return
    bounds = np.flatnonzero(np.diff(region)) + 1
    for a, b in zip(np.concatenate(([0], bounds)), np.concatenate((bounds, [zero.size]))):
        r = region[a]
        sign[zero[a:b]] = _numpy_signs(signal[offset[r]:offset[r + 1]], j[a:b], window)


def _summits(a_start, a_end, a_treat, g0, g1, p: Params):
    """Summits of the regions ``[g0, g1)`` of runs above the cutoff: (run, position) per summit."""
    if not p.call_summits:
        return _single_summits(a_start, a_end, a_treat, g0, g1)
    window = p.min_length // 2 * 2 + 1
    ps, pe = a_start[g0], a_end[g1 - 1]
    starts = np.maximum(ps - 10, 0)
    lengths = pe + 10 - starts
    runs, positions = [], []
    # Regions in chunks of ~32M bases, to bound the host buffers
    cum = np.cumsum(lengths)
    cuts = np.searchsorted(cum, np.arange(2**25, cum[-1], 2**25), side="right")
    edges = np.unique(np.concatenate(([0], cuts, [g0.size])))
    for lo, hi in zip(edges[:-1], edges[1:]):
        sl = slice(lo, hi)
        offset = np.concatenate(([0], np.cumsum(lengths[sl])))
        total = int(offset[-1])
        signal = np.empty(total, np.float32)
        index = np.empty(total, np.int64)
        sign = np.empty(total, np.int8)
        version31 = p.macs3_version == "3.1"
        _fill_regions(a_start, a_end, a_treat, g0[sl], g1[sl], offset, signal, index, sign,
                      window // 2, version31)
        _resolve_zero_slopes(signal, sign, offset, ps[sl] - starts[sl], pe[sl] - ps[sl], window)
        mark = np.zeros(total, np.int64)
        _mark_summits(a_start, a_end, a_treat, g0[sl], g1[sl], offset, signal, index, sign,
                      mark, version31)
        hits = np.flatnonzero(mark)
        region = np.searchsorted(offset, hits, side="right") - 1
        runs.append(g0[sl][region] + mark[hits] - 1)
        positions.append(starts[sl][region] + hits - offset[region])
    return np.concatenate(runs), np.concatenate(positions)


# =============================================================================
# Driver
# =============================================================================


def _batches(names, sizes, budget_bytes):
    """Chromosomes in batches of at most ``budget_bytes`` of GPU work (one chromosome may exceed it)."""
    cap = max(1, budget_bytes // _BYTES_PER_FRAGMENT)
    out, cur, n = [], [], 0
    for c in names:
        if cur and n + sizes[c] > cap:
            out.append(cur)
            cur, n = [], 0
        cur.append(c)
        n += sizes[c]
    if cur:
        out.append(cur)
    return out


def _tags(fragments, names):
    """+ strand tags at fragment starts, - strand tags at ends - 1, on the GPU."""
    n = np.array([len(fragments[c][0]) for c in names])
    total = int(n.sum())
    pos = cp.asarray(np.concatenate([fragments[c][0] for c in names] + [fragments[c][1] for c in names]))
    pos = pos.astype(cp.int64)
    pos[total:] -= 1
    cid = cp.repeat(cp.arange(len(names), dtype=cp.int64), n.tolist())
    cid = cp.concatenate((cid, cid))
    minus = cp.zeros(pos.size, dtype=bool)
    minus[total:] = True
    return pos, cid, minus


def _run_starts(end):
    prev = cp.concatenate((cp.full(1, -1, cp.int64), end[:-1]))
    same = (prev >> _POS_BITS) == (end >> _POS_BITS)
    return cp.where(same, prev, (end >> _POS_BITS) << _POS_BITS)


def _empty() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=t) for c, t in zip(
        PEAK_COLUMNS, [object, np.int64, np.int64, object, np.int64, object,
                       np.float64, np.float64, np.float64, np.int64])})


def callpeak(fragments: dict, p: Params, budget_bytes: int | None = None) -> pd.DataFrame:
    """
    Narrow peaks of one group of cells.

    Parameters
    ----------
    fragments : dict
        ``{chrom: (starts, ends)}`` numpy arrays of the group's fragments.
    p : Params
        MACS3 options.
    budget_bytes : int, optional
        GPU memory for one batch of chromosomes (default: 60% of the free memory).

    Returns
    -------
    pd.DataFrame
        One row per summit, with the columns of SnapATAC2's ``macs3``
        (``chrom, start, end, name, score, strand, signal_value, p_value,
        q_value, peak``), ``peak`` being the summit offset from ``start``.
    """
    # cupy's pool: with RMM's default (non-pooled) resource, each of the
    # many temporaries here would be a cudaMalloc
    pool = cp.get_default_memory_pool()
    try:
        with cp.cuda.using_allocator(pool.malloc):
            return _callpeak(fragments, p, budget_bytes)
    finally:
        pool.free_all_blocks()


def _callpeak(fragments: dict, p: Params, budget_bytes: int | None) -> pd.DataFrame:
    names = sorted(c for c, (s, _) in fragments.items() if len(s))
    sizes = {c: len(fragments[c][0]) for c in names}
    n_fragments = sum(sizes.values())
    if n_fragments == 0:
        return _empty()
    lambda_bg = np.float32(float(p.extsize) * (2 * n_fragments) / p.gsize)
    ctrl_scale = np.float32(float(p.extsize) / p.llocal)
    if budget_bytes is None:
        free = cp.cuda.Device().mem_info[0] + cp.get_default_memory_pool().free_bytes()
        budget_bytes = int(0.6 * free)
    batches = _batches(names, sizes, budget_bytes)
    keep_on_gpu = len(batches) == 1

    # Pass 1: runs of every chromosome, and the genome-wide length of each
    # distinct (pileup, lambda) pair
    runs, tables = [], []
    for names_b in batches:
        pos, cid, minus = _tags(fragments, names_b)
        end, start, treat, ctrl = _segments(pos, cid, minus, len(names_b), p, lambda_bg, ctrl_scale)
        del pos, cid, minus
        key = (treat.astype(cp.int64) << 32) | ctrl.view(cp.uint32).astype(cp.int64)
        del treat, ctrl
        ukey, inv = cp.unique(key, return_inverse=True)
        del key
        lengths = cp.bincount(inv.ravel(), weights=(end - start).astype(cp.float64), minlength=ukey.size)
        inv = inv.ravel().astype(cp.int32)
        tables.append((ukey.get(), lengths.get().astype(np.int64)))
        runs.append((end, inv) if keep_on_gpu else (end.get(), inv.get()))
        del start, ukey, lengths
    keys = np.unique(np.concatenate([k for k, _ in tables]))
    lengths = np.zeros(keys.size, np.int64)
    for k, l in tables:
        np.add.at(lengths, np.searchsorted(keys, k), l)
    key_treat = (keys >> 32).astype(np.int64)
    key_ctrl = (keys & _POS_MASK).astype(np.uint32).view(np.float32)
    key_p = pscores(key_treat, key_ctrl)
    key_q = qscores(key_p, lengths)
    cutoff = cp.float32(p.log_qvalue)
    key_q_gpu = cp.asarray(key_q)

    # Pass 2: runs above the cutoff, joined into peaks, and their summits
    out = []
    for names_b, (end, inv), (ukey, _) in zip(batches, runs, tables):
        end, inv = cp.asarray(end), cp.asarray(inv)
        to_global = cp.asarray(np.searchsorted(keys, ukey))
        q = key_q_gpu[to_global][inv]
        above = cp.flatnonzero(q > cutoff)
        del q
        if above.size == 0:
            continue
        a_end = end[above]
        a_start = _run_starts(end)[above]
        a_key = to_global[inv[above]]
        del end, inv, above
        brk = cp.flatnonzero((a_start[1:] - a_end[:-1]) > p.max_gap) + 1
        g0 = cp.concatenate((cp.zeros(1, brk.dtype), brk))
        g1 = cp.concatenate((brk, cp.full(1, a_end.size, brk.dtype)))
        keep = (a_end[g1 - 1] - a_start[g0]) >= p.min_length
        g0, g1 = g0[keep].get(), g1[keep].get()
        if g0.size == 0:
            continue
        chrom = (a_start.get() >> _POS_BITS)
        a_start_h = (a_start & _POS_MASK).get()
        a_end_h = (a_end & _POS_MASK).get()
        a_key = a_key.get()
        a_treat = key_treat[a_key].astype(np.float32)
        run, summit = _summits(a_start_h, a_end_h, a_treat, g0, g1, p)
        region = np.searchsorted(g0, run, side="right") - 1
        t, c, k = a_treat[run], key_ctrl[a_key[run]], a_key[run]
        fc = ((t.astype(np.float64) + 1.0) / (c.astype(np.float64) + 1.0)).astype(np.float32)
        q = key_q[k]
        start = a_start_h[g0[region]]
        df = pd.DataFrame({
            "chrom": np.asarray(names_b, dtype=object)[chrom[g0[region]]],
            "start": start,
            "end": a_end_h[g1[region] - 1],
            "name": ".",
            "score": np.minimum(np.floor(np.clip(q.astype(np.float64) * 10.0, 0, 65535)), 1000).astype(np.int64),
            "strand": ".",
            "signal_value": fc.astype(np.float64),
            "p_value": key_p[k].astype(np.float64),
            "q_value": q.astype(np.float64),
            "peak": summit - start,
        })
        out.append(df[fc.astype(np.float64) >= p.fe_cutoff])
    if not out:
        return _empty()
    return pd.concat(out, ignore_index=True)
