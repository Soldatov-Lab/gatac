"""
GPU differential TF binding: TOBIAS BINDetect on :class:`~gatac.tl.footprint.FootprintTracks`.

Reimplements TOBIAS 0.17.5 ``BINDetect`` with default parameters. Every step
gives the same numbers as TOBIAS given the same footprint scores:

* motif matrices (TOBIAS PSSMs, including its reverse-strand matrix) and MOODS
  match thresholds (``threshold_from_p`` ported, no MOODS dependency);
* site scanning on the GPU with MOODS's summation order, so match scores and
  ``>= threshold`` decisions are bit-identical;
* per-TF overlap resolution and the motif-clustering overlap counts per chunk of
  peaks, as TOBIAS's 100 worker chunks do;
* background sampling, quantile normalisation, bound/unbound threshold,
  background log2FC fits and the per-TF statistics; the p-value resampling uses
  NumPy's legacy ``RandomState`` seeded like TOBIAS.

``tobias_compat=True`` (default) reproduces two TOBIAS behaviours that are
arguably bugs; ``False`` uses the corrected behaviour:

* for MEME input the reverse-strand matrix is rebuilt from ``round(p × nsites)``;
  with ``nsites=1`` (as in cisBP MEME files) the reverse strand is scanned with a
  coarse consensus matrix. Corrected: the exact reverse complement.
* the "mean log2FC per peak" groups sites by the *TFBS* coordinates, so every
  site is its own group. Corrected: sites are averaged per peak.
"""

from __future__ import annotations

import itertools
import logging
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Sequence, Union

import cupy as cp
import numba
import numpy as np
import pandas as pd
import polars as pl

from .footprint import FootprintTracks, _fai, _region_bytes

logger = logging.getLogger(__name__)

__all__ = ["bindetect", "BINDetectResult"]

MOODS_WINDOW = 7            # MOODS.scan.Scanner(7)
N_CHUNKS = 100              # BINDetect --split
RAND_WINDOW = 200           # one background position per 200 bp of peak


def _n_threads() -> int:
    return max(1, len(os.sched_getaffinity(0)))


# =============================================================================
# Motifs: TOBIAS-compatible matrices and MOODS thresholds
# =============================================================================


@dataclass
class _Motif:
    id: str
    name: str
    pfm: np.ndarray          # (4, length) probabilities, rows A C G T
    n: int = 20              # nsites (TOBIAS default 20)


def _read_meme(path: Union[str, Path]) -> list[_Motif]:
    """MEME reader following TOBIAS's MotifList.from_file (id, optional name, nsites)."""
    motifs, rows = [], None
    cur = None
    for line in open(path).read().split("\n") + [""]:
        if line.startswith("MOTIF"):
            if rows is not None:
                cur.pfm = np.array(rows).T
                motifs.append(cur)
                rows = None
            parts = line.split()
            cur = _Motif(id=parts[1], name=parts[2] if len(parts) > 2 else "", pfm=np.zeros((4, 0)))
        elif line.startswith("letter-probability matrix"):
            kv = re.sub(r"letter-probability matrix:\s*", "", line.rstrip())
            rows = []
            if kv:
                info = dict(re.split(r"=\s*", p)[:2] for p in re.split(r"(?<!=)\s+", kv) if p)
                if "nsites" in info:
                    cur.n = int(float(info["nsites"]))
        elif rows is not None:
            if re.match(r"^\s*(?![-\s]+)([\d\-\.\se\-]+?)$", line):
                rows.append(list(map(float, line.split())))
            else:
                cur.pfm = np.array(rows).T
                motifs.append(cur)
                rows = None
    return motifs


def _as_motifs(motifs) -> list[_Motif]:
    if isinstance(motifs, (str, Path)):
        return _read_meme(motifs)
    out = []
    for m in motifs:                       # gatac DNAMotif: pwm (length, 4), columns A C G T
        out.append(_Motif(id=str(m.id), name=str(m.name or ""), pfm=np.asarray(m.pwm, np.float64).T.copy(),
                          n=int(getattr(m, "nsites", 20) or 20)))
    return out


def _prefixes(motifs: list[_Motif]) -> list[str]:
    """``filafy(name + "_" + id)``, then TOBIAS's case-insensitive de-duplication with ``_1``, ``_2``."""
    valid = set("-_.abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    pre = ["".join(ch for ch in f"{m.name}_{m.id}" if ch in valid) for m in motifs]
    upper = [p.upper() for p in pre]
    counts = pd.Series(upper).value_counts()
    dup = set(counts.index[counts > 1])
    seen = dict.fromkeys(dup, 1)
    for i, p in enumerate(pre):
        if upper[i] in dup:
            pre[i] = f"{p}_{seen[upper[i]]}"
            seen[upper[i]] += 1
    return pre


def _pssm(pfm: np.ndarray, bg: np.ndarray, ps: float = 0.01) -> np.ndarray:
    bgc = bg.reshape(-1, 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = np.true_divide(pfm + ps * bgc, np.sum(pfm + ps * bgc, axis=0))
        return np.nan_to_num(np.log(p) - np.log(bgc))


def _motif_matrices(m: _Motif, bg: np.ndarray, tobias_compat: bool):
    """Forward and reverse scanner matrices.

    TOBIAS builds the reverse matrix from the rounded counts ``round(p × nsites)``
    (OneMotif.get_reverse); ``tobias_compat=False`` uses the reverse complement of
    the forward matrix (identical score distribution, so the same threshold).
    """
    fwd = _pssm(m.pfm, bg)
    if not tobias_compat:
        return fwd, fwd[::-1, ::-1].copy()
    counts = np.round(m.pfm * m.n).astype(int)
    rc = counts[::-1, ::-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        rpfm = rc / np.sum(rc, axis=0)
    return fwd, _pssm(rpfm, bg[[3, 2, 1, 0]])


@numba.njit(cache=True)
def _threshold_from_p(pssm, bg, p, precision=2000.0):
    """MOODS threshold_from_p_with_precision (alphabet size 4)."""
    a, n = pssm.shape
    mat = np.empty((a, n), np.int64)
    for i in range(n):
        for j in range(a):
            v = precision * pssm[j, i]
            mat[j, i] = np.int64(v + 0.5) if pssm[j, i] > 0.0 else np.int64(v - 0.5)   # C cast truncates
    maxT = 0
    minV = np.iinfo(np.int64).max
    for i in range(n):
        mx = mat[0, i]
        mn = mx
        for j in range(1, a):
            v = mat[j, i]
            if mx < v:
                mx = v
            elif mn > v:
                mn = v
        maxT += mx
        if minV > mn:
            minV = mn
    R = maxT - n * minV
    t0 = np.zeros(R + 1)
    t1 = np.zeros(R + 1)
    for j in range(a):
        t0[mat[j, 0] - minV] += bg[j]
    for i in range(1, n):
        for j in range(a):
            s = mat[j, i] - minV
            for r in range(s, R + 1):
                t1[r] += bg[j] * t0[r - s]
        for r in range(R + 1):
            t0[r] = t1[r]
            t1[r] = 0.0
    total = t0[R]
    if total > p:
        mxs = 0.0
        for i in range(n):
            m_ = -np.inf
            for j in range(a):
                m_ = max(pssm[j, i], m_)
            mxs += m_
        md = np.inf
        for i in range(n):
            m1 = -np.inf
            m2 = -np.inf
            for j in range(a):
                if pssm[j, i] > m1:
                    m2 = m1
                    m1 = pssm[j, i]
                elif pssm[j, i] < m1 and pssm[j, i] > m2:
                    m2 = pssm[j, i]
            md = min(md, m1 - m2)
        return mxs - md / 2
    for r in range(R - 1, -1, -1):
        total += t0[r]
        if total > p:
            return (r + n * minV + 1) / precision
    mns = 0.0
    for i in range(n):
        m_ = np.inf
        for j in range(a):
            m_ = min(pssm[j, i], m_)
        mns += m_
    return mns - 1.0


@numba.njit(parallel=True, cache=True)
def _thresholds_batch(flat, offs, lens, bg, p):
    out = np.empty(len(lens))
    for k in numba.prange(len(lens)):
        out[k] = _threshold_from_p(flat[:, offs[k]:offs[k] + lens[k]].copy(), bg, p)
    return out


def _thresholds(pssms: list[np.ndarray], bg: np.ndarray, p: float) -> np.ndarray:
    lens = np.array([m.shape[1] for m in pssms], np.int64)
    offs = np.concatenate([[0], np.cumsum(lens)[:-1]]).astype(np.int64)
    flat = np.ascontiguousarray(np.concatenate([np.asarray(m, np.float64) for m in pssms], axis=1))
    prev = numba.get_num_threads()
    numba.set_num_threads(min(prev, _n_threads()))
    try:
        return _thresholds_batch(flat, offs, lens, np.asarray(bg, np.float64), p)
    finally:
        numba.set_num_threads(prev)


def _moods_order(pssm: np.ndarray, bg: np.ndarray, window: int = MOODS_WINDOW) -> np.ndarray:
    """MOODS Motif0 summation order: best expected-difference window first, then the
    remaining columns by decreasing expected difference (std::sort, stable below 16)."""
    a, m = pssm.shape
    ed = np.empty(m)
    for i in range(m):
        mx = -np.inf
        for j in range(a):
            mx = max(mx, pssm[j, i])
        v = mx
        for j in range(a):
            v -= bg[j] * pssm[j, i]
        ed[i] = v
    if window >= m:
        return np.arange(m, dtype=np.int32)
    cur = 0.0
    for i in range(window):
        cur += ed[i]
    best, wp = cur, 0
    for i in range(m - window):
        cur -= ed[i]
        cur += ed[i + window]
        if cur > best:
            best, wp = cur, i + 1
    rest = sorted(list(range(wp)) + list(range(wp + window, m)), key=lambda i: -ed[i])
    return np.array(list(range(wp, wp + window)) + rest, dtype=np.int32)


_scan_kernel = cp.RawKernel(r"""
extern "C" __global__ void scan(const unsigned char* seq, const long long n, const double* mats, const int* moff,
                                const int* mlen, const int* order, const double* thr, int* hit_m, long long* hit_p,
                                double* hit_s, unsigned long long* counter, const unsigned long long cap) {
    long long p = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    int k = blockIdx.y;
    int L = mlen[k];
    if (p + L > n) return;
    for (int t = 0; t < L; t++) if (seq[p + t] > 3) return;
    if (L < 7) {
        // MOODS scans windows of 7 inside each run of ACGT. A short motif is a hit at p when the
        // window p..p+7 lies in the run, or, in the run's last 6 windows, when p + L < run end
        // (strict). Runs shorter than 7 go through another code path, emulated on the host.
        int f = L;
        while (f < 7 && p + f < n && seq[p + f] <= 3) f++;
        if (f < 7) {
            int need = 7 - f, b = 0;
            while (b < need && p - 1 - b >= 0 && seq[p - 1 - b] <= 3) b++;
            if (b < need) return;           // run shorter than 7: host
            if (!(L < f)) return;
        }
    }
    const double* M = mats + moff[k] * 4;   // column-major: M[col*4 + base]
    const int* ord = order + moff[k];
    double s = 0.0;
    for (int t = 0; t < L; t++) {
        int col = ord[t];
        unsigned char b = seq[p + col];
        if (b > 3) return;
        s += M[col * 4 + b];
    }
    if (s >= thr[k]) {
        unsigned long long i = atomicAdd(counter, 1ULL);
        if (i < cap) { hit_m[i] = k; hit_p[i] = p; hit_s[i] = s; }
    }
}
""", "scan", options=("--fmad=false",))


def _scan(seq: np.ndarray, mats: list[np.ndarray], orders: list[np.ndarray], thr: np.ndarray):
    """All ``(matrix, position, score)`` with score ≥ threshold over the encoded layout."""
    lens = np.array([m.shape[1] for m in mats], np.int32)
    offs = np.concatenate([[0], np.cumsum(lens)[:-1]]).astype(np.int32)
    seq_d = cp.asarray(seq)
    mats_d = cp.asarray(np.concatenate([np.asarray(m, np.float64).T.reshape(-1) for m in mats]))
    order_d = cp.asarray(np.concatenate(orders).astype(np.int32))
    offs_d, lens_d = cp.asarray(offs), cp.asarray(lens)
    thr_d = cp.asarray(np.asarray(thr, np.float64))
    n = len(seq)
    cap = max(4_000_000, n // 4)
    while True:
        hm = cp.empty(cap, cp.int32)
        hp = cp.empty(cap, cp.int64)
        hs = cp.empty(cap, cp.float64)
        cnt = cp.zeros(1, cp.uint64)
        _scan_kernel(((n + 255) // 256, len(mats)), (256,),
                     (seq_d, cp.int64(n), mats_d, offs_d, lens_d, order_d, thr_d, hm, hp, hs, cnt, cp.uint64(cap)))
        nh = int(cnt.get()[0])
        if nh <= cap:
            return hm[:nh].get(), hp[:nh].get(), hs[:nh].get()
        cap = nh


def _short_run_hits(seq: np.ndarray, mats: list[np.ndarray], orders: list[np.ndarray], thr: np.ndarray,
                    n_mot: int, window: int = MOODS_WINDOW):
    """Hits of MOODS's code path for runs of ACGT shorter than the window (between Ns).

    MOODS (scanner.cpp, ``end - start < l``) packs the run plus dummy ``A``s into a window
    code and, for each position ``i`` of the run, advances the window only if *any* matrix of
    the scanner (one scanner per strand) has a hit for the current code; matrices of length
    <= ``window`` whose window score passes are then reported at ``i`` if they fit in the run.
    """
    valid = seq < 4
    edges = np.flatnonzero(np.diff(np.concatenate([[0], valid.astype(np.int8), [0]])))
    starts, ends = edges[0::2], edges[1::2]
    short = (ends - starts) < window
    if not short.any():
        return np.zeros(0, np.int32), np.zeros(0, np.int64), np.zeros(0)
    out_m, out_p, out_s = [], [], []
    for lo, hi in ((0, n_mot), (n_mot, 2 * n_mot)):              # forward, reverse scanner
        ks = np.arange(lo, hi)
        mlen = np.array([mats[k].shape[1] for k in ks])
        maxlen = mlen.max()
        M = np.zeros((len(ks), 4, maxlen))
        for i, k in enumerate(ks):
            M[i, :, :mlen[i]] = mats[k]
        wp = np.array([orders[k][0] if mlen[i] > window else 0 for i, k in enumerate(ks)])
        la0 = np.zeros(len(ks))                                   # lookahead_scores[0]
        for i, k in enumerate(ks):
            if mlen[i] > window:
                tot = 0.0
                for c in orders[k][window:][::-1]:
                    tot += max(-np.inf, *mats[k][:, c])
                la0[i] = tot
        T = thr[ks]
        rows = np.arange(len(ks))
        cache = {}

        def window_scores(code):
            if code not in cache:
                sc = np.zeros(len(ks))
                for t in range(window):
                    col = np.where(mlen > window, wp + t, t)
                    use = (mlen > window) | (t < mlen)
                    sc = np.where(use, sc + M[rows, code[t], np.minimum(col, maxlen - 1)], sc)
                full = (mlen <= window) & (sc >= T)
                cache[code] = (sc, full, bool(full.any() or ((mlen > window) & (sc + la0 >= T)).any()))
            return cache[code]

        for s0, e0 in zip(starts[short], ends[short]):
            def code_at(w):
                return tuple(int(seq[j]) if s0 <= j < e0 else 0 for j in range(w, w + window))
            w = s0 - 1                                            # initial code: run + dummies, 0 on top
            for i in range(s0, e0):
                if window_scores(code_at(w))[2]:
                    w += 1
                    sc, full, _ = window_scores(code_at(w))
                    for j in np.flatnonzero(full & (mlen <= e0 - i)):
                        out_m.append(ks[j]); out_p.append(i); out_s.append(sc[j])
    return np.array(out_m, np.int32), np.array(out_p, np.int64), np.array(out_s, np.float64)


# =============================================================================
# Site bookkeeping (numba)
# =============================================================================


@numba.njit(cache=True)
def _resolve_overlaps(gstart, chrom, start, end, score, keep):
    """RegionList.resolve_overlaps(priority="higher") on every loc-sorted (TF, chunk) group."""
    for g in range(len(gstart) - 1):
        o, n = gstart[g], gstart[g + 1] - gstart[g]
        i, j = 0, 1
        while i + j < n:
            a, b = o + i, o + i + j
            if not keep[a]:
                i += 1
                j = 1
            elif not keep[b]:
                j += 1
            elif chrom[a] == chrom[b] and start[b] < end[a]:
                if score[a] <= score[b]:      # scores.index(min(scores)) -> 0 on ties
                    keep[a] = False
                else:
                    keep[b] = False
            else:
                i += 1
                j = 1


@numba.njit(cache=True)
def _count_overlaps(gstart, chrom, start, end, name, n_mot, ov, tot):
    """RegionList.count_overlaps per chunk of loc-sorted sites (all TFs): merge same-name
    overlapping neighbours (end = current end), then bp overlap with each following site
    while ``s1_start < s2_end and s1_end > s2_start + 1``."""
    for g in range(len(gstart) - 1):
        o, n = gstart[g], gstart[g + 1] - gstart[g]
        mc = np.empty(n, np.int64)
        ms = np.empty(n, np.int64)
        me = np.empty(n, np.int64)
        mn = np.empty(n, np.int64)
        k = 0
        for t in range(o, o + n):
            if k > 0 and chrom[t] == mc[k - 1] and start[t] < me[k - 1] and name[t] == mn[k - 1]:
                me[k - 1] = end[t]
            else:
                mc[k] = chrom[t]
                ms[k] = start[t]
                me[k] = end[t]
                mn[k] = name[t]
                k += 1
        for i in range(k):
            tot[mn[i]] += me[i] - ms[i]
            j = 1
            while i + j < k:
                b = i + j
                if mc[i] == mc[b] and ms[i] < me[b] and me[i] > ms[b] + 1:
                    bp = min(me[i], me[b]) - max(ms[i], ms[b])
                    ov[mn[i], mn[b]] += bp
                    ov[mn[b], mn[i]] += bp
                    j += 1
                else:
                    break


def _round5(x: np.ndarray) -> np.ndarray:
    """Python ``round(x, 5)`` / ``'{:.5f}'`` semantics (exact formatting where NumPy could differ)."""
    r = np.round(x, 5)
    amb = np.abs(np.abs(x * 1e5 - np.trunc(x * 1e5)) - 0.5) < 1e-6
    if amb.any():
        r[amb] = [float(f"{v:.5f}") for v in x[amb]]
    return r


def _gpu_argsort(key: np.ndarray) -> np.ndarray:
    return cp.argsort(cp.asarray(key)).get()          # stable


# =============================================================================
# Motif clustering (RegionCluster)
# =============================================================================


def _cluster_motifs(prefixes: list[str], ov: np.ndarray, tot: np.ndarray, threshold: float):
    """RegionCluster.cluster(threshold, "average"): returns {prefix: cluster name} and the distance matrix."""
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform

    order = sorted(range(len(prefixes)), key=lambda i: prefixes[i])
    names = [prefixes[i] for i in order]
    n = len(names)
    o = np.array(order)
    O = ov[np.ix_(o, o)].astype(np.float64)
    t = tot[o].astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = np.where(t[:, None] > 0, 1 - O / t[:, None], 1.0)
        d2 = np.where(t[None, :] > 0, 1 - O / t[None, :], 1.0)
    D = np.minimum(d1, d2)
    np.fill_diagonal(D, 0.0)
    if n > 1:
        Z = linkage(squareform(D), "average")
        clusters = {i: [i] for i in range(n)}
        for i, row in enumerate(Z):
            a, b, dist = int(row[0]), int(row[1]), row[2]
            if dist <= threshold:
                clusters[n + i] = clusters[a] + clusters[b] + [n + i]
                del clusters[a], clusters[b]
        members = [[m for m in v if m < n] for v in clusters.values()]
    else:
        members = [[0]]
    rowsum = D.sum(axis=1)
    name2cluster = {}
    for mem in members:
        if len(mem) > 1:
            rep = names[sorted(mem, key=lambda m: rowsum[m])[0]]
        else:
            rep = names[mem[0]]
        for m in mem:
            name2cluster[names[m]] = "C_" + rep
    return name2cluster, pd.DataFrame(D, index=names, columns=names)


# =============================================================================
# Statistics (bindetect / bindetect_functions)
# =============================================================================


def _fast_rolling_mean(arr: np.ndarray, w: int) -> np.ndarray:
    """fast_rolling_math(arr, w, "mean"): float32 accumulator, value at ``i + w // 2``."""
    n = len(arr)
    out = np.full(n, np.nan)
    lf = w // 2
    for i in range(n - w + 1):
        v = np.float32(0.0)
        for j in range(w):
            v = np.float32(np.float64(v) + arr[i + j])
        out[i + lf] = np.float64(v) / (w * 1.0)
    return out


def _sigmoid(x, a, b, L, shift):
    return L / (1 + np.exp(-b * (x - a))) + shift


class _ArrayNorm:
    def __init__(self, func, popt, vmin, vmax):
        self.func, self.popt, self.vmin, self.vmax = func, popt, vmin, vmax

    def normalize(self, arr):
        c = arr * (arr <= self.vmax) + self.vmax * (arr > self.vmax)
        c = c * (c >= self.vmin) + self.vmin * (c < self.vmin)
        f = _sigmoid(c, *self.popt) if self.func == "sigmoid" else c * 0 + self.popt
        return arr * f


def _quantile_normalization(arrays: list[np.ndarray]) -> list[_ArrayNorm]:
    from scipy import interpolate
    from scipy.optimize import curve_fit

    n = len(arrays)
    quantiles = np.linspace(0.05, 0.99, 1000, endpoint=True)
    aq = [np.quantile([0] if sum(a > 0) == 0 else a[a > 0], quantiles) for a in arrays]
    mq = [np.mean([aq[i][j] for i in range(n)]) for j in range(len(quantiles))]
    out = []
    for i in range(n):
        x = aq[i]
        y = mq / x
        pad = 50
        ys = _fast_rolling_mean(np.pad(y, pad, "edge"), pad)[pad:-pad]
        p = interpolate.interp1d(x, ys, kind="linear", fill_value="extrapolate")
        xv = np.linspace(np.min(x), np.max(x), 1000)
        yi = p(xv)
        xv, yi = xv[~np.isnan(yi)], yi[~np.isnan(yi)]
        try:
            L_range = np.diff(np.percentile(yi, [5, 95]))[0]
            bounds = ((np.min(x), -np.inf, 0, 0), (np.max(x), np.inf, L_range, np.inf))
            popt, _ = curve_fit(_sigmoid, xv, yi, bounds=bounds)
            func = "sigmoid"
        except Exception:
            popt, func = np.mean(yi), "constant"
        out.append(_ArrayNorm(func, popt, np.min(x), np.max(x)))
    return out


def _bound_threshold(bgv: np.ndarray, bound_pvalue: float):
    """Bound/unbound threshold from the pooled normalised background (lognormal + mirrored normal)."""
    import scipy.optimize
    import scipy.stats
    import sklearn.mixture
    from scipy.optimize import curve_fit

    bgv = bgv[np.logical_not(np.isclose(bgv, 0.0))]
    x_max = np.percentile(bgv, [99])
    bgv = bgv[bgv < x_max]
    log_vals = np.log(bgv).reshape(-1, 1)
    gmm = sklearn.mixture.GaussianMixture(n_components=2, random_state=1).fit(log_vals)
    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_).flatten()
    ci = np.argmax(means)
    log_params = scipy.stats.lognorm.fit(bgv, f0=stds[ci], fscale=np.exp(means[ci]))
    mode = scipy.optimize.fmin(lambda x: -scipy.stats.lognorm.pdf(x, *log_params), 0, disp=False)[0]
    lx = np.linspace(scipy.stats.lognorm(*log_params).ppf([0.01]), mode, 100)
    lp = scipy.stats.lognorm.pdf(lx, *log_params)
    mx = np.concatenate([lx, np.max(lx) + (lx - np.min(lx))]).flatten()
    mp = np.concatenate([lp, lp[::-1]]).flatten()
    popt, _ = curve_fit(lambda x, std, sc: sc * scipy.stats.norm.pdf(x, mode, std), mx, mp)
    threshold = round(scipy.stats.norm.ppf(1 - bound_pvalue, mode, popt[0]), 5)
    return threshold, mode


def _sample_changes(n: int, bg_mean: float, bg_std: float) -> np.ndarray:
    """TOBIAS: ``np.random.seed(n_obs)``; 100 × ``norm.rvs(bg_mean, bg_std, n_obs)`` → standardised changes.
    One ``standard_normal`` stream of ``100 * n_obs`` draws equals the 100 successive calls."""
    rs = np.random.RandomState(n)
    out = np.empty(100)
    rows = max(1, min(100, 20_000_000 // max(n, 1)))     # bound memory; the stream stays continuous
    for r0 in range(0, 100, rows):
        r1 = min(100, r0 + rows)
        s = (rs.standard_normal((r1 - r0) * n) * bg_std + bg_mean).reshape(r1 - r0, n)
        out[r0:r1] = (s.mean(axis=1) - bg_mean) / ((s.std(axis=1) + bg_std) / 2.0)
    return out


# =============================================================================
# Public API
# =============================================================================


@dataclass
class BINDetectResult:
    """Output of :func:`bindetect`.

    Attributes
    ----------
    results : pandas.DataFrame
        One row per motif, columns as TOBIAS ``bindetect_results.txt``:
        ``output_prefix``, ``name``, ``motif_id``, ``cluster``, ``total_tfbs``,
        ``{group}_mean_score``, ``{group}_bound``, ``{g1}_{g2}_change``,
        ``{g1}_{g2}_pvalue`` (full precision; TOBIAS prints 6 digits) and
        ``{g1}_{g2}_highlighted``.
    sites : pandas.DataFrame or None
        One row per TFBS (TOBIAS ``*_overview.txt`` content for all TFs): position,
        motif, match score, strand, peak, normalised footprint score and bound
        call per group, log2FC per comparison.
    clusters : pandas.DataFrame
        Mean of each numeric column per motif cluster (TOBIAS "Motif clusters" sheet).
    distances : pandas.DataFrame
        Motif distance matrix from TFBS overlap (``*_distances.txt``).
    threshold : float
        Bound/unbound footprint score threshold.
    pseudo : float
        Pseudocount of the log2FCs.
    groups : list of str
    comparisons : list of tuple
    params : dict
    """

    results: pd.DataFrame
    sites: Optional[pd.DataFrame]
    clusters: pd.DataFrame
    distances: pd.DataFrame
    threshold: float
    pseudo: float
    groups: list[str]
    comparisons: list[tuple[str, str]]
    params: dict

    def __repr__(self) -> str:
        n_sites = "no sites kept" if self.sites is None else f"{len(self.sites):,} sites"
        return (f"BINDetectResult({len(self.results)} motifs, {n_sites}, groups {self.groups}, "
                f"threshold {self.threshold})")


def bindetect(
    tracks: FootprintTracks,
    motifs,
    genome_fasta: Union[None, str, Path] = None,
    *,
    groups: Optional[Sequence[str]] = None,
    comparisons: Union[Literal["all", "time_series"], Sequence[tuple[str, str]]] = "all",
    motif_pvalue: float = 1e-4,
    bound_pvalue: float = 0.001,
    pseudo: Optional[float] = None,
    cluster_threshold: float = 0.5,
    tobias_compat: bool = True,
    return_sites: bool = True,
    verbose: bool = True,
) -> BINDetectResult:
    """
    Differential TF binding from footprints (GPU TOBIAS BINDetect).

    Scans the peaks of ``tracks`` for every motif, scores each binding site by
    the footprint score at its centre in each group, calls bound sites and
    tests per motif whether footprints change between groups, reproducing
    TOBIAS 0.17.5 ``BINDetect`` with default parameters.

    Parameters
    ----------
    tracks : FootprintTracks
        Output of :func:`gatac.tl.footprint`.
    motifs : str, Path or list of DNAMotif
        MEME file (read like TOBIAS, including ``nsites``) or GATAC motifs
        (e.g. a subset from :func:`gatac.tl.read_motifs`). Pass the MEME path to
        reproduce TOBIAS exactly: :class:`~gatac.tl.DNAMotif` stores clipped,
        renormalised probabilities and no ``nsites`` (20 is assumed).
    genome_fasta : str or Path, optional
        Genome FASTA (default: the one used for ``tracks``).
    groups : sequence of str, optional
        Groups (conditions) to compare, default all groups of ``tracks``.
    comparisons : "all", "time_series" or list of (group1, group2)
        All pairs in group order (TOBIAS default), consecutive pairs
        (``--time-series``), or explicit pairs.
    motif_pvalue : float, default 1e-4
        Motif match p-value (MOODS threshold).
    bound_pvalue : float, default 0.001
        P-value of the bound/unbound footprint threshold.
    pseudo : float, optional
        Log2FC pseudocount (default: half the mode of the background scores).
    cluster_threshold : float, default 0.5
        Distance threshold of the motif clustering by TFBS overlap.
    tobias_compat : bool, default True
        Reproduce TOBIAS's reverse-strand matrix and per-site (not per-peak)
        log2FC averaging; False uses the corrected behaviours.
    return_sites : bool, default True
        Return the per-site table.
    verbose : bool, default True
        Log progress.

    Returns
    -------
    BINDetectResult

    Examples
    --------
    >>> tracks = ga.tl.footprint(adata, "cell_type", "fragments.parquet", "GRCh38.fa",
    ...                          groups=["CD14 Mono", "CD4 Memory"])
    >>> res = ga.tl.bindetect(tracks, "cisBP_human.meme")
    >>> res.results.sort_values("CD14 Mono_CD4 Memory_change").head()
    """
    import scipy.stats

    T = {}
    t0 = time.perf_counter()
    fasta = Path(genome_fasta or tracks.params["genome_fasta"])
    fai = _fai(fasta)
    groups = [str(g) for g in (groups or tracks.groups)]
    missing = set(groups) - set(tracks.groups)
    if missing:
        raise ValueError(f"groups not in tracks: {sorted(missing)}")
    if comparisons == "all":
        comps = list(itertools.combinations(groups, 2))
    elif comparisons == "time_series":
        comps = list(zip(groups[:-1], groups[1:]))
    else:
        comps = [(str(a), str(b)) for a, b in comparisons]
        bad = {g for c in comps for g in c} - set(groups)
        if bad:
            raise ValueError(f"comparison groups not in groups: {sorted(bad)}")

    # ---- merged peaks in loc_sort order; concatenated layout with one N after each peak ----
    corder = sorted(tracks.peaks)
    reg_c = np.concatenate([np.full(len(tracks.peaks[c]), i, np.int64) for i, c in enumerate(corder)])
    reg_s = np.concatenate([tracks.peaks[c][:, 0] for c in corder])
    reg_e = np.concatenate([tracks.peaks[c][:, 1] for c in corder])
    reg_len = reg_e - reg_s
    n_reg = len(reg_s)
    if n_reg == 0:
        raise ValueError("tracks have no peaks")
    rstart = np.concatenate([[0], np.cumsum(reg_len + 1)[:-1]])
    ntot = int(rstart[-1] + reg_len[-1] + 1)
    per_chunk = int(np.ceil(n_reg / float(N_CHUNKS)))
    raw = np.full(ntot, ord("N"), np.uint8)
    for ci, c in enumerate(corder):
        sel = np.flatnonzero(reg_c == ci)
        if len(sel) == 0:
            continue
        if reg_e[sel].max() > fai[c][0]:
            raise ValueError(f"peaks beyond the end of {c}")
        lens = reg_len[sel]
        within = np.arange(lens.sum()) - np.repeat(np.cumsum(lens) - lens, lens)
        raw[np.repeat(rstart[sel], lens) + within] = _region_bytes(fasta, fai, c, reg_s[sel], reg_e[sel])
    lut = np.full(256, 4, np.uint8)
    for ch, v in zip(b"ACGTacgt", [0, 1, 2, 3, 0, 1, 2, 3]):
        lut[ch] = v
    seq = lut[raw]

    # get_gc_content per chunk (G/C = 1, A/T = 0, other = 0.5), mean over chunks
    is_gc = np.isin(raw, np.frombuffer(b"GgCc", np.uint8))
    is_at = np.isin(raw, np.frombuffer(b"AaTt", np.uint8))
    w = is_gc + 0.5 * (~is_at & ~is_gc)
    w[rstart + reg_len] = 0.0
    cs = np.concatenate([[0.0], np.cumsum(w)])             # exact: sums of halves
    gcs = []
    for i0 in range(0, n_reg, per_chunk):
        i1 = min(i0 + per_chunk, n_reg)
        gcs.append(float(cs[rstart[i1 - 1] + reg_len[i1 - 1]] - cs[rstart[i0]]) / float(reg_len[i0:i1].sum()))
    gc = np.mean(gcs)
    del raw, is_gc, is_at, w, cs
    T["layout"] = time.perf_counter() - t0

    # ---- motifs ----
    t0 = time.perf_counter()
    mlist = _as_motifs(motifs)
    if not mlist:
        raise ValueError("no motifs")
    bg = np.array([(1 - gc) / 2.0, gc / 2.0, gc / 2.0, (1 - gc) / 2.0])
    prefixes = _prefixes(mlist)
    fwd, rev = zip(*[_motif_matrices(m, bg, tobias_compat) for m in mlist])
    n_mot = len(mlist)
    thr = _thresholds(list(fwd), bg, motif_pvalue)
    rev_bg = bg[[3, 2, 1, 0]]
    mats = list(fwd) + list(rev)
    orders = [_moods_order(P, bg) for P in fwd] + [_moods_order(P, rev_bg) for P in rev]
    mlen = np.array([P.shape[1] for P in mats])
    T["motifs"] = time.perf_counter() - t0

    # ---- footprints over the layout (uncovered = 0, like a bigwig gap) and background sample ----
    t0 = time.perf_counter()
    flay = tracks.layouts["footprint"]
    lay_idx = np.full(ntot, -1, np.int64)
    for ci, c in enumerate(corder):
        sel = np.flatnonzero(reg_c == ci)
        lens = reg_len[sel]
        within = np.arange(lens.sum()) - np.repeat(np.cumsum(lens) - lens, lens)
        gpos = np.repeat(reg_s[sel], lens) + within
        lay_idx[np.repeat(rstart[sel], lens) + within] = flay.index(c, gpos) if c in flay.intervals else -1
    covered = lay_idx >= 0
    fpl = {}
    for g in groups:
        v = np.zeros(ntot, np.float32)
        v[covered] = tracks.values["footprint"][g][lay_idx[covered]]
        fpl[g] = v
    del lay_idx, covered
    # random.seed(reglen); random.sample(range(reglen), max(1, reglen / 200)) depends only on the length
    cache = {}
    for L in np.unique(reg_len):
        random.seed(int(L))
        cache[int(L)] = np.array(random.sample(range(int(L)), max(1, int(int(L) / RAND_WINDOW))), np.int64)
    bg_idx = np.concatenate([rstart[r] + cache[int(reg_len[r])] for r in range(n_reg)])
    bg_raw = {g: fpl[g][bg_idx].astype(np.float64) for g in groups}
    T["footprints"] = time.perf_counter() - t0

    # ---- scan, then sites in TOBIAS append order (peak, strand, motif, position) ----
    t0 = time.perf_counter()
    hm, hp, hs = _scan(seq, mats, orders, np.concatenate([thr, thr]))
    sm, sp, ss = _short_run_hits(seq, mats, orders, np.concatenate([thr, thr]), n_mot)
    if len(sm):
        hm, hp, hs = np.concatenate([hm, sm]), np.concatenate([hp, sp]), np.concatenate([hs, ss])
    del seq
    T["scan"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    rid = np.searchsorted(rstart, hp, side="right") - 1
    minus = hm >= n_mot
    motif = (hm % n_mot).astype(np.int64)
    within = hp - rstart[rid]
    if n_reg >= 2**24 or within.max(initial=0) >= 2**27 or n_mot >= 2**12:
        raise ValueError("too many peaks, too long peaks or too many motifs for the site sort keys")
    o = _gpu_argsort((rid << 40) | (minus.astype(np.int64) << 39) | (motif << 27) | within)
    rid, minus, motif, hp, hs = rid[o], minus[o], motif[o], hp[o], hs[o]
    L = mlen[hm[o]]
    start = reg_s[rid] + (hp - rstart[rid])
    end = start + L
    chromc = reg_c[rid]
    score = _round5(hs)                                    # round(match.score, 5)
    chunk = rid // per_chunk
    # resolve_overlaps per (motif, chunk), loc-sorted; end = start + motif length never breaks a tie
    if start.max(initial=0) >= 2**30 or len(corder) >= 2**6:
        raise ValueError("chromosome too long or too many chromosomes for the site sort keys")
    o2 = _gpu_argsort((motif << 44) | (chunk << 36) | (chromc << 30) | start)
    brk = np.flatnonzero((np.diff(motif[o2]) != 0) | (np.diff(chunk[o2]) != 0)) + 1
    gstart = np.concatenate([[0], brk, [len(o2)]]).astype(np.int64)
    k2 = np.ones(len(o2), np.bool_)
    _resolve_overlaps(gstart, chromc[o2], start[o2], end[o2], score[o2], k2)
    keep = np.empty(len(o2), np.bool_)
    keep[o2] = k2

    # TFBS overlap between motifs per chunk (count_overlaps), for the motif clustering
    sel = np.flatnonzero(keep)
    name_rank = np.empty(n_mot, np.int64)
    name_rank[sorted(range(n_mot), key=lambda i: prefixes[i])] = np.arange(n_mot)
    nr = name_rank[motif[sel]]
    o3 = _gpu_argsort((L[sel].astype(np.int64) << 12) | nr)
    o3 = o3[_gpu_argsort(((chunk[sel] << 36) | (chromc[sel] << 30) | start[sel])[o3])]
    s3 = sel[o3]
    brk = np.flatnonzero(np.diff(chunk[s3]) != 0) + 1
    gstart = np.concatenate([[0], brk, [len(s3)]]).astype(np.int64)
    ov = np.zeros((n_mot, n_mot), np.int64)
    tot = np.zeros(n_mot, np.int64)
    _count_overlaps(gstart, chromc[s3], start[s3], end[s3], motif[s3], n_mot, ov, tot)
    T["sites"] = time.perf_counter() - t0

    # ---- normalisation, bound threshold, background log2FC ----
    t0 = time.perf_counter()
    if len(groups) > 1:
        norms = dict(zip(groups, _quantile_normalization([bg_raw[g] for g in groups])))
        bgn = {}
        for g in groups:
            v = norms[g].normalize(bg_raw[g])
            v[v < 0] = 0
            bgn[g] = v
    else:
        norms = {g: _ArrayNorm("constant", 1.0, 0, 1) for g in groups}
        bgn = dict(bg_raw)
    if len(bg_idx) < 1000:
        logger.warning(f"only {len(bg_idx)} background positions; threshold and normalisation may be unstable")
    threshold, mode = _bound_threshold(np.array([bgn[g] for g in groups]).flatten(), bound_pvalue)
    pseudo = mode / 2.0 if pseudo is None else float(pseudo)
    log2fc_params = {}
    for g1, g2 in comps:
        s1, s2 = bgn[g1], bgn[g2]
        inc = (s1 > 0) | (s2 > 0)
        l2 = np.log2((s1[inc] + pseudo) / (s2[inc] + pseudo))
        lo, hi = np.percentile(l2, [1, 99])
        log2fc_params[(g1, g2)] = scipy.stats.norm.fit(l2[(l2 >= lo) & (l2 <= hi)])
    T["background"] = time.perf_counter() - t0

    # ---- per-site table, sorted per TF by (chrom, start, end) as process_tfbs sorts it ----
    t0 = time.perf_counter()
    mid = hp[sel] + (L[sel] // 2)
    o4 = _gpu_argsort((motif[sel] << 40) | (chromc[sel] << 32) | start[sel])
    s4 = sel[o4]
    mid = mid[o4]
    cols = {}
    for g in groups:
        v = norms[g].normalize(_round5(fpl[g][mid].astype(np.float64)))
        cols[f"{g}_score"] = np.round(np.where(v > 0, v, 0), 5)
        cols[f"{g}_bound"] = (cols[f"{g}_score"] > threshold).astype(np.int8)
    for g1, g2 in comps:
        cols[f"{g1}_{g2}_log2fc"] = np.round(np.log2((cols[f"{g1}_score"] + pseudo) / (cols[f"{g2}_score"] + pseudo)), 5)
    m4 = motif[s4]
    bounds = np.concatenate([[0], np.flatnonzero(np.diff(m4)) + 1, [len(m4)]])
    tf_ids = m4[bounds[:-1]]
    T["site_table"] = time.perf_counter() - t0

    # ---- per-TF summaries ----
    t0 = time.perf_counter()
    res = pd.DataFrame({
        "output_prefix": prefixes,
        "name": [m.name for m in mlist],
        "motif_id": [m.id for m in mlist],
        "total_tfbs": np.zeros(n_mot, np.int64),
    })
    res.loc[tf_ids, "total_tfbs"] = np.diff(bounds)
    for g in groups:
        sc, bd = cols[f"{g}_score"], cols[f"{g}_bound"]
        ms = np.full(n_mot, np.nan)
        ms[tf_ids] = [np.round(np.mean(sc[a:b]), 5) for a, b in zip(bounds[:-1], bounds[1:])]
        nb = np.zeros(n_mot, np.int64)
        nb[tf_ids] = np.add.reduceat(bd.astype(np.int64), bounds[:-1]) if len(bd) else []
        res[f"{g}_mean_score"] = ms
        res[f"{g}_bound"] = nb
    chrom_names = np.array(corder)
    pstart, pend = reg_s[rid[s4]], reg_e[rid[s4]]
    sample_cache: dict[tuple, np.ndarray] = {}
    for g1, g2 in comps:
        base = f"{g1}_{g2}"
        inc = np.flatnonzero((cols[f"{g1}_score"] > 0) | (cols[f"{g2}_score"] > 0))
        if tobias_compat:
            # groupby("peak_id") on "chrom_start_end" of the *TFBS*: one site per group, string order
            key = (pl.DataFrame({"m": m4[inc], "c": chrom_names[chromc[s4[inc]]], "s": start[s4[inc]],
                                 "e": end[s4[inc]]})
                   .with_columns(k=pl.concat_str([pl.col("c"), pl.col("s").cast(pl.String),
                                                  pl.col("e").cast(pl.String)], separator="_"))
                   .with_row_index("i").sort(["m", "k"], maintain_order=True))
            o5 = key["i"].to_numpy()
            ii = inc[o5]
            obs_all = cols[f"{base}_log2fc"][ii]
            mm = m4[ii]
        else:
            # mean log2FC per peak
            df = pd.DataFrame({"m": m4[inc], "c": chromc[s4[inc]], "s": pstart[inc], "e": pend[inc],
                               "v": cols[f"{base}_log2fc"][inc]})
            agg = df.groupby(["m", "c", "s", "e"], sort=True)["v"].mean().reset_index()
            obs_all, mm = agg["v"].to_numpy(), agg["m"].to_numpy()
        mb = np.concatenate([[0], np.flatnonzero(np.diff(mm)) + 1, [len(mm)]]) if len(mm) else np.array([0])
        obs_tf = mm[mb[:-1]]
        bg_mean, bg_std = log2fc_params[(g1, g2)]
        change, n_obs = {}, {}
        for a, b, m in zip(mb[:-1], mb[1:], obs_tf):
            om, osd = scipy.stats.norm.fit(obs_all[a:b])
            change[m] = np.round((om - bg_mean) / np.mean([osd, bg_std]), 5) if om != bg_mean else 0
            n_obs[m] = b - a
        uniq = sorted({n for n in n_obs.values() if (n, bg_mean, bg_std) not in sample_cache}, reverse=True)
        with ThreadPoolExecutor(_n_threads()) as ex:
            for n, s in zip(uniq, ex.map(lambda n: _sample_changes(n, bg_mean, bg_std), uniq)):
                sample_cache[(n, bg_mean, bg_std)] = s
        res[f"{base}_change"] = [change.get(m, np.nan) for m in range(n_mot)]
        res[f"{base}_pvalue"] = [scipy.stats.ttest_1samp(sample_cache[(n_obs[m], bg_mean, bg_std)], change[m])[1]
                                 if m in change else np.nan for m in range(n_mot)]
    T["per_tf"] = time.perf_counter() - t0

    # ---- clusters and highlighting ----
    name2cluster, distances = _cluster_motifs(prefixes, ov, tot, cluster_threshold)
    if tobias_compat:
        # TOBIAS replaces every prefix by its motif name inside each cluster name
        convert = [(p, m.name) for p, m in zip(prefixes, mlist)]
        renamed = {}
        for cname in set(name2cluster.values()):
            new = cname
            for p, nm in convert:
                new = new.replace(p, nm)
            renamed[cname] = new
        cluster = [renamed[name2cluster[p]] for p in prefixes]
    else:
        disp = {p: (m.name or m.id) for p, m in zip(prefixes, mlist)}
        cluster = ["C_" + disp[name2cluster[p][2:]] for p in prefixes]
    res.insert(3, "cluster", cluster)
    clusters = res.groupby("cluster").mean(numeric_only=True).reset_index()
    for g1, g2 in comps:
        base = f"{g1}_{g2}"
        changes = res[f"{base}_change"].to_numpy(float)
        pv = np.array([float(f"{p:.5E}") if np.isfinite(p) else np.nan for p in res[f"{base}_pvalue"]])
        pos = pv[pv > 0]
        pval_min = np.percentile(pos, 5) if len(pos) >= 1 else 1.0
        cmin, cmax = np.percentile(changes, [5, 95])
        res[f"{base}_highlighted"] = (changes < cmin) | (changes > cmax) | (pv < pval_min)

    sites = None
    if return_sites:
        sites = pd.DataFrame({
            "chrom": pd.Categorical.from_codes(chromc[s4], categories=corder),
            "start": start[s4],
            "end": end[s4],
            "motif": pd.Categorical.from_codes(m4, categories=prefixes),
            "score": score[s4],
            "strand": pd.Categorical.from_codes(minus[s4].astype(np.int8), categories=["+", "-"]),
            "peak_start": pstart,
            "peak_end": pend,
            **cols,
        }, copy=False)                     # no consolidation copy of a table with 10^7+ rows
    T["total"] = sum(T.values())
    logger.info("bindetect timings: " + ", ".join(f"{k} {v:.1f}s" for k, v in T.items()))
    return BINDetectResult(
        results=res,
        sites=sites,
        clusters=clusters,
        distances=distances,
        threshold=threshold,
        pseudo=pseudo,
        groups=groups,
        comparisons=comps,
        params=dict(motif_pvalue=motif_pvalue, bound_pvalue=bound_pvalue, gc=gc, background=bg,
                    motif_thresholds=dict(zip(prefixes, thr)), log2fc_params=log2fc_params,
                    cluster_threshold=cluster_threshold, tobias_compat=tobias_compat,
                    n_background=len(bg_idx), timings=T),
    )
