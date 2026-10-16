"""
GPU differential TF binding: TOBIAS BINDetect on :class:`~gatac.tobias.footprint.FootprintTracks`.

Reimplements TOBIAS 0.17.5 ``BINDetect`` with default parameters. Every step
gives the same numbers as TOBIAS given the same footprint scores:

* motif matrices (TOBIAS PSSMs, including its reverse-strand matrix) and MOODS
  match thresholds (``threshold_from_p`` ported, no MOODS dependency);
* site scanning on the GPU with MOODS's summation order, so match scores and
  ``>= threshold`` decisions are bit-identical;
* per-TF overlap resolution and the motif-clustering overlap counts per chunk of
  peaks, as TOBIAS's 100 worker chunks do;
* the sites depend only on peaks and motifs: :func:`bindetect_sites` builds them
  once (:class:`SiteSet`) for any number of :func:`bindetect` calls;
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

from .._cupy_compat import to_device
from .footprint import FootprintTracks, _fai, _region_bytes

logger = logging.getLogger(__name__)

__all__ = ["bindetect", "bindetect_sites", "BINDetectResult", "SiteSet"]

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


def _scan_device(seq_d: cp.ndarray, mats: list[np.ndarray], orders: list[np.ndarray], thr: np.ndarray):
    """All ``(matrix, position, score)`` with score ≥ threshold over the encoded layout (device arrays)."""
    lens = np.array([m.shape[1] for m in mats], np.int32)
    offs = np.concatenate([[0], np.cumsum(lens)[:-1]]).astype(np.int32)
    mats_d = cp.asarray(np.concatenate([np.asarray(m, np.float64).T.reshape(-1) for m in mats]))
    order_d = cp.asarray(np.concatenate(orders).astype(np.int32))
    offs_d, lens_d = cp.asarray(offs), cp.asarray(lens)
    thr_d = cp.asarray(np.asarray(thr, np.float64))
    n = len(seq_d)
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
            return hm[:nh], hp[:nh], hs[:nh]
        del hm, hp, hs
        cap = nh


def _scan(seq: np.ndarray, mats: list[np.ndarray], orders: list[np.ndarray], thr: np.ndarray):
    """``_scan_device`` on a host sequence, returning host arrays."""
    return tuple(a.get() for a in _scan_device(cp.asarray(seq), mats, orders, thr))


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
# Site bookkeeping
# =============================================================================


@numba.njit(cache=True)
def _resolve_overlaps(gstart, chrom, start, end, score, keep):
    """RegionList.resolve_overlaps(priority="higher") on every loc-sorted (TF, chunk) group.
    Host reference of ``_resolve_kernel`` (kept for tests)."""
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
    while ``s1_start < s2_end and s1_end > s2_start + 1``. Host reference of ``_count_kernel``."""
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


# resolve_overlaps, one thread per loc-sorted (TF, chunk) group; groups are independent
_resolve_kernel = cp.RawKernel(r"""
extern "C" __global__ void resolve(const long long* gstart, const long long n_groups, const int* chrom,
                                   const long long* start, const long long* end, const double* score,
                                   unsigned char* keep) {
    long long g = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (g >= n_groups) return;
    long long o = gstart[g], n = gstart[g + 1] - o, i = 0, j = 1;
    while (i + j < n) {
        long long a = o + i, b = o + i + j;
        if (!keep[a]) { i++; j = 1; }
        else if (!keep[b]) { j++; }
        else if (chrom[a] == chrom[b] && start[b] < end[a]) {
            if (score[a] <= score[b]) keep[a] = 0; else keep[b] = 0;
        } else { i++; j = 1; }
    }
}
""", "resolve")

# count_overlaps pair scan, one thread per merged site; integer sums, so atomics keep it exact
_count_kernel = cp.RawKernel(r"""
extern "C" __global__ void count_ov(const long long n, const int* chunk, const int* chrom, const long long* ms,
                                    const long long* me, const int* mn, const long long n_mot,
                                    unsigned long long* ov, unsigned long long* tot) {
    long long i = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    atomicAdd(&tot[mn[i]], (unsigned long long)(me[i] - ms[i]));
    for (long long b = i + 1; b < n; b++) {
        if (chunk[b] != chunk[i] || chrom[b] != chrom[i] || !(ms[i] < me[b] && me[i] > ms[b] + 1)) break;
        long long bp = (me[i] < me[b] ? me[i] : me[b]) - (ms[i] > ms[b] ? ms[i] : ms[b]);
        atomicAdd(&ov[mn[i] * n_mot + mn[b]], (unsigned long long)bp);
        atomicAdd(&ov[mn[b] * n_mot + mn[i]], (unsigned long long)bp);
    }
}
""", "count_ov")

# Python string order of "start_end" (TOBIAS groups sites by that id): 21 characters of 4 bits,
# pad 0 < digits 1..10 < "_" 11, packed into two words compared high then low
_strkey_kernel = cp.ElementwiseKernel("int64 s, int64 e", "uint64 hi, uint64 lo", r"""
    unsigned char buf[21], tmp[20];
    int n = 0, t = 0;
    long long v = s;
    do { tmp[t++] = (unsigned char)(v % 10); v /= 10; } while (v > 0);
    while (t > 0) buf[n++] = 1 + tmp[--t];
    buf[n++] = 11;
    v = e;
    do { tmp[t++] = (unsigned char)(v % 10); v /= 10; } while (v > 0);
    while (t > 0) buf[n++] = 1 + tmp[--t];
    while (n < 21) buf[n++] = 0;
    unsigned long long h = 0, l = 0;
    for (int k = 0; k < 16; k++) h = (h << 4) | buf[k];
    for (int k = 16; k < 21; k++) l = (l << 4) | buf[k];
    hi = h; lo = l;
""", "tobias_strkey")


def _launch(kernel, n, args):
    kernel(((int(n) + 255) // 256,), (256,), args)


def _round5(x: np.ndarray) -> np.ndarray:
    """Python ``round(x, 5)`` / ``'{:.5f}'`` semantics (exact formatting where NumPy could differ)."""
    r = np.round(x, 5)
    amb = np.abs(np.abs(x * 1e5 - np.trunc(x * 1e5)) - 0.5) < 1e-6
    if amb.any():
        r[amb] = [float(f"{v:.5f}") for v in x[amb]]
    return r


# Elements per chunk of the host-side elementwise steps over all sites (32 MB of float64)
_CHUNK = 1 << 22


def _gather_dev(values: cp.ndarray, idx: cp.ndarray) -> cp.ndarray:
    """``values[idx]`` as float64, 0 where ``idx`` is -1 (a position no track covers)."""
    x = cp.zeros(len(idx), cp.float64)
    ok = idx >= 0
    x[ok] = values[idx[ok]]
    return x


def _log2fc(s1: np.ndarray, s2: np.ndarray, pseudo: float) -> np.ndarray:
    """Per-site ``round(log2((s1 + pseudo) / (s2 + pseudo)), 5)``, in chunks."""
    out = np.empty(len(s1))
    for a in range(0, len(s1), _CHUNK):
        b = a + _CHUNK
        out[a:b] = np.round(np.log2((s1[a:b] + pseudo) / (s2[a:b] + pseudo)), 5)
    return out


def _round5_dev(x: cp.ndarray) -> cp.ndarray:
    """``_round5`` on the device: ``cp.around`` equals ``np.round`` bit for bit; the rare values
    within 1e-6 of a rounding tie take the host string path."""
    r = cp.around(x, 5)
    y = x * 1e5
    amb = cp.flatnonzero(cp.abs(cp.abs(y - cp.trunc(y)) - 0.5) < 1e-6)
    if len(amb):
        r[amb] = cp.asarray([float(f"{v:.5f}") for v in x[amb].get()])
    return r


def _segments(key: cp.ndarray) -> cp.ndarray:
    """Start offsets of runs of equal values in a sorted key, plus the end."""
    brk = cp.flatnonzero(key[1:] != key[:-1]) + 1
    return cp.concatenate([cp.zeros(1, cp.int64), brk, cp.asarray([len(key)], cp.int64)])


def _stable_order(*keys: cp.ndarray) -> cp.ndarray:
    """Lexicographic order by ``keys`` (most significant first), by successive stable sorts."""
    o = cp.argsort(keys[-1])
    for k in keys[-2::-1]:
        o = o[cp.argsort(k[o])]
    return o


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
    # bound memory (one chunk per thread of the pool); the stream stays continuous
    rows = max(1, min(100, 2_000_000 // max(n, 1)))
    for r0 in range(0, 100, rows):
        r1 = min(100, r0 + rows)
        s = (rs.standard_normal((r1 - r0) * n) * bg_std + bg_mean).reshape(r1 - r0, n)
        out[r0:r1] = (s.mean(axis=1) - bg_mean) / ((s.std(axis=1) + bg_std) / 2.0)
    return out


# =============================================================================
# Public API
# =============================================================================


@dataclass
class SiteSet:
    """Motif binding sites in a peak set: everything BINDetect computes that does not depend on
    the footprints. Build once with :func:`bindetect_sites` and pass to :func:`bindetect` for
    every set of pseudobulks footprinted on the same peaks.

    Site arrays are sorted per motif by (chromosome, start), the order TOBIAS processes them in;
    ``motif_offsets[k]:motif_offsets[k + 1]`` are the sites of motif ``k``.
    """

    chroms: list[str]
    peaks: dict[str, np.ndarray]
    peak_chrom: np.ndarray
    peak_start: np.ndarray
    peak_end: np.ndarray
    motifs: list[_Motif]
    prefixes: list[str]
    thresholds: np.ndarray
    gc: float
    background: np.ndarray
    chrom: np.ndarray
    start: np.ndarray
    end: np.ndarray
    motif: np.ndarray
    score: np.ndarray
    minus: np.ndarray
    peak: np.ndarray
    motif_offsets: np.ndarray
    bg_chrom: np.ndarray
    bg_pos: np.ndarray
    overlap: np.ndarray
    overlap_total: np.ndarray
    tobias_compat: bool
    motif_pvalue: float
    timings: dict

    def __len__(self) -> int:
        return len(self.start)

    def __repr__(self) -> str:
        return (f"SiteSet({len(self):,} sites, {len(self.motifs)} motifs, {len(self.peak_start):,} peaks, "
                f"tobias_compat={self.tobias_compat})")


def _merged_peaks(peaks, fai: dict, tobias_compat: bool) -> dict[str, np.ndarray]:
    from .footprint import _merge, _peaks_from_input

    if isinstance(peaks, FootprintTracks):
        return dict(peaks.peaks)
    df = _peaks_from_input(peaks, None)
    df = df[df["chrom"].isin(fai)]
    return {c: _merge(sub[["start", "end"]].to_numpy(np.int64), tobias_compat)
            for c, sub in df.groupby("chrom", sort=False)}


def bindetect_sites(
    peaks,
    motifs,
    genome_fasta: Union[None, str, Path] = None,
    *,
    motif_pvalue: float = 1e-4,
    tobias_compat: bool = True,
    verbose: bool = True,
) -> SiteSet:
    """
    Motif binding sites of a peak set, for :func:`bindetect` (GPU).

    Scans the peaks for every motif with MOODS's thresholds and summation order,
    resolves overlapping sites of the same motif and counts the overlap between
    motifs for the motif clustering, exactly as TOBIAS BINDetect does. None of
    this depends on the footprints, so one site set serves every call of
    :func:`bindetect` on tracks footprinted over the same peaks.

    Parameters
    ----------
    peaks : FootprintTracks, BED path, DataFrame or sequence of ``chrom:start-end``
        The peaks (merged as TOBIAS does). Passing the tracks guarantees they
        match the footprinted regions.
    motifs : str, Path or list of DNAMotif
        MEME file (read like TOBIAS, including ``nsites``) or GATAC motifs.
    genome_fasta : str or Path, optional
        Genome FASTA (default: the one of the tracks, when ``peaks`` are tracks).
    motif_pvalue : float, default 1e-4
        Motif match p-value (MOODS threshold).
    tobias_compat : bool, default True
        TOBIAS's reverse-strand motif matrix (see :func:`bindetect`); must match
        the ``tobias_compat`` of the :func:`bindetect` calls.
    verbose : bool, default True
        Log timings.

    Returns
    -------
    SiteSet

    Examples
    --------
    >>> sites = ga.tobias.bindetect_sites(tracks, "cisBP_human.meme")
    >>> for tracks in per_patient_tracks:
    ...     res = ga.tobias.bindetect(tracks, sites=sites)
    """
    T = {}
    t0 = time.perf_counter()
    if genome_fasta is None:
        if not isinstance(peaks, FootprintTracks):
            raise ValueError("genome_fasta is required unless peaks are FootprintTracks")
        genome_fasta = peaks.params["genome_fasta"]
    fasta = Path(genome_fasta)
    fai = _fai(fasta)
    pk = _merged_peaks(peaks, fai, tobias_compat)

    # ---- merged peaks in loc_sort order; concatenated layout with one N after each peak ----
    corder = sorted(c for c in pk if len(pk[c]))
    if not corder:
        raise ValueError("no peaks")
    reg_c = np.concatenate([np.full(len(pk[c]), i, np.int16) for i, c in enumerate(corder)])
    reg_s = np.concatenate([pk[c][:, 0] for c in corder]).astype(np.int64)
    reg_e = np.concatenate([pk[c][:, 1] for c in corder]).astype(np.int64)
    reg_len = reg_e - reg_s
    n_reg = len(reg_s)
    rstart = np.concatenate([[0], np.cumsum(reg_len + 1)[:-1]])
    ntot = int(rstart[-1] + reg_len[-1] + 1)
    per_chunk = int(np.ceil(n_reg / float(N_CHUNKS)))
    raw = np.full(ntot, ord("N"), np.uint8)
    for ci, c in enumerate(corder):
        sel = np.flatnonzero(reg_c == ci)
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
    w = np.isin(raw, np.frombuffer(b"GgCc", np.uint8)).astype(np.float64)
    w += 0.5 * (seq == 4)
    w[rstart + reg_len] = 0.0
    cs = np.concatenate([[0.0], np.cumsum(w)])             # exact: sums of halves
    gcs = []
    for i0 in range(0, n_reg, per_chunk):
        i1 = min(i0 + per_chunk, n_reg)
        gcs.append(float(cs[rstart[i1 - 1] + reg_len[i1 - 1]] - cs[rstart[i0]]) / float(reg_len[i0:i1].sum()))
    gc = np.mean(gcs)
    del raw, w, cs
    T["layout"] = time.perf_counter() - t0

    # ---- motifs ----
    t0 = time.perf_counter()
    mlist = _as_motifs(motifs)
    if not mlist:
        raise ValueError("no motifs")
    bg = np.array([(1 - gc) / 2.0, gc / 2.0, gc / 2.0, (1 - gc) / 2.0])
    prefixes = _prefixes(mlist)
    fwd, rev = zip(*[_motif_matrices(m, bg, tobias_compat) for m in mlist])
    fwd, rev = list(fwd), list(rev)
    n_mot = len(mlist)
    thr = _thresholds(fwd, bg, motif_pvalue)
    rev_bg = bg[[3, 2, 1, 0]]
    ord_f = [_moods_order(P, bg) for P in fwd]
    ord_r = [_moods_order(P, rev_bg) for P in rev]
    mlen = np.array([P.shape[1] for P in fwd])
    T["motifs"] = time.perf_counter() - t0

    # ---- background positions: random.seed(reglen); random.sample(range(reglen), reglen // 200) ----
    cache = {}
    for L in np.unique(reg_len):
        random.seed(int(L))
        cache[int(L)] = np.array(random.sample(range(int(L)), max(1, int(int(L) / RAND_WINDOW))), np.int64)
    bg_rid = np.concatenate([np.full(len(cache[int(L)]), r) for r, L in enumerate(reg_len)])
    bg_pos = np.concatenate([reg_s[r] + cache[int(L)] for r, L in enumerate(reg_len)])

    # ---- scan in motif batches; per batch, sites of a motif in TOBIAS loc order and resolve_overlaps ----
    t0 = time.perf_counter()
    t_scan = 0.0
    sm, sp, ss = _short_run_hits(seq, fwd + rev, ord_f + ord_r, np.concatenate([thr, thr]), n_mot)
    s_mot = sm % n_mot
    seq_d = to_device(seq)
    del seq
    rstart_d, reg_s_d, reg_c_d = cp.asarray(rstart), cp.asarray(reg_s), cp.asarray(reg_c.astype(np.int32))
    hp_bits = max(1, int(ntot).bit_length())
    batch = max(1, min(n_mot, int(4e10 // max(ntot, 1)) // 2))
    kept = []
    for m0 in range(0, n_mot, batch):
        m1 = min(n_mot, m0 + batch)
        nb = m1 - m0
        ts = time.perf_counter()
        hm, hp, hs = _scan_device(seq_d, fwd[m0:m1] + rev[m0:m1], ord_f[m0:m1] + ord_r[m0:m1],
                                  np.concatenate([thr[m0:m1], thr[m0:m1]]))
        sel = np.flatnonzero((s_mot >= m0) & (s_mot < m1))
        if len(sel):
            loc = np.where(sm[sel] < n_mot, sm[sel] - m0, sm[sel] - n_mot - m0 + nb)
            hm = cp.concatenate([hm, cp.asarray(loc.astype(np.int32))])
            hp = cp.concatenate([hp, cp.asarray(sp[sel])])
            hs = cp.concatenate([hs, cp.asarray(ss[sel])])
        cp.cuda.Device().synchronize()
        t_scan += time.perf_counter() - ts
        if len(hm) == 0:
            continue
        minus = hm >= nb
        mloc = (hm % nb).astype(cp.int64)
        # TOBIAS appends sites per (peak, strand, motif, position); per motif and chunk it then
        # loc-sorts them stably, i.e. by (layout position, strand)
        o = cp.argsort((mloc << (hp_bits + 1)) | (hp << 1) | minus.astype(cp.int64))
        mloc, hp, hs, minus = mloc[o], hp[o], hs[o], minus[o]
        del o
        rid = cp.searchsorted(rstart_d, hp, side="right") - 1
        chrom = reg_c_d[rid]
        start = reg_s_d[rid] + (hp - rstart_d[rid])
        end = start + cp.asarray(mlen[m0:m1])[mloc]
        score = _round5_dev(hs)                            # round(match.score, 5)
        del hp, hs
        gstart = _segments(mloc * (n_reg // per_chunk + 2) + rid // per_chunk)
        keep = cp.ones(len(mloc), cp.uint8)
        _launch(_resolve_kernel, len(gstart) - 1, (gstart, cp.int64(len(gstart) - 1), chrom, start, end, score, keep))
        k = cp.flatnonzero(keep)
        kept.append(dict(chrom=chrom[k], start=start[k], end=end[k], motif=(mloc[k] + m0).astype(cp.int32),
                         score=score[k], minus=minus[k], rid=rid[k].astype(cp.int32)))
        del mloc, minus, rid, chrom, start, end, score, keep, k, gstart
    del seq_d
    S = {key: cp.concatenate([b[key] for b in kept]) for key in kept[0]} if kept else None
    del kept
    T["scan"] = t_scan
    T["resolve"] = time.perf_counter() - t0 - t_scan

    # ---- TFBS overlap between motifs per chunk (count_overlaps), for the motif clustering ----
    t0 = time.perf_counter()
    ov = np.zeros((n_mot, n_mot), np.int64)
    tot = np.zeros(n_mot, np.int64)
    if S is not None:
        name_rank = np.empty(n_mot, np.int64)
        name_rank[sorted(range(n_mot), key=lambda i: prefixes[i])] = np.arange(n_mot)
        rel = S["start"] - reg_s_d[S["rid"]]
        L = S["end"] - S["start"]
        # loc_sort by (chrom, start, end, name) within each chunk; peaks are in (chrom, start) order
        o = _stable_order(S["rid"].astype(cp.int64) * (1 << 27) + rel, L * n_mot + cp.asarray(name_rank)[S["motif"]])
        c, s, e, nm, rid = S["chrom"][o], S["start"][o], S["end"][o], S["motif"][o], S["rid"][o]
        chunk = (rid // per_chunk).astype(cp.int32)
        del o, rel, L, rid
        # merge(name=True): a site joins the previous one if same chunk, chrom and name and it
        # starts before the previous site's end; the merged region ends at its last member's end
        join = cp.zeros(len(s), cp.bool_)
        join[1:] = (chunk[1:] == chunk[:-1]) & (c[1:] == c[:-1]) & (nm[1:] == nm[:-1]) & (s[1:] < e[:-1])
        first = cp.flatnonzero(~join)
        last = cp.concatenate([first[1:], cp.asarray([len(s)], cp.int64)]) - 1
        ov_d = cp.zeros(n_mot * n_mot, cp.uint64)
        tot_d = cp.zeros(n_mot, cp.uint64)
        _launch(_count_kernel, len(first), (cp.int64(len(first)), chunk[first], c[first], s[first], e[last],
                                            nm[first], cp.int64(n_mot), ov_d, tot_d))
        ov = ov_d.get().view(np.int64).reshape(n_mot, n_mot)
        tot = tot_d.get().view(np.int64)
        del c, s, e, nm, chunk, join, first, last, ov_d, tot_d
    T["overlaps"] = time.perf_counter() - t0

    # ---- final order: per motif by (chrom, start), as process_tfbs sorts its sites ----
    t0 = time.perf_counter()
    if S is None:
        S = dict(chrom=cp.zeros(0, cp.int32), start=cp.zeros(0, cp.int64), end=cp.zeros(0, cp.int64),
                 motif=cp.zeros(0, cp.int32), score=cp.zeros(0), minus=cp.zeros(0, cp.bool_),
                 rid=cp.zeros(0, cp.int32))
    o = _stable_order(S["motif"], S["rid"].astype(cp.int64) * (1 << 27) + (S["start"] - reg_s_d[S["rid"]]))
    pos_t = np.int32 if int(reg_e.max()) < 2**31 else np.int64
    out = {k: v[o].get() for k, v in S.items()}
    del S, o
    offsets = np.searchsorted(out["motif"], np.arange(n_mot + 1))
    T["order"] = time.perf_counter() - t0
    if verbose:
        logger.info("bindetect_sites timings: " + ", ".join(f"{k} {v:.1f}s" for k, v in T.items()))
    return SiteSet(
        chroms=corder,
        peaks={c: pk[c] for c in corder},
        peak_chrom=reg_c.astype(np.int16), peak_start=reg_s, peak_end=reg_e,
        motifs=mlist, prefixes=prefixes, thresholds=thr, gc=float(gc), background=bg,
        chrom=out["chrom"].astype(np.int16), start=out["start"].astype(pos_t), end=out["end"].astype(pos_t),
        motif=out["motif"], score=out["score"], minus=out["minus"], peak=out["rid"],
        motif_offsets=offsets, bg_chrom=reg_c[bg_rid].astype(np.int16), bg_pos=bg_pos,
        overlap=ov, overlap_total=tot, tobias_compat=tobias_compat, motif_pvalue=motif_pvalue, timings=T,
    )


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
        motif, match score, strand, peak, and the normalised footprint score
        (``{group}_score``) and bound call (``{group}_bound``) per group. The
        per-site log2FC of a comparison is :meth:`site_log2fc`.
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

    def site_log2fc(self, group1: str, group2: str) -> pd.Series:
        """Per-site log2 fold change of the footprint scores, ``group1`` over ``group2``.

        ``round(log2((score1 + pseudo) / (score2 + pseudo)), 5)`` on the columns of
        :attr:`sites`, as in TOBIAS's ``*_overview.txt``. It is computed on request
        rather than stored: one column per comparison would double the size of the
        site table at a few groups.

        Returns
        -------
        pandas.Series
            Aligned with :attr:`sites`, named ``{group1}_{group2}_log2fc``.
        """
        if self.sites is None:
            raise ValueError("no site table; run bindetect(..., return_sites=True)")
        missing = [g for g in (group1, group2) if g not in self.groups]
        if missing:
            raise ValueError(f"groups not in this result: {missing}; groups are {self.groups}")
        s1 = self.sites[f"{group1}_score"].to_numpy()
        s2 = self.sites[f"{group2}_score"].to_numpy()
        return pd.Series(_log2fc(s1, s2, self.pseudo), index=self.sites.index,
                         name=f"{group1}_{group2}_log2fc")

    def __repr__(self) -> str:
        n_sites = "no sites kept" if self.sites is None else f"{len(self.sites):,} sites"
        return (f"BINDetectResult({len(self.results)} motifs, {n_sites}, groups {self.groups}, "
                f"threshold {self.threshold})")


def _layout_index_dev(layout, chroms: list[str], chrom: cp.ndarray, pos: cp.ndarray) -> cp.ndarray:
    """Index of each (chrom, pos) into a track over ``layout``, -1 where not covered."""
    out = cp.full(len(pos), -1, cp.int64)
    for ci, c in enumerate(chroms):
        iv = layout.intervals.get(c)
        if iv is None or len(iv) == 0:
            continue
        sel = cp.flatnonzero(chrom == ci)
        if len(sel) == 0:
            continue
        lens = iv[:, 1] - iv[:, 0]
        cum = cp.asarray(np.concatenate([[0], np.cumsum(lens)[:-1]]) + layout.offsets[c])
        st, en = cp.asarray(iv[:, 0]), cp.asarray(iv[:, 1])
        p = pos[sel]
        r = cp.maximum(cp.searchsorted(st, p, side="right") - 1, 0)
        ok = (p >= st[r]) & (p < en[r])
        out[sel[ok]] = cum[r[ok]] + (p[ok] - st[r[ok]])
    return out


def _sample_changes_exact_null(n: int, bg_mean: float, bg_std: float, seed=None, size: int = 100) -> np.ndarray:
    """The null of ``_sample_changes`` drawn from its exact sampling distribution: the mean of
    ``n`` normal draws is N(bg_mean, bg_std²/n) and n·var/bg_std² is an independent χ²(n-1),
    so 2 draws per replicate replace ``n`` (only the random stream differs from TOBIAS)."""
    rng = np.random.default_rng(n if seed is None else seed)
    z = rng.standard_normal(size)
    x = 2.0 * rng.standard_gamma((n - 1) / 2.0, size) if n > 1 else np.zeros(size)
    mean = bg_mean + bg_std * z / np.sqrt(n)
    std = bg_std * np.sqrt(x / n)
    return (mean - bg_mean) / ((std + bg_std) / 2.0)


def _tobias_site_order(sites: SiteSet, idx: Optional[np.ndarray] = None) -> np.ndarray:
    """Order of sites ``idx`` (sorted by motif; default: all sites) by TOBIAS's groupby key
    "chrom_start_end" within each motif: Python string order of the id, computed on the device
    from its characters. The order indexes ``idx`` and has its smallest integer dtype."""
    def col(a):
        return a if idx is None else a[idx]

    n = len(sites.start) if idx is None else len(idx)
    out_t = np.int32 if n < 2**31 else np.int64
    if any("_" in c for c in sites.chroms):            # a chrom name could prefix another's id
        key = (pl.DataFrame({"m": col(sites.motif), "c": np.asarray(sites.chroms)[col(sites.chrom)],
                             "s": col(sites.start), "e": col(sites.end)})
               .with_columns(k=pl.concat_str([pl.col("c"), pl.col("s").cast(pl.String),
                                              pl.col("e").cast(pl.String)], separator="_"))
               .with_row_index("i").sort(["m", "k"], maintain_order=True))
        return key["i"].to_numpy().astype(out_t)
    # without "_" in names, "chrA_..." vs "chrB_..." is decided by chrA + "_" vs chrB + "_"
    rank = np.empty(len(sites.chroms), np.int64)
    rank[sorted(range(len(sites.chroms)), key=lambda i: sites.chroms[i] + "_")] = np.arange(len(sites.chroms))
    # columns go to the device as stored and are widened there, with no host copies
    hi, lo = _strkey_kernel(to_device(col(sites.start)).astype(cp.int64),
                            to_device(col(sites.end)).astype(cp.int64))
    chrom_rank = cp.asarray(rank)[to_device(col(sites.chrom))]
    return _stable_order(to_device(col(sites.motif)), chrom_rank, hi, lo).astype(out_t).get()


def bindetect(
    tracks: FootprintTracks,
    motifs=None,
    genome_fasta: Union[None, str, Path] = None,
    *,
    sites: Optional[SiteSet] = None,
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

    The binding sites depend only on the peaks and motifs: build them once with
    :func:`bindetect_sites` and pass ``sites=`` to reuse them across calls.

    Parameters
    ----------
    tracks : FootprintTracks
        Output of :func:`gatac.tobias.footprint`.
    motifs : str, Path or list of DNAMotif, optional
        MEME file (read like TOBIAS, including ``nsites``) or GATAC motifs
        (e.g. a subset from :func:`gatac.motif.read`). Pass the MEME path to
        reproduce TOBIAS exactly: :class:`~gatac.motif.DNAMotif` stores clipped,
        renormalised probabilities and no ``nsites`` (20 is assumed). Not
        needed with ``sites``.
    genome_fasta : str or Path, optional
        Genome FASTA (default: the one used for ``tracks``).
    sites : SiteSet, optional
        Precomputed binding sites (:func:`bindetect_sites`) on the peaks of ``tracks``.
    groups : sequence of str, optional
        Groups (conditions) to compare, default all groups of ``tracks``.
    comparisons : "all", "time_series" or list of (group1, group2)
        All pairs in group order (TOBIAS default), consecutive pairs
        (``--time-series``), or explicit pairs.
    motif_pvalue : float, default 1e-4
        Motif match p-value (MOODS threshold); ignored with ``sites``.
    bound_pvalue : float, default 0.001
        P-value of the bound/unbound footprint threshold.
    pseudo : float, optional
        Log2FC pseudocount (default: half the mode of the background scores).
    cluster_threshold : float, default 0.5
        Distance threshold of the motif clustering by TFBS overlap.
    tobias_compat : bool, default True
        Reproduce TOBIAS exactly: its reverse-strand matrix, per-site (not
        per-peak) log2FC averaging, and its p-value resampling (``n`` draws per
        replicate from NumPy's legacy generator, the CPU floor of this
        function). False uses the corrected behaviours and draws the same
        null from its exact sampling distribution (2 draws per replicate).
    return_sites : bool, default True
        Return the per-site table (scores and bound calls per group; the per-site
        log2FCs come from :meth:`BINDetectResult.site_log2fc`).
    verbose : bool, default True
        Log timings.

    Returns
    -------
    BINDetectResult

    Examples
    --------
    >>> tracks = ga.tobias.footprint(adata, "cell_type", "fragments.parquet", "GRCh38.fa",
    ...                          groups=["CD14 Mono", "CD4 Memory"])
    >>> res = ga.tobias.bindetect(tracks, "cisBP_human.meme")
    >>> res.results.sort_values("CD14 Mono_CD4 Memory_change").head()
    """
    import scipy.stats

    T = {}
    if sites is None:
        if motifs is None:
            raise ValueError("pass motifs or a precomputed SiteSet (sites=)")
        sites = bindetect_sites(tracks, motifs, genome_fasta, motif_pvalue=motif_pvalue,
                                tobias_compat=tobias_compat, verbose=verbose)
        T.update({f"sites_{k}": v for k, v in sites.timings.items()})
    else:
        if sites.tobias_compat != tobias_compat:
            raise ValueError(f"sites were built with tobias_compat={sites.tobias_compat}")
        same = sorted(c for c in tracks.peaks if len(tracks.peaks[c])) == sites.chroms and all(
            np.array_equal(tracks.peaks[c], sites.peaks[c]) for c in sites.chroms)
        if not same:
            raise ValueError("sites were built on other peaks than the tracks'")
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
    mlist, prefixes = sites.motifs, sites.prefixes
    n_mot = len(mlist)
    off = sites.motif_offsets

    # ---- footprint score at each background position (uncovered = 0, a bigwig gap); the sites
    # come after the normalisation, which depends on the background only ----
    t0 = time.perf_counter()
    flay = tracks.layouts["footprint"]
    idx_bg = _layout_index_dev(flay, sites.chroms, to_device(sites.bg_chrom), to_device(sites.bg_pos))
    bg_raw = {g: _gather_dev(to_device(tracks.values["footprint"][g]), idx_bg).get() for g in groups}
    del idx_bg
    T["footprints"] = time.perf_counter() - t0

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
    if len(sites.bg_pos) < 1000:
        logger.warning(f"only {len(sites.bg_pos)} background positions; threshold and normalisation may be unstable")
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

    # ---- per-site scores, one group at a time (exp and log2 stay in NumPy: CUDA's differ in the
    # last bit); the elementwise steps run in chunks, in place, to bound their temporaries ----
    t0 = time.perf_counter()
    chrom_d = to_device(sites.chrom)
    start_d = to_device(sites.start)
    mid_d = start_d.astype(cp.int64)
    mid_d += (to_device(sites.end) - start_d) // 2
    del start_d
    idx_site = _layout_index_dev(flay, sites.chroms, chrom_d, mid_d)
    del chrom_d, mid_d
    cols = {}
    for g in groups:
        # BINDetect writes "{:.5f}" of the bigwig value
        s = _round5_dev(_gather_dev(to_device(tracks.values["footprint"][g]), idx_site)).get()
        for a in range(0, len(s), _CHUNK):
            v = norms[g].normalize(s[a:a + _CHUNK])
            v[~(v > 0)] = 0                                  # np.where(v > 0, v, 0)
            s[a:a + _CHUNK] = np.round(v, 5)
        cols[f"{g}_score"] = s
        cols[f"{g}_bound"] = (s > threshold).astype(np.int8)
    del idx_site
    T["site_table"] = time.perf_counter() - t0

    # ---- per-TF summaries ----
    t0 = time.perf_counter()
    has = np.flatnonzero(np.diff(off) > 0)
    res = pd.DataFrame({
        "output_prefix": prefixes,
        "name": [m.name for m in mlist],
        "motif_id": [m.id for m in mlist],
        "total_tfbs": np.diff(off).astype(np.int64),
    })
    for g in groups:
        sc, bd = cols[f"{g}_score"], cols[f"{g}_bound"]
        ms = np.full(n_mot, np.nan)
        ms[has] = [np.round(np.mean(sc[off[m]:off[m + 1]]), 5) for m in has]
        nb = np.zeros(n_mot, np.int64)
        if len(has):
            nb[has] = np.add.reduceat(bd, off[has], dtype=np.int64)
        res[f"{g}_mean_score"] = ms
        res[f"{g}_bound"] = nb
    sample_cache: dict[tuple, np.ndarray] = {}
    sampler = _sample_changes if tobias_compat else _sample_changes_exact_null
    order_all = None
    if tobias_compat and comps and len(sites.start):
        # TOBIAS's site order within each motif is one total order for every comparison: sort all
        # sites once and keep the included ones per comparison (the sort is stable, so this is
        # the order of sorting them alone)
        order_all = _tobias_site_order(sites)
    for g1, g2 in comps:
        base = f"{g1}_{g2}"
        s1, s2 = cols[f"{g1}_score"], cols[f"{g2}_score"]
        included = (s1 > 0) | (s2 > 0)
        l2 = _log2fc(s1, s2, pseudo)                     # not kept: see BINDetectResult.site_log2fc
        if tobias_compat:
            # groupby("peak_id") on "chrom_start_end" of the *TFBS*: one site per group, string order
            ii = order_all[included[order_all]] if order_all is not None else np.zeros(0, np.int64)
            obs_all, mm = l2[ii], sites.motif[ii]
            del ii
        else:
            inc = np.flatnonzero(included)
            if len(inc):
                # mean log2FC per peak: sites of a motif in a peak are contiguous
                key = sites.motif[inc].astype(np.int64) * (len(sites.peak_start) + 1) + sites.peak[inc]
                seg = np.flatnonzero(np.concatenate([[True], key[1:] != key[:-1]]))
                cnt = np.diff(np.append(seg, len(inc)))
                obs_all = np.add.reduceat(l2[inc], seg) / cnt
                mm = sites.motif[inc][seg]
            else:
                obs_all, mm = np.zeros(0), np.zeros(0, np.int32)
            del inc
        del included, l2
        mb = np.concatenate([[0], np.flatnonzero(np.diff(mm)) + 1, [len(mm)]]) if len(mm) else np.array([0])
        bg_mean, bg_std = log2fc_params[(g1, g2)]
        change, n_obs = {}, {}
        for a, b in zip(mb[:-1], mb[1:]):
            m = int(mm[a])
            om, osd = scipy.stats.norm.fit(obs_all[a:b])
            change[m] = np.round((om - bg_mean) / np.mean([osd, bg_std]), 5) if om != bg_mean else 0
            n_obs[m] = b - a
        uniq = sorted({n for n in n_obs.values() if (n, bg_mean, bg_std) not in sample_cache}, reverse=True)
        if tobias_compat:
            with ThreadPoolExecutor(_n_threads()) as ex:
                drawn = list(ex.map(lambda n: sampler(n, bg_mean, bg_std), uniq))
        else:
            drawn = [sampler(n, bg_mean, bg_std) for n in uniq]
        sample_cache.update({(n, bg_mean, bg_std): s for n, s in zip(uniq, drawn)})
        res[f"{base}_change"] = [change.get(m, np.nan) for m in range(n_mot)]
        res[f"{base}_pvalue"] = [scipy.stats.ttest_1samp(sample_cache[(n_obs[m], bg_mean, bg_std)], change[m])[1]
                                 if m in change else np.nan for m in range(n_mot)]
    T["per_tf"] = time.perf_counter() - t0

    # ---- clusters and highlighting ----
    t0 = time.perf_counter()
    name2cluster, distances = _cluster_motifs(prefixes, sites.overlap, sites.overlap_total, cluster_threshold)
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
    T["clusters"] = time.perf_counter() - t0

    out_sites = None
    if return_sites:
        out_sites = pd.DataFrame({
            "chrom": pd.Categorical.from_codes(sites.chrom, categories=sites.chroms),
            "start": sites.start,
            "end": sites.end,
            "motif": pd.Categorical.from_codes(sites.motif, categories=prefixes),
            "score": sites.score,
            "strand": pd.Categorical.from_codes(sites.minus.astype(np.int8), categories=["+", "-"]),
            "peak_start": sites.peak_start[sites.peak],
            "peak_end": sites.peak_end[sites.peak],
            **cols,
        }, copy=False)                     # no consolidation copy of a table with 10^7+ rows
    T["total"] = sum(T.values())
    if verbose:
        logger.info("bindetect timings: " + ", ".join(f"{k} {v:.1f}s" for k, v in T.items()))
    return BINDetectResult(
        results=res,
        sites=out_sites,
        clusters=clusters,
        distances=distances,
        threshold=threshold,
        pseudo=pseudo,
        groups=groups,
        comparisons=comps,
        params=dict(motif_pvalue=sites.motif_pvalue, bound_pvalue=bound_pvalue, gc=sites.gc,
                    background=sites.background, motif_thresholds=dict(zip(prefixes, sites.thresholds)),
                    log2fc_params=log2fc_params, cluster_threshold=cluster_threshold,
                    tobias_compat=tobias_compat, n_background=len(sites.bg_pos), timings=T),
    )
