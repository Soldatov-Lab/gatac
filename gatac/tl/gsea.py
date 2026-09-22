"""
GPU-accelerated preranked GSEA using CuPy.

Implements the same algorithm as GSEApy's ``prerank`` (Rust backend) but
runs the enrichment-score computation and permutation testing entirely on
the GPU for large-scale motif enrichment.

Algorithm
---------
For a ranked list of N features with ranking metric r[i] and a feature set S of
size N_H:

    P_hit(i) = |r[i]|^w / N_R   if feature i ∈ S    (N_R = Σ_{j∈S} |r[j]|^w)
    P_miss(i) = 1 / (N - N_H)   if feature i ∉ S

    RES(i) = cumsum(P_hit - P_miss)
    ES = max(RES) if |max(RES)| > |min(RES)| else min(RES)

Permutation null: shuffle feature labels (feature-set permutation), recompute ES.
NES, p-value, FDR follow the GSEA paper / GSEApy implementation.

Hits-only evaluation
--------------------
RES only ever *increases* at a hit and decreases linearly between hits, so
its extrema are pinned to hit positions. Writing the sorted hit positions as
p_0 < ... < p_{H-1}, and C_j for the normalised cumulative hit weight
Σ_{m<=j} w[p_m] / N_R, the running score at hit j and at the position
immediately preceding it are

    top_j = C_j - (p_j - j) / (N - H)          RES(p_j)
    bot_j = top_j - w[p_j] / N_R               RES(p_j - 1)

(the miss count at or before p_j is p_j + 1 - (j + 1) = p_j - j, and position
p_j itself contributes no miss, so both share the same miss term). Then

    max(RES) = max(max_j top_j, RES(N-1))    min(RES) = min(min_j bot_j, RES(N-1))

exactly. The endpoint RES(N-1) = C_{H-1} - 1 is carried as an explicit
candidate: normally C_{H-1} = 1 so it is 0 and never binds, but when N_R = 0
(an all-zero metric, or a set whose members all have metric exactly 0) the
normalisation is clamped, C_{H-1} = 0, and the true extremum is the endpoint
rather than any hit.

This makes scoring O(H) per (set, permutation) rather than O(N), which for
typical N=20k and H~100 is a ~200x reduction in both work and memory.
``_enrichment_scores_gpu`` below retains the naive O(N) cumsum as a reference
oracle for the tests.
"""

from __future__ import annotations

import logging

import cupy as cp
import numpy as np

logger = logging.getLogger(__name__)

#: Target number of int32 elements in a (permutation × set × H_max) scoring
#: block. Caps feature-set batch width so that very large sets do not blow up
#: device memory; 64M elements is ~256 MB per intermediate.
_SCORING_ELEM_BUDGET = 64_000_000


# =============================================================================
# Core GPU kernels
# =============================================================================


def _es_from_hits_gpu(
    weighted_metric: cp.ndarray,
    hit_pos: cp.ndarray,
    n_hits: cp.ndarray,
    with_peaks: bool = False,
):
    """
    Compute enrichment scores from sorted hit positions alone.

    Evaluates the running enrichment score only at hit positions, which is
    exact (see module docstring) and costs O(H) instead of O(N) per row.

    Parameters
    ----------
    weighted_metric : cp.ndarray, shape (N,)
        |r[i]|^weight for each feature, in the original ranked order.
    hit_pos : cp.ndarray, shape (B, Hmax), int
        Hit positions per row, sorted ascending, with any padding at the end
        of the row. Padding values are ignored (they may be arbitrary, as
        long as they sort last).
    n_hits : cp.ndarray, shape (B, 1), float32
        True number of hits per row.
    with_peaks : bool, default False
        Also return the hit index at which the ES peak occurs, for
        leading-edge extraction.

    Returns
    -------
    es : cp.ndarray, shape (B,)
    peak_j : cp.ndarray, shape (B,), int32
        Only when ``with_peaks``. For ES >= 0 the index j of the maximising
        ``top_j``; for ES < 0 the index j of the minimising ``bot_j``.
    """
    N = weighted_metric.shape[0]
    Hmax = hit_pos.shape[1]

    ar = cp.arange(Hmax, dtype=cp.float32)[None, :]
    mask = ar < n_hits

    # Padding-safe gather: index at 0 where masked out, then zero the weight.
    pos_safe = cp.where(mask, hit_pos, 0)
    w_h = weighted_metric[pos_safe] * mask

    n_r = cp.maximum(w_h.sum(axis=1, keepdims=True), 1e-10)
    cum = cp.cumsum(w_h, axis=1) / n_r

    # Padded entries inherit the last real hit position, so their top_j
    # duplicates top_{H-1} and cannot shift either extremum.
    last_j = (n_hits.astype(cp.int32) - 1).ravel()
    last_pos = hit_pos[cp.arange(hit_pos.shape[0]), last_j][:, None]
    pos_eff = cp.where(mask, hit_pos, last_pos).astype(cp.float32)

    eff_j = cp.minimum(ar, n_hits - 1.0)
    miss = pos_eff - eff_j
    denom = cp.maximum(N - n_hits, 1.0)

    top = cum - miss / denom
    bot = top - w_h / n_r

    # RES at the final position, N-1. This is 0 whenever the hit weights sum
    # to something positive (C_{H-1} = 1 and all N-H misses have accrued), but
    # if N_R is 0 -- an all-zero metric, or a set whose members all have metric
    # exactly 0 -- the normalisation is clamped, C_{H-1} = 0, and the tail
    # falls to -1. It is a real position, so it must be an explicit candidate.
    tail = cum[:, -1] - 1.0

    max_body = top.max(axis=1)
    min_body = bot.min(axis=1)
    max_es = cp.maximum(max_body, tail)
    min_es = cp.minimum(min_body, tail)
    take_max = cp.abs(max_es) > cp.abs(min_es)
    es = cp.where(take_max, max_es, min_es)

    if not with_peaks:
        return es

    # Sentinel n_hits means "the extremum is the tail at position N-1". Ties
    # resolve to the earlier (body) candidate, matching np.argmax/argmin's
    # first-occurrence semantics in the reference.
    tail_j = n_hits.ravel().astype(cp.int32)
    peak_j = cp.where(
        take_max,
        cp.where(tail > max_body, tail_j, top.argmax(axis=1)),
        cp.where(tail < min_body, tail_j, bot.argmin(axis=1)),
    ).astype(cp.int32)
    return es, peak_j


def _enrichment_scores_gpu(
    weighted_metric: cp.ndarray,
    tag_indicators: cp.ndarray,
) -> cp.ndarray:
    """
    Reference O(N) implementation: enrichment scores from full tag indicators.

    Superseded by :func:`_es_from_hits_gpu` in the hot path; retained as the
    oracle that the hits-only kernel is validated against.

    Parameters
    ----------
    weighted_metric : cp.ndarray, shape (N,)
        |r[i]|^weight for each feature, in the original ranked order.
    tag_indicators : cp.ndarray, shape (n_perm, N)
        Binary indicators: 1 if feature is in set, 0 otherwise.

    Returns
    -------
    cp.ndarray, shape (n_perm,)
        Enrichment scores for each permutation.
    """
    N = weighted_metric.shape[0]

    # Number of hits per permutation → shape (n_perm, 1)
    n_hits = tag_indicators.sum(axis=1, keepdims=True)
    n_miss = N - n_hits

    # Sum of weighted metric at hit positions → shape (n_perm, 1)
    sum_correl_tag = (tag_indicators * weighted_metric[None, :]).sum(
        axis=1, keepdims=True
    )

    # Normalisation factors
    norm_tag = 1.0 / cp.maximum(sum_correl_tag, 1e-10)
    norm_no_tag = 1.0 / cp.maximum(n_miss, 1.0)

    no_tag = 1.0 - tag_indicators

    # Per-position increment
    increments = (
        tag_indicators * weighted_metric[None, :] * norm_tag
        - no_tag * norm_no_tag
    )

    # Running enrichment score = cumulative sum
    run_es = cp.cumsum(increments, axis=1)

    # ES = max deviation from zero
    max_es = run_es.max(axis=1)
    min_es = run_es.min(axis=1)

    es = cp.where(cp.abs(max_es) > cp.abs(min_es), max_es, min_es)

    return es


def _enrichment_scores_and_running_gpu_batch(
    weighted_metric: cp.ndarray,
    tag_indicators: cp.ndarray,
) -> tuple[cp.ndarray, cp.ndarray]:
    """
    Compute enrichment scores and full running ES for multiple feature sets.

    Reference implementation, retained as the oracle for leading-edge tests.

    Parameters
    ----------
    weighted_metric : cp.ndarray, shape (N,)
    tag_indicators : cp.ndarray, shape (n_sets, N)
        Binary indicators: 1 if feature is in set, 0 otherwise.

    Returns
    -------
    es : cp.ndarray, shape (n_sets,)
    run_es : cp.ndarray, shape (n_sets, N)
    """
    N = weighted_metric.shape[0]

    n_hits = tag_indicators.sum(axis=1, keepdims=True)
    n_miss = N - n_hits

    sum_correl_tag = (tag_indicators * weighted_metric[None, :]).sum(
        axis=1, keepdims=True
    )

    norm_tag = 1.0 / cp.maximum(sum_correl_tag, 1e-10)
    norm_no_tag = 1.0 / cp.maximum(n_miss, 1.0)

    no_tag = 1.0 - tag_indicators

    increments = (
        tag_indicators * weighted_metric[None, :] * norm_tag
        - no_tag * norm_no_tag
    )

    run_es = cp.cumsum(increments, axis=1)

    max_es = run_es.max(axis=1)
    min_es = run_es.min(axis=1)

    es = cp.where(cp.abs(max_es) > cp.abs(min_es), max_es, min_es)

    return es, run_es


# =============================================================================
# Permutation generation
# =============================================================================


def _permutation_pool_gpu(
    n_features: int,
    n_perm: int,
    rs: cp.random.RandomState,
    row_chunk: int = 256,
) -> cp.ndarray:
    """
    Generate a pool of uniform random permutations on the device.

    Feature-set permutation is a relabelling of positions: under permutation
    ``pi``, a feature set sitting at original positions ``P`` moves to
    ``pi[P]``. Building the pool once means each (set, permutation) costs a
    gather of H entries rather than a length-N shuffle, and the pool is
    shared across all feature sets exactly as the previous implementation
    shared one permutation matrix.

    Parameters
    ----------
    n_features : int
    n_perm : int
    rs : cp.random.RandomState
    row_chunk : int, default 256
        Permutations generated per argsort call, bounding transient memory.

    Returns
    -------
    cp.ndarray, shape (n_perm, n_features), dtype int32
    """
    pool = cp.empty((n_perm, n_features), dtype=cp.int32)
    for start in range(0, n_perm, row_chunk):
        end = min(start + row_chunk, n_perm)
        keys = rs.random_sample(size=(end - start, n_features), dtype=cp.float32)
        pool[start:end] = cp.argsort(keys, axis=1).astype(cp.int32)
        del keys
    return pool


# =============================================================================
# Statistical functions
# =============================================================================


def _normalize_es(
    es: float,
    esnull: np.ndarray,
) -> tuple[float, np.ndarray]:
    """
    Normalize ES and null distribution following GSEA convention.
    """
    pos_mask = esnull >= 0
    neg_mask = esnull < 0

    pos_mean = esnull[pos_mask].mean() if pos_mask.any() else es
    neg_mean = esnull[neg_mask].mean() if neg_mask.any() else es

    if pos_mean == 0:
        pos_mean = 1e-10
    if neg_mean == 0:
        neg_mean = -1e-10

    nes = es / pos_mean if es >= 0 else es / abs(neg_mean)

    nesnull = np.where(
        esnull >= 0,
        esnull / pos_mean,
        esnull / abs(neg_mean),
    )

    return nes, nesnull


def _compute_pval(es: float, esnull: np.ndarray) -> float:
    """
    Compute nominal p-value from null distribution.
    """
    if es >= 0:
        denom = (esnull >= 0).sum()
        numer = (esnull >= es).sum()
    else:
        denom = (esnull < 0).sum()
        numer = (esnull <= es).sum()

    if denom == 0:
        return 1.0
    return float(numer / denom)


def _compute_fdr(
    nes_observed: np.ndarray,
    nesnull_concat: np.ndarray,
) -> np.ndarray:
    """
    Compute FDR q-values for all gene sets.

    Following GSEApy/GSEA convention:
    FDR(NES) = (fraction of nesnull >= NES among same-sign nulls) /
               (fraction of nes_observed >= NES among same-sign observed)
    """
    nes_observed = np.asarray(nes_observed, dtype=np.float64)
    # Upcast the null once. A float64 probe against a float32 null makes
    # searchsorted upcast the whole (n_sets * n_perm)-element array on every
    # call; done per set that dominates the entire run once the sets number in
    # the thousands. Probing in float32 instead would avoid the copy but round
    # the search boundary, so pay the one-off cast and stay exact.
    nvals = np.sort(np.asarray(nesnull_concat, dtype=np.float64))
    nnes = np.sort(nes_observed)

    all_neg_idx = np.searchsorted(nvals, 0, side="left")
    nes_neg_idx = np.searchsorted(nnes, 0, side="left")

    pos = nes_observed >= 0

    all_higher = np.where(
        pos,
        len(nvals) - np.searchsorted(nvals, nes_observed, side="left"),
        np.searchsorted(nvals, nes_observed, side="right"),
    )
    nes_higher = np.where(
        pos,
        len(nnes) - np.searchsorted(nnes, nes_observed, side="left"),
        np.searchsorted(nnes, nes_observed, side="right"),
    )
    all_pos = np.where(pos, len(nvals) - all_neg_idx, all_neg_idx)
    nes_pos = np.where(pos, len(nnes) - nes_neg_idx, nes_neg_idx)

    fdrs = np.ones(len(nes_observed))
    ok = (all_pos > 0) & (nes_pos > 0)
    zeros = np.zeros(len(nes_observed))
    phi_norm = np.divide(all_higher, all_pos, out=zeros.copy(), where=ok)
    phi_obs = np.divide(nes_higher, nes_pos, out=zeros.copy(), where=ok)

    good = ok & (phi_obs > 0)
    fdrs[good] = np.minimum(phi_norm[good] / phi_obs[good], 1.0)

    return fdrs


# =============================================================================
# Leading edge
# =============================================================================


def _leading_edge_from_peak(
    hit_indices: np.ndarray,
    es: float,
    peak_j: int,
    n_features: int,
) -> np.ndarray:
    """
    Leading-edge hits, derived from the index of the peak hit.

    ``peak_j`` is the position within ``hit_indices`` at which the ES extremum
    is attained (see :func:`_es_from_hits_gpu`). For a positive ES the peak
    sits *at* hit ``peak_j``, so the leading edge is hits ``0..peak_j``. For a
    negative ES the extremum sits at the position immediately *before* hit
    ``peak_j``; that position is itself a hit whenever the preceding hit is
    adjacent, in which case it joins the leading edge. ``peak_j == len(hits)``
    is the sentinel for an extremum at the final position, N-1.

    Returns
    -------
    np.ndarray
        Indices (into the ranked list) of the leading-edge features.
    """
    if len(hit_indices) == 0:
        return hit_indices[:0]

    if peak_j >= len(hit_indices):  # extremum at position N-1
        if es >= 0:
            return hit_indices
        return hit_indices[hit_indices >= n_features - 1]

    if es >= 0:
        return hit_indices[: peak_j + 1]

    start = peak_j
    if peak_j > 0 and hit_indices[peak_j - 1] == hit_indices[peak_j] - 1:
        start = peak_j - 1
    return hit_indices[start:]


def _leading_edge_size(run_es_np: np.ndarray, es: float, hit_indices: np.ndarray) -> int:
    """
    Reference implementation: count leading-edge genes from the full running ES.

    Superseded by :func:`_leading_edge_from_peak`; retained as a test oracle.
    """
    if len(hit_indices) == 0:
        return 0

    if es >= 0:
        peak_idx = np.argmax(run_es_np)
        return int((hit_indices <= peak_idx).sum())
    else:
        peak_idx = np.argmin(run_es_np)
        return int((hit_indices >= peak_idx).sum())


# =============================================================================
# Main GPU prerank function
# =============================================================================


def prerank_gpu(
    feature_names: list[str],
    ranking_values: np.ndarray,
    feature_sets: dict[str, list[str]],
    weight: float = 1.0,
    min_size: int = 15,
    max_size: int = 2000,
    permutation_num: int = 1000,
    seed: int = 42,
    perm_batch_size: int = 512,
    gs_batch_size: int = 128,
) -> list[dict]:
    """
    GPU-accelerated preranked GSEA.

    Implements the same algorithm as GSEApy's ``prerank`` but runs the
    enrichment-score computation entirely on the GPU using CuPy.

    A pool of ``perm_batch_size`` random permutations is generated on the
    device at a time, and each chunk is scored against every feature set
    before the next chunk is drawn. Feature sets are grouped into batches of
    ``gs_batch_size``, sorted by size so that padding within a batch stays
    small, and all (set × permutation) pairs in a batch are evaluated in a
    single kernel call.

    Scoring evaluates the running enrichment score only at hit positions
    (see module docstring), so the cost is set-size- rather than
    N-proportional. Peak GPU memory is roughly
    ``perm_batch_size * gs_batch_size * H_max * 4`` bytes for the scoring
    plus ``perm_batch_size * N * 4`` bytes for the permutation pool.

    Parameters
    ----------
    feature_names : list[str]
        Feature (or peak) names in ranked order (descending by ranking_values).
    ranking_values : np.ndarray, shape (N,)
        Ranking metric values corresponding to feature_names (already sorted
        descending).
    feature_sets : dict[str, list[str]]
        Feature sets to test. Keys are set names, values are lists of feature names.
    weight : float, default 1.0
        Weighting exponent for the ranking metric.
    min_size : int, default 15
        Minimum feature set size (after intersection with ranked list).
    max_size : int, default 2000
        Maximum feature set size.
    permutation_num : int, default 1000
        Number of permutations for the null distribution.
    seed : int, default 42
        Random seed for permutation reproducibility.
    perm_batch_size : int, default 512
        Number of permutations held on the device at once. Controls GPU
        memory usage. Reduce if OOM.
    gs_batch_size : int, default 128
        Number of feature sets scored simultaneously on the GPU.
        Larger values improve throughput but increase memory usage.
        Reduce if OOM.

    Returns
    -------
    list[dict]
        List of result dicts with keys:
        - term: feature set name
        - es: enrichment score
        - nes: normalized enrichment score
        - pval: nominal p-value
        - fdr: FDR q-value
        - lead_edge_n: number of leading-edge features
        - lead_edge_idx: indices of the leading-edge features in the ranked list
        - hits: indices of feature-set members in the ranked list
    """
    from tqdm.auto import tqdm

    N = len(feature_names)
    if N == 0:
        return []
    if permutation_num < 1:
        raise ValueError(
            f"permutation_num must be >= 1, got {permutation_num}; "
            "NES, p-values and FDR are all defined against the null."
        )

    # Build feature-name → index lookup
    feature_to_idx = {g: i for i, g in enumerate(feature_names)}

    # Weight the metric: |r|^weight
    ranking_values = np.asarray(ranking_values, dtype=np.float64)
    weighted_metric_np = (np.abs(ranking_values) ** weight).astype(np.float32)
    weighted_metric_gpu = cp.asarray(weighted_metric_np)

    # Filter feature sets and record their hit positions in the ranked list.
    # ``set`` dedupes members repeated within a feature set, which would
    # otherwise be double-counted in both the size filter and N_R.
    valid_sets = []  # (name, hit_indices_np)
    for term, members in feature_sets.items():
        hit_idx = sorted({feature_to_idx[g] for g in members if g in feature_to_idx})
        if min_size <= len(hit_idx) <= max_size:
            valid_sets.append((term, np.array(hit_idx, dtype=np.int32)))

    if not valid_sets:
        logger.warning("No feature sets passed size filter.")
        return []

    n_sets = len(valid_sets)

    logger.info(
        f"GPU GSEA: {n_sets} feature sets, {N} features, "
        f"{permutation_num} permutations  "
        f"(gs_batch={gs_batch_size}, perm_batch={perm_batch_size})"
    )

    # -----------------------------------------------------------------
    # Group feature sets into rectangular batches, sorted by size so that
    # the padding needed to make a batch rectangular stays small. Batch
    # width is capped so that the (permutation × set × H_max) scoring block
    # stays within a fixed element budget regardless of set size.
    # -----------------------------------------------------------------
    sizes = np.array([len(h) for _, h in valid_sets], dtype=np.int64)
    size_order = np.argsort(sizes, kind="stable")

    effective_perm_batch = min(perm_batch_size, permutation_num)
    effective_gs_batch = min(gs_batch_size, n_sets)
    budget_cols = max(1, _SCORING_ELEM_BUDGET // effective_perm_batch)

    batches = []  # (orig_indices, pos_gpu, pos_clamped_gpu, n_hits_gpu)
    cursor = 0
    while cursor < n_sets:
        h_max = 0
        take = 0
        while cursor + take < n_sets and take < effective_gs_batch:
            cand_h = max(h_max, int(sizes[size_order[cursor + take]]))
            if take > 0 and (take + 1) * cand_h > budget_cols:
                break
            h_max = cand_h
            take += 1
        idxs = size_order[cursor : cursor + take]
        cursor += take

        # Sentinel N sorts after every real position and is masked out.
        pos_np = np.full((take, h_max), N, dtype=np.int32)
        nh_np = np.empty((take, 1), dtype=np.float32)
        for k, i in enumerate(idxs):
            hits = valid_sets[i][1]
            pos_np[k, : len(hits)] = hits
            nh_np[k, 0] = len(hits)

        pos_gpu = cp.asarray(pos_np)
        batches.append((
            idxs,
            pos_gpu,
            cp.minimum(pos_gpu, N - 1),  # in-bounds gather index for the pool
            cp.asarray(nh_np),
        ))

    # -----------------------------------------------------------------
    # Observed ES and leading-edge peak (unpermuted)
    # -----------------------------------------------------------------
    all_es = np.empty(n_sets, dtype=np.float64)
    all_peak = np.empty(n_sets, dtype=np.int64)
    for idxs, pos_gpu, _pos_clamped, nh_gpu in batches:
        es_b, peak_b = _es_from_hits_gpu(
            weighted_metric_gpu, pos_gpu, nh_gpu, with_peaks=True
        )
        all_es[idxs] = cp.asnumpy(es_b)
        all_peak[idxs] = cp.asnumpy(peak_b)

    # -----------------------------------------------------------------
    # Permutation null: one device-resident permutation pool at a time,
    # scored against every feature set before the next pool is drawn.
    # -----------------------------------------------------------------
    esnull = np.empty((n_sets, permutation_num), dtype=np.float32)
    rs = cp.random.RandomState(seed)

    n_perm_chunks = -(-permutation_num // effective_perm_batch)
    progress = tqdm(
        total=n_perm_chunks * len(batches), desc="GSEA scoring blocks"
    )
    for p_start in range(0, permutation_num, effective_perm_batch):
        n_perm = min(effective_perm_batch, permutation_num - p_start)
        pool = _permutation_pool_gpu(N, n_perm, rs)

        for idxs, _pos_gpu, pos_clamped, nh_gpu in batches:
            n_gs, h_max = pos_clamped.shape

            # Where each set's members land under each permutation
            # → (n_perm, n_gs, h_max)
            perm_pos = pool[:, pos_clamped]
            keep = (
                cp.arange(h_max, dtype=cp.float32)[None, None, :]
                < nh_gpu[None, :, :]
            )
            perm_pos = cp.where(keep, perm_pos, N)
            perm_pos = cp.sort(perm_pos, axis=2).reshape(n_perm * n_gs, h_max)
            nh_flat = cp.tile(nh_gpu.reshape(1, n_gs), (n_perm, 1)).reshape(-1, 1)

            es_p = _es_from_hits_gpu(weighted_metric_gpu, perm_pos, nh_flat)
            esnull[idxs, p_start : p_start + n_perm] = cp.asnumpy(
                es_p.reshape(n_perm, n_gs)
            ).T

            del perm_pos, keep, nh_flat, es_p
            progress.update(1)

        del pool
        cp.get_default_memory_pool().free_all_blocks()
    progress.close()

    # -----------------------------------------------------------------
    # Per-feature-set statistics (CPU – negligible cost)
    # -----------------------------------------------------------------
    all_nes = np.empty(n_sets, dtype=np.float64)
    all_pvals = np.empty(n_sets, dtype=np.float64)
    all_lead_edge: list[np.ndarray] = [None] * n_sets
    nesnull_parts = []

    for i, (_term, hit_idx) in enumerate(valid_sets):
        es_obs = float(all_es[i])
        null_i = esnull[i]

        all_pvals[i] = _compute_pval(es_obs, null_i)
        nes, nesnull = _normalize_es(es_obs, null_i)
        all_nes[i] = nes
        nesnull_parts.append(nesnull)
        all_lead_edge[i] = _leading_edge_from_peak(
            hit_idx, es_obs, int(all_peak[i]), N
        )

    # FDR across all feature sets
    nesnull_concat = np.concatenate(nesnull_parts)
    fdrs = _compute_fdr(all_nes, nesnull_concat)

    # Build results
    results = []
    for i, (term, hit_idx) in enumerate(valid_sets):
        results.append({
            "term": term,
            "es": float(all_es[i]),
            "nes": float(all_nes[i]),
            "pval": float(all_pvals[i]),
            "fdr": float(fdrs[i]),
            "lead_edge_n": len(all_lead_edge[i]),
            "lead_edge_idx": all_lead_edge[i].tolist(),
            "hits": hit_idx.tolist(),
        })

    # Free GPU memory
    del weighted_metric_gpu, batches
    cp.get_default_memory_pool().free_all_blocks()

    return results
