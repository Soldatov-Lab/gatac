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
"""

from __future__ import annotations

import logging
from typing import Optional

import cupy as cp
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Core GPU kernels
# =============================================================================


def _enrichment_scores_gpu(
    weighted_metric: cp.ndarray,
    tag_indicators: cp.ndarray,
) -> cp.ndarray:
    """
    Compute enrichment scores for multiple tag indicators (permutations).

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


def _enrichment_scores_and_running_gpu(
    weighted_metric: cp.ndarray,
    tag_indicator: cp.ndarray,
) -> tuple[float, cp.ndarray]:
    """
    Compute enrichment score and full running ES for a single feature set.

    Parameters
    ----------
    weighted_metric : cp.ndarray, shape (N,)
    tag_indicator : cp.ndarray, shape (N,)

    Returns
    -------
    es : float
    run_es : cp.ndarray, shape (N,)
    """
    N = weighted_metric.shape[0]
    n_hits = float(tag_indicator.sum())
    n_miss = N - n_hits

    sum_correl_tag = float((tag_indicator * weighted_metric).sum())

    norm_tag = 1.0 / max(sum_correl_tag, 1e-10)
    norm_no_tag = 1.0 / max(n_miss, 1.0)

    no_tag = 1.0 - tag_indicator

    increments = (
        tag_indicator * weighted_metric * norm_tag
        - no_tag * norm_no_tag
    )

    run_es = cp.cumsum(increments)
    max_es = float(run_es.max())
    min_es = float(run_es.min())

    es = max_es if abs(max_es) > abs(min_es) else min_es

    return es, run_es


def _enrichment_scores_and_running_gpu_batch(
    weighted_metric: cp.ndarray,
    tag_indicators: cp.ndarray,
) -> tuple[cp.ndarray, cp.ndarray]:
    """
    Compute enrichment scores and full running ES for multiple feature sets.

    Vectorised version of :func:`_enrichment_scores_and_running_gpu`.

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


def _generate_permutation_indices(
    n_features: int,
    n_perm: int,
    seed: int,
) -> np.ndarray:
    """
    Generate permutation index arrays (feature-set permutation).

    Row 0 = identity (original order), rows 1..n_perm = shuffled.

    Parameters
    ----------
    n_features : int
    n_perm : int
    seed : int

    Returns
    -------
    np.ndarray, shape (n_perm + 1, n_features), dtype int32
    """
    perm_indices = np.empty((n_perm + 1, n_features), dtype=np.int32)
    perm_indices[0] = np.arange(n_features, dtype=np.int32)

    rs = np.random.RandomState(seed)
    for i in range(1, n_perm + 1):
        perm_indices[i] = perm_indices[0].copy()
        rs.shuffle(perm_indices[i])

    return perm_indices


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
    nvals = np.sort(nesnull_concat)
    nnes = np.sort(nes_observed)

    all_neg_idx = np.searchsorted(nvals, 0, side="left")
    nes_neg_idx = np.searchsorted(nnes, 0, side="left")

    fdrs = np.ones(len(nes_observed))

    for i, nes in enumerate(nes_observed):
        if nes >= 0:
            all_pos = len(nvals) - all_neg_idx
            all_higher = len(nvals) - np.searchsorted(nvals, nes, side="left")
            nes_pos = len(nnes) - nes_neg_idx
            nes_higher = len(nnes) - np.searchsorted(nnes, nes, side="left")
        else:
            all_pos = all_neg_idx
            all_higher = np.searchsorted(nvals, nes, side="right")
            nes_pos = nes_neg_idx
            nes_higher = np.searchsorted(nnes, nes, side="right")

        if all_pos > 0 and nes_pos > 0:
            phi_norm = all_higher / all_pos
            phi_obs = nes_higher / nes_pos
            if phi_obs > 0:
                fdr = phi_norm / phi_obs
                fdrs[i] = min(fdr, 1.0)
            else:
                fdrs[i] = 1.0
        else:
            fdrs[i] = 1.0

    return fdrs


# =============================================================================
# Leading edge
# =============================================================================


def _leading_edge_size(run_es_np: np.ndarray, es: float, hit_indices: np.ndarray) -> int:
    """
    Count leading-edge genes (hits before the ES peak).
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
    perm_batch_size: int = 256,
    gs_batch_size: int = 16,
) -> list[dict]:
    """
    GPU-accelerated preranked GSEA.

    Implements the same algorithm as GSEApy's ``prerank`` but runs the
    enrichment-score computation entirely on the GPU using CuPy.

    Feature sets are processed in batches of ``gs_batch_size`` and, within
    each batch, permutations are chunked into groups of ``perm_batch_size``.
    All feature-set × permutation combinations in a chunk are evaluated in
    a **single GPU kernel call**, which dramatically increases GPU
    utilisation compared to processing one feature set at a time.

    Peak GPU memory for the ES computation is approximately
    ``gs_batch_size * perm_batch_size * N * 4`` bytes (float32).

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
    perm_batch_size : int, default 256
        Number of permutations to process together on GPU per feature-set
        batch. Controls GPU memory usage. Reduce if OOM.
    gs_batch_size : int, default 16
        Number of feature sets to process simultaneously on the GPU.
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
        - hits: indices of feature-set members in the ranked list
    """
    from tqdm.auto import tqdm

    N = len(feature_names)
    if N == 0:
        return []

    # Build feature-name → index lookup
    feature_to_idx = {g: i for i, g in enumerate(feature_names)}

    # Weight the metric: |r|^weight
    ranking_values = np.asarray(ranking_values, dtype=np.float64)
    weighted_metric_np = (np.abs(ranking_values) ** weight).astype(np.float32)
    weighted_metric_gpu = cp.asarray(weighted_metric_np)

    # Filter and build tag indicators for each feature set
    valid_sets = []  # (name, hit_indices_np, tag_np)
    for term, members in feature_sets.items():
        hit_idx = sorted([feature_to_idx[g] for g in members if g in feature_to_idx])
        if min_size <= len(hit_idx) <= max_size:
            tag = np.zeros(N, dtype=np.float32)
            tag[hit_idx] = 1.0
            valid_sets.append((term, np.array(hit_idx, dtype=np.int32), tag))

    if not valid_sets:
        logger.warning("No feature sets passed size filter.")
        return []

    n_sets = len(valid_sets)
    n_total_perms = permutation_num + 1  # including the unpermuted original

    logger.info(
        f"GPU GSEA: {n_sets} feature sets, {N} features, "
        f"{permutation_num} permutations  "
        f"(gs_batch={gs_batch_size}, perm_batch={perm_batch_size})"
    )

    # Generate permutation indices ONCE on CPU (shared across all feature sets)
    logger.info("Generating permutation indices...")
    perm_indices = _generate_permutation_indices(N, permutation_num, seed)
    # Keep on CPU; we'll transfer batches to GPU as needed

    # Pre-allocate result arrays
    all_es = np.empty(n_sets, dtype=np.float64)
    all_nes = np.empty(n_sets, dtype=np.float64)
    all_pvals = np.empty(n_sets, dtype=np.float64)
    all_lead_edge_n = np.empty(n_sets, dtype=np.int32)
    all_hits = []
    nesnull_parts = []

    # Determine effective batch sizes
    effective_perm_batch = min(perm_batch_size, n_total_perms)
    effective_gs_batch = min(gs_batch_size, n_sets)

    n_gs_batches = (n_sets + effective_gs_batch - 1) // effective_gs_batch

    # Process feature sets in batches
    for gs_start in tqdm(
        range(0, n_sets, effective_gs_batch),
        desc="GSEA feature-set batches",
        total=n_gs_batches,
    ):
        gs_end = min(gs_start + effective_gs_batch, n_sets)
        gs_batch = valid_sets[gs_start:gs_end]
        n_gs = len(gs_batch)

        # Stack tag indicators for all feature sets in this batch → (n_gs, N)
        tags_np = np.stack([t for _, _, t in gs_batch])
        tags_gpu = cp.asarray(tags_np)

        # -----------------------------------------------------------
        # Compute ES for every (feature-set, permutation) pair in chunks
        # -----------------------------------------------------------
        es_all = np.empty((n_gs, n_total_perms), dtype=np.float32)

        for perm_start in range(0, n_total_perms, effective_perm_batch):
            perm_end = min(perm_start + effective_perm_batch, n_total_perms)
            n_perm = perm_end - perm_start

            # Transfer permutation indices once for all feature sets
            perm_batch_gpu = cp.asarray(perm_indices[perm_start:perm_end])

            # Apply permutations to all feature sets at once:
            #   tags_gpu[:, perm_batch_gpu] → (n_gs, n_perm, N)
            # Flatten to (n_gs * n_perm, N) for a single kernel call
            perm_tags = tags_gpu[:, perm_batch_gpu].reshape(
                n_gs * n_perm, N
            )

            # One GPU kernel for the entire (gene-set × permutation) block
            es_batch_flat = _enrichment_scores_gpu(
                weighted_metric_gpu, perm_tags
            )
            es_all[:, perm_start:perm_end] = cp.asnumpy(
                es_batch_flat.reshape(n_gs, n_perm)
            )

            del perm_batch_gpu, perm_tags, es_batch_flat

        # -----------------------------------------------------------
        # Running ES for leading edge (all feature sets in batch at once)
        # -----------------------------------------------------------
        _, run_es_batch_gpu = _enrichment_scores_and_running_gpu_batch(
            weighted_metric_gpu, tags_gpu
        )
        run_es_batch_np = cp.asnumpy(run_es_batch_gpu)
        del tags_gpu, run_es_batch_gpu

        # -----------------------------------------------------------
        # Per-feature-set statistics (CPU – negligible cost)
        # -----------------------------------------------------------
        for j in range(n_gs):
            idx = gs_start + j
            _term, hit_idx, _ = gs_batch[j]

            es_obs = float(es_all[j, 0])
            esnull = es_all[j, 1:]

            pval = _compute_pval(es_obs, esnull)
            nes, nesnull = _normalize_es(es_obs, esnull)
            le_n = _leading_edge_size(
                run_es_batch_np[j], es_obs, hit_idx
            )

            all_es[idx] = es_obs
            all_nes[idx] = nes
            all_pvals[idx] = pval
            all_lead_edge_n[idx] = le_n
            all_hits.append(hit_idx)
            nesnull_parts.append(nesnull)

    # FDR across all feature sets
    nesnull_concat = np.concatenate(nesnull_parts)
    fdrs = _compute_fdr(all_nes, nesnull_concat)

    # Build results
    results = []
    for i, (term, hit_idx, _) in enumerate(valid_sets):
        results.append({
            "term": term,
            "es": float(all_es[i]),
            "nes": float(all_nes[i]),
            "pval": float(all_pvals[i]),
            "fdr": float(fdrs[i]),
            "lead_edge_n": int(all_lead_edge_n[i]),
            "hits": all_hits[i].tolist(),
        })

    # Free GPU memory
    del weighted_metric_gpu
    cp.get_default_memory_pool().free_all_blocks()

    return results
