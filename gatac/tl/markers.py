"""
GPU-accelerated marker peak detection using binomial test.

This module provides GPU-accelerated differential accessibility analysis
for single-cell ATAC-seq data. It uses the binomial test to compare
peak accessibility between foreground and background cell groups.

The binomial test is particularly suitable for peak marker detection
because accessibility data is naturally binary (accessible vs not accessible).

Approach based on ArchR's getMarkerFeatures with testMethod="binomial":
- Binarizes the peak matrix (accessible vs not accessible)
- Compares proportion of cells with accessible peaks between groups
- Uses two-sided binomial test for differential detection
"""

from __future__ import annotations

import gc
import logging
from typing import Optional, Sequence, Union

import cupy as cp
import cupyx.scipy.sparse as cusp
import numpy as np
import pandas as pd
import polars as pl
import scipy.sparse as sp
from cupyx.scipy.special import betainc

logger = logging.getLogger(__name__)

# GPU memory pools
mempool = cp.get_default_memory_pool()
pinned_mempool = cp.get_default_pinned_memory_pool()


# =============================================================================
# GPU-accelerated Binomial Test Functions
# =============================================================================


def _binomial_sf_gpu(k: cp.ndarray, n: cp.ndarray, p: cp.ndarray) -> cp.ndarray:
    """
    GPU-accelerated binomial survival function (1 - CDF).
    
    Computes P(X > k) where X ~ Binomial(n, p) using the regularized
    incomplete beta function.
    
    P(X > k) = I_{p}(k+1, n-k) where I is the regularized incomplete beta.
    
    Parameters
    ----------
    k : cp.ndarray
        Number of successes (0 <= k <= n)
    n : cp.ndarray  
        Number of trials
    p : cp.ndarray
        Probability of success (0 < p < 1)
        
    Returns
    -------
    cp.ndarray
        Survival function values P(X > k)
    """
    k = cp.asarray(k, dtype=cp.float64)
    n = cp.asarray(n, dtype=cp.float64)
    p = cp.asarray(p, dtype=cp.float64)
    
    result = cp.zeros_like(k, dtype=cp.float64)
    
    # Normal case: 0 <= k < n
    valid = (k >= 0) & (k < n) & (p > 0) & (p < 1)
    if cp.any(valid):
        k_valid = k[valid]
        n_valid = n[valid]
        p_valid = p[valid]
        result[valid] = betainc(k_valid + 1, n_valid - k_valid, p_valid)
    
    # Edge cases
    result[k >= n] = 0.0
    result[p <= 0] = 0.0
    result[(p >= 1) & (k < n)] = 1.0
    result[(p >= 1) & (k >= n)] = 0.0
    
    return result


def _p_adjust_bh_gpu(p: cp.ndarray) -> cp.ndarray:
    """
    GPU-accelerated Benjamini-Hochberg FDR correction.
    
    Following R's p.adjust(method="BH"):
    1. Sort p-values in descending order
    2. Multiply by n/rank (rank from n down to 1)
    3. Apply forward cumulative minimum to ensure monotonicity
    4. Reorder back to original order
    """
    n = len(p)
    if n == 0:
        return p
    
    # Sort p-values in descending order
    by_descend = cp.argsort(p)[::-1]
    by_orig = cp.argsort(by_descend)
    
    # Steps: n/n, n/(n-1), ..., n/1
    steps = float(n) / cp.arange(n, 0, -1, dtype=cp.float64)
    p_sorted = p[by_descend]
    q_raw = steps * p_sorted
    
    # Forward cumulative minimum (transfer to CPU since cupy doesn't support it)
    # This ensures q[i] <= q[i+1] in the sorted order
    q_cpu = q_raw.get()
    q_cummin = np.minimum.accumulate(q_cpu)
    q = cp.asarray(q_cummin)
    
    # Clip to [0, 1]
    q = cp.minimum(q, 1.0)
    
    return q[by_orig]


def _binarize_matrix_gpu(X) -> cusp.csr_matrix:
    """Binarize a sparse matrix on GPU."""
    if isinstance(X, cusp.csr_matrix):
        X_gpu = X
    elif isinstance(X, cusp.csc_matrix):
        X_gpu = X.tocsr()
    elif sp.issparse(X):
        X_csr = X.tocsr() if not sp.isspmatrix_csr(X) else X
        X_gpu = cusp.csr_matrix(
            (cp.asarray(X_csr.data, dtype=cp.float32),
             cp.asarray(X_csr.indices),
             cp.asarray(X_csr.indptr)),
            shape=X_csr.shape
        )
    else:
        raise TypeError(f"Unsupported matrix type: {type(X)}")
    
    X_binary = cusp.csr_matrix(
        (cp.ones_like(X_gpu.data, dtype=cp.float32),
         X_gpu.indices.copy(),
         X_gpu.indptr.copy()),
        shape=X_gpu.shape
    )
    
    return X_binary


def _compute_group_stats_gpu(
    X_binary: cusp.csr_matrix,
    fg_indices: cp.ndarray,
    bg_indices: cp.ndarray,
) -> tuple[cp.ndarray, cp.ndarray, int, int]:
    """Compute accessibility counts for foreground and background groups."""
    n_fg = len(fg_indices)
    n_bg = len(bg_indices)
    
    X_fg = X_binary[fg_indices.get(), :]
    X_bg = X_binary[bg_indices.get(), :]
    
    fg_counts = cp.asarray(X_fg.sum(axis=0)).ravel()
    bg_counts = cp.asarray(X_bg.sum(axis=0)).ravel()
    
    return fg_counts, bg_counts, n_fg, n_bg


def _binomial_test_gpu(
    fg_counts: cp.ndarray,
    bg_counts: cp.ndarray,
    n_fg: int,
    n_bg: int,
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]:
    """
    GPU-accelerated binomial test for differential accessibility.
    
    Following ArchR's approach (.sparseMatBinomTest):
    - If fg_mean >= bg_mean: test P(X >= fg_count) with p = bg_rate
    - If fg_mean < bg_mean: test P(X >= bg_count) with p = fg_rate
    - Two-sided p-value = min(2 * one_sided_p, 1)
    
    Returns
    -------
    tuple
        (p_values, log2_fc, mean_fg, mean_bg, mean_diff)
    """
    n_features = len(fg_counts)
    
    # Compute means (proportions)
    mean_fg = fg_counts / n_fg
    mean_bg = bg_counts / n_bg
    mean_diff = mean_fg - mean_bg
    
    # Log2 fold change with offset (following ArchR: offset = 1)
    offset = 1.0
    log2_fc = cp.log2((fg_counts + offset) / (bg_counts + offset))
    
    # Binomial test (two-sided, following ArchR's .sparseMatBinomTest)
    p_values = cp.zeros(n_features, dtype=cp.float64)
    
    # Where foreground has higher proportion
    fg_higher = mean_fg >= mean_bg
    
    # P(X >= k) = sf(k-1) where X ~ Binom(n_fg, p_bg)
    # Use max(bg_counts, 1) / n_bg as background probability (ArchR approach)
    if cp.any(fg_higher):
        p_bg = cp.maximum(bg_counts[fg_higher], 1.0) / n_bg
        p_one_sided = _binomial_sf_gpu(
            fg_counts[fg_higher] - 1,
            cp.full(int(fg_higher.sum()), n_fg, dtype=cp.float64),
            p_bg
        )
        p_values[fg_higher] = cp.minimum(2.0 * p_one_sided, 1.0)
    
    # Where background has higher proportion  
    bg_higher = ~fg_higher
    if cp.any(bg_higher):
        p_fg = cp.maximum(fg_counts[bg_higher], 1.0) / n_fg
        p_one_sided = _binomial_sf_gpu(
            bg_counts[bg_higher] - 1,
            cp.full(int(bg_higher.sum()), n_bg, dtype=cp.float64),
            p_fg
        )
        p_values[bg_higher] = cp.minimum(2.0 * p_one_sided, 1.0)
    
    p_values = cp.clip(p_values, 1e-300, 1.0)
    
    return p_values, log2_fc, mean_fg, mean_bg, mean_diff


def marker_peaks(
    adata,
    groupby: str,
    groups: Optional[Union[str, Sequence[str]]] = None,
    reference: str = "rest",
    max_cells: int = 500,
    min_pct: float = 0.05,
    min_log2_fc: float = 1.0,
    use_raw: bool = False,
    key_added: str = "marker_peaks",
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """
    GPU-accelerated marker peak detection using binomial test.
    
    Identifies differentially accessible peaks between cell groups using
    the binomial test on binarized accessibility data. This approach follows
    ArchR's getMarkerFeatures with testMethod="binomial".
    
    For each group, compares the proportion of cells with accessible peaks
    in the foreground group vs the background (all other cells or a specific
    reference group).
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with cells × peaks.
    groupby : str
        Column name in adata.obs containing group labels.
    groups : str or list of str, optional
        Groups to test. If None, test all groups in groupby.
    reference : str
        Reference group for comparison:
        - "rest": compare each group against all other cells (default)
        - group name: compare each group against the specified group
    max_cells : int
        Maximum number of cells to sample from each group (foreground and 
        background). This controls test sensitivity - larger values detect
        smaller differences but may flag many biologically irrelevant peaks.
        Default: 500 (following ArchR).
    min_pct : float
        Minimum fraction of cells (0-1) with accessible peak in either
        group to include the peak in results. Default: 0.05
    min_log2_fc : float
        Minimum absolute log2 fold change threshold for results. Default: 1.0
    use_raw : bool
        If True, use adata.raw.X. Default: False
    key_added : str
        Key to store results in adata.uns. Default: "marker_peaks"
    seed : int
        Random seed for cell subsampling reproducibility. Default: 42
        
    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary mapping group names to DataFrames with columns:
        - "feature": Peak/feature name
        - "log2_fc": Log2 fold change (foreground vs background)
        - "mean_fg": Mean accessibility in foreground (proportion of cells)
        - "mean_bg": Mean accessibility in background
        - "mean_diff": mean_fg - mean_bg
        - "p_value": Raw p-value from two-sided binomial test
        - "fdr": Benjamini-Hochberg adjusted p-value
        
    Notes
    -----
    The binomial test compares the observed count of accessible cells in the
    foreground to what would be expected under the null hypothesis that the
    foreground has the same accessibility rate as the background.
    
    **Important**: The test is highly sensitive to sample size. With many cells,
    even tiny differences become statistically significant. The `max_cells` 
    parameter (default 500, following ArchR) subsamples both foreground and 
    background to control sensitivity. Adjust `min_log2_fc` and `min_pct` to
    focus on biologically meaningful differences.
    
    Two-sided test is performed (following ArchR):
    - If fg_rate >= bg_rate: tests for enrichment
    - If fg_rate < bg_rate: tests for depletion
    
    Results are sorted by FDR (ascending) then by absolute log2_fc (descending).
    
    Examples
    --------
    >>> import gatac as ga
    >>> # Find marker peaks for all clusters
    >>> results = ga.tl.marker_peaks(adata, groupby="cluster")
    >>>
    >>> # Get top markers for a specific cluster
    >>> cd8_markers = results["CD8_T"].filter(pl.col("fdr") < 0.05).head(100)
    >>>
    >>> # For more sensitive detection (more hits), increase max_cells:
    >>> results = ga.tl.marker_peaks(adata, groupby="cluster", max_cells=1000, min_log2_fc=0.5)
    """
    logger.info(f"Running marker peak detection for groups in '{groupby}'")
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Get matrix
    X = adata.raw.X if use_raw and adata.raw is not None else adata.X
    var_names = adata.raw.var_names if use_raw and adata.raw is not None else adata.var_names
    var_names = np.asarray(var_names)
    
    # Get group labels
    if groupby not in adata.obs:
        raise ValueError(f"'{groupby}' not found in adata.obs")
    
    group_labels = np.asarray(adata.obs[groupby])
    unique_groups = np.unique(group_labels)
    
    # Determine which groups to test
    if groups is None:
        test_groups = list(unique_groups)
    elif isinstance(groups, str):
        test_groups = [groups]
    else:
        test_groups = list(groups)
    
    # Validate groups
    for g in test_groups:
        if g not in unique_groups:
            raise ValueError(f"Group '{g}' not found in '{groupby}'")
    
    # Binarize matrix on GPU
    logger.debug("Binarizing matrix on GPU")
    X_binary = _binarize_matrix_gpu(X)
    n_cells, n_features = X_binary.shape
    
    logger.info(f"Testing {len(test_groups)} groups, {n_features:,} features, {n_cells:,} cells")
    logger.info(f"Using max_cells={max_cells} per group for balanced comparison")
    
    # Store results
    results = {}
    
    for group in test_groups:
        logger.debug(f"Processing group: {group}")
        
        # Get cell indices for foreground and background
        fg_mask = group_labels == group
        fg_indices_all = np.where(fg_mask)[0]
        
        if reference == "rest":
            bg_mask = ~fg_mask
        else:
            if reference not in unique_groups:
                raise ValueError(f"Reference group '{reference}' not found")
            bg_mask = group_labels == reference
        
        bg_indices_all = np.where(bg_mask)[0]
        
        n_fg_total = len(fg_indices_all)
        n_bg_total = len(bg_indices_all)
        
        if n_fg_total == 0:
            logger.warning(f"No cells in group '{group}', skipping")
            continue
        if n_bg_total == 0:
            logger.warning(f"No cells in background for group '{group}', skipping")
            continue
        
        # Subsample to max_cells (following ArchR approach)
        # This balances foreground and background and controls test sensitivity
        n_fg_sample = min(n_fg_total, max_cells)
        n_bg_sample = min(n_bg_total, max_cells)
        
        # For fair comparison, use equal sample sizes when possible
        n_sample = min(n_fg_sample, n_bg_sample)
        
        if n_fg_total > n_sample:
            fg_indices = np.random.choice(fg_indices_all, n_sample, replace=False)
        else:
            fg_indices = fg_indices_all
            
        if n_bg_total > n_sample:
            bg_indices = np.random.choice(bg_indices_all, n_sample, replace=False)
        else:
            bg_indices = bg_indices_all
        
        fg_indices = cp.asarray(fg_indices)
        bg_indices = cp.asarray(bg_indices)
        
        n_fg = len(fg_indices)
        n_bg = len(bg_indices)
        
        logger.debug(f"  Foreground: {n_fg} cells (from {n_fg_total}), Background: {n_bg} cells (from {n_bg_total})")
        
        # Compute group statistics
        fg_counts, bg_counts, n_fg, n_bg = _compute_group_stats_gpu(
            X_binary, fg_indices, bg_indices
        )
        
        # Run binomial test
        p_values, log2_fc, mean_fg, mean_bg, mean_diff = _binomial_test_gpu(
            fg_counts, bg_counts, n_fg, n_bg
        )
        
        # FDR correction
        fdr = _p_adjust_bh_gpu(p_values)
        
        # Create result DataFrame
        df = pl.DataFrame({
            "feature": var_names.tolist(),
            "log2_fc": log2_fc.get().tolist(),
            "mean_fg": mean_fg.get().tolist(),
            "mean_bg": mean_bg.get().tolist(),
            "mean_diff": mean_diff.get().tolist(),
            "p_value": p_values.get().tolist(),
            "fdr": fdr.get().tolist(),
        })
        
        # Apply filters
        if min_pct > 0 or min_log2_fc > 0:
            df = df.filter(
                ((pl.col("mean_fg") >= min_pct) | (pl.col("mean_bg") >= min_pct)) &
                (pl.col("log2_fc").abs() >= min_log2_fc)
            )
        
        # Sort by FDR, then by absolute log2_fc
        df = df.with_columns(
            pl.col("log2_fc").abs().alias("_abs_log2_fc")
        ).sort(["fdr", "_abs_log2_fc"], descending=[False, True]).drop("_abs_log2_fc")
        
        results[group] = df
        logger.debug(f"  Found {len(df)} features for '{group}'")
    
    # Store in adata.uns
    adata.uns[key_added] = {
        "params": {
            "groupby": groupby,
            "reference": reference,
            "max_cells": max_cells,
            "min_pct": min_pct,
            "min_log2_fc": min_log2_fc,
            "seed": seed,
        },
        "results": {k: v.to_pandas() for k, v in results.items()},
    }
    
    # Clean up GPU memory
    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()
    gc.collect()
    
    logger.info(f"Marker peak detection complete. Results stored in adata.uns['{key_added}']")
    
    return {k: v.to_pandas() for k, v in results.items()}


def get_marker_peaks(
    adata,
    key: str = "marker_peaks",
    group: Optional[str] = None,
    fdr_threshold: float = 0.05,
    log2_fc_threshold: float = 0.0,
    n_peaks: Optional[int] = None,
) -> Union[dict[str, list[str]], list[str]]:
    """
    Extract marker peak names from stored results.
    
    Convenience function to get peak names passing significance thresholds.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data with marker peak results in uns.
    key : str
        Key in adata.uns containing results from marker_peaks.
    group : str, optional
        Specific group to extract. If None, return all groups.
    fdr_threshold : float
        FDR threshold (default: 0.05).
    log2_fc_threshold : float
        Minimum absolute log2 fold change (default: 0.0).
    n_peaks : int, optional
        Maximum number of peaks per group.
        
    Returns
    -------
    dict or list
        Dictionary mapping group names to peak lists, or list if group specified.

    Examples
    --------
    >>> import gatac as ga
    >>> # All groups at once
    >>> markers_by_group = ga.tl.get_marker_peaks(adata, fdr_threshold=0.05)
    >>> # A single group's top peaks
    >>> top_peak_names = ga.tl.get_marker_peaks(adata, group="CD8_T", fdr_threshold=0.05, n_peaks=100)
    """

    if key not in adata.uns:
        raise KeyError(f"'{key}' not found in adata.uns. Run marker_peaks first.")
    
    results_dict = adata.uns[key]["results"]
    
    def filter_peaks(df):
        if not isinstance(df, pl.DataFrame):
            df = pl.from_pandas(df)
        
        filtered = df.filter(
            (pl.col("fdr") <= fdr_threshold) &
            (pl.col("log2_fc").abs() >= log2_fc_threshold)
        )
        
        if n_peaks is not None:
            filtered = filtered.head(n_peaks)
        
        return filtered["feature"].to_list()
    
    if group is not None:
        if group not in results_dict:
            raise KeyError(f"Group '{group}' not found in results")
        return filter_peaks(results_dict[group])
    
    return {g: filter_peaks(df) for g, df in results_dict.items()}
