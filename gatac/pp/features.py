"""
GPU-accelerated feature selection for ATAC-seq tile matrices.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Literal

import cupy as cp
import cupyx.scipy.sparse as cusp
import numpy as np
import scipy.sparse as sp

from .._cupy_compat import bincount

logger = logging.getLogger(__name__)


def _is_binary_matrix(X) -> bool:
    """
    Detect if a sparse matrix is binary: bool dtype, or only 0/1 stored.

    The value check is what catches GATAC's own binary matrices:
    ``make_tile_matrix(count_strategy="binarize")`` stores its 0/1 values in
    a numeric dtype.

    Parameters
    ----------
    X : sparse matrix
        Matrix to check (scipy or cupyx sparse)

    Returns
    -------
    bool
        True if matrix is binary, False otherwise
    """
    if isinstance(X, (cusp.csr_matrix, cusp.csc_matrix)):
        xp = cp
    elif sp.issparse(X):
        xp = np
    else:
        return False
    if X.data.dtype == bool:
        return True
    return bool(xp.all((X.data == 0) | (X.data == 1)))


def _find_most_accessible_features_gpu(
    feature_count: cp.ndarray,
    filter_lower_quantile: float,
    filter_upper_quantile: float,
    total_features: int,
    is_binary: bool = False,
) -> cp.ndarray:
    """
    Find most accessible features.

    For binary matrices (following ArchR): select top N features by total accessibility.
    For count matrices: select top N features excluding quantile tails.

    Note: Zero-count features are always excluded before quantile filtering,
    matching SnapATAC2's behavior.

    Parameters
    ----------
    feature_count : cp.ndarray
        Array of counts per feature
    filter_lower_quantile : float
        Lower quantile to filter out (ignored for binary matrices)
    filter_upper_quantile : float
        Upper quantile to filter out (ignored for binary matrices)
    total_features : int
        Number of features to select
    is_binary : bool
        Whether the matrix is binary (default: False)

    Returns
    -------
    cp.ndarray
        Indices of selected features
    """
    n = len(feature_count)

    if is_binary:
        # For binary matrices: select top N most accessible features.
        # argpartition is O(n) vs argsort's O(n log n) — meaningful for large genomes.
        n_to_select = min(total_features, n)
        if n_to_select < n:
            # Partial sort: only the top-N need to be identified, not fully ordered
            partition_idx = cp.argpartition(feature_count, n - n_to_select)
            selected = partition_idx[n - n_to_select:]
        else:
            selected = cp.arange(n)
    else:
        # For count matrices: first exclude zero-count features (matching SnapATAC2)
        sorted_indices = cp.argsort(feature_count)
        # Find the first non-zero index in sorted order
        sorted_counts = feature_count[sorted_indices]
        nonzero_mask = sorted_counts > 0
        first_nonzero_idx = int(cp.argmax(nonzero_mask))
        
        # Filter to non-zero features only
        nonzero_indices = sorted_indices[first_nonzero_idx:]
        n_nonzero = len(nonzero_indices)
        
        # Apply quantile filtering on non-zero features only
        n_lower = int(filter_lower_quantile * n_nonzero)
        n_upper = int(filter_upper_quantile * n_nonzero)
        
        valid_range = nonzero_indices[n_lower:n_nonzero - n_upper]
        
        # Select top N from valid range (reversed to get highest counts first)
        n_to_select = min(total_features, len(valid_range))
        selected = valid_range[-n_to_select:]

    return selected


def _compute_feature_counts_gpu(
    X,
    chunk_size: int = 2000
) -> cp.ndarray:
    """
    Compute per-feature (column) counts from a sparse matrix.

    For CPU sparse matrices, column sums are computed entirely on CPU and only
    the small result vector is transferred to GPU, avoiding the expensive PCIe
    bulk data transfer that bottlenecks consumer GPUs.

    Parameters
    ----------
    X : sparse matrix
        Cell x feature count matrix (GPU or CPU sparse)
    chunk_size : int
        Unused; kept for API compatibility.

    Returns
    -------
    cp.ndarray
        Sum of counts per feature
    """
    n_features = X.shape[1]

    if isinstance(X, cusp.csr_matrix):
        # Already on GPU — bincount is O(nnz) and avoids format conversion
        if X.data.dtype == cp.bool_:
            return bincount(X.indices, minlength=n_features).astype(cp.float32)
        else:
            return cp.bincount(
                X.indices,
                weights=X.data.astype(cp.float32),
                minlength=n_features,
            )
    elif isinstance(X, cusp.csc_matrix):
        if X.data.dtype == cp.bool_:
            # For CSC, column counts are directly encoded in indptr differences
            return cp.diff(X.indptr).astype(cp.float32)
        else:
            return cp.array(X.sum(axis=0)).ravel().astype(cp.float32)
    elif sp.issparse(X):
        # CPU sparse: compute on CPU, transfer only the small result vector.
        # This avoids shipping the full matrix over PCIe (critical for consumer GPUs).
        if not isinstance(X, sp.csr_matrix):
            X = X.tocsr()
        if X.data.dtype in (np.bool_, bool):
            # np.bincount on indices alone — O(nnz), no data array needed
            counts = np.bincount(X.indices, minlength=n_features).astype(np.float32)
        else:
            counts = np.asarray(X.sum(axis=0)).ravel().astype(np.float32)
        return cp.asarray(counts)
    else:
        # Dense array
        arr = X if isinstance(X, cp.ndarray) else cp.asarray(X)
        return arr.sum(axis=0).astype(cp.float32)


def _tile_stats_usable(adata) -> bool:
    """
    Whether var holds tile statistics that match ``adata``'s matrix: summed
    over exactly its cells, with as many entries as it has and, when the
    matrix is in memory, the same total. A matrix binarized, normalized or
    subset after it was built fails one of these; a backed matrix is only
    checked for its entry count, so its values are not read.
    """
    stats = adata.uns.get("tile_stats")
    if (
        stats is None
        or not {"count", "n_cells"} <= set(adata.var.columns)
        or int(stats.get("n_obs", -1)) != adata.n_obs
    ):
        return False
    X = adata.X
    if X is None:
        return True
    if adata.isbacked:
        nnz = X.group["indptr"][-1] if hasattr(X, "group") else None
        return nnz is not None and int(nnz) == int(adata.var["n_cells"].sum())
    if not sp.issparse(X) or X.nnz != int(adata.var["n_cells"].sum()):
        return False
    total = X.data.sum(dtype=np.float64)
    return bool(np.isclose(total, adata.var["count"].to_numpy(dtype=np.float64).sum(), rtol=0, atol=0.5))


def select_features(
    adata,
    n_features: int = 500000,
    filter_lower_quantile: float = 0.005,
    filter_upper_quantile: float = 0.005,
    inplace: bool = True,
    output_path: Optional[str | Path] = None,
):
    """
    GPU-accelerated feature selection for ATAC-seq tile matrices.

    For binary matrices: selects top N most accessible features, as ArchR does
    :cite:p:`Granja2021`.
    For count matrices: selects top N accessible features, excluding quantile tails.

    A tile matrix from :func:`~gatac.pp.make_tile_matrix` or
    :func:`~gatac.pp.scan_tiles` carries per-tile statistics in ``var``
    (``count``, ``n_cells``); they stand for the matrix as long as every cell
    they were summed over is kept (``uns["tile_stats"]["n_obs"]``), so the
    matrix is not read, and a :func:`~gatac.pp.scan_tiles` result needs none.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with cells × features, or a
        :func:`~gatac.pp.scan_tiles` result
    n_features : int
        Target number of features to select (default: 500000)
    filter_lower_quantile : float
        Lower quantile threshold for filtering (ignored for binary matrices) (default: 0.005)
    filter_upper_quantile : float
        Upper quantile threshold for filtering (ignored for binary matrices) (default: 0.005)
    inplace : bool
        Whether to modify adata in place (default: True)
    output_path : str or Path, optional
        If provided, save the result to this path

    Returns
    -------
    adata : AnnData or None
        Modified AnnData if inplace=False, else None

    Examples
    --------
    >>> import gatac as ga
    >>> ga.pp.select_features(adata, n_features=500_000)
    >>> # adata.var["selected"] is now a boolean mask of the chosen features
    """

    logger.info(f"Selecting features from {adata.shape[1]:,} total")

    if _tile_stats_usable(adata):
        logger.info("Using the tile statistics of var")
        # every entry is 1 exactly when the counts equal the cell counts
        is_binary = bool((adata.var["count"] == adata.var["n_cells"]).all())
        feature_counts = cp.asarray(adata.var["count"].to_numpy(dtype=np.float64)).astype(cp.float32)
    elif adata.X is None:
        raise ValueError(
            "`adata` has no X and no tile statistics ('count', 'n_cells' in var, "
            "uns['tile_stats']); see `pp.scan_tiles`"
        )
    else:
        is_binary = None

    # Detect if matrix is binary
    if is_binary is None:
        is_binary = _is_binary_matrix(adata.X)
    if is_binary:
        logger.info("Detected binary matrix - using top-N selection (ArchR approach)")
    else:
        logger.info("Detected count matrix - using quantile-filtered selection")

    # Compute feature counts
    logger.debug("Computing feature accessibility")
    if not _tile_stats_usable(adata):
        feature_counts = _compute_feature_counts_gpu(adata.X)
    logger.debug(f"Count range: {float(feature_counts.min()):.0f} - {float(feature_counts.max()):.0f}")

    # Select features
    selected_indices = _find_most_accessible_features_gpu(
        feature_counts,
        filter_lower_quantile,
        filter_upper_quantile,
        n_features,
        is_binary=is_binary
    )

    n_selected = len(selected_indices)
    logger.debug(f"Selected {n_selected:,} features")

    # Build selection mask on CPU — avoids a GPU alloc + PCIe round-trip
    selected_mask_cpu = np.zeros(adata.shape[1], dtype=bool)
    selected_mask_cpu[selected_indices.get()] = True

    if inplace:
        adata.var['selected'] = selected_mask_cpu
        adata.var['accessibility_count'] = feature_counts.get()
        result = None
    else:
        adata = adata.copy()
        adata.var['selected'] = selected_mask_cpu
        adata.var['accessibility_count'] = feature_counts.get()
        result = adata

    if output_path:
        output_path = Path(output_path)
        adata.write_h5ad(str(output_path))
        logger.info(f"Saved to {output_path.name}")

    logger.info(f"Selected {n_selected:,} features")

    return result
