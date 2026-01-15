"""
GPU-accelerated feature selection for ATAC-seq tile matrices.
"""

import logging
from pathlib import Path
from typing import Optional, Literal

import cupy as cp
import cupyx.scipy.sparse as cusp
import numpy as np
import scipy.sparse as sp

logger = logging.getLogger(__name__)


def _find_most_accessible_features_gpu(
    feature_count: cp.ndarray,
    filter_lower_quantile: float,
    filter_upper_quantile: float,
    total_features: int,
) -> cp.ndarray:
    """
    Find most accessible features, excluding quantile tails.

    Parameters
    ----------
    feature_count : cp.ndarray
        Array of counts per feature
    filter_lower_quantile : float
        Lower quantile to filter out
    filter_upper_quantile : float
        Upper quantile to filter out
    total_features : int
        Number of features to select

    Returns
    -------
    cp.ndarray
        Indices of selected features
    """
    sorted_indices = cp.argsort(feature_count)
    n = len(feature_count)

    lower_idx = int(n * filter_lower_quantile)
    upper_idx = int(n * (1 - filter_upper_quantile))

    valid_range = sorted_indices[lower_idx:upper_idx]
    n_to_select = min(total_features, len(valid_range))
    selected = valid_range[-n_to_select:]

    return selected


def _compute_feature_counts_gpu(
    X,
    chunk_size: int = 2000
) -> cp.ndarray:
    """
    Compute per-feature (column) counts from a sparse matrix.

    Parameters
    ----------
    X : sparse matrix
        Cell x feature count matrix (GPU or CPU sparse)
    chunk_size : int
        Chunk size for processing

    Returns
    -------
    cp.ndarray
        Sum of counts per feature
    """
    if isinstance(X, (cusp.csr_matrix, cusp.csc_matrix)):
        X_gpu = X if isinstance(X, cusp.csc_matrix) else X.tocsc()
        return cp.array(X_gpu.sum(axis=0)).ravel()
    else:
        # CPU sparse matrix - process in chunks
        n_features = X.shape[1]
        feature_counts = cp.zeros(n_features, dtype=cp.float32)

        for start in range(0, n_features, chunk_size):
            end = min(start + chunk_size, n_features)
            chunk = X[:, start:end]
            if sp.issparse(chunk):
                chunk_sum = np.array(chunk.sum(axis=0)).ravel()
            else:
                chunk_sum = chunk.sum(axis=0)
            feature_counts[start:end] = cp.asarray(chunk_sum)
            logger.debug(f"Processed features {start}-{end}")

        return feature_counts


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

    Select the most accessible features (tiles/bins) across all cells,
    filtering out very rare and very common features.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with cells × features
    n_features : int
        Target number of features to select (default: 500000)
    filter_lower_quantile : float
        Lower quantile threshold for filtering (default: 0.005)
    filter_upper_quantile : float
        Upper quantile threshold for filtering (default: 0.005)
    inplace : bool
        Whether to modify adata in place (default: True)
    output_path : str or Path, optional
        If provided, save the result to this path

    Returns
    -------
    adata : AnnData or None
        Modified AnnData if inplace=False, else None
    """
    logger.info(f"Selecting features from {adata.shape[1]:,} total")

    # Compute feature counts
    logger.debug("Computing feature accessibility")
    feature_counts = _compute_feature_counts_gpu(adata.X)
    logger.debug(f"Count range: {float(feature_counts.min()):.0f} - {float(feature_counts.max()):.0f}")

    # Select features
    selected_indices = _find_most_accessible_features_gpu(
        feature_counts,
        filter_lower_quantile,
        filter_upper_quantile,
        n_features
    )

    n_selected = len(selected_indices)
    logger.debug(f"Selected {n_selected:,} features")

    # Create selection mask
    selected_mask = cp.zeros(adata.shape[1], dtype=bool)
    selected_mask[selected_indices] = True
    selected_mask_cpu = selected_mask.get()

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
