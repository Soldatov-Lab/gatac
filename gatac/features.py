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
        Chunk size for processing (used for very large matrices)

    Returns
    -------
    cp.ndarray
        Sum of counts per feature
    """
    if isinstance(X, (cusp.csr_matrix, cusp.csc_matrix)):
        # Already on GPU
        X_gpu = X if isinstance(X, cusp.csc_matrix) else X.tocsc()
        return cp.array(X_gpu.sum(axis=0)).ravel()
    elif sp.issparse(X):
        # CPU sparse matrix - transfer to GPU for fast column sum
        try:
            # Convert to float32 for GPU compatibility
            X_f32 = X.astype(np.float32)
            X_gpu = cusp.csr_matrix(X_f32)
            return cp.array(X_gpu.sum(axis=0)).ravel()
        except Exception as e:
            # Fallback to chunked CPU processing if GPU transfer fails
            logger.warning(f"GPU transfer failed, falling back to CPU: {e}")
            n_features = X.shape[1]
            feature_counts = cp.zeros(n_features, dtype=cp.float32)
            for start in range(0, n_features, chunk_size):
                end = min(start + chunk_size, n_features)
                chunk_sum = np.array(X[:, start:end].sum(axis=0)).ravel()
                feature_counts[start:end] = cp.asarray(chunk_sum)
            return feature_counts
    else:
        # Dense array
        return cp.asarray(X.sum(axis=0)).ravel()


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


def select_features_multi(
    input_paths: list[str | Path],
    output_path: str | Path,
    n_features: int = 500000,
    filter_lower_quantile: float = 0.005,
    filter_upper_quantile: float = 0.005,
    chunk_size: int = 6000,
    binarize: bool = True,
):
    """
    Streaming feature selection across multiple h5ad files.

    Processes files one at a time to avoid OOM, producing a single
    combined output with selected features from all inputs.

    Parameters
    ----------
    input_paths : list of str or Path
        List of paths to h5ad files
    output_path : str or Path
        Output path for combined h5ad file
    n_features : int
        Target number of features to select (default: 500000)
    filter_lower_quantile : float
        Lower quantile threshold for filtering (default: 0.005)
    filter_upper_quantile : float
        Upper quantile threshold for filtering (default: 0.005)
    chunk_size : int
        Chunk size for row processing (default: 6000)
    binarize : bool
        Whether to binarize the output matrix (default: True)
    """
    import anndata as ad
    import scanpy as sc
    from tqdm import tqdm

    input_paths = [Path(p) for p in input_paths]
    output_path = Path(output_path)

    if len(input_paths) == 0:
        raise ValueError("No input files provided")

    if len(input_paths) == 1:
        # Single file - use regular function
        logger.info("Single file provided, using standard feature selection")
        adata = sc.read_h5ad(str(input_paths[0]))
        select_features(
            adata,
            n_features=n_features,
            filter_lower_quantile=filter_lower_quantile,
            filter_upper_quantile=filter_upper_quantile,
            inplace=True,
            output_path=output_path,
        )
        return

    logger.info(f"Processing {len(input_paths)} h5ad files")

    # Get reference var from first file
    first_adata = sc.read_h5ad(str(input_paths[0]))
    n_vars = first_adata.n_vars
    var_df = first_adata.var.copy()
    del first_adata
    cp.get_default_memory_pool().free_all_blocks()

    # =========================================================================
    # PASS 1: Aggregate feature counts across all files
    # =========================================================================
    logger.info("Pass 1: Aggregating feature counts...")
    feature_counts = cp.zeros(n_vars, dtype=cp.float32)
    total_cells = 0
    total_nnz = 0

    for fpath in tqdm(input_paths, desc="Counting features"):
        adata = sc.read_h5ad(str(fpath))
        if adata.n_vars != n_vars:
            raise ValueError(
                f"Feature mismatch: {fpath.name} has {adata.n_vars} features, "
                f"expected {n_vars}"
            )
        
        # Accumulate feature counts
        file_counts = _compute_feature_counts_gpu(adata.X)
        feature_counts += file_counts
        
        # Count cells and estimate nnz for preallocation
        total_cells += adata.n_obs
        if sp.issparse(adata.X):
            total_nnz += adata.X.nnz
        else:
            total_nnz += np.count_nonzero(adata.X)
        
        del adata
        cp.get_default_memory_pool().free_all_blocks()

    logger.debug(f"Total cells: {total_cells:,}, total features: {n_vars:,}")

    # Select features based on aggregated counts
    selected_indices = _find_most_accessible_features_gpu(
        feature_counts,
        filter_lower_quantile,
        filter_upper_quantile,
        n_features
    )
    selected_indices_sorted = cp.sort(selected_indices)
    selected_mask = cp.zeros(n_vars, dtype=bool)
    selected_mask[selected_indices] = True
    selected_mask_cpu = selected_mask.get()
    selected_indices_cpu = selected_indices_sorted.get()
    n_selected = len(selected_indices)
    logger.info(f"Selected {n_selected:,} features from aggregated counts")

    # =========================================================================
    # PASS 2: Build combined sparse matrix with selected features
    # =========================================================================
    logger.info("Pass 2: Building combined matrix...")

    # Estimate nnz for selected features (rough approximation)
    # We'll count actual nnz in first pass through
    actual_nnz = 0
    for fpath in tqdm(input_paths, desc="Counting nnz"):
        adata = sc.read_h5ad(str(fpath))
        X = adata.X
        if sp.issparse(X):
            X_sel = X[:, selected_indices_cpu]
            actual_nnz += X_sel.nnz
        else:
            actual_nnz += np.count_nonzero(X[:, selected_indices_cpu])
        del adata

    # Preallocate arrays with optimal dtypes
    dtype = bool if binarize else np.float32
    indices_dtype = np.uint32 if n_selected > 65535 else np.uint16
    indptr_dtype = np.uint64 if actual_nnz > 4294967295 else np.uint32

    logger.debug(f"Allocating: {actual_nnz:,} nnz, indices={indices_dtype}, indptr={indptr_dtype}")
    all_data = np.empty(actual_nnz, dtype=dtype)
    all_indices = np.empty(actual_nnz, dtype=indices_dtype)
    all_indptr = np.zeros(total_cells + 1, dtype=indptr_dtype)

    # Collect obs metadata
    all_obs = []
    current_nnz = 0
    current_row = 0

    for fpath in tqdm(input_paths, desc="Building matrix"):
        adata = sc.read_h5ad(str(fpath))
        X = adata.X[:, selected_indices_cpu]
        
        if not sp.issparse(X):
            X = sp.csr_matrix(X)
        elif not isinstance(X, sp.csr_matrix):
            X = X.tocsr()

        n_rows = X.shape[0]
        nnz = X.nnz

        if binarize:
            all_data[current_nnz:current_nnz + nnz] = X.data.astype(bool)
        else:
            all_data[current_nnz:current_nnz + nnz] = X.data.astype(dtype)

        all_indices[current_nnz:current_nnz + nnz] = X.indices.astype(indices_dtype)
        all_indptr[current_row + 1:current_row + n_rows + 1] = (
            X.indptr[1:].astype(indptr_dtype) + current_nnz
        )

        # Collect obs with source file info
        obs = adata.obs.copy()
        obs['source_file'] = fpath.name
        all_obs.append(obs)

        current_nnz += nnz
        current_row += n_rows
        del adata

    # Build final sparse matrix
    combined_X = sp.csr_matrix(
        (all_data, all_indices, all_indptr),
        shape=(total_cells, n_selected)
    )

    # Build combined obs
    import pandas as pd
    combined_obs = pd.concat(all_obs, axis=0)
    combined_obs.index = combined_obs.index.astype(str)
    # Handle duplicate indices by making them unique
    if combined_obs.index.duplicated().any():
        combined_obs.index = pd.Index(
            [f"{idx}_{i}" for i, idx in enumerate(combined_obs.index)]
        )

    # Build var for selected features
    combined_var = var_df.iloc[selected_indices_cpu].copy()
    combined_var['selected'] = True
    combined_var['accessibility_count'] = feature_counts.get()[selected_indices_cpu]

    # Create combined AnnData
    combined_adata = ad.AnnData(
        X=combined_X,
        obs=combined_obs,
        var=combined_var,
    )

    # Save
    combined_adata.write_h5ad(str(output_path))
    logger.info(f"Saved combined matrix ({total_cells:,} cells × {n_selected:,} features) to {output_path.name}")
