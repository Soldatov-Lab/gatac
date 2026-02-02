"""
GPU-accelerated feature selection for ATAC-seq tile matrices.
"""

import gc
import logging
from pathlib import Path
from typing import Optional, Literal

import cupy as cp
import cupyx.scipy.sparse as cusp
import numpy as np
import scipy.sparse as sp

logger = logging.getLogger(__name__)


def _is_binary_matrix(X) -> bool:
    """
    Detect if a sparse matrix is binary by checking its dtype.

    Parameters
    ----------
    X : sparse matrix
        Matrix to check (scipy or cupyx sparse)

    Returns
    -------
    bool
        True if matrix is binary (bool dtype), False otherwise
    """
    if isinstance(X, (cusp.csr_matrix, cusp.csc_matrix)):
        return X.data.dtype == cp.bool_ or X.data.dtype == bool
    elif sp.issparse(X):
        return X.data.dtype == np.bool_ or X.data.dtype == bool
    
    return False


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
    sorted_indices = cp.argsort(feature_count)
    n = len(feature_count)

    if is_binary:
        # For binary matrices: simply select top N most accessible features
        n_to_select = min(total_features, n)
        selected = sorted_indices[-n_to_select:]
    else:
        # For count matrices: first exclude zero-count features (matching SnapATAC2)
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

    For binary matrices: selects top N most accessible features (ArchR approach).
    For count matrices: selects top N accessible features, excluding quantile tails.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with cells × features
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
    """
    logger.info(f"Selecting features from {adata.shape[1]:,} total")

    # Detect if matrix is binary
    is_binary = _is_binary_matrix(adata.X)
    if is_binary:
        logger.info("Detected binary matrix - using top-N selection (ArchR approach)")
    else:
        logger.info("Detected count matrix - using quantile-filtered selection")

    # Compute feature counts
    logger.debug("Computing feature accessibility")
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
    import pandas as pd
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
    max_count = 0

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
        
        # Track max count value for dtype optimization
        if sp.issparse(adata.X):
            file_max = adata.X.data.max() if adata.X.nnz > 0 else 0
        else:
            file_max = adata.X.max()
        max_count = max(max_count, int(file_max))
        
        # Count cells and estimate nnz for preallocation
        total_cells += adata.n_obs
        if sp.issparse(adata.X):
            total_nnz += adata.X.nnz
        else:
            total_nnz += np.count_nonzero(adata.X)
        
        del adata
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()

    logger.debug(f"Total cells: {total_cells:,}, total features: {n_vars:,}")

    # Detect if matrices are binary (check first file's dtype)
    first_adata_check = sc.read_h5ad(str(input_paths[0]))
    is_binary = _is_binary_matrix(first_adata_check.X)
    del first_adata_check
    cp.get_default_memory_pool().free_all_blocks()
    
    if is_binary:
        logger.info("Detected binary matrices - using top-N selection (ArchR approach)")
    else:
        logger.info("Detected count matrices - using quantile-filtered selection")

    # Select features based on aggregated counts
    selected_indices = _find_most_accessible_features_gpu(
        feature_counts,
        filter_lower_quantile,
        filter_upper_quantile,
        n_features,
        is_binary=is_binary
    )
    selected_indices_sorted = cp.sort(selected_indices)
    selected_mask = cp.zeros(n_vars, dtype=bool)
    selected_mask[selected_indices] = True
    selected_mask_cpu = selected_mask.get()
    selected_indices_cpu = selected_indices_sorted.get()
    n_selected = len(selected_indices)
    logger.info(f"Selected {n_selected:,} features from aggregated counts")

    # Determine optimal dtype based on max count value
    if binarize:
        optimal_dtype = bool
        logger.info("Using bool dtype (binarized)")
    elif max_count <= 65535:
        optimal_dtype = np.uint16
        logger.info(f"Using uint16 dtype (max_count={max_count})")
    else:
        optimal_dtype = np.int32
        logger.info(f"Using int32 dtype (max_count={max_count})")

    # =========================================================================
    # PASS 2: Build combined sparse matrix with selected features
    # =========================================================================
    logger.info("Pass 2: Building combined matrix...")

    data_list = []
    indices_list = []
    indptr_list = []
    obs_list = []
    
    current_nnz = 0
    total_cells_processed = 0

    for fpath in tqdm(input_paths, desc="Building matrix"):
        # Read metadata first to avoid loading data if not needed
        adata_meta = sc.read_h5ad(str(fpath), backed='r')
        n_obs = adata_meta.n_obs
        
        # Pull only selected features into memory
        X = adata_meta[:, selected_indices_cpu].to_memory().X
        
        if not sp.issparse(X):
            X = sp.csr_matrix(X)
        elif not isinstance(X, sp.csr_matrix):
            X = X.tocsr()

        nnz = X.nnz
        
        # Collect matrix components with optimal dtype
        data_list.append(X.data.astype(optimal_dtype))
            
        indices_list.append(X.indices.astype(np.uint32 if n_selected > 65535 else np.uint16))
        
        # Adjust indptr for concatenation
        if total_cells_processed == 0:
            indptr_list.append(X.indptr.astype(np.uint64))
        else:
            # Drop the first 0 to append to existing indptr
            # Cast to uint64 BEFORE addition to prevent int32 overflow
            chunk_indptr = X.indptr[1:].astype(np.uint64)
            indptr_list.append(chunk_indptr + current_nnz)

        # Collect obs metadata
        obs_df = adata_meta.obs.copy()
        obs_df['source_file'] = fpath.name
        obs_list.append(obs_df)

        current_nnz += nnz
        total_cells_processed += n_obs
        
        # Explicit cleanup per file
        del adata_meta, X, obs_df
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()

    # =========================================================================
    # Final Assembly
    # =========================================================================
    logger.info("Final assembly of matrix and metadata...")
    
    # Concatenate sparse components
    all_data = np.concatenate(data_list)
    del data_list
    all_indices = np.concatenate(indices_list)
    del indices_list
    all_indptr = np.concatenate(indptr_list)
    del indptr_list
    gc.collect()

    # Build final sparse matrix
    logger.info("Building final sparse matrix...")
    combined_X = sp.csr_matrix(
        (all_data, all_indices, all_indptr),
        shape=(total_cells_processed, n_selected)
    )
    del all_data, all_indices, all_indptr
    gc.collect()

    # Build combined obs
    combined_obs = pd.concat(obs_list)
    del obs_list
    gc.collect()
    
    # Make barcodes unique
    combined_obs.index.name = 'barcode'
    if 'barcode' in combined_obs.columns:
        combined_obs.drop(columns=['barcode'], inplace=True)
    combined_obs.reset_index(inplace=True)

    if not combined_obs['barcode'].is_unique:
        n_dups = combined_obs['barcode'].duplicated().sum()
        logger.warning(f"Detected {n_dups:,} duplicate barcodes. Making barcodes unique.")

        # Using anndata helper for efficiency if available, or manual uniqueness
        def make_unique(indices):
            seen = {}
            out = []
            for x in indices:
                if x in seen:
                    seen[x] += 1
                    out.append(f"{x}-{seen[x]}")
                else:
                    seen[x] = 0
                    out.append(x)
            return out
        
        combined_obs.index = make_unique(combined_obs['barcode'])
        combined_obs.drop(columns=['barcode'], inplace=True)
    else:
        # Use barcodes as index and drop the column
        combined_obs.index = combined_obs['barcode'].values
        combined_obs.drop(columns=['barcode'], inplace=True)

    # Build var for selected features
    combined_var = var_df.iloc[selected_indices_cpu].copy()
    combined_var['selected'] = True
    combined_var['accessibility_count'] = feature_counts.get()[selected_indices_cpu]
    logger.info("Building Anndata...")
    # Create combined AnnData
    combined_adata = ad.AnnData(
        X=combined_X,
        obs=combined_obs,
        var=combined_var,
    )

    # Save
    combined_adata.write_h5ad(str(output_path))
    logger.info(f"Saved combined matrix ({total_cells_processed :,} cells × {n_selected:,} features) to {output_path.name}")
