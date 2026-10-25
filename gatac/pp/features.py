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

    Examples
    --------
    >>> import gatac as ga
    >>> ga.pp.select_features(adata, n_features=500_000)
    >>> # adata.var["selected"] is now a boolean mask of the chosen features
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
    binarize : bool
        Whether to binarize the output matrix (default: True)

    Examples
    --------
    >>> import gatac as ga
    >>> ga.pp.select_features_multi(
    ...     ["sampleA.h5ad", "sampleB.h5ad"],
    ...     output_path="combined.h5ad",
    ...     n_features=500_000,
    ... )
    """

    import anndata as ad
    import h5py
    import pandas as pd
    from anndata.experimental import read_elem
    from tqdm import tqdm

    input_paths = [Path(p) for p in input_paths]
    output_path = Path(output_path)

    if len(input_paths) == 0:
        raise ValueError("No input files provided")

    logger.info(f"Processing {len(input_paths)} h5ad files")

    # Only the sparse arrays of X are read, never whole AnnData objects, and
    # the features are selected on the GPU: a remap of the column ids, whose
    # running count of kept entries at each row start is the new indptr.
    with h5py.File(input_paths[0], "r") as f:
        var_df = read_elem(f["var"])
    n_vars = len(var_df)

    # =========================================================================
    # PASS 1: Aggregate feature counts across all files
    # =========================================================================
    logger.info("Pass 1: Aggregating feature counts...")
    feature_counts = cp.zeros(n_vars, dtype=cp.float64)
    feature_nnz = cp.zeros(n_vars, dtype=cp.int64)
    total_cells = 0
    max_count = 0
    is_binary = True

    for fpath in tqdm(input_paths, desc="Counting features"):
        indptr, indices, data, shape = _read_csr(fpath)
        if shape[1] != n_vars:
            raise ValueError(
                f"Feature mismatch: {fpath.name} has {shape[1]} features, "
                f"expected {n_vars}"
            )
        indices_gpu = cp.asarray(indices)
        nnz = bincount(indices_gpu, minlength=n_vars)
        feature_nnz += nnz
        if data.dtype == bool:
            feature_counts += nnz
            file_max = int(data.any())
        else:
            data_gpu = cp.asarray(data)
            feature_counts += cp.bincount(indices_gpu, weights=data_gpu.astype(cp.float64), minlength=n_vars)
            file_max = int(data_gpu.max()) if data_gpu.size else 0
            is_binary &= bool(cp.all((data_gpu == 0) | (data_gpu == 1)))
            del data_gpu
        max_count = max(max_count, file_max)
        total_cells += shape[0]
        del indices_gpu, nnz, indptr, indices, data

    logger.debug(f"Total cells: {total_cells:,}, total features: {n_vars:,}")

    if is_binary:
        logger.info("Detected binary matrices - using top-N selection (ArchR approach)")
    else:
        logger.info("Detected count matrices - using quantile-filtered selection")

    # Select features based on aggregated counts
    selected_indices = _find_most_accessible_features_gpu(
        feature_counts.astype(cp.float32),
        filter_lower_quantile,
        filter_upper_quantile,
        n_features,
        is_binary=is_binary
    )
    selected_indices_sorted = cp.sort(selected_indices)
    selected_indices_cpu = selected_indices_sorted.get()
    n_selected = len(selected_indices)
    logger.info(f"Selected {n_selected:,} features from aggregated counts")
    # Old column id -> new one, -1 if dropped. Increasing, so every row's
    # column ids stay sorted.
    remap = cp.full(n_vars, -1, dtype=cp.int32)
    remap[selected_indices_sorted] = cp.arange(n_selected, dtype=cp.int32)
    total_nnz = int(feature_nnz[selected_indices_sorted].sum())

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
    # Metadata, and the output file without X
    # =========================================================================
    obs_list = []
    for fpath in input_paths:
        with h5py.File(fpath, "r") as f:
            obs_df = read_elem(f["obs"])
        obs_df['source_file'] = fpath.name
        obs_list.append(obs_df)
    combined_obs = pd.concat(obs_list)
    del obs_list

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
    combined_var['accessibility_count'] = feature_counts.get()[selected_indices_cpu].astype(np.float32)
    ad.AnnData(obs=combined_obs, var=combined_var).write_h5ad(str(output_path))

    # =========================================================================
    # PASS 2: Write the selected columns of each file into the output's X
    # =========================================================================
    logger.info("Pass 2: Writing combined matrix...")
    with h5py.File(output_path, "a") as out:
        X = out.create_group("X")
        X.attrs.update({"encoding-type": "csr_matrix", "encoding-version": "0.1.0",
                        "shape": np.array([total_cells, n_selected])})
        out_data = X.create_dataset("data", shape=(total_nnz,), dtype=optimal_dtype)
        # one index dtype for both arrays, as scipy picks it: anndata reads them
        # as stored, and mixed dtypes make every sparse product copy the indices
        idx_dtype = np.int32 if max(total_nnz, total_cells, n_selected) <= np.iinfo(np.int32).max else np.int64
        out_indices = X.create_dataset("indices", shape=(total_nnz,), dtype=idx_dtype)
        out_indptr = X.create_dataset("indptr", shape=(total_cells + 1,), dtype=idx_dtype)
        out_indptr[0] = 0
        row, pos = 0, 0
        for fpath in tqdm(input_paths, desc="Building matrix"):
            indptr, indices, data, shape = _read_csr(fpath)
            new_col = remap[cp.asarray(indices)]
            keep = new_col >= 0
            kept_before = cp.zeros(len(keep) + 1, dtype=cp.int64)
            cp.cumsum(keep, out=kept_before[1:])
            k = int(kept_before[-1])
            out_indptr[row + 1:row + shape[0] + 1] = (kept_before[cp.asarray(indptr[1:])] + pos).get()
            out_indices[pos:pos + k] = new_col[keep].get()
            data_gpu = cp.asarray(data)[keep]
            out_data[pos:pos + k] = (data_gpu != 0 if optimal_dtype is bool else data_gpu.astype(optimal_dtype)).get()
            row += shape[0]
            pos += k
            del new_col, keep, kept_before, data_gpu, indptr, indices, data
        if pos != total_nnz:
            raise RuntimeError(f"Wrote {pos:,} entries, expected {total_nnz:,}")
    cp.get_default_memory_pool().free_all_blocks()

    logger.info(f"Saved combined matrix ({total_cells:,} cells × {n_selected:,} features) to {output_path.name}")


def _read_csr(path: Path):
    """``indptr``, ``indices``, ``data`` and ``shape`` of an h5ad's X, as CSR."""
    import h5py
    from anndata.experimental import read_elem

    with h5py.File(path, "r") as f:
        X = f["X"]
        if isinstance(X, h5py.Group) and X.attrs.get("encoding-type") == "csr_matrix":
            shape = tuple(int(n) for n in X.attrs["shape"])
            return X["indptr"][:], X["indices"][:], X["data"][:], shape
        X = sp.csr_matrix(read_elem(X))
    return X.indptr, X.indices, X.data, X.shape
