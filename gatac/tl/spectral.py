"""
GPU-accelerated spectral embedding for ATAC-seq tile matrices.

Implements Laplacian Eigenmaps via matrix-free cosine similarity on the GPU,
ported from SnapATAC2's spectral embedding and accelerated with CuPy.

The algorithm:
    1. IDF-weight the cell × feature matrix and L2-normalize each row so that
       dot products equal cosine similarities.
    2. Define the normalized graph Laplacian implicitly through a matrix-free
       linear operator: A v = X (Xᵀ v) − D v, where D is the degree vector.
       This avoids ever materializing the N × N similarity matrix.
    3. Compute the top-k eigenpairs of A using Lanczos (eigsh) on the GPU.
    4. Optionally weight eigenvectors by √(eigenvalue) to reduce sensitivity
       to the number of components chosen.
"""

from __future__ import annotations

import logging
from typing import Literal

import cupy as cp
import cupyx.scipy.sparse as cusp
import cupyx.scipy.sparse.linalg as cusla
import numpy as np
import scipy.sparse as sp
from anndata import AnnData

logger = logging.getLogger(__name__)

__all__ = ["spectral"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_gpu_csr(X) -> cusp.csr_matrix:
    """Convert a scipy or cupy sparse matrix to a CuPy CSR float64 matrix."""
    if isinstance(X, cusp.csr_matrix) and X.dtype == cp.float64:
        return X
    if isinstance(X, (cusp.spmatrix,)):
        return cusp.csr_matrix(X, dtype=cp.float64)
    if sp.issparse(X):
        X_csr = sp.csr_matrix(X, dtype=np.float64)
        return cusp.csr_matrix(X_csr)
    raise TypeError(f"Unsupported matrix type: {type(X)}")


def _idf(X_csr: cusp.csr_matrix) -> cp.ndarray:
    """
    Compute inverse-document-frequency weights on the GPU.

    IDF_j = ln(N / df_j), with boundary handling.
    """
    n = X_csr.shape[0]
    n_features = X_csr.shape[1]

    # Document frequency: number of cells where feature j is nonzero
    # Use binarized copy so repeated indices in data don't inflate counts
    binary = X_csr.copy()
    binary.data = cp.ones_like(binary.data)
    df = cp.asarray(binary.sum(axis=0)).ravel()  # shape (n_features,)

    # Check if all features have the same doc frequency → uniform weights
    if cp.all(df == df[0]):
        return cp.ones(n_features, dtype=cp.float64)

    # Boundary cases
    df = cp.where(df == 0, cp.float64(1.0), df)
    df = cp.where(df == n, cp.float64(n - 1), df)

    return cp.log(cp.float64(n) / df)


def _normalize_rows(X_csr: cusp.csr_matrix, weights: cp.ndarray) -> cusp.csr_matrix:
    """
    Apply IDF feature weights and L2-normalize each row, in-place.

    After this, dot(row_i, row_j) == cosine_similarity(row_i, row_j).
    """
    X = X_csr.copy().astype(cp.float64)

    # Multiply each element by its feature weight
    # CuPy CSR: data, indices, indptr
    X.data *= weights[X.indices]

    # L2-normalize each row
    # Compute row norms via squaring data, then segment-sum per row
    sq = X.copy()
    sq.data = X.data ** 2
    row_norms = cp.asarray(sq.sum(axis=1)).ravel()
    row_norms = cp.sqrt(row_norms)
    row_norms = cp.where(row_norms == 0, cp.float64(1.0), row_norms)

    # Divide each row's data by its norm
    # Build per-element row assignment from indptr
    row_idx = cp.zeros(X.nnz, dtype=cp.int32)
    indptr = X.indptr
    # Use a kernel to map each nonzero to its row index
    _assign_row_indices(row_idx, indptr, X.shape[0])
    X.data /= row_norms[row_idx]

    return X


def _assign_row_indices(row_idx: cp.ndarray, indptr: cp.ndarray, n_rows: int):
    """Populate row_idx[k] = i for indptr[i] <= k < indptr[i+1]."""
    _kernel = cp.ElementwiseKernel(
        "raw int32 indptr, int32 n_rows",
        "int32 row_idx",
        """
        // binary search for the row index
        int lo = 0, hi = n_rows;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (indptr[mid + 1] <= i) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        row_idx = lo;
        """,
        "assign_row_indices",
    )
    _kernel(indptr, np.int32(n_rows), row_idx)


def _spectral_mf(
    X: cusp.csr_matrix,
    n_comps: int,
    random_state: int,
) -> tuple[cp.ndarray, cp.ndarray]:
    """
    Matrix-free spectral embedding on GPU.

    Parameters
    ----------
    X : cusp.csr_matrix
        IDF-weighted, L2-normalized CSR matrix (n_cells × n_features) on GPU.
    n_comps : int
        Number of eigenpairs to compute.
    random_state : int
        Seed for the random starting vector.

    Returns
    -------
    evals : cp.ndarray, shape (n_comps,)
        Eigenvalues in descending order.
    evecs : cp.ndarray, shape (n_cells, n_comps)
        Corresponding eigenvectors.
    """
    n_cells, n_features = X.shape

    # Column sums of the normalized matrix
    col_sum = cp.asarray(X.sum(axis=0)).ravel()  # (n_features,)

    # Degree vector: d_i = (X @ col_sum)_i - 1
    # This equals sum_j!=i cos(x_i, x_j)
    degree = cp.asarray(X.dot(col_sum.reshape(-1, 1))).ravel() - 1.0

    # Cells with no features in the selected set produce degree = -1,
    # which makes the Laplacian ill-defined.  Fail fast with a clear message.
    n_zero = int(cp.sum(degree <= 0))
    if n_zero > 0:
        raise ValueError(
            f"{n_zero} cell(s) have no features in the selected set (degree ≤ 0). "
            "Please filter out empty cells before running spectral embedding, e.g.:\n"
            "    sc.pp.filter_cells(adata, min_features=1)\n"
            "or, for a custom feature mask:\n"
            "    mask = adata.X[:, adata.var['selected']].sum(axis=1).A1 > 0\n"
            "    adata = adata[mask].copy()"
        )

    # D^{-1}: inverse degree
    degree_inv = 1.0 / degree

    # Scale each row by sqrt(degree_inv)
    sqrt_dinv = cp.sqrt(degree_inv)
    # Build per-element row assignment
    row_idx = cp.zeros(X.nnz, dtype=cp.int32)
    _assign_row_indices(row_idx, X.indptr, n_cells)
    X.data *= sqrt_dinv[row_idx]

    # Linear operator: A v = X (X^T v) - D_inv * v
    # where D_inv = 1/degree (the degree_inv vector).
    # After row-scaling by sqrt(D_inv), this implements
    # D^{-1/2} (S - I) D^{-1/2} where S = X X^T (cosine sim).

    def matvec(v):
        # v is a cupy 1-d array
        # X^T v
        XTv = X.T.dot(v)
        # X (X^T v)
        XXTv = X.dot(XTv)
        return XXTv - degree_inv * v

    A = cusla.LinearOperator(
        shape=(n_cells, n_cells),
        matvec=matvec,
        dtype=cp.float64,
    )

    # Generate a deterministic random starting vector
    rng = cp.random.RandomState(seed=random_state)
    v0 = rng.rand(n_cells).astype(cp.float64)

    # Compute top-k eigenpairs
    evals, evecs = cusla.eigsh(A, k=n_comps, which="LM", v0=v0)

    # Sort descending
    ix = cp.argsort(evals)[::-1]
    evals = evals[ix]
    evecs = evecs[:, ix]

    return evals, evecs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def spectral(
    adata: AnnData,
    n_comps: int = 30,
    features: str | np.ndarray | None = "selected",
    random_state: int = 0,
    weighted_by_sd: bool = True,
    feature_weights: np.ndarray | None = None,
    inplace: bool = True,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    GPU-accelerated spectral embedding via Laplacian Eigenmaps.

    Converts the cell × feature count matrix into a lower-dimensional
    representation using the spectrum of the normalized graph Laplacian
    defined by pairwise cosine similarity between cells. The entire
    computation runs on the GPU via CuPy, using a matrix-free approach
    that never materializes the N × N similarity matrix.

    This is a GPU-accelerated port of SnapATAC2's ``tl.spectral``.

    Parameters
    ----------
    adata
        AnnData object. ``adata.X`` should be a sparse cell × tile (or peak)
        matrix — binarized or count-valued.
    n_comps
        Number of spectral dimensions to compute. When ``weighted_by_sd=True``
        (default) the result is insensitive to this value as long as it is
        large enough (e.g. 30).
    features
        Which features (columns) to use:
        - ``"selected"`` (default): use ``adata.var["selected"]`` boolean mask
          (requires a prior call to ``pp.select_features``).
        - A numpy boolean array of length ``n_vars``.
        - ``None``: use all features.
    random_state
        Seed for reproducibility of the Lanczos starting vector.
    weighted_by_sd
        If True (default), weight each eigenvector by the square root of its
        eigenvalue and discard components with non-positive eigenvalues. This
        typically eliminates the need to manually choose the number of
        components.
    feature_weights
        Optional per-feature IDF weights. If ``None``, IDF weights are
        computed automatically from the data.
    inplace
        If True, store the embedding in ``adata.obsm["X_spectral"]`` and
        eigenvalues in ``adata.uns["spectral_eigenvalue"]``.
        If False, return ``(eigenvalues, eigenvectors)`` as numpy arrays.

    Returns
    -------
    tuple[np.ndarray, np.ndarray] | None
        If ``inplace=True``: stores results in *adata* and returns ``None``.
        If ``inplace=False``: returns ``(eigenvalues, eigenvectors)``.

    Notes
    -----
    The algorithm:

    1. Apply IDF weights and L2-normalize each row of the selected feature
       matrix so that row dot-products equal cosine similarities.
    2. Define a matrix-free linear operator
       ``A v = X (X^T v) - D v``  where ``D`` is the degree vector
       (sum of cosine similarities per cell). This implicitly represents
       ``D^{-1/2} (S - I) D^{-1/2}`` with ``S = X X^T``.
    3. Compute the top-k eigenpairs via CuPy's Lanczos (``eigsh``).
    4. Optionally weight eigenvectors by ``sqrt(eigenvalue)``.

    Examples
    --------
    >>> import gatac
    >>> # After tile matrix creation and feature selection:
    >>> gatac.pp.select_features(adata)
    >>> gatac.tl.spectral(adata)
    >>> adata.obsm["X_spectral"].shape
    (n_cells, n_effective_comps)
    """
    logger.info("Running GPU-accelerated spectral embedding ...")

    # ---- Resolve feature mask ----
    if isinstance(features, str):
        if features in adata.var.columns:
            feat_mask = adata.var[features].to_numpy().astype(bool)
        else:
            raise KeyError(
                f"Column '{features}' not found in adata.var. "
                "Call `pp.select_features` first or set `features=None`."
            )
    elif features is not None:
        feat_mask = np.asarray(features, dtype=bool)
    else:
        feat_mask = None

    # ---- Extract & transfer to GPU ----
    X = adata.X
    if feat_mask is not None:
        if sp.issparse(X):
            X = X[:, feat_mask]
        else:
            X = X[:, feat_mask]

    X_gpu = _to_gpu_csr(X)

    # Clamp n_comps
    max_comps = min(X_gpu.shape[0] - 1, X_gpu.shape[1] - 1)
    if n_comps > max_comps:
        logger.warning(
            f"Requested n_comps={n_comps} exceeds matrix rank; "
            f"clamping to {max_comps}."
        )
        n_comps = max_comps

    # ---- IDF weights ----
    if feature_weights is not None:
        weights = cp.asarray(feature_weights, dtype=cp.float64)
        if feat_mask is not None:
            weights = weights[cp.asarray(feat_mask)]
    else:
        weights = _idf(X_gpu)
    logger.info("IDF weights computed.")

    # ---- Normalize rows ----
    X_norm = _normalize_rows(X_gpu, weights)
    del X_gpu
    cp.get_default_memory_pool().free_all_blocks()
    logger.info("Row normalization done.")

    # ---- Matrix-free spectral embedding ----
    evals, evecs = _spectral_mf(X_norm, n_comps, random_state)
    del X_norm
    cp.get_default_memory_pool().free_all_blocks()
    logger.info("Eigen-decomposition done.")

    # ---- SD weighting ----
    if weighted_by_sd:
        pos_mask = evals > 0
        evals = evals[pos_mask]
        evecs = evecs[:, pos_mask] * cp.sqrt(evals)

    # ---- Transfer back to CPU ----
    evals_np = cp.asnumpy(evals)
    evecs_np = cp.asnumpy(evecs)
    cp.get_default_memory_pool().free_all_blocks()

    logger.info(
        f"Spectral embedding complete: {evecs_np.shape[1]} components "
        f"(top eigenvalue = {evals_np[0]:.4f})."
    )

    if inplace:
        adata.uns["spectral_eigenvalue"] = evals_np
        adata.obsm["X_spectral"] = evecs_np
        return None
    else:
        return evals_np, evecs_np
