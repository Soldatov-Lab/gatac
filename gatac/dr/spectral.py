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

Memory management
-----------------
The matrix is kept in its **original integer dtype** (int8, uint16, …) until
the moment of normalisation.  Only the ``data`` array is converted to
**float32** in-place at that point — the ``indices`` and ``indptr`` arrays
(which are typically the bulk of GPU memory) stay int32.  All arithmetic
(IDF weighting, row normalisation, Lanczos eigsh) runs in float32.

The default (full-GPU) path loads the matrix to the GPU as-is and converts
only the ``data`` array.  For larger datasets, the ``chunk_size`` parameter
activates a *chunked* code-path: the count matrix is used as it is, in row
chunks, and the IDF weighting, L2 normalisation and degree scaling are
applied to the vectors of each product instead of the matrix. Chunks stay
on the GPU while they fit in half of its free memory; the others are
streamed from page-locked host memory at every product.
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

from .._cupy_compat import bincount
from ._gpu import (
    _launch_1d,
    _row_norms_sq_kernel,
    _scale_rows_kernel,
    _to_gpu_csr,
)
from ._stream import StreamedCounts
from .lsi import _pinned_signs, _spmv

logger = logging.getLogger(__name__)

__all__ = ["spectral"]


# ---------------------------------------------------------------------------
# CUDA kernels – avoid materialising O(nnz) index arrays
#
# _scale_rows_kernel, _row_norms_sq_kernel, _launch_1d, _to_gpu_csr and
# _PinnedChunk now live in gatac.dr._gpu, shared with gatac.dr.lsi.
# ---------------------------------------------------------------------------

_idf_l2_normalize_kernel = cp.RawKernel(
    r"""
    extern "C" __global__
    void idf_l2_normalize(float* data, const int* indices, const int* indptr,
                          const float* idf, int n_rows) {
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        if (row < n_rows) {
            int start = indptr[row];
            int end   = indptr[row + 1];

            // Pass 1: apply IDF weights and accumulate squared L2 norm
            float norm_sq = 0.0f;
            for (int j = start; j < end; j++) {
                float val = data[j] * idf[indices[j]];
                data[j]   = val;
                norm_sq  += val * val;
            }

            // Pass 2: L2 normalize (rsqrtf = fast reciprocal sqrt)
            float inv_norm = (norm_sq > 0.0f) ? rsqrtf(norm_sq) : 0.0f;
            for (int j = start; j < end; j++) {
                data[j] *= inv_norm;
            }
        }
    }
    """,
    "idf_l2_normalize",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _idf_gpu(X_csr: cusp.csr_matrix) -> cp.ndarray:
    """
    Compute IDF weights on GPU without copying the matrix.

    Uses ``bincount`` on column indices instead of materialising a binary
    copy (saves one full matrix worth of GPU memory).
    """
    n, n_features = X_csr.shape

    # In a valid CSR each (row, col) pair appears at most once, so
    # bincount(indices) == document-frequency per column.
    df = bincount(X_csr.indices, minlength=n_features).astype(cp.float32)

    if cp.all(df == df[0]):
        return cp.ones(n_features, dtype=cp.float32)

    df = cp.where(df == 0, cp.float32(1.0), df)
    df = cp.where(df == n, cp.float32(n - 1), df)
    return cp.log(cp.float32(n) / df)


def _normalize_rows_inplace(X: cusp.csr_matrix, weights: cp.ndarray) -> None:
    """
    Apply IDF feature weights and L2-normalize each row **in-place**.

    After this, dot(row_i, row_j) == cosine_similarity(row_i, row_j).

    Converts ``X.data`` to **float32** in-place (only the non-zero values,
    not the index arrays), then uses CUDA kernels for all arithmetic.
    Peak extra memory is O(n_cells) — no full-matrix copies.
    """
    n_rows = X.shape[0]

    # Convert only the data array to float32; keep indices/indptr as-is.
    X.data = X.data.astype(cp.float32, copy=False)
    weights = weights.astype(cp.float32, copy=False)

    # Feature weighting (in-place on the data array)
    X.data *= weights[X.indices]

    # Row norms via CUDA kernel — no temporary squared-data matrix
    norms_sq = cp.empty(n_rows, dtype=cp.float32)
    indptr32 = X.indptr.astype(cp.int32, copy=False)
    _launch_1d(
        _row_norms_sq_kernel, n_rows,
        X.data, indptr32, norms_sq, np.int32(n_rows),
    )

    norms = cp.sqrt(norms_sq)
    del norms_sq
    norms = cp.where(norms == 0, cp.float32(1.0), norms)

    # Row scaling via CUDA kernel — no O(nnz) row_idx array
    inv_norms = (cp.float32(1.0) / norms).astype(cp.float32)
    del norms
    _launch_1d(
        _scale_rows_kernel, n_rows,
        X.data, indptr32, inv_norms, np.int32(n_rows),
    )


# ---------------------------------------------------------------------------
# Full-GPU spectral (memory-optimised)
# ---------------------------------------------------------------------------


def _spectral_mf(
    X: cusp.csr_matrix,
    n_comps: int,
    random_state: int,
    deterministic: bool = True,
) -> tuple[cp.ndarray, cp.ndarray]:
    """
    Matrix-free spectral embedding on GPU (full matrix in VRAM).

    Parameters
    ----------
    X : cusp.csr_matrix
        IDF-weighted, L2-normalized CSR matrix (n_cells × n_features) on GPU.
    n_comps : int
        Number of eigenpairs to compute.
    random_state : int
        Seed for the random starting vector.
    deterministic : bool
        Apply ``X`` and ``Xᵀ`` with the fixed-order product of
        :func:`gatac.dr.lsi._spmv` instead of cuSPARSE, whose SpMV is not
        run-to-run reproducible. Costs a materialised copy of ``Xᵀ``.

    Returns
    -------
    evals : cp.ndarray, shape (n_comps,)
        Eigenvalues in descending order.
    evecs : cp.ndarray, shape (n_cells, n_comps)
        Corresponding eigenvectors.
    """
    n_cells = X.shape[0]
    if deterministic:
        XT = X.T.tocsr()

        def dot(A, v):
            return _spmv(A, cp.ascontiguousarray(v, dtype=cp.float32),
                         cp.empty(A.shape[0], dtype=cp.float32))
    else:
        XT = X.T

        def dot(A, v):
            return A.dot(v)

    # Column sums of the normalized matrix
    col_sum = dot(XT, cp.ones(n_cells, dtype=cp.float32))

    # Degree vector: d_i = (X @ col_sum)_i - 1   (== Σ_{j≠i} cos(x_i, x_j))
    degree = dot(X, col_sum) - cp.float32(1.0)

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

    degree_inv = (cp.float32(1.0) / degree).astype(cp.float32)

    # Scale each row by sqrt(D^{-1}) via kernel — no O(nnz) row_idx array
    sqrt_dinv = cp.sqrt(degree_inv).astype(cp.float32)
    indptr32 = X.indptr.astype(cp.int32, copy=False)
    _launch_1d(
        _scale_rows_kernel, n_cells,
        X.data, indptr32, sqrt_dinv, np.int32(n_cells),
    )
    if deterministic:
        # Xᵀ was materialised before the scaling; scale its columns to match
        XT.data *= sqrt_dinv[XT.indices]
    del sqrt_dinv

    # Linear operator: A v = X (Xᵀ v) − D⁻¹ v
    def matvec(v):
        shp = v.shape
        y = dot(X, dot(XT, v.ravel())) - degree_inv * v.ravel()
        return y.reshape(shp)

    A = cusla.LinearOperator(
        shape=(n_cells, n_cells),
        matvec=matvec,
        dtype=cp.float32,
    )

    rng = cp.random.RandomState(seed=random_state)
    v0 = rng.rand(n_cells).astype(cp.float32)

    evals, evecs = cusla.eigsh(A, k=n_comps, which="LM", v0=v0)

    ix = cp.argsort(evals)[::-1]
    return evals[ix], evecs[:, ix]


# ---------------------------------------------------------------------------
# Chunked spectral — the count matrix streamed as it is, normalized on vectors
# ---------------------------------------------------------------------------
#
# With C the raw count matrix, w the IDF weights and r_i the norm of row i of
# C diag(w), the normalized rows are x_i = c_i w / r_i, and with the degrees
# d_i = x_i . sum_k x_k - 1 the scaled matrix is Y = diag(a) C diag(w),
# a_i = 1 / (r_i sqrt(d_i)). Every normalization is a row or column scaling,
# so it is applied to the vectors and C is used unchanged:
#
#     Y u = a * C (w * u)        Yt v = w * Ct (a * v)
#     A v = a * C (w^2 * Ct (a * v)) - v / d
#
# so the matrix is never rewritten, on the host or the GPU.


def _spectral_chunked(
    X_cpu: sp.csr_matrix,
    n_comps: int,
    random_state: int,
    feature_weights: np.ndarray | None,
    chunk_size: int,
) -> tuple[cp.ndarray, cp.ndarray]:
    """
    Chunked spectral embedding for large datasets.

    The count matrix stays as it is in host memory; its row chunks are kept
    on the GPU while they fit in half of its free memory, and the others
    are streamed at every product (see :class:`gatac.dr._stream.StreamedCounts`).
    """
    n_cells, n_features = X_cpu.shape
    C = StreamedCounts(X_cpu, chunk_size)
    try:
        # ---- IDF weights: from the column entry counts, or caller-supplied ----
        if feature_weights is not None:
            w = cp.asarray(feature_weights, dtype=cp.float32)
        else:
            df = C.col_nnz().astype(cp.float32)
            if bool(cp.all(df == df[0])):
                w = cp.ones(n_features, dtype=cp.float32)
            else:
                df = cp.where(df == 0, cp.float32(1.0), df)
                df = cp.where(df == n_cells, cp.float32(n_cells - 1), df)
                w = cp.log(cp.float32(n_cells) / df).astype(cp.float32)
        logger.info("IDF weights computed.")

        # ---- L2 row norms of C diag(w), column sums and degrees ----
        r = cp.sqrt(C.row_sq_norms(w))
        rinv = (cp.float32(1.0) / cp.where(r == 0, cp.float32(1.0), r)).astype(cp.float32)
        del r
        col_sum = w * C.rdot(rinv)
        degree = rinv * C.dot(w * col_sum) - cp.float32(1.0)
        n_zero = int(cp.sum(degree <= 0))
        if n_zero > 0:
            raise ValueError(
                f"{n_zero} cell(s) have degree ≤ 0. "
                "Filter empty cells before running spectral embedding."
            )
        logger.info("Degree computation done.")

        a = (rinv / cp.sqrt(degree)).astype(cp.float32)
        dinv = (cp.float32(1.0) / degree).astype(cp.float32)
        w2 = (w * w).astype(cp.float32)
        del rinv, col_sum, degree

        # ---- eigsh over A v = a * C (w^2 * Ct (a * v)) - v / d ----
        def matvec(v):
            shp = v.shape
            vf = v.ravel()
            y = a * C.dot(w2 * C.rdot(a * vf)) - dinv * vf
            return y.reshape(shp)

        A = cusla.LinearOperator(shape=(n_cells, n_cells), matvec=matvec, dtype=cp.float32)
        rng = cp.random.RandomState(seed=random_state)
        v0 = rng.rand(n_cells).astype(cp.float32)
        # Larger Krylov subspace -> fewer matvec calls (each a pass over the matrix).
        ncv = min(n_cells - 1, max(4 * n_comps + 1, n_comps + 60))
        logger.info(f"Starting eigsh (chunked matvec, ncv={ncv}) ...")
        evals, evecs = cusla.eigsh(A, k=n_comps, ncv=ncv, which="LM", v0=v0)
    finally:
        C.close()

    ix = cp.argsort(evals)[::-1]
    return evals[ix], evecs[:, ix]


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
    chunk_size: int | None = None,
    deterministic: bool = True,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    GPU-accelerated spectral embedding via Laplacian Eigenmaps.

    Converts the cell × feature count matrix into a lower-dimensional
    representation using the spectrum of the normalized graph Laplacian
    defined by pairwise cosine similarity between cells. The entire
    computation runs on the GPU via CuPy, using a matrix-free approach
    that never materializes the N × N similarity matrix.

    This is a GPU-accelerated port of SnapATAC2's ``dr.spectral``
    :cite:p:`Zhang2024`, itself an implementation of Laplacian Eigenmaps
    :cite:p:`Belkin2003`.

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
        Which features (columns) to use. ``"selected"`` uses the boolean
        mask in ``adata.var["selected"]`` and requires a prior call to
        ``pp.select_features``. You can also pass a NumPy boolean array of
        length ``n_vars`` or ``None`` to use all features.
    random_state
        Seed of the Lanczos starting vector. With ``deterministic=True`` and
        no ``chunk_size``, the same input and seed give a bitwise-identical
        embedding; each component's sign is pinned in every mode.
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
    chunk_size
        When set, use the matrix in row chunks of this many cells. Chunks
        are kept on the GPU while they fit in half of its free memory, and
        the others are streamed from host memory at every Lanczos product,
        so any matrix that fits in host memory can be embedded; the matrix
        itself is not modified. ``None`` (default) loads the full matrix to
        the GPU, or uses chunks of 50,000 cells above 100,000 cells.
        The chunked products are not run-to-run reproducible in the last
        bits, whatever ``deterministic`` says.
    deterministic
        Use a fixed-order sparse product instead of cuSPARSE, whose SpMV is
        not run-to-run reproducible. Costs a second copy of the normalized
        matrix in GPU memory; ``False`` saves it.

    Returns
    -------
    tuple[np.ndarray, np.ndarray] | None
        If ``inplace=True``: stores results in *adata* and returns ``None``.
        If ``inplace=False``: returns ``(eigenvalues, eigenvectors)``.

    Notes
    -----
    In the default full-GPU path, the matrix is uploaded in its original
    integer dtype and only ``X.data`` is converted to float32 in place during
    normalization. The sparse index arrays stay int32, and peak GPU memory is
    roughly one float32 copy of the ``data`` array, plus the transposed copy
    of the matrix the deterministic product needs. A matrix already on the GPU
    is copied first rather than normalized in place.

    When ``chunk_size`` is set, the count matrix is used as it is: its values
    and int32 column indices, about 6 bytes per entry for counts stored as
    uint16. Chunks take at most half of the GPU's free memory; each Lanczos
    product reads the matrix twice, and the chunks that did not fit are
    copied from host memory both times, so it runs at the speed of the host
    to GPU link.

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
    >>> gatac.dr.spectral(adata)
    >>> adata.obsm["X_spectral"].shape
    (n_cells, n_effective_comps)

    For large datasets that do not fit on the GPU:

    >>> gatac.dr.spectral(adata, chunk_size=30_000)
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

    # ---- Extract sparse matrix (stays on CPU until needed) ----
    X = adata.X
    if feat_mask is not None:
        if sp.issparse(X):
            X = X[:, feat_mask]
        else:
            X = X[:, feat_mask]

    # ---- Clamp n_comps ----
    n, m = X.shape
    max_comps = min(n - 1, m - 1)
    if n_comps > max_comps:
        logger.warning(
            f"Requested n_comps={n_comps} exceeds matrix rank; "
            f"clamping to {max_comps}."
        )
        n_comps = max_comps

    # ---- Set chunk_size if number of cells exceeds 100k ----
    if n > 100_000 and chunk_size is None:
        chunk_size = 50_000
        logger.info(
            f"Number of cells ({n:,}) exceeds 100k; "
            f"automatically enabling chunked processing with chunk_size={chunk_size:,}."
        )

    # ---- Dispatch to chunked or full-GPU path ----
    if chunk_size is not None:
        # ---- Chunked path: data stays on CPU, chunks stream to GPU ----
        X_cpu = X

        # Pre-slice feature_weights if provided; IDF otherwise computed on GPU
        # inside _spectral_chunked from the column entry counts.
        fw = None
        if feature_weights is not None:
            fw = np.asarray(feature_weights, dtype=np.float32)
            if feat_mask is not None:
                fw = fw[feat_mask]

        evals, evecs = _spectral_chunked(
            X_cpu, n_comps, random_state, fw, chunk_size,
        )
        del X_cpu
    else:
        # ---- Full-GPU path (memory-optimised) ----
        X_gpu = _to_gpu_csr(X)
        if X_gpu is adata.X:
            # normalisation below rewrites X_gpu.data in place
            X_gpu = X_gpu.copy()

        if feature_weights is not None:
            weights = cp.asarray(feature_weights, dtype=cp.float64)
            if feat_mask is not None:
                weights = weights[cp.asarray(feat_mask)]
        else:
            weights = _idf_gpu(X_gpu)
        logger.info("IDF weights computed.")

        _normalize_rows_inplace(X_gpu, weights)
        del weights
        cp.get_default_memory_pool().free_all_blocks()
        logger.info("Row normalization done.")

        evals, evecs = _spectral_mf(X_gpu, n_comps, random_state, deterministic)
        del X_gpu
        cp.get_default_memory_pool().free_all_blocks()

    logger.info("Eigen-decomposition done.")

    # eigsh leaves each eigenvector's sign arbitrary; pin it to the data
    evecs = evecs * _pinned_signs(evecs)

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
