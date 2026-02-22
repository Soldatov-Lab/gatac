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
only the ``data`` array.  For datasets that still exceed GPU memory, the
``chunk_size`` parameter activates a *chunked* code-path: the matrix stays
on the CPU but IDF weighting, L2 normalisation, and degree scaling are
performed on the GPU via a fused CUDA kernel (one row-chunk at a time).
Only one chunk resides on the GPU during each Lanczos matvec.
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
# CUDA kernels – avoid materialising O(nnz) index arrays
# ---------------------------------------------------------------------------

_scale_rows_kernel = cp.RawKernel(
    r"""
    extern "C" __global__
    void scale_rows(float* data, const int* indptr,
                    const float* scale, int n_rows) {
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        if (row < n_rows) {
            float s = scale[row];
            for (int j = indptr[row]; j < indptr[row + 1]; j++) {
                data[j] *= s;
            }
        }
    }
    """,
    "scale_rows",
)

_row_norms_sq_kernel = cp.RawKernel(
    r"""
    extern "C" __global__
    void row_norms_sq(const float* data, const int* indptr,
                      float* norms, int n_rows) {
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        if (row < n_rows) {
            float s = 0.0f;
            for (int j = indptr[row]; j < indptr[row + 1]; j++) {
                s += data[j] * data[j];
            }
            norms[row] = s;
        }
    }
    """,
    "row_norms_sq",
)

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


def _launch_1d(kernel, n: int, *args):
    """Launch a 1-D grid CUDA kernel with *n* threads."""
    block = 256
    grid = (int(n) + block - 1) // block
    kernel((grid,), (block,), args)


def _to_gpu_csr(X) -> cusp.csr_matrix:
    """
    Upload a sparse matrix to the GPU as a CuPy CSR float32 matrix.

    The ``data`` array is cast to float32 if needed; ``indices`` and
    ``indptr`` (which dominate memory for ATAC-seq matrices) are shared
    without copying when the source is already in CSR format.
    No full CPU-side float copy is made.
    """
    if isinstance(X, cusp.csr_matrix):
        if X.dtype == cp.float32:
            return X
        # Convert only the data array; share indices/indptr
        return cusp.csr_matrix(
            (X.data.astype(cp.float32), X.indices, X.indptr), shape=X.shape
        )
    if isinstance(X, cusp.spmatrix):
        return _to_gpu_csr(X.tocsr())
    if sp.issparse(X):
        X_csr = X.tocsr()
        if X_csr.dtype == np.float32:
            return cusp.csr_matrix(X_csr)
        # Build a scipy CSR with float32 data sharing index arrays — avoids a
        # full-matrix float conversion on the CPU before the GPU upload.
        data_f32 = X_csr.data.astype(np.float32)
        return cusp.csr_matrix(
            sp.csr_matrix(
                (data_f32, X_csr.indices, X_csr.indptr), shape=X_csr.shape
            )
        )
    raise TypeError(f"Unsupported matrix type: {type(X)}")


def _idf_gpu(X_csr: cusp.csr_matrix) -> cp.ndarray:
    """
    Compute IDF weights on GPU without copying the matrix.

    Uses ``bincount`` on column indices instead of materialising a binary
    copy (saves one full matrix worth of GPU memory).
    """
    n, n_features = X_csr.shape

    # In a valid CSR each (row, col) pair appears at most once, so
    # bincount(indices) == document-frequency per column.
    df = cp.bincount(
        X_csr.indices.astype(cp.intp), minlength=n_features
    ).astype(cp.float32)

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

    Returns
    -------
    evals : cp.ndarray, shape (n_comps,)
        Eigenvalues in descending order.
    evecs : cp.ndarray, shape (n_cells, n_comps)
        Corresponding eigenvectors.
    """
    n_cells = X.shape[0]

    # Column sums of the normalized matrix
    col_sum = cp.asarray(X.sum(axis=0), dtype=cp.float32).ravel()

    # Degree vector: d_i = (X @ col_sum)_i - 1   (== Σ_{j≠i} cos(x_i, x_j))
    degree = cp.asarray(
        X.dot(col_sum.reshape(-1, 1)), dtype=cp.float32
    ).ravel() - cp.float32(1.0)

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
    del sqrt_dinv

    # Linear operator: A v = X (Xᵀ v) − D⁻¹ v
    def matvec(v):
        return X.dot(X.T.dot(v)) - degree_inv * v

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
# Chunked spectral — GPU-accelerated TF-IDF + row-chunk streaming for eigsh
# ---------------------------------------------------------------------------


class _PinnedChunk:
    """
    CSR chunk with arrays stored in page-locked (pinned) memory.

    Pinned memory allows the GPU copy engine to transfer data via direct
    PCIe DMA (bypassing the OS staging buffer), roughly doubling H2D
    bandwidth compared to pageable host memory.

    Parameters
    ----------
    csr : sp.csr_matrix
        The CSR chunk to pin (data must already be float32).
    row_start, row_end : int
        Row range of this chunk in the full matrix.
    """

    __slots__ = ("data", "indices", "indptr", "shape", "row_start", "row_end", "_pins")

    def __init__(self, csr: sp.csr_matrix, row_start: int, row_end: int) -> None:
        self.shape     = csr.shape
        self.row_start = row_start
        self.row_end   = row_end
        self._pins: list = []

        for name, src in (
            ("data",    csr.data.astype(np.float32, copy=False)),
            ("indices", csr.indices.astype(np.int32,   copy=False)),
            ("indptr",  csr.indptr.astype(np.int32,    copy=False)),
        ):
            # PinnedMemory → cudaMallocHost (no pool cap).
            # PinnedMemoryPointer exposes the buffer protocol for np.frombuffer.
            mem = cp.cuda.PinnedMemory(src.nbytes, 0)
            ptr = cp.cuda.PinnedMemoryPointer(mem, 0)
            arr = np.frombuffer(ptr, dtype=src.dtype).reshape(src.shape)
            arr[:] = src
            setattr(self, name, arr)
            self._pins.append(mem)


def _idf_gpu_chunked(
    X_cpu: sp.csr_matrix,
    chunk_size: int,
) -> cp.ndarray:
    """
    Compute IDF weights on the GPU using chunked bincount over column indices.

    Only the ``indices`` slice of each chunk is transferred to the GPU;
    no data values are needed.  Peak extra GPU memory is O(n_features).
    """
    n, n_features = X_cpu.shape
    df = cp.zeros(n_features, dtype=cp.float32)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        idx_s = int(X_cpu.indptr[start])
        idx_e = int(X_cpu.indptr[end])
        d_indices = cp.asarray(X_cpu.indices[idx_s:idx_e])
        df += cp.bincount(d_indices.astype(cp.intp), minlength=n_features)
        del d_indices
        cp.get_default_memory_pool().free_all_blocks()

    if cp.all(df == df[0]):
        return cp.ones(n_features, dtype=cp.float32)

    df = cp.where(df == 0, cp.float32(1.0), df)
    df = cp.where(df == n, cp.float32(n - 1), df)
    return cp.log(cp.float32(n) / df)


def _spectral_chunked(
    X_cpu: sp.csr_matrix,
    n_comps: int,
    random_state: int,
    feature_weights: np.ndarray | None,
    chunk_size: int,
) -> tuple[cp.ndarray, cp.ndarray]:
    """
    Chunked spectral embedding for large datasets.

    The full matrix stays on the CPU; row-chunks are transferred to the
    GPU for IDF + L2 normalisation (via a fused CUDA kernel), degree
    scaling, and Lanczos matvec.  This trades throughput for reduced peak
    GPU memory (≈ 1 chunk instead of the full matrix).
    """
    n_cells, n_features = X_cpu.shape
    n_chunks = (n_cells + chunk_size - 1) // chunk_size
    logger.info(
        f"Chunked mode: {n_cells:,} cells in {n_chunks} chunks "
        f"of ≤ {chunk_size:,}"
    )

    # ---- Ensure CSR with float32 data (convert only the data array) ----
    X_cpu = X_cpu.tocsr()
    if X_cpu.dtype != np.float32:
        X_cpu.data = X_cpu.data.astype(np.float32, copy=False)

    # ---- IDF weights: GPU-accelerated chunked bincount or caller-supplied ----
    if feature_weights is not None:
        idf_gpu = cp.asarray(feature_weights, dtype=cp.float32)
    else:
        idf_gpu = _idf_gpu_chunked(X_cpu, chunk_size)
        logger.info("IDF weights computed (GPU, chunked bincount).")

    # ---- Phase 1: IDF + L2 normalisation ----
    col_sum = cp.zeros(n_features, dtype=cp.float32)

    for start in range(0, n_cells, chunk_size):
        end = min(start + chunk_size, n_cells)
        data_s = int(X_cpu.indptr[start])
        data_e = int(X_cpu.indptr[end])
        n_rows = end - start

        d_data = cp.asarray(X_cpu.data[data_s:data_e])
        d_indices = cp.asarray(X_cpu.indices[data_s:data_e])
        d_indptr = cp.asarray(
            (X_cpu.indptr[start : end + 1] - data_s).astype(np.int32)
        )

        _launch_1d(
            _idf_l2_normalize_kernel, n_rows,
            d_data, d_indices, d_indptr, idf_gpu, np.int32(n_rows),
        )

        col_sum += cp.bincount(
            d_indices.astype(cp.intp), weights=d_data, minlength=n_features,
        ).astype(cp.float32)

        X_cpu.data[data_s:data_e] = cp.asnumpy(d_data)

        del d_data, d_indices, d_indptr
        cp.get_default_memory_pool().free_all_blocks()

    del idf_gpu
    logger.info("IDF + L2 normalisation done (GPU-accelerated).")

    # ---- Phase 2: Degree computation ----
    col_sum_cpu = cp.asnumpy(col_sum)
    del col_sum
    cp.get_default_memory_pool().free_all_blocks()

    degree = np.asarray(
        X_cpu.dot(col_sum_cpu.reshape(-1, 1)), dtype=np.float32
    ).ravel() - np.float32(1.0)
    del col_sum_cpu

    n_zero = int(np.sum(degree <= 0))
    if n_zero > 0:
        raise ValueError(
            f"{n_zero} cell(s) have degree ≤ 0. "
            "Filter empty cells before running spectral embedding."
        )
    logger.info("Degree computation done.")

    # ---- Phase 3: GPU-accelerated degree scaling ----
    sqrt_dinv = np.sqrt(np.float32(1.0) / degree).astype(np.float32)

    for start in range(0, n_cells, chunk_size):
        end = min(start + chunk_size, n_cells)
        data_s = int(X_cpu.indptr[start])
        data_e = int(X_cpu.indptr[end])
        n_rows = end - start

        d_data = cp.asarray(X_cpu.data[data_s:data_e])
        d_indptr = cp.asarray(
            (X_cpu.indptr[start : end + 1] - data_s).astype(np.int32)
        )
        d_scale = cp.asarray(sqrt_dinv[start:end])

        _launch_1d(
            _scale_rows_kernel, n_rows,
            d_data, d_indptr, d_scale, np.int32(n_rows),
        )

        X_cpu.data[data_s:data_e] = cp.asnumpy(d_data)

        del d_data, d_indptr, d_scale
        cp.get_default_memory_pool().free_all_blocks()

    del sqrt_dinv
    logger.info("Degree scaling done (GPU-accelerated).")

    # ---- Phase 4: eigsh with pinned memory + double-buffered uploads ----
    # Pinned chunks enable DMA transfers; a non-blocking upload stream
    # pre-fetches chunk i+1 while the GPU computes with chunk i.
    pinned_chunks: list[_PinnedChunk] = []
    for start in range(0, n_cells, chunk_size):
        end = min(start + chunk_size, n_cells)
        pinned_chunks.append(_PinnedChunk(X_cpu[start:end], start, end))
    del X_cpu
    logger.info(
        f"Chunk arrays moved to pinned memory ({len(pinned_chunks)} chunks)."
    )

    degree_inv_gpu = cp.asarray(np.float32(1.0) / degree, dtype=cp.float32)
    del degree

    upload_stream = cp.cuda.Stream(non_blocking=True)

    def _upload(pc: "_PinnedChunk") -> cusp.csr_matrix:
        with upload_stream:
            d_data    = cp.asarray(pc.data)
            d_indices = cp.asarray(pc.indices)
            d_indptr  = cp.asarray(pc.indptr)
        return cusp.csr_matrix(
            (d_data, d_indices, d_indptr), shape=pc.shape
        )

    def _chunked_XTv(v: cp.ndarray) -> cp.ndarray:
        w = cp.zeros(n_features, dtype=cp.float32)
        cur_gpu = _upload(pinned_chunks[0])
        cur_ev  = cp.cuda.Event()
        cur_ev.record(upload_stream)

        for i, pc in enumerate(pinned_chunks):
            if i + 1 < len(pinned_chunks):
                nxt_gpu = _upload(pinned_chunks[i + 1])
                nxt_ev  = cp.cuda.Event()
                nxt_ev.record(upload_stream)

            cur_ev.synchronize()
            w += cur_gpu.T.dot(v[pc.row_start : pc.row_end])
            del cur_gpu

            if i + 1 < len(pinned_chunks):
                cur_gpu, cur_ev = nxt_gpu, nxt_ev

        return w

    def _chunked_Xw(w: cp.ndarray) -> cp.ndarray:
        y = cp.empty(n_cells, dtype=cp.float32)
        cur_gpu = _upload(pinned_chunks[0])
        cur_ev  = cp.cuda.Event()
        cur_ev.record(upload_stream)

        for i, pc in enumerate(pinned_chunks):
            if i + 1 < len(pinned_chunks):
                nxt_gpu = _upload(pinned_chunks[i + 1])
                nxt_ev  = cp.cuda.Event()
                nxt_ev.record(upload_stream)

            cur_ev.synchronize()
            y[pc.row_start : pc.row_end] = cur_gpu.dot(w)
            del cur_gpu

            if i + 1 < len(pinned_chunks):
                cur_gpu, cur_ev = nxt_gpu, nxt_ev

        return y

    def matvec(v):
        w = _chunked_XTv(v)
        y = _chunked_Xw(w)
        y -= degree_inv_gpu * v
        return y

    A = cusla.LinearOperator(
        shape=(n_cells, n_cells), matvec=matvec, dtype=cp.float32,
    )
    rng = cp.random.RandomState(seed=random_state)
    v0 = rng.rand(n_cells).astype(cp.float32)

    # Larger Krylov subspace → fewer matvec calls (each expensive due to PCIe).
    ncv = min(n_cells - 1, max(4 * n_comps + 1, n_comps + 60))

    logger.info(
        f"Starting eigsh (chunked double-buffered matvec, ncv={ncv}) ..."
    )
    evals, evecs = cusla.eigsh(A, k=n_comps, ncv=ncv, which="LM", v0=v0)

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
    chunk_size
        When set, process the matrix in row-batches of this many cells
        instead of loading the full matrix into GPU memory at once.  Only
        one chunk resides on the GPU at a time during each Lanczos
        iteration, trading throughput for reduced peak GPU memory.
        Recommended values: 20 000 – 50 000.  ``None`` (default) loads the
        full matrix to the GPU (fastest, but requires enough VRAM).

    Returns
    -------
    tuple[np.ndarray, np.ndarray] | None
        If ``inplace=True``: stores results in *adata* and returns ``None``.
        If ``inplace=False``: returns ``(eigenvalues, eigenvectors)``.

    Notes
    -----
    **Memory behaviour (full-GPU path, default):**

    The matrix is uploaded to the GPU in its original integer dtype
    (int8, uint16, …).  Only the ``data`` array is converted to float32
    in-place during normalisation; the index arrays (typically dominant
    in sparse matrices) stay int32.  All arithmetic runs in float32.
    Peak GPU memory ≈ 1 × the **data array** in float32.

    **Memory behaviour (chunked path, ``chunk_size`` set):**

    Peak GPU memory ≈ 1 chunk.  The full matrix stays on the CPU
    (data converted to float32 in-place, indices untouched) and
    row-chunks are streamed to the GPU during each eigsh matvec call.
    Each matvec requires two passes (Xᵀv then Xw), so this path is
    slower than the full-GPU path by a factor that depends on PCIe
    bandwidth.

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

    For large datasets that cause GPU OOM:

    >>> gatac.tl.spectral(adata, chunk_size=30_000)
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

    # ---- Dispatch to chunked or full-GPU path ----
    if chunk_size is not None:
        # ---- Chunked path: data stays on CPU, chunks stream to GPU ----
        X_cpu = X

        # Pre-slice feature_weights if provided; IDF otherwise computed on GPU
        # inside _spectral_chunked via chunked bincount.
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

        evals, evecs = _spectral_mf(X_gpu, n_comps, random_state)
        del X_gpu
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
