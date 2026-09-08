"""
Shared GPU sparse-matrix helpers for ``gatac.tl``.

These were originally private to :mod:`gatac.tl.spectral`; they are shared with
:mod:`gatac.tl.lsi`, which needs the same primitives:

* upload a sparse matrix to the device converting only the ``data`` array,
* row-wise scaling and row-norm kernels that avoid materialising ``O(nnz)``
  index arrays,
* page-locked row chunks plus a double-buffered upload stream, so a matrix
  larger than VRAM can be streamed through an iterative solver.

The streaming path is exposed as :class:`ChunkedMatrix`, which can present
itself either as ``X`` (for :func:`cupyx.scipy.sparse.linalg.svds`) or as the
Gram operator ``X Xᵀ`` (for :func:`cupyx.scipy.sparse.linalg.eigsh`).
"""

from __future__ import annotations

import logging

import cupy as cp
import cupyx.scipy.sparse as cusp
import cupyx.scipy.sparse.linalg as cusla
import numpy as np
import scipy.sparse as sp

logger = logging.getLogger(__name__)

__all__ = [
    "_launch_1d",
    "_to_gpu_csr",
    "_scale_rows_kernel",
    "_row_norms_sq_kernel",
    "_PinnedChunk",
    "ChunkedMatrix",
    "chunked_operator",
]


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


# ---------------------------------------------------------------------------
# Page-locked row chunks
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


class ChunkedMatrix:
    """
    A host-resident CSR matrix streamed to the GPU in row chunks.

    Holds the matrix as page-locked row chunks and pre-fetches chunk ``i+1``
    on a non-blocking stream while the GPU computes with chunk ``i``, so only
    one chunk (plus the prefetch) occupies VRAM at a time.

    Parameters
    ----------
    X : sp.csr_matrix
        Host CSR matrix, ``n_rows x n_cols``. Converted to float32 data if
        needed.
    chunk_size : int
        Rows per chunk.

    Notes
    -----
    What this buys, and what it does not: the *matrix* lives in host memory, so
    a matrix larger than VRAM becomes workable at all. But peak device memory
    does **not** fall off monotonically with *chunk_size*. Measured on a
    60,000 x 200,000 matrix (48 M nonzeros, k=20): 1.45 GB resident,
    0.45 GB at ``chunk_size=20_000``, and 1.63 GB at ``chunk_size=5_000`` --
    *worse* than resident. The dominant term is per-spmv cuSPARSE workspace,
    which is allocated outside CuPy's pool (the pool itself reported no growth)
    and is paid once per chunk per matvec, so halving the chunk size doubles
    the number of those allocations. There is a sweet spot, not a knob that
    trades memory for speed linearly; a few large chunks beat many small ones.

    ``matvec`` and ``rmatvec`` **preserve the dimensionality of their input**:
    a ``(n, 1)`` column vector returns ``(n, 1)``, a flat ``(n,)`` returns
    ``(n,)``. This is required by CuPy's iterative solvers — returning a flat
    array for a 2-D input surfaces as a bare ``ValueError: Out shape is
    mismatched`` raised from inside the eigensolver, with nothing indicating
    the cause. Callers should always go through :meth:`as_operator` or
    :meth:`gram_operator` rather than wiring the methods up by hand.
    """

    def __init__(self, X: sp.csr_matrix, chunk_size: int) -> None:
        X = X.tocsr()
        if X.dtype != np.float32:
            X = sp.csr_matrix(
                (X.data.astype(np.float32), X.indices, X.indptr), shape=X.shape
            )
        self.shape = X.shape
        self.dtype = cp.float32
        n_rows = X.shape[0]

        self.chunks: list[_PinnedChunk] = []
        for start in range(0, n_rows, chunk_size):
            end = min(start + chunk_size, n_rows)
            self.chunks.append(_PinnedChunk(X[start:end], start, end))
        logger.info(
            f"ChunkedMatrix: {n_rows:,} rows in {len(self.chunks)} pinned "
            f"chunks of <= {chunk_size:,}"
        )
        self._stream = cp.cuda.Stream(non_blocking=True)

    # -- internals ---------------------------------------------------------

    def _upload(self, pc: _PinnedChunk) -> cusp.csr_matrix:
        with self._stream:
            d_data    = cp.asarray(pc.data)
            d_indices = cp.asarray(pc.indices)
            d_indptr  = cp.asarray(pc.indptr)
        return cusp.csr_matrix((d_data, d_indices, d_indptr), shape=pc.shape)

    def _stream_chunks(self):
        """Yield ``(chunk, gpu_csr)`` with chunk i+1 prefetched during i."""
        cur_gpu = self._upload(self.chunks[0])
        cur_ev = cp.cuda.Event()
        cur_ev.record(self._stream)

        for i, pc in enumerate(self.chunks):
            nxt_gpu = nxt_ev = None
            if i + 1 < len(self.chunks):
                nxt_gpu = self._upload(self.chunks[i + 1])
                nxt_ev = cp.cuda.Event()
                nxt_ev.record(self._stream)

            cur_ev.synchronize()
            yield pc, cur_gpu
            del cur_gpu

            if nxt_gpu is not None:
                cur_gpu, cur_ev = nxt_gpu, nxt_ev

    # -- products ----------------------------------------------------------

    def matvec(self, v: cp.ndarray) -> cp.ndarray:
        """``X @ v``, preserving the input's dimensionality."""
        shp = v.shape
        v = v.ravel()
        y = cp.empty(self.shape[0], dtype=cp.float32)
        for pc, gpu in self._stream_chunks():
            y[pc.row_start : pc.row_end] = gpu.dot(v)
        return y.reshape(-1, 1) if len(shp) == 2 else y

    def rmatvec(self, v: cp.ndarray) -> cp.ndarray:
        """``Xᵀ @ v``, preserving the input's dimensionality."""
        shp = v.shape
        v = v.ravel()
        w = cp.zeros(self.shape[1], dtype=cp.float32)
        for pc, gpu in self._stream_chunks():
            w += gpu.T.dot(v[pc.row_start : pc.row_end])
        return w.reshape(-1, 1) if len(shp) == 2 else w

    # -- operators ---------------------------------------------------------

    def as_operator(self) -> cusla.LinearOperator:
        """
        ``X`` as an ``(n_rows, n_cols)`` LinearOperator, for ``svds``.

        ``cupyx.scipy.sparse.linalg.svds`` composes ``a.H @ a`` internally and
        runs Lanczos on the smaller Gram side, so a streamed operator needs no
        separate eigensolver path.
        """
        return cusla.LinearOperator(
            shape=self.shape,
            matvec=self.matvec,
            rmatvec=self.rmatvec,
            dtype=cp.float32,
        )

    def gram_operator(self, diag_shift: cp.ndarray | None = None):
        """
        ``X Xᵀ`` (optionally minus ``diag(diag_shift)``) as an
        ``(n_rows, n_rows)`` symmetric LinearOperator, for ``eigsh``.

        Parameters
        ----------
        diag_shift
            Per-row values subtracted from the diagonal, i.e. the operator
            becomes ``X (Xᵀ v) - diag_shift * v``. ``None`` leaves the plain
            Gram product. :mod:`gatac.tl.spectral` passes the inverse degree
            vector here; LSI passes nothing.
        """

        def matvec(v):
            shp = v.shape
            vf = v.ravel()
            y = self.matvec(self.rmatvec(vf))
            if diag_shift is not None:
                y = y - diag_shift * vf
            return y.reshape(-1, 1) if len(shp) == 2 else y

        n = self.shape[0]
        return cusla.LinearOperator(
            shape=(n, n), matvec=matvec, dtype=cp.float32
        )


def chunked_operator(
    X: sp.csr_matrix,
    chunk_size: int,
    diag_shift: cp.ndarray | None = None,
    gram: bool = False,
):
    """
    Convenience factory: stream *X* from host memory as a LinearOperator.

    Parameters
    ----------
    X
        Host CSR matrix.
    chunk_size
        Rows per streamed chunk.
    diag_shift
        Only meaningful with ``gram=True``; see
        :meth:`ChunkedMatrix.gram_operator`.
    gram
        ``False`` (default) returns ``X`` itself, for ``svds``. ``True``
        returns the symmetric ``X Xᵀ`` operator, for ``eigsh``.

    Returns
    -------
    tuple[ChunkedMatrix, cupyx.scipy.sparse.linalg.LinearOperator]
        The backing store and the operator. Keep a reference to the store for
        as long as the operator is in use — it owns the pinned host memory.
    """
    store = ChunkedMatrix(X, chunk_size)
    op = store.gram_operator(diag_shift) if gram else store.as_operator()
    return store, op
