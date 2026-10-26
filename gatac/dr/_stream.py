"""
A host CSR count matrix used on the GPU in row chunks, as it is.

The matrix is never normalized or copied into new chunk objects: its raw
values and int32 column indices are uploaded per row chunk, and the
kernels index rows through the matrix's global ``indptr`` (kept on the GPU),
offset by the chunk's first entry. Normalizations that scale rows and
columns are applied to the vectors instead (see :mod:`gatac.dr.spectral`).

Chunks are kept on the GPU while they fit in a share of its free memory;
the others are streamed from page-locked host memory at every product,
the next chunk copied while the current one is used.
"""
from __future__ import annotations

import logging

import cupy as cp
import numpy as np
import scipy.sparse as sp

from .._cupy_compat import bincount

logger = logging.getLogger(__name__)

_DTYPES = {np.dtype(np.uint8): "unsigned char", np.dtype(np.uint16): "unsigned short",
           np.dtype(np.int32): "int", np.dtype(np.float32): "float"}

# One warp per row; ``indptr`` is the whole matrix's, ``base`` the index of the
# chunk's first entry in it, ``r0`` the chunk's first row.
_SOURCE = r"""
template <typename T>
__device__ __forceinline__ float val(const T* data, long long j) { return (float)data[j]; }

// y[r0 + i] = sum_j data[j] * x[indices[j]]
template <typename T>
__global__ void gather(const T* __restrict__ data, const int* __restrict__ indices,
                       const long long* __restrict__ indptr, long long base, int r0,
                       int n_rows, const float* __restrict__ x, float* __restrict__ y) {
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp >= n_rows) return;
    long long s = indptr[r0 + warp] - base, e = indptr[r0 + warp + 1] - base;
    double acc = 0.0;
    for (long long j = s + lane; j < e; j += 32)
        acc += (double)(val(data, j) * x[indices[j]]);
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffff, acc, o);
    if (lane == 0) y[r0 + warp] = (float)acc;
}

// y[r0 + i] = sum_j (data[j] * w[indices[j]])^2
template <typename T>
__global__ void row_sq(const T* __restrict__ data, const int* __restrict__ indices,
                       const long long* __restrict__ indptr, long long base, int r0,
                       int n_rows, const float* __restrict__ w, float* __restrict__ y) {
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp >= n_rows) return;
    long long s = indptr[r0 + warp] - base, e = indptr[r0 + warp + 1] - base;
    double acc = 0.0;
    for (long long j = s + lane; j < e; j += 32) {
        float v = val(data, j) * w[indices[j]];
        acc += (double)(v * v);
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffff, acc, o);
    if (lane == 0) y[r0 + warp] = (float)acc;
}

// out[indices[j]] += data[j] * z[r0 + i]
template <typename T>
__global__ void scatter(const T* __restrict__ data, const int* __restrict__ indices,
                        const long long* __restrict__ indptr, long long base, int r0,
                        int n_rows, const float* __restrict__ z, float* __restrict__ out) {
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp >= n_rows) return;
    float zi = z[r0 + warp];
    if (zi == 0.0f) return;
    long long s = indptr[r0 + warp] - base, e = indptr[r0 + warp + 1] - base;
    for (long long j = s + lane; j < e; j += 32)
        atomicAdd(out + indices[j], val(data, j) * zi);
}
"""

_module = None


def _kernel(name: str, dtype: np.dtype):
    global _module
    if _module is None:
        exprs = [f"{k}<{t}>" for k in ("gather", "row_sq", "scatter") for t in _DTYPES.values()]
        _module = cp.RawModule(code=_SOURCE, options=("-std=c++14",), name_expressions=exprs)
    return _module.get_function(f"{name}<{_DTYPES[np.dtype(dtype)]}>")


class _Chunk:
    __slots__ = ("r0", "r1", "e0", "e1", "data", "indices")

    def __init__(self, r0, r1, e0, e1):
        self.r0, self.r1, self.e0, self.e1 = r0, r1, e0, e1
        self.data = self.indices = None   # device arrays while resident


class StreamedCounts:
    """
    A host CSR matrix (``n_rows × n_cols``) for row-chunked GPU products.

    Parameters
    ----------
    X
        Host CSR matrix. Its values are used as they are when their dtype is
        uint8, uint16, int32 or float32 (otherwise converted to float32);
        its column indices are converted to int32 when stored wider.
    chunk_size
        Rows per chunk.
    memory_fraction
        Share of the GPU's free memory the resident chunks may take.
    """

    def __init__(self, X: sp.csr_matrix, chunk_size: int, memory_fraction: float = 0.5) -> None:
        if hasattr(X, "get"):   # a CuPy sparse matrix
            X = X.get()
        X = X.tocsr()
        self.shape = X.shape
        n_rows, n_cols = X.shape
        if n_cols > np.iinfo(np.int32).max or n_rows > np.iinfo(np.int32).max:
            raise ValueError("StreamedCounts needs fewer than 2**31 rows and columns")
        data = X.data if X.data.dtype in _DTYPES else X.data.astype(np.float32)
        indices = X.indices if X.indices.dtype == np.int32 else X.indices.astype(np.int32)
        self._data, self._indices = data, indices
        self.dtype = data.dtype
        indptr = np.asarray(X.indptr, dtype=np.int64)
        self.indptr = cp.asarray(indptr)

        entry_bytes = data.itemsize + 4
        budget = memory_fraction * cp.cuda.Device().mem_info[0]
        self.chunks: list[_Chunk] = []
        used = 0
        for r0 in range(0, n_rows, chunk_size):
            r1 = min(r0 + chunk_size, n_rows)
            c = _Chunk(r0, r1, int(indptr[r0]), int(indptr[r1]))
            size = (c.e1 - c.e0) * entry_bytes
            if used + size <= budget:
                c.data = cp.asarray(data[c.e0:c.e1])
                c.indices = cp.asarray(indices[c.e0:c.e1])
                used += size
            self.chunks.append(c)
        self._streamed = [c for c in self.chunks if c.data is None]
        self._registered: list = []
        self._stream = None
        if self._streamed:
            # Page-lock the streamed part of the host arrays in place, once,
            # and stage it through two device buffers.
            e0, e1 = self._streamed[0].e0, self._streamed[-1].e1
            for arr in (data, indices):
                part = arr[e0:e1]
                if part.nbytes:
                    cp.cuda.runtime.hostRegister(part.ctypes.data, part.nbytes, 0)
                    self._registered.append(part)
            biggest = max(c.e1 - c.e0 for c in self._streamed)
            self._buffers = [(cp.empty(biggest, dtype=data.dtype), cp.empty(biggest, dtype=np.int32))
                             for _ in range(2)]
            self._stream = cp.cuda.Stream(non_blocking=True)
        logger.info(
            f"Count matrix: {len(self.chunks)} chunks of <= {chunk_size:,} rows, "
            f"{len(self.chunks) - len(self._streamed)} on the GPU "
            f"({used / 1e9:.1f} GB), {len(self._streamed)} streamed"
        )

    def close(self) -> None:
        """Unpin the host memory and free the GPU copies."""
        for part in self._registered:
            cp.cuda.runtime.hostUnregister(part.ctypes.data)
        self._registered = []
        for c in self.chunks:
            c.data = c.indices = None
        self._buffers = []

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # -- chunk iteration -------------------------------------------------------

    def _copy(self, c: _Chunk, slot: int):
        d, i = self._buffers[slot]
        n = c.e1 - c.e0
        with self._stream:
            d[:n].set(self._data[c.e0:c.e1], stream=self._stream)
            i[:n].set(self._indices[c.e0:c.e1], stream=self._stream)
            ev = cp.cuda.Event()
            ev.record(self._stream)
        return d[:n], i[:n], ev

    def _iter(self):
        """Yield ``(chunk, data, indices)`` on the GPU, streamed chunks prefetched."""
        pending = None
        streamed = iter(self._streamed)
        nxt = next(streamed, None)
        slot = 0
        if nxt is not None:
            pending = (nxt, *self._copy(nxt, slot))
        for c in self.chunks:
            if c.data is not None:
                yield c, c.data, c.indices
                continue
            chunk, d, i, ev = pending
            # The compute stream waits for this chunk's copy. The next chunk
            # goes to the other buffer, once the kernels queued so far (those
            # of the chunk that last used that buffer) are done.
            compute = cp.cuda.get_current_stream()
            compute.wait_event(ev)
            nxt = next(streamed, None)
            slot ^= 1
            if nxt is not None:
                done = cp.cuda.Event()
                done.record(compute)
                self._stream.wait_event(done)
                pending = (nxt, *self._copy(nxt, slot))
            yield chunk, d, i

    def _launch(self, name, c, d, i, *args):
        n = c.r1 - c.r0
        block = 256
        grid = (n * 32 + block - 1) // block
        _kernel(name, self.dtype)((grid,), (block,), (d, i, self.indptr, np.int64(c.e0),
                                                      np.int32(c.r0), np.int32(n), *args))

    # -- products --------------------------------------------------------------

    def dot(self, x: cp.ndarray) -> cp.ndarray:
        """``X @ x`` for a vector over the columns."""
        y = cp.empty(self.shape[0], dtype=cp.float32)
        x = cp.ascontiguousarray(x, dtype=cp.float32)
        for c, d, i in self._iter():
            self._launch("gather", c, d, i, x, y)
        return y

    def rdot(self, z: cp.ndarray) -> cp.ndarray:
        """``Xᵀ @ z`` for a vector over the rows (atomic sums: not bitwise reproducible)."""
        out = cp.zeros(self.shape[1], dtype=cp.float32)
        z = cp.ascontiguousarray(z, dtype=cp.float32)
        for c, d, i in self._iter():
            self._launch("scatter", c, d, i, z, out)
        return out

    def row_sq_norms(self, w: cp.ndarray) -> cp.ndarray:
        """``sum_j (X_ij w_j)^2`` per row."""
        y = cp.empty(self.shape[0], dtype=cp.float32)
        w = cp.ascontiguousarray(w, dtype=cp.float32)
        for c, d, i in self._iter():
            self._launch("row_sq", c, d, i, w, y)
        return y

    def col_nnz(self) -> cp.ndarray:
        """Number of stored entries per column."""
        df = cp.zeros(self.shape[1], dtype=cp.int64)
        for _, _, i in self._iter():
            df += bincount(i, minlength=self.shape[1])
        return df
