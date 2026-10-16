"""
Workarounds for CuPy and CUDA driver behaviours that GATAC hits at genome scale.

``cupy.bincount`` without weights dispatches to CUB's histogram when the CUB
accelerator is enabled (the default). In CuPy 14.0.1 that path raises
``CUDA_ERROR_ILLEGAL_ADDRESS`` once there are more than ~5M bins, independent
of the input size, which a genome-wide 500 bp tile matrix (~6.2M features on
hg38) exceeds. CuPy 13.6 is unaffected. The weighted path uses a plain atomic
kernel and is safe.

A host -> device copy of a large pageable array (``cupy.asarray``) can leave a
host buffer the size of the copy allocated for the rest of the process: with
the WSL2 CUDA driver, a single copy of about 1 GB or more keeps one, reused
only by copies of that exact size. ``to_device`` copies in 64 MB chunks, which
leave nothing behind.
"""

from __future__ import annotations

import cupy as cp
import numpy as np

# Same algorithm as CuPy's non-CUB bincount kernel: one atomic add per input.
_bincount_kernel = cp.ElementwiseKernel(
    "T x",
    "raw uint64 counts",
    "atomicAdd(&counts[x], 1ULL)",
    "gatac_bincount",
)


def bincount(x: cp.ndarray, minlength: int = 0) -> cp.ndarray:
    """Unweighted ``cupy.bincount`` that never takes the CUB histogram path.

    Returns int64 counts of length ``max(minlength, x.max() + 1)``, like
    :func:`numpy.bincount`.
    """
    x = cp.asarray(x).ravel()
    if x.dtype.kind not in "iub":
        raise TypeError(f"bincount needs integer input, got {x.dtype}")
    size = int(minlength)
    if x.size:
        if x.dtype.kind == "i" and int(x.min()) < 0:
            raise ValueError("bincount input must be non-negative")
        size = max(size, int(x.max()) + 1)
    counts = cp.zeros(size, dtype=cp.uint64)
    if x.size:
        _bincount_kernel(x, counts)
    return counts.view(cp.int64)


# Bytes per chunk of to_device
_H2D_CHUNK = 64 << 20


def to_device(a, dtype=None) -> cp.ndarray:
    """``cupy.asarray(a, dtype)`` of a host array, copied in chunks of 64 MB.

    Arrays already on the device are returned as they are (cast to ``dtype``
    if given); arrays of up to one chunk go through ``cupy.asarray``.
    """
    if isinstance(a, cp.ndarray):
        return a if dtype is None else a.astype(dtype, copy=False)
    h = np.asarray(a, dtype=dtype)
    if h.nbytes <= _H2D_CHUNK:
        return cp.asarray(h)
    h = np.ascontiguousarray(h)
    d = cp.empty(h.shape, h.dtype)
    flat_h, flat_d = h.reshape(-1), d.reshape(-1)
    step = max(1, _H2D_CHUNK // h.itemsize)
    for i in range(0, flat_h.size, step):
        flat_d[i:i + step].set(flat_h[i:i + step])
    return d
