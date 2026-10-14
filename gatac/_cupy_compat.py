"""
Workarounds for CuPy bugs that GATAC hits at genome scale.

``cupy.bincount`` without weights dispatches to CUB's histogram when the CUB
accelerator is enabled (the default). In CuPy 14.0.1 that path raises
``CUDA_ERROR_ILLEGAL_ADDRESS`` once there are more than ~5M bins, independent
of the input size, which a genome-wide 500 bp tile matrix (~6.2M features on
hg38) exceeds. CuPy 13.6 is unaffected. The weighted path uses a plain atomic
kernel and is safe.
"""

from __future__ import annotations

import cupy as cp

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
