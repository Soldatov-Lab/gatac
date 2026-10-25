"""Private utilities shared across ``gatac.pp`` modules."""
from __future__ import annotations

import gc
from contextlib import contextmanager

import cupy as cp


def cleanup_gpu_memory():
    """Force cleanup of GPU memory."""
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


@contextmanager
def async_memory_pool():
    """
    Use RMM's stream-ordered pool (cudaMallocAsync) unless a memory resource
    is already set. Without a pool every cuDF and CuPy allocation is a
    cudaMalloc, which takes about as long as the work on fragment batches.
    """
    import rmm

    previous = rmm.mr.get_current_device_resource()
    if type(previous) is not rmm.mr.CudaMemoryResource:
        yield
        return
    rmm.mr.set_current_device_resource(rmm.mr.CudaAsyncMemoryResource())
    try:
        yield
    finally:
        rmm.mr.set_current_device_resource(previous)
