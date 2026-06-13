"""Private utilities shared across ``gatac.pp`` modules."""
from __future__ import annotations

import gc

import cupy as cp


def cleanup_gpu_memory():
    """Force cleanup of GPU memory."""
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
