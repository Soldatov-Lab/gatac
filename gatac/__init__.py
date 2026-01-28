"""
GATAC - GPU-accelerated ATAC-seq processing toolkit.

Public API:
    - make_parquet: Convert ATAC fragments TSV.GZ to Parquet
    - make_tile_matrix: Process fragments to tile matrix (AnnData)
    - select_features: GPU-accelerated feature selection
    - compute_metrics: GPU-accelerated quality metrics
"""

from .convert import make_parquet
from .process import make_tile_matrix, read_fragments_parquet
from .features import select_features, select_features_multi
from .metrics import (
    load_tss_from_gtf, 
    compute_metrics,
    cleanup_gpu_memory
)

__version__ = "0.1.0"
__all__ = [
    "make_parquet", 
    "make_tile_matrix", 
    "read_fragments_parquet", 
    "select_features",
    "select_features_multi",
    "load_tss_from_gtf", 
    "compute_metrics",
    "cleanup_gpu_memory",
]

