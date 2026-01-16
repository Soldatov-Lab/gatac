"""
GATAC - GPU-accelerated ATAC-seq processing toolkit.

Public API:
    - make_parquet: Convert ATAC fragments TSV.GZ to Parquet
    - make_tile_matrix: Process fragments to tile matrix (AnnData)
    - select_features: GPU-accelerated feature selection
"""

from .convert import make_parquet
from .process import make_tile_matrix
from .features import select_features
from .metrics import load_tss_from_gtf, compute_tsse

__version__ = "0.1.0"
__all__ = ["make_parquet", "make_tile_matrix", "select_features", "load_tss_from_gtf", "compute_tsse"]
