"""
GATAC - GPU-accelerated ATAC-seq processing toolkit.

Public API:
    - pp: Preprocessing functions
    - tl: Tools (analysis functions)
"""

from . import pp
from . import tl

__version__ = "0.1.0"
__all__ = ["pp", "tl"]


