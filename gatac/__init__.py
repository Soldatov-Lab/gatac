"""
GATAC - GPU-accelerated ATAC-seq processing toolkit.

Public API:
    - pp: Preprocessing functions
    - tl: Tools (analysis functions)
"""

import os
import sys

# Disable tqdm progress bars when not running in an interactive terminal
# (e.g. when called from Snakemake), to keep logs clean.
if not sys.stderr.isatty():
    os.environ.setdefault("TQDM_DISABLE", "1")

from . import pp
from . import tl

__version__ = "0.1.0"
__all__ = ["pp", "tl"]


