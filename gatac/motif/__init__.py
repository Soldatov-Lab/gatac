"""
Motifs: reading and scanning motifs, enrichment in peak sets, GSEA on motif
rankings, and chromVAR TF-activity scores.
"""

from .motif import (
    enrichment,
    enrichment_regression,
    presence_matrix,
    read,
    DNAMotif,
    gsea,
    sample_gc_matched_background,
)
from .chromvar import (
    chromvar,
    compute_deviations,
    scan,
    sample_bg_peaks,
    compute_peak_bias,
)

__all__ = [
    "read",
    "DNAMotif",
    "sample_gc_matched_background",
    "enrichment",
    "presence_matrix",
    "enrichment_regression",
    "gsea",
    "chromvar",
    "compute_deviations",
    "scan",
    "sample_bg_peaks",
    "compute_peak_bias",
]
