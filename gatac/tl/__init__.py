"""
GATAC tools module - analysis functions for ATAC-seq data.
"""

from .peaks import call_peaks, merge_peaks, make_peak_matrix
from .motif import motif_enrichment, read_motifs, parse_meme, DNAMotif, gsea_motif_enrichment
from .markers import marker_peaks, get_marker_peaks
from .chromvar import (
    chromvar,
    scan_motifs,
    sample_bg_peaks,
    compute_peak_bias,
)
from .spectral import spectral

__all__ = [
    "call_peaks", 
    "merge_peaks", 
    "make_peak_matrix",
    "motif_enrichment",
    "gsea_motif_enrichment",
    "read_motifs",
    "parse_meme",
    "DNAMotif",
    "marker_peaks",
    "get_marker_peaks",
    "chromvar",
    "scan_motifs",
    "sample_bg_peaks",
    "compute_peak_bias",
    "spectral",
]
