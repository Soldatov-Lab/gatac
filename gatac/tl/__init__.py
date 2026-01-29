"""
GATAC tools module - analysis functions for ATAC-seq data.
"""

from .peaks import call_peaks, merge_peaks, make_peak_matrix
from .motif import motif_enrichment, read_motifs, parse_meme, DNAMotif

__all__ = [
    "call_peaks", 
    "merge_peaks", 
    "make_peak_matrix",
    "motif_enrichment",
    "read_motifs",
    "parse_meme",
    "DNAMotif",
]
