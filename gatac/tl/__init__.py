"""
GATAC tools module - analysis functions for ATAC-seq data.
"""

from .peaks import call_peaks, merge_peaks, make_peak_matrix

__all__ = ["call_peaks", "merge_peaks", "make_peak_matrix"]
