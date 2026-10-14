"""
GATAC tools module - analysis functions for ATAC-seq data.
"""

from .peaks import call_peaks, merge_peaks, make_peak_matrix
from .motif import (
    motif_enrichment,
    motif_enrichment_regression,
    motif_presence_matrix,
    read_motifs,
    parse_meme,
    DNAMotif,
    gsea_motif_enrichment,
    sample_gc_matched_background,
)
from .markers import marker_peaks, get_marker_peaks
from .chromvar import (
    chromvar,
    compute_deviations,
    scan_motifs,
    sample_bg_peaks,
    compute_peak_bias,
)
from .spectral import spectral
from .lsi import (
    lsi,
    iterative_lsi,
    project_lsi,
    model_from_adata,
    LSIModel,
    initial_features,
    accessibility_pool,
    cluster_var_features,
    feature_accessibility,
    scale_dims,
)
from .lda import lda, MiniBatchLDA
from .footprint import footprint, FootprintTracks, TrackLayout
from .bindetect import bindetect, bindetect_sites, BINDetectResult, SiteSet

__all__ = [
    "call_peaks", 
    "merge_peaks", 
    "make_peak_matrix",
    "motif_enrichment",
    "motif_enrichment_regression",
    "motif_presence_matrix",
    "gsea_motif_enrichment",
    "sample_gc_matched_background",
    "read_motifs",
    "parse_meme",
    "DNAMotif",
    "marker_peaks",
    "get_marker_peaks",
    "chromvar",
    "compute_deviations",
    "scan_motifs",
    "sample_bg_peaks",
    "compute_peak_bias",
    "spectral",
    "lsi",
    "iterative_lsi",
    "project_lsi",
    "model_from_adata",
    "LSIModel",
    "initial_features",
    "accessibility_pool",
    "cluster_var_features",
    "feature_accessibility",
    "scale_dims",
    "lda",
    "MiniBatchLDA",
    "footprint",
    "FootprintTracks",
    "TrackLayout",
    "bindetect",
    "bindetect_sites",
    "BINDetectResult",
    "SiteSet",
]
