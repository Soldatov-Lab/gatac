# Tools — `gatac.tl`

The `gatac.tl` namespace provides downstream analysis tools: dimensionality
reduction, peak calling, marker detection, motif scanning, chromVAR deviation
scoring, and topic modelling.

---

## Dimensionality reduction

### Spectral embedding

Compute a spectral decomposition of the cell × feature matrix — the standard
entry point for UMAP and clustering in ATAC-seq workflows.

```{eval-rst}
.. currentmodule:: gatac.tl

.. autosummary::
   :toctree: generated/
   :nosignatures:

   spectral
```

---

### Latent Dirichlet Allocation

Topic modelling of the peak-accessibility matrix using GPU-accelerated
mini-batch Online Variational Bayes.

```{eval-rst}
.. autosummary::
   :toctree: generated/
   :nosignatures:

   lda
   MiniBatchLDA
```

---

## Peak calling

Call ATAC peaks per cell-type group using the MACS3 algorithm under the hood,
merge them into a non-overlapping set, and count fragments over peaks.

```{eval-rst}
.. autosummary::
   :toctree: generated/
   :nosignatures:

   call_peaks
   merge_peaks
   make_peak_matrix
```

---

## Marker peaks

Identify differentially accessible peaks between groups using a GPU-
accelerated binomial test with Benjamini–Hochberg correction.

```{eval-rst}
.. autosummary::
   :toctree: generated/
   :nosignatures:

   marker_peaks
   get_marker_peaks
```

---

## Motif analysis

Read motifs from MEME-format files, test for over-representation in peak
sets, and run GSEA on motif rankings.

```{eval-rst}
.. autosummary::
   :toctree: generated/
   :nosignatures:

   read_motifs
   parse_meme
   DNAMotif
   sample_gc_matched_background
   motif_enrichment
   gsea_motif_enrichment
```

---

## chromVAR

Compute transcription-factor activity deviation scores following the chromVAR
algorithm.  All compute-intensive steps are executed on GPU.

```{eval-rst}
.. autosummary::
   :toctree: generated/
   :nosignatures:

   chromvar
   compute_peak_bias
   sample_bg_peaks
   scan_motifs
   compute_deviations
```

To run the four steps individually, see the docstring of
`compute_deviations` (which lists them end-to-end).
