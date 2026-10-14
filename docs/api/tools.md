# Tools — `gatac.tl`

The `gatac.tl` namespace provides downstream analysis tools: dimensionality
reduction, peak calling, marker detection, motif scanning, chromVAR deviation
scoring, footprinting with differential TF binding, and topic modelling.

---

## Dimensionality reduction

Compute a spectral decomposition of the cell × feature matrix — the standard
entry point for UMAP and clustering in ATAC-seq workflows (`spectral`) — run
TF-IDF followed by a truncated SVD, the Signac/ArchR latent semantic indexing
(`lsi`, with `project_lsi` to place new cells on a fitted embedding) — or model
topics over the peak-accessibility matrix with GPU-accelerated mini-batch
Online Variational Bayes (`lda`, `MiniBatchLDA`).

`lsi` reproduces `ArchR:::.computeLSI` component for component, including its
three `LSIMethod` variants, its depth-outlier hold-out, and the
depth-correlation dimension filter. `iterative_lsi` is the port of
`ArchR::addIterativeLSI`: it clusters on a first embedding, keeps the features
whose per-cluster accessibility varies most, and redoes the decomposition on
those — so each round selects for the structure the previous round found. It
requires cuGraph for the clustering step; `lsi` does not.

```{eval-rst}
.. currentmodule:: gatac.tl

.. autosummary::
   :toctree: generated/
   :nosignatures:

   spectral
   lsi
   iterative_lsi
   project_lsi
   model_from_adata
   lda
   MiniBatchLDA
```

The individual stages of the LSI pipeline are exported too, for building a
custom iterative scheme or inspecting one step in isolation. `scale_dims` is
ArchR's `scaleDims` — note it standardises each *cell* across its dimensions,
not each dimension across cells.

```{eval-rst}
.. autosummary::
   :toctree: generated/
   :nosignatures:

   feature_accessibility
   initial_features
   accessibility_pool
   cluster_var_features
   scale_dims
```

---

## Peak calling & marker peaks

Call ATAC peaks per cell-type group using the MACS3 algorithm under the hood,
merge them into a non-overlapping set, count fragments over peaks, and
identify differentially accessible peaks between groups using a GPU-
accelerated binomial test with Benjamini–Hochberg correction.

```{eval-rst}
.. autosummary::
   :toctree: generated/
   :nosignatures:

   call_peaks
   merge_peaks
   make_peak_matrix
   marker_peaks
   get_marker_peaks
```

---

## Motif analysis

Read motifs from MEME-format files, test for over-representation in peak
sets, run GSEA on motif rankings, and score per-cell TF activity with
chromVAR.

```{eval-rst}
.. autosummary::
   :toctree: generated/
   :nosignatures:

   read_motifs
   parse_meme
   DNAMotif
   sample_gc_matched_background
   motif_enrichment
   motif_enrichment_regression
   gsea_motif_enrichment
```

### chromVAR

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

---

## Footprinting & differential TF binding

GPU port of the [TOBIAS](https://github.com/loosolab/TOBIAS) pipeline.
`footprint` corrects the Tn5 insertion bias of each pseudobulk
(TOBIAS `ATACorrect`) and computes footprint scores (`ScoreBigwig`), reading
cut sites straight from the fragments parquet — no BAM or bigwig files.
`bindetect` scans the peaks for motifs and tests for differential binding
between groups (`BINDetect`). The binding sites depend only on the peaks and
motifs: `bindetect_sites` builds them once, and `bindetect(..., sites=)` reuses
them for every set of pseudobulks footprinted on the same peaks.

Every stage matches TOBIAS 0.17.5 exactly when given the same inputs. End to
end, the DWM bias score differs from TOBIAS only in its last bits (floating-point
summation order). That is about half the deviation TOBIAS itself shows when
its bias score is perturbed by 1e-13. `tobias_compat=True` (the default) also
reproduces TOBIAS's quirks; `tobias_compat=False` fixes them:

* the bias model depends on the BAM read length (`read_length`, 50 by
  default);
* for MEME motifs the reverse-strand matrix is rebuilt from
  `round(p × nsites)`, which is coarse for `nsites=1` files such as cisBP;
* the per-peak log2FC average actually groups by binding site.

```{eval-rst}
.. autosummary::
   :toctree: generated/
   :nosignatures:

   footprint
   bindetect_sites
   bindetect
   FootprintTracks
   SiteSet
   BINDetectResult
```
