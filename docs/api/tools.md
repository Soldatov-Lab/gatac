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

After computing the embedding, the typical downstream steps are:

```python
import scanpy as sc

sc.pp.neighbors(adata, use_rep="X_spectral")
sc.tl.umap(adata)
sc.pl.umap(adata, color="cell_type")
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

### Call peaks

Call ATAC peaks per cell-type group using the MACS3 algorithm under the hood.

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

### Output columns

| Column | Description |
|--------|-------------|
| `feature` | Peak / tile name |
| `log2_fc` | Log₂ fold change (foreground vs background) |
| `mean_fg` | Mean accessibility in foreground group |
| `mean_bg` | Mean accessibility in background group |
| `mean_diff` | `mean_fg − mean_bg` |
| `p_value` | Raw two-sided binomial p-value |
| `fdr` | Benjamini–Hochberg adjusted p-value |

---

## Motif analysis

### Reading motifs

```{eval-rst}
.. autosummary::
   :toctree: generated/
   :nosignatures:

   read_motifs
   parse_meme
   DNAMotif
```

#### DNAMotif attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `id` | `str` | Unique MEME identifier |
| `name` | `str` | Human-readable TF name |
| `family` | `str` | TF family |
| `pwm` | `ndarray` (L×4) | Position weight matrix [A, C, G, T] |
| `pfm` | `ndarray` (L×4) | Raw position frequency matrix |

---

### Motif enrichment

Test whether motifs are over-represented in a set of marker peaks relative to
background peaks.

```{eval-rst}
.. autosummary::
   :toctree: generated/
   :nosignatures:

   sample_gc_matched_background
   motif_enrichment
```

---

### GSEA motif enrichment

Run preranked GSEA using motif-to-gene-set memberships as gene sets.  GPU-
accelerated with CuPy.

```{eval-rst}
.. autosummary::
   :toctree: generated/
   :nosignatures:

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

### Output

The pipeline writes to several `adata` slots:

| Slot | Set by | Contents |
|------|--------|----------|
| `adata.obsm["chromvar"]` | `compute_deviations` / `chromvar` | Per-cell, per-motif deviation scores *(cells × motifs)* |
| `adata.uns["motif_name"]` | `scan_motifs` | Motif identifiers in column order |
| `adata.varm["bg_peaks"]` | `sample_bg_peaks` | Background peak indices per peak *(n_peaks × n_iterations)* |

To run the four steps individually, see the docstring of
`compute_deviations` (which lists them end-to-end).
