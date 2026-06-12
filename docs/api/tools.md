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

#### Usage example

```python
import gatac as ga
import scanpy as sc

ga.tl.spectral(adata, n_comps=30)

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

#### Usage example

```python
model = ga.tl.lda(adata, n_topics=20, n_epochs=10)
# Cell × topic matrix stored in adata.obsm["X_lda"]
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

#### Usage example

```python
# 1. Call peaks per group
peaks = ga.tl.call_peaks(
    adata,
    group_by="leiden",
    fragment_source="pbmc.parquet",
    genome="hg38",
)

# 2. Merge overlapping peaks across groups
merged = ga.tl.merge_peaks(adata)

# 3. Build cell × peak count matrix
peak_adata = ga.tl.make_peak_matrix(
    peaks=merged,
    fragments="pbmc.parquet",
)
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

### Usage example

```python
markers = ga.tl.marker_peaks(
    adata,
    groupby="leiden",
    reference="rest",
    min_pct=0.05,
    min_log2_fc=1.0,
)

# markers["0"] → Polars DataFrame for cluster 0
print(markers["0"].head())
```

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

```python
motifs = ga.tl.read_motifs("cisBP_human.meme", unique=True)
print(f"Loaded {len(motifs)} motifs")
```

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

#### Usage example

```python
motifs = ga.tl.read_motifs("cisBP_human.meme")
matched_bg = ga.tl.sample_gc_matched_background(
   marker_peaks,
   genome_fasta="GRCh38.fa",
   background_pool=list(peak_adata.var_names),
)

enrichment_df = ga.tl.motif_enrichment(
   motifs,
   marker_peaks,
   genome_fasta="GRCh38.fa",
   background=matched_bg,
)
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

#### Usage example

```python
gsea_df = ga.tl.gsea_motif_enrichment(
    rankings=marker_scores,
    motif_terms=motif_dict,
    n_perm=1000,
)
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

### Workflow

The high-level `ga.tl.chromvar()` runs the full pipeline in a single call —
peak bias → background sampling → motif scanning → deviation scoring.

```python
import gatac as ga

ga.tl.chromvar(
    adata,
    genome_fasta="GRCh38.fa",
    motifs_path="cisBP_human.meme",
)
# → stores deviation scores in adata.obsm["chromvar"]
```

The same result can be obtained by running the four steps individually:

```python
import gatac as ga

# 1. Compute peak GC content (used for background sampling)
ga.tl.compute_peak_bias(adata, genome_fasta="GRCh38.fa")

# 2. Sample background peaks matched on GC content + accessibility
ga.tl.sample_bg_peaks(adata, method="knn", n_iterations=50)

# 3. Load motifs and scan peaks
motifs = ga.tl.read_motifs("cisBP_human.meme")
ga.tl.scan_motifs(adata, motifs, "GRCh38.fa")

# 4. Compute per-cell, per-motif deviation scores
ga.tl.compute_deviations(adata)
```

### Output

Deviation scores are stored in `adata.obsm["chromvar"]` as a cell × motif
matrix.  Motif names (written by `scan_motifs`) are stored in
`adata.uns["motif_name"]`.  Background peak indices (written by
`sample_bg_peaks`) are stored in `adata.varm["bg_peaks"]`.
