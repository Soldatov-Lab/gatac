---
sd_hide_title: true
---

# GATAC

```{toctree}
:hidden:
:maxdepth: 1

installation
changelog
contributing
```

```{toctree}
:hidden:
:caption: CLI Reference
:maxdepth: 2

cli/index
```

```{toctree}
:hidden:
:caption: Python API
:maxdepth: 2

api/index
```

```{toctree}
:hidden:
:caption: Tutorials
:maxdepth: 2

tutorials/index
```

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} GPU-Accelerated ATAC-seq Processing
:shadow: none

**GATAC** is a GPU-accelerated toolkit for ATAC-seq data processing — from raw
fragment files to analysis-ready matrices.  It leverages **RAPIDS cuDF**,
**CuPy**, and **cuML** to deliver 10–50× speedups over CPU-only tools.

+++
```{button-ref} installation
:ref-type: doc
:color: primary
:outline:
Get started
```
:::

:::{grid-item-card} Scanpy-compatible
:shadow: none

GATAC outputs standard **AnnData** objects that drop straight into any
Scanpy/SnapATAC2 downstream workflow.  Quality metrics, tile matrices, peak
matrices, spectral embeddings, and chromVAR deviations are all first-class
citizens.

+++
```{button-ref} api/index
:ref-type: doc
:color: secondary
:outline:
Browse the API
```
:::

::::

---

## What can GATAC do?

::::{grid} 2 2 3 3
:gutter: 2

:::{grid-item-card} {fas}`file-arrow-down` Fragment I/O
:shadow: none

Convert raw TSV.GZ fragment files to columnar Parquet for fast GPU streaming,
with optional barcode prefixing for multi-sample projects.

[→ CLI: convert](cli/convert)
:::

:::{grid-item-card} {fas}`chart-bar` Quality Metrics
:shadow: none

Stream-compute **TSS enrichment**, unique fragment count, duplicate rate, and
mito fraction entirely on GPU without loading the full file into memory.

[→ CLI: metrics](cli/metrics)
:::

:::{grid-item-card} {fas}`filter` Fragment Filtering
:shadow: none

Filter barcodes by arbitrary metric thresholds (e.g. `tsse_score > 5`) using a
Polars query engine backed by GPU execution.

[→ CLI: filter](cli/filter)
:::

:::{grid-item-card} {fas}`th` Tile Matrix
:shadow: none

Bin the genome into fixed-size tiles and produce a sparse cell × bin count
matrix compatible with SnapATAC2.

[→ CLI: tile](cli/tile)
:::

:::{grid-item-card} {fas}`dna` Gene Activity
:shadow: none

Score gene activity from paired insertion counts over promoter + gene body
regions using a GTF annotation.

[→ CLI: gene](cli/gene)
:::

:::{grid-item-card} {fas}`sliders` Feature Selection
:shadow: none

GPU-accelerated selection of the most accessible genomic features across one
or many h5ad files using streaming aggregation.

[→ CLI: features](cli/features)
:::

:::{grid-item-card} {fas}`project-diagram` Spectral Embedding
:shadow: none

Spectral decomposition of the cell × feature matrix for dimensionality
reduction, UMAP, and clustering.

[→ API: tl.spectral](api/tools.md#gatac.tl.spectral)
:::

:::{grid-item-card} {fas}`mountain` Peak Calling
:shadow: none

Call peaks per cell-type group, merge overlapping peaks across groups, and
build a cell × peak count matrix.

[→ API: tl.call_peaks](api/tools.md#gatac.tl.call_peaks)
:::

:::{grid-item-card} {fas}`dna` Motif Analysis
:shadow: none

Scan peaks for TF binding motifs (MEME format), run motif enrichment tests,
and compute chromVAR deviation scores.

[→ API: tl.chromvar](api/tools.md#gatac.tl.chromvar)
:::

::::
