---
sd_hide_title: true
---

# GATAC

```{toctree}
:hidden:
:caption: General
:maxdepth: 1

installation
tutorials/index
changelog
contributing
```

```{toctree}
:hidden:
:caption: Reference
:maxdepth: 2

api/index
cli/index
```

```{toctree}
:hidden:
:caption: About
:maxdepth: 1

../reproducibility/README
license
credits
```
# GATAC: GPU-Accelerated scATACseq Analysis

GATAC is a GPU-accelerated toolkit for end-to-end ATAC-seq data processing.
Starting from raw fragment files, it produces analysis-ready sparse matrices
(tile, peak, and gene activity) that integrate directly with the
[scverse](https://scverse.org) ecosystem — AnnData, Scanpy, and SnapATAC2.
Computations are offloaded to the GPU via [RAPIDS cuDF](https://rapids.ai),
[CuPy](https://cupy.dev), and [cuML](https://docs.rapids.ai/api/cuml/stable/),
delivering 10× speedups over CPU-only workflows on typical single-cell
datasets.

GATAC workflow takes advantage of [Apache Parquet](https://parquet.apache.org) file format. Parquet's columnar layout and
built-in compression make it an ideal staging format for GPU-based pipelines:
columns are read independently, so only the genomic coordinates actually needed
for a given operation are transferred to device memory. RAPIDS cuDF can read
Parquet directly into GPU memory with zero CPU round-trips, enabling streaming
aggregation over datasets that far exceed the size of available GPU RAM.

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

:::{grid-item-card} Ecosystem Integration
:shadow: none

GATAC [reproduces](../reproducibility/README) the core operations and functions of established tools like **SnapATAC2**, **ArchR**, **MACS3**, and **chromVAR** within a unified framework. It produces standard **AnnData** objects that are fully compatible with the **scverse** ecosystem.

+++
```{button-ref} tutorials/index
:ref-type: doc
:color: success
:outline:
View Tutorials
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

[→ API: tl.spectral](api/tools.md#spectral-embedding)
:::

:::{grid-item-card} {fas}`mountain` Peak Calling
:shadow: none

Call peaks per cell-type group, merge overlapping peaks across groups, and
build a cell × peak count matrix.

[→ API: tl.call_peaks](api/tools.md#call-peaks)
:::

:::{grid-item-card} {fas}`dna` Motif Analysis
:shadow: none

Scan peaks for TF binding motifs (MEME format), run motif enrichment tests,
and compute chromVAR deviation scores.

[→ API: tl.chromvar](api/tools.md#chromvar)
:::

::::
