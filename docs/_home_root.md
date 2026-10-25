# GATAC: GPU-Accelerated scATACseq Analysis

GATAC is a GPU-accelerated toolkit for end-to-end ATAC-seq data processing.
Starting from raw fragment files, it produces analysis-ready sparse matrices
(tile, peak, and gene activity) that integrate directly with the
[scverse](https://scverse.org) ecosystem {cite:p}`Virshup2023` — AnnData,
Scanpy {cite:p}`Wolf2018`, and SnapATAC2 {cite:p}`Zhang2024`.
Computations are offloaded to the GPU via [RAPIDS cuDF](https://rapids.ai),
[CuPy](https://cupy.dev), and [cuML](https://docs.rapids.ai/api/cuml/stable/),
delivering 10–50× speedups over CPU-only workflows on typical single-cell
datasets.

---

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Parquet-Native Pipeline
:shadow: none

GATAC uses [Apache Parquet](https://parquet.apache.org) as its staging format.
Columnar layout and built-in compression let RAPIDS cuDF stream data directly
into GPU memory with zero CPU round-trips — enabling aggregation over datasets
that far exceed available GPU RAM.

+++
```{button-link} api/io.html#parquet-fragment-format
:color: primary
:outline:
Inside the Parquet file
```
:::

:::{grid-item-card} Ecosystem Integration
:shadow: none

GATAC [reproduces](reproducibility) the core operations of established tools
like **SnapATAC2** {cite:p}`Zhang2024`, **ArchR** {cite:p}`Granja2021`,
**MACS3** {cite:p}`Zhang2008`, and **chromVAR** {cite:p}`Schep2017` within a unified
framework. It produces standard **AnnData** objects fully compatible with the
**scverse** ecosystem.

+++
```{button-link} installation.html
:color: primary
:outline:
Get started
```
```{button-link} tutorials/index.html
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

<a href="cli/convert.html">→ CLI: convert</a> · <a href="api/io.html">→ API: io</a> · <a href="api/io.html#parquet-fragment-format">→ File format</a>
:::

:::{grid-item-card} {fas}`filter` Quality & Filtering
:shadow: none

GPU QC metrics (**TSSe**, unique fragments, duplicate & mito fraction) plus
threshold-based barcode filtering with a query on the metrics.

<a href="cli/metrics.html">→ CLI: metrics</a> · <a href="cli/filter.html">→ CLI: filter</a> · <a href="api/preprocessing.html#quality-metrics-filtering">→ API: pp</a>
:::

:::{grid-item-card} {fas}`clone` Doublet Detection
:shadow: none

Flag doublet / multiplet cells via the **AMULET** Poisson overlap test.

<a href="cli/doublets.html">→ CLI: doublets</a> · <a href="api/preprocessing.html#quality-metrics-filtering">→ API: pp</a>
:::

:::{grid-item-card} {fas}`th` Tile Matrix
:shadow: none

Bin the genome into fixed-size tiles and produce a sparse cell × bin count
matrix compatible with SnapATAC2.

<a href="cli/tile.html">→ CLI: tile</a> · <a href="api/preprocessing.html#matrix-processing">→ API: pp</a>
:::

:::{grid-item-card} {fas}`dna` Gene Processing
:shadow: none

Score gene activity from paired insertion counts over promoter + gene body
regions, or compute ArchR-style distance-weighted gene scores — all from a GTF
annotation.

<a href="cli/gene.html">→ CLI: gene</a> · <a href="cli/genescore.html">→ CLI: genescore</a> · <a href="api/preprocessing.html#matrix-processing">→ API: pp</a>
:::

:::{grid-item-card} {fas}`sliders` Feature Selection
:shadow: none

GPU-accelerated selection of the most accessible genomic features, from a
matrix or from the tile statistics, so that a cohort's tile matrix can be
built with the selected tiles only.

<a href="cli/tile.html">→ CLI: tile</a> · <a href="api/preprocessing.html#matrix-processing">→ API: pp</a>
:::

:::{grid-item-card} {fas}`project-diagram` Dimensionality Reduction
:shadow: none

Embed cells for UMAP and clustering via spectral decomposition or ArchR-style
(iterative) **LSI** with projection of new cells.

<a href="api/dr.html">→ API: dr</a>
:::

:::{grid-item-card} {fas}`mountain` Peak Calling
:shadow: none

Call peaks per cell-type group, merge overlapping peaks across groups, and
build a cell × peak count matrix.

<a href="api/peak.html">→ API: peak</a>
:::

:::{grid-item-card} {fas}`dna` Motif Analysis
:shadow: none

Scan peaks for TF binding motifs (MEME format), run motif enrichment tests,
and compute chromVAR deviation scores.

<a href="api/motif.html">→ API: motif</a>
:::

::::
