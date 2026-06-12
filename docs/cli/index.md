# CLI Reference

GATAC ships a command-line interface for the preprocessing pipeline.  Every
step that turns raw fragment files into analysis-ready matrices is available
as a sub-command.

```bash
gatac [--verbose] <command> [options]
```

Global option:

| Flag | Description |
|------|-------------|
| `-v, --verbose` | Enable debug-level logging |


```{toctree}
:maxdepth: 1
:hidden:

convert
metrics
filter
doublets
tile
gene
features
combine
```

## Commands at a glance

::::{grid} 2 2 3 3
:gutter: 2

:::{grid-item-card} {fas}`file-arrow-down` convert
:link: convert
:link-type: doc
:shadow: none

Convert TSV.GZ fragment files to Parquet.
:::

:::{grid-item-card} {fas}`chart-bar` metrics
:link: metrics
:link-type: doc
:shadow: none

Compute TSS enrichment and fragment QC statistics.
:::

:::{grid-item-card} {fas}`filter` filter
:link: filter
:link-type: doc
:shadow: none

Filter barcodes by quality thresholds.
:::

:::{grid-item-card} {fas}`clone` doublets
:link: doublets
:link-type: doc
:shadow: none

Detect doublet / multiplet cells (AMULET Poisson method).
:::

:::{grid-item-card} {fas}`th` tile
:link: tile
:link-type: doc
:shadow: none

Build a cell × genomic-tile count matrix.
:::

:::{grid-item-card} {fas}`dna` gene
:link: gene
:link-type: doc
:shadow: none

Build a cell × gene activity matrix.
:::

:::{grid-item-card} {fas}`sliders` features
:link: features
:link-type: doc
:shadow: none

Select highly accessible features.
:::

:::{grid-item-card} {fas}`layer-group` combine
:link: combine
:link-type: doc
:shadow: none

Merge multiple h5ad files into one.
:::

::::
