# API Reference

GATAC's Python API is organized into namespaces that mirror the workflow:
fragment files, preprocessing, then one namespace per mode of analysis, and
plotting.

```{toctree}
:maxdepth: 2
:hidden:

io
preprocessing
dr
peak
motif
plotting
```

::::{grid} 1 1 3 3
:gutter: 2

:::{grid-item-card} {fas}`file-import` Fragment I/O — `gatac.io`
:link: io
:link-type: doc
:shadow: none

Convert fragment TSV.GZ files to Parquet and read them back.
:::

:::{grid-item-card} {fas}`cogs` Preprocessing — `gatac.pp`
:link: preprocessing
:link-type: doc
:shadow: none

Tile/gene matrix construction, QC metrics, filtering, and feature
selection.
:::

:::{grid-item-card} {fas}`project-diagram` Dimensionality reduction — `gatac.dr`
:link: dr
:link-type: doc
:shadow: none

Spectral embedding, ArchR-style (iterative) LSI, and LDA topic modelling.
:::

:::{grid-item-card} {fas}`mountain` Peaks — `gatac.peak`
:link: peak
:link-type: doc
:shadow: none

Peak calling, merging, the cell × peak matrix, and marker peaks.
:::

:::{grid-item-card} {fas}`dna` Motifs — `gatac.motif`
:link: motif
:link-type: doc
:shadow: none

Motif enrichment, GSEA on motif rankings, and chromVAR.
:::

:::{grid-item-card} {fas}`chart-bar` Plotting — `gatac.pl`
:link: plotting
:link-type: doc
:shadow: none

Fragment size distribution, QC metrics, and motif enrichment plots.
:::

::::

---

## Quick overview

```python
import gatac as ga

# ── Fragment I/O ──────────────────────────────────────────────────────────
# 1. Convert fragment TSV → Parquet
ga.io.make_parquet("sample.tsv.gz")

# ── Preprocessing ─────────────────────────────────────────────────────────
# 2. Compute QC metrics
metrics = ga.pp.compute_metrics("sample.parquet", "GRCh38.gtf.gz")

# 3. Build tile matrix (filtered by QC)
adata = ga.pp.make_tile_matrix(
    "sample.parquet",
    chrom_sizes="hg38",
    metrics=metrics,
    filter_query="tsse_score > 5 and n_unique > 1000",
)

# 4. Select most accessible features
ga.pp.select_features(adata, n_features=500_000)

# ── Dimensionality reduction ──────────────────────────────────────────────
# 5. Spectral embedding
ga.dr.spectral(adata)

# ── Peaks ─────────────────────────────────────────────────────────────────
# 6. Call peaks
ga.peak.call(adata, groupby="cell_type", parquet_path="sample.parquet")

# ── Motifs ────────────────────────────────────────────────────────────────
# 7. chromVAR TF activity
ga.motif.chromvar(
    adata,
    genome_fasta="GRCh38.fa",
    motifs_path="cisBP_human.meme",
)
```

