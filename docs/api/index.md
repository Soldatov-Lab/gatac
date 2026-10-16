# API Reference

GATAC's Python API is organized into three namespaces that mirror the
fragment files → preprocessing → analysis workflow.

```{toctree}
:maxdepth: 2
:hidden:

io
preprocessing
tools
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

:::{grid-item-card} {fas}`chart-line` Tools — `gatac.tl`
:link: tools
:link-type: doc
:shadow: none

Spectral embedding, LDA, peak calling, marker peaks, motif enrichment,
chromVAR, and GSEA.
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

# ── Tools ──────────────────────────────────────────────────────────────────
# 5. Spectral embedding
ga.tl.spectral(adata)

# 6. Call peaks
ga.tl.call_peaks(adata, groupby="cell_type", parquet_path="sample.parquet")

# 7. Motif enrichment
ga.tl.chromvar(
    adata,
    genome_fasta="GRCh38.fa",
    motifs_path="cisBP_human.meme",
)
```

