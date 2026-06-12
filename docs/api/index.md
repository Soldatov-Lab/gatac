# API Reference

GATAC's Python API is organized into two namespaces that mirror the
preprocessing → analysis workflow:

| Namespace | Description |
|-----------|-------------|
| `gatac.pp` | **Preprocessing** — fragment I/O, matrix building, QC metrics, filtering |
| `gatac.tl` | **Tools** — spectral embedding, peak calling, motif analysis, chromVAR, LDA |

```{toctree}
:maxdepth: 2
:hidden:

preprocessing
tools
```

---

## Quick overview

```python
import gatac as ga

# ── Preprocessing ─────────────────────────────────────────────────────────
# 1. Convert fragment TSV → Parquet
ga.pp.make_parquet("sample.tsv.gz")

# 2. Compute QC metrics
tss = ga.pp.load_tss_from_gtf("GRCh38.gtf.gz")
metrics = ga.pp.compute_metrics("sample.parquet", tss)

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

---

## Submodule pages

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item-card} {fas}`cogs` Preprocessing — `gatac.pp`
:link: preprocessing
:link-type: doc
:shadow: none

Fragment I/O, tile/gene matrix construction, QC metrics, filtering, and
feature selection.
:::

:::{grid-item-card} {fas}`chart-line` Tools — `gatac.tl`
:link: tools
:link-type: doc
:shadow: none

Spectral embedding, LDA, peak calling, marker peaks, motif enrichment,
chromVAR, and GSEA.
:::

::::
