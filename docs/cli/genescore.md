# `gatac genescore`

Build a cell × gene **score** matrix from an ATAC-seq fragment Parquet file.
This is a faithful GPU port of ArchR's `addGeneScoreMatrix`: tile insertion
counts are weighted by their signed distance to each gene and an inverse
gene-width factor, then summed and per-cell normalised.

For the simpler SnapATAC2-style paired-insertion count, see
[`gatac gene`](gene.md) instead.

---

## Synopsis

```
gatac genescore <input.parquet> -g <annotations.gtf>
                [-o OUTPUT] [--gene-model EXPR] [--tile-size N]
                [--extend-upstream MIN MAX] [--extend-downstream MIN MAX]
                [--gene-upstream N] [--gene-downstream N]
                [--no-gene-boundaries] [--use-tss]
                [--ceiling N] [--gene-scale-factor F] [--scale-to F]
                [-m MIN_FRAGS] [-e CHROMS ...]
                [--metrics METRICS] [--filter QUERY]
                [--barcode-prefix PREFIX] [--low-memory]
                [--cell-batch-size N]
```

---

## Arguments

### Positional

| Argument | Description |
|----------|-------------|
| `input.parquet` | Path to the (filtered) fragment Parquet file |

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `-g`, `--gtf` | **required** | GTF/GFF annotation, or CSV with columns `symbol,seqnames,start,end,strand` |
| `-o`, `--output` | `<input>_gene_score_matrix.h5ad` | Output h5ad path |
| `--gene-model` | `exp(-abs(x)/5000) + exp(-1)` | ArchR `geneModel` expression in `x` (signed distance to TSS) |
| `--tile-size` | `500` | Tile size (bp) |
| `--extend-upstream` | `1000 100000` | Min/max bp upstream extension of the regulatory window |
| `--extend-downstream` | `1000 100000` | Min/max bp downstream extension of the regulatory window |
| `--gene-upstream` | `5000` | bp the gene body is grown upstream before the model is applied |
| `--gene-downstream` | `0` | bp the gene body is grown downstream before the model is applied |
| `--no-gene-boundaries` | off | Disable neighbouring-gene boundary clipping |
| `--use-tss` | off | Build the model on the 1bp TSS instead of the gene body |
| `--ceiling` | `4` | Max insertions counted per tile (limits pileup bias) |
| `--gene-scale-factor` | `5.0` | Inverse-gene-width weighting scale factor |
| `--scale-to` | `10000.0` | Per-cell normalisation target |
| `-m`, `--min-fragments` | `100` | Minimum unique fragments per barcode |
| `-e`, `--exclude-chroms` | `chrY chrM` | Chromosomes to exclude |
| `--metrics` | — | Metrics CSV for quality-based filtering |
| `--filter` | — | Polars query string applied to metrics |
| `--barcode-prefix` | — | String prepended to barcodes |
| `--low-memory` | off | Process one Parquet row-group at a time |
| `--cell-batch-size` | — | Process cells in column batches (lower GPU memory) |

---

## Scoring strategy

The gene score is a **distance-weighted** activity score, computed per
chromosome:

1. Both Tn5 insertion ends of every fragment are binned into `--tile-size`
   tiles and accumulated per cell, capped at `--ceiling`.
2. For each gene, an extended regulatory window is built (gene body grown by
   `--gene-upstream`/`--gene-downstream`, then out to
   `--extend-upstream`/`--extend-downstream`), optionally clipped at
   neighbouring genes unless `--no-gene-boundaries` is set.
3. Each (gene, tile) pair is weighted by `--gene-model` evaluated on the signed
   distance to the TSS, times a per-gene inverse-width weight.
4. Gene scores are the weighted sum of tile counts, then each cell is
   normalised to `--scale-to`.

This matches ArchR's `addGeneScoreMatrix` defaults and output exactly.

---

## Examples

### Basic usage

```bash
gatac genescore pbmc.parquet -g GRCh38.gtf.gz
```

### With quality filtering

```bash
gatac genescore pbmc.parquet -g GRCh38.gtf.gz \
    --metrics pbmc_metrics.csv \
    --filter "tsse_score > 5" \
    -o pbmc_gene_score.h5ad
```

### TSS-centred model

```bash
gatac genescore pbmc.parquet -g GRCh38.gtf.gz \
    --use-tss --gene-model "exp(-abs(x)/5000) + exp(-1)"
```

### Lower GPU memory

```bash
gatac genescore pbmc.parquet -g GRCh38.gtf.gz \
    --low-memory --cell-batch-size 5000
```

---

## Python equivalent

```python
import gatac as ga

adata_score = ga.pp.make_gene_score_matrix(
    "pbmc.parquet",
    gene_anno="GRCh38.gtf.gz",
    gene_model="exp(-abs(x)/5000) + exp(-1)",
    tile_size=500,
    extend_upstream=(1000, 100000),
    extend_downstream=(1000, 100000),
    metrics="pbmc_metrics.csv",
    filter_query="tsse_score > 5",
)
adata_score.write_h5ad("pbmc_gene_score.h5ad")
```

---

## Output AnnData structure

| Slot | Content |
|------|---------|
| `adata.X` | Sparse cell × gene normalised score matrix |
| `adata.obs` | Barcode metadata |
| `adata.var` | Gene metadata: `name`, `chrom`, `start`, `end`, `strand`, `gene_idx` (indexed by gene `name`) |
