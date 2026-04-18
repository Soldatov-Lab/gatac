# `gatac gene`

Build a cell × gene activity matrix from an ATAC-seq fragment Parquet file.
Accessibility is scored by counting paired fragment insertions over promoter
and gene-body regions defined by a GTF annotation.

---

## Synopsis

```
gatac gene <input.parquet> -g <annotations.gtf>
           [-o OUTPUT] [--id-type TYPE]
           [--upstream N] [--downstream N]
           [--include-gene-body | --no-gene-body]
           [-m MIN_FRAGS] [-e CHROMS ...]
           [--metrics METRICS] [--filter QUERY]
           [--barcode-prefix PREFIX] [--low-memory]
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
| `-g`, `--gtf` | **required** | GTF (or GFF3) gene annotation file |
| `-o`, `--output` | `<input>_gene.h5ad` | Output h5ad path |
| `--id-type` | `gene` | Aggregation level: `gene` or `transcript` |
| `--upstream` | `2000` | Promoter extension upstream of TSS (bp) |
| `--downstream` | `0` | Extension downstream of TSS (bp) |
| `--include-gene-body` | on | Include gene body in scoring region |
| `--no-gene-body` | — | Score promoter region only |
| `-m`, `--min-fragments` | `100` | Minimum unique fragments per barcode |
| `-e`, `--exclude-chroms` | `chrM M` | Chromosomes to exclude |
| `--metrics` | — | Metrics CSV for quality-based filtering |
| `--filter` | — | Polars query string applied to metrics |
| `--barcode-prefix` | — | String prepended to barcodes |
| `--low-memory` | off | Process one Parquet row-group at a time |

---

## Counting strategy

Gene activity is scored using the **paired-insertion counting** method:

1. For each fragment, both insertion sites (start + 1 and end) are considered.
2. Insertions overlapping the scoring region (promoter ± gene body) are
   counted per gene per barcode.

This is conceptually equivalent to the SnapATAC2 gene-activity approach.

---

## Examples

### Basic usage

```bash
gatac gene pbmc.parquet -g GRCh38.gtf.gz
```

### With quality filtering

```bash
gatac gene pbmc.parquet -g GRCh38.gtf.gz \
    --metrics pbmc_metrics.csv \
    --filter "tsse_score > 5" \
    -o pbmc_gene.h5ad
```

### Promoter-only scoring

```bash
gatac gene pbmc.parquet -g GRCh38.gtf.gz \
    --upstream 2000 --downstream 200 \
    --no-gene-body \
    -o pbmc_promoter.h5ad
```

### Transcript-level aggregation

```bash
gatac gene pbmc.parquet -g GRCh38.gtf.gz --id-type transcript
```

---

## Python equivalent

```python
import gatac as ga

adata_gene = ga.pp.make_gene_matrix(
    "pbmc.parquet",
    gene_anno="GRCh38.gtf.gz",
    id_type="gene",
    upstream=2000,
    downstream=0,
    include_gene_body=True,
    metrics="pbmc_metrics.csv",
    filter_query="tsse_score > 5",
)
adata_gene.write_h5ad("pbmc_gene.h5ad")
```

---

## Output AnnData structure

| Slot | Content |
|------|---------|
| `adata.X` | Sparse cell × gene count matrix |
| `adata.obs` | Barcode metadata |
| `adata.var` | Gene metadata: `gene_name`, `gene_id`, `chrom`, `start`, `end` |
