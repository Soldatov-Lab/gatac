# `gatac metrics`

Compute per-barcode quality metrics from a fragment Parquet file.  All
computation is GPU-accelerated using streaming row-group processing, so files
larger than GPU VRAM are handled transparently.

---

## Synopsis

```
gatac metrics <input.parquet> -g <annotations.gtf>
              [-o OUTPUT] [--min-frags N]
              [--batch-size N] [--memory-resource RESOURCE]
```

---

## Arguments

### Positional

| Argument | Description |
|----------|-------------|
| `input.parquet` | Path to the fragment Parquet file |

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `-g`, `--gtf` | **required** | GTF annotation file (used to extract TSS positions) |
| `-o`, `--output` | `<input>_metrics.csv` | Output CSV path |
| `--min-frags` | `100` | Minimum unique fragments; barcodes below this are excluded |
| `--batch-size` | `64` | Parquet row-groups processed per GPU batch |
| `--memory-resource` | `cuda-async` | RAPIDS memory resource (`cuda-async`, `managed`, `managed-pool`, `cuda`) |

---

## Computed metrics

| Column | Description |
|--------|-------------|
| `barcode` | Cell barcode |
| `tsse_score` | TSS enrichment score (signal/background ratio) |
| `n_unique` | Number of unique (deduplicated) fragments |
| `duplicate_fraction` | Fraction of total reads that are duplicates |
| `mito_fraction` | Fraction of fragments mapping to mitochondrial contigs |

---

## Examples

### Basic usage

```bash
gatac metrics pbmc.parquet -g GRCh38.gtf.gz -o pbmc_metrics.csv
```

### Adjust batch size for limited VRAM

```bash
gatac metrics pbmc.parquet -g GRCh38.gtf.gz --batch-size 32
```

### Use managed memory (helpful for very large files)

```bash
gatac metrics pbmc.parquet -g GRCh38.gtf.gz \
    --memory-resource managed-pool
```

---

## Python equivalent

```python
import gatac as ga

tss = ga.pp.load_tss_from_gtf("GRCh38.gtf.gz")
metrics = ga.pp.compute_metrics(
    "pbmc.parquet",
    tss_df=tss,
    min_unique_frags=100,
    row_groups_per_batch=64,
)
metrics.to_pandas().to_csv("pbmc_metrics.csv", index=False)
```

---

## Downstream use

The output CSV is accepted by `gatac filter`, `gatac tile`, and `gatac gene`
via the `--metrics` flag for on-the-fly quality filtering:

```bash
gatac filter pbmc.parquet \
    --metrics pbmc_metrics.csv \
    --filter "tsse_score > 5 and n_unique > 1000" \
    -o pbmc_filtered.parquet
```
