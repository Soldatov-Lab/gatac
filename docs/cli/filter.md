# `gatac filter`

Filter an ATAC-seq fragment Parquet file by barcode quality thresholds.
Accepts pre-computed metrics and a Polars query string for flexible filtering.
Output is written as a new Parquet file using GPU-accelerated streaming.

---

## Synopsis

```
gatac filter <input.parquet> [-o OUTPUT]
             [--metrics METRICS] [-m MIN_FRAGS]
             [--filter QUERY] [-g GENOME]
             [--barcode-prefix PREFIX] [--batch-size N]
```

---

## Arguments

### Positional

| Argument | Description |
|----------|-------------|
| `input.parquet` | Path to the fragment Parquet file (glob patterns supported) |

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `-o`, `--output` | `<input>_filtered.parquet` | Output Parquet path |
| `--metrics` | — | Path to metrics CSV (from `gatac metrics`) |
| `-m`, `--min-fragments` | `100` | Minimum unique fragments per barcode |
| `--filter` | — | Polars filter query string (e.g. `"tsse_score > 5"`) |
| `-g`, `--genome` | — | Genome name (`hg38`, `mm10`, …) for chromosome allowlist filtering |
| `--barcode-prefix` | — | String prepended to barcodes |
| `--batch-size` | `64` | Row-groups per GPU batch |

---

## Filter query syntax

The `--filter` argument accepts any valid Polars expression string evaluated
against the metrics DataFrame columns.  Supported operators: `>`, `<`, `>=`,
`<=`, `==`, `!=`, `and`, `or`, `not`.

```bash
# Keep cells with TSSe > 5 AND at least 1000 unique fragments
--filter "tsse_score > 5 and n_unique > 1000"

# Remove high-duplication barcodes
--filter "duplicate_fraction < 0.8"

# Combine conditions
--filter "tsse_score > 4 and n_unique > 500 and mito_fraction < 0.1"
```

---

## Examples

### Filter by minimum fragment count only

```bash
gatac filter pbmc.parquet -m 500 -o pbmc_filtered.parquet
```

### Filter using pre-computed metrics

```bash
gatac filter pbmc.parquet \
    --metrics pbmc_metrics.csv \
    --filter "tsse_score > 5 and n_unique > 1000" \
    -o pbmc_filtered.parquet
```

### Filter with chromosome allowlist (removes alt contigs)

```bash
gatac filter pbmc.parquet -g hg38 -m 500 -o pbmc_filtered.parquet
```

### Batch filtering (glob)

```bash
gatac filter "samples/*.parquet" \
    --metrics "metrics/*.csv" \
    --filter "tsse_score > 5" \
    --output-dir filtered/
```

---

## Python equivalent

```python
import gatac as ga

ga.pp.filter_fragments(
    "pbmc.parquet",
    output_parquet="pbmc_filtered.parquet",
    metrics="pbmc_metrics.csv",
    filter_query="tsse_score > 5 and n_unique > 1000",
    chrom_sizes="hg38",
)
```
