# `gatac convert`

Convert raw ATAC-seq fragment files from TSV.GZ to Parquet format.  Uses
DuckDB for parallel decompression and Parquet row-group sizing tuned for
efficient GPU streaming.

---

## Synopsis

```
gatac convert <input> [--output OUTPUT] [--output-dir DIR]
              [-j WORKERS] [--barcode-prefix PREFIX]
              [--row-group-size N] [--separator SEP]
```

---

## Arguments

### Positional

| Argument | Description |
|----------|-------------|
| `input` | Path (or glob pattern) to one or more TSV.GZ fragment files |

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--output`, `-o` | `<input>.parquet` | Output Parquet file path (single-file mode only) |
| `--output-dir` | same dir as input | Directory for output files in batch mode |
| `-j`, `--workers` | CPU count | Worker processes for parallel batch conversion |
| `--barcode-prefix` | — | String prepended to every barcode (e.g., `"Sample1_"`) |
| `--row-group-size` | `1_000_000` | Row-group size written to Parquet |
| `--separator` | `\t` | Field separator in the input TSV |

---

## Input format

Expected TSV columns (no header):

```
chrom   start   end     barcode   count
chr1    10001   10200   ATCG...   1
```

Compressed with gzip (`.tsv.gz`) or uncompressed (`.tsv`) are both supported.

---

## Examples

### Single file

```bash
gatac convert pbmc_fragments.tsv.gz
# → pbmc_fragments.parquet
```

### Single file with explicit output

```bash
gatac convert pbmc_fragments.tsv.gz --output pbmc.parquet
```

### Batch conversion with barcode prefixes

```bash
gatac convert "data/*/fragments.tsv.gz" \
    --output-dir parquets/ \
    --barcode-prefix "$(basename {})"
```

### Parallel conversion with N workers

```bash
gatac convert "samples/*.tsv.gz" --output-dir out/ -j 8
```

---

## Python equivalent

```python
import gatac as ga

# Single file
ga.pp.make_parquet("pbmc_fragments.tsv.gz", barcode_prefix="PBMC_")

# Batch
ga.pp.make_parquet_batch(
    ["sampleA.tsv.gz", "sampleB.tsv.gz"],
    output_dir="parquets/",
    workers=4,
)
```
