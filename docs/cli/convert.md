# `gatac convert`

Convert raw ATAC-seq fragment files from TSV.GZ to Parquet format.  Uses
DuckDB for parallel decompression and Parquet row-group sizing tuned for
efficient GPU streaming.  Files are written with ZSTD and delta-encoded
coordinates, with chromosomes and barcodes stored as integer ids (below).

---

## Synopsis

```
gatac convert <input> [--output OUTPUT] [--output-dir DIR]
              [-j WORKERS] [--barcode-prefix PREFIX]
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

`--row-group-size` and `--separator` are available from Python
(`make_parquet(row_group_size=..., separator=...)`).

---

## Input format

Expected TSV columns (no header):

```
chrom   start   end     barcode   count
chr1    10001   10200   ATCG...   1
```

Compressed with gzip (`.tsv.gz`) or uncompressed (`.tsv`) are both supported.

---

## Output

Columns: `chrom_id`, `start`, `end`, `barcode_id`, `count`, plus the
per-barcode table `bc_name`, `bc_n_unique`, `bc_n_total`.

Row *i* of the `bc_*` columns describes barcode id *i* (its full original
name, number of fragments and read count); they are null after the last
barcode. Barcode ids are ordered by decreasing fragment count. Chromosome
names are stored in the file footer (`gatac.chroms`). GATAC joins cells and
chromosomes by integer lookups, so these files stream several times faster on
the GPU than names in every row. To read a file outside GATAC, use
`gatac.io.FragmentFile` (e.g. `FragmentFile(path).register(duckdb_con)` gives
a DuckDB view with `chrom`, `start`, `end`, `barcode`, `count`). See
[Parquet fragment format](../api/io.md#parquet-fragment-format) for how the format works.

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
ga.io.make_parquet("pbmc_fragments.tsv.gz", barcode_prefix="PBMC_")

# Batch
ga.io.make_parquet_batch(
    ["sampleA.tsv.gz", "sampleB.tsv.gz"],
    output_dir="parquets/",
    workers=4,
)
```
