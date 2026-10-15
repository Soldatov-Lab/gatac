# `gatac convert`

Convert raw ATAC-seq fragment files from TSV.GZ to Parquet format, or repack
an existing GATAC Parquet file.  Uses DuckDB for parallel decompression and
Parquet row-group sizing tuned for efficient GPU streaming.  Files are written
with ZSTD and delta-encoded coordinates.

---

## Synopsis

```
gatac convert <input> [--output OUTPUT] [--output-dir DIR]
              [-j WORKERS] [--barcode-prefix PREFIX]
              [--layout {standard,compact}] [--chroms FILE]
```

---

## Arguments

### Positional

| Argument | Description |
|----------|-------------|
| `input` | Path (or glob pattern) to one or more TSV.GZ fragment files, or GATAC `.parquet` files to repack |

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--output`, `-o` | `<input>.parquet` | Output Parquet file path (single-file mode only) |
| `--output-dir` | same dir as input | Directory for output files in batch mode |
| `-j`, `--workers` | CPU count | Worker processes for parallel batch conversion |
| `--barcode-prefix` | — | String prepended to every barcode (e.g., `"Sample1_"`) |
| `--layout` | `standard` | `standard` (names in every row) or `compact` (integer ids plus an in-file barcode table) |
| `--chroms` | — | Compact layout: file listing the chromosome order for the ids; use the same file for every library of a cohort |

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

## Layouts

| Layout | Columns |
|--------|---------|
| `standard` | `chrom`, `start`, `end`, `barcode`, `count` |
| `compact` | `chrom_id`, `start`, `end`, `barcode_id`, `count`, plus the per-barcode table `bc_name`, `bc_n_unique`, `bc_n_total` |

In the compact layout, row *i* of the `bc_*` columns describes barcode id *i*
(its full original name, number of fragments and read count); they are null
after the last barcode. Barcode ids are ordered by decreasing fragment count.
Chromosome names are stored in the file footer (`gatac.chroms`). All GATAC
functions read both layouts and return the same results; compact files stream
several times faster on the GPU because cells and chromosomes are joined by
integer lookups. To read a compact file outside GATAC, use
`gatac.pp.FragmentFile` (e.g. `FragmentFile(path).register(duckdb_con)` gives
a DuckDB view with the standard columns).

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

### Compact layout, or repack existing Parquet files

```bash
gatac convert pbmc_fragments.tsv.gz --layout compact
# Repack a cohort with shared chromosome ids
gatac convert "parquets/*.parquet" --output-dir compact/ --layout compact --chroms chroms.txt
```

---

## Python equivalent

```python
import gatac as ga

# Single file
ga.pp.make_parquet("pbmc_fragments.tsv.gz", barcode_prefix="PBMC_")

# Compact layout, or repack an existing Parquet file
ga.pp.make_parquet("pbmc_fragments.tsv.gz", layout="compact")
ga.pp.make_parquet("pbmc.parquet", "pbmc_compact.parquet", layout="compact")

# Batch
ga.pp.make_parquet_batch(
    ["sampleA.tsv.gz", "sampleB.tsv.gz"],
    output_dir="parquets/",
    workers=4,
)
```
