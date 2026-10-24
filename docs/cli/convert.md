# `gatac convert`

Convert raw ATAC-seq fragment files from TSV.GZ to Parquet format.  Uses
DuckDB for parallel decompression and Parquet row-group sizing tuned for
efficient GPU streaming.  Files are written with ZSTD, delta-encoded starts
and byte-stream split lengths, barcode ids and counts, with chromosomes and
barcodes stored as integer ids (below).
Given `.parquet` inputs, it writes them back to fragments TSV.GZ
([Back to TSV](#back-to-tsv)).

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
| `input` | Path (or glob pattern) to one or more TSV.GZ fragment files, or `.parquet` files to write back to TSV.GZ |

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--output`, `-o` | `<input>.parquet` (`<input>.tsv.gz` for a `.parquet` input) | Output file path (single-file mode only) |
| `--output-dir` | same dir as input | Directory for output files in batch mode |
| `-j`, `--workers` | CPU count | Worker processes for parallel batch conversion |
| `--barcode-prefix` | — | String prepended to every barcode (e.g., `"Sample1_"`); TSV inputs only |
| `--overwrite` | off | Replace existing TSV outputs; `.parquet` inputs only |

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

Columns: `chrom_id`, `start`, `length` (`end - start`), `barcode_id`,
`count`, plus the per-barcode table `bc_name`, `bc_n_unique`, `bc_n_total`.

Row *i* of the `bc_*` columns describes barcode id *i* (its full original
name, number of fragments and read count); they are null after the last
barcode. Barcode ids are ordered by decreasing fragment count. Chromosome
names are stored in the file footer (`gatac.chroms`). GATAC joins cells and
chromosomes by integer lookups, so these files stream several times faster on
the GPU than names in every row. To read a file outside GATAC, use
`gatac.io.FragmentFile` (e.g. `FragmentFile(path).register(duckdb_con)` gives
a DuckDB (2.0 or later) view with `chrom`, `start`, `end`, `barcode`, `count`). See
[Parquet fragment format](../api/io.md#parquet-fragment-format) for how the format works.

---

## Back to TSV

A `.parquet` input is written back to a fragments TSV with the same five
columns, in the order of the original file, and barcode names as stored
(including any `--barcode-prefix`). A `.tsv.gz` output is BGZF-compressed,
so `tabix -p bed` can index it; a `.tsv` output is plain text.

The default output, `<input>.tsv.gz`, is usually the original fragments
file the Parquet was made from, so an existing output is never replaced
without `--overwrite` (`overwrite=True` in Python). The file is written next
to the target and renamed once complete.

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

### Back to TSV

```bash
gatac convert pbmc_fragments.parquet --output pbmc_roundtrip.tsv.gz
# → pbmc_roundtrip.tsv.gz (BGZF)
tabix -p bed pbmc_roundtrip.tsv.gz
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

# Back to TSV
ga.io.to_tsv("pbmc_fragments.parquet", "pbmc_roundtrip.tsv.gz")
```
