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
gatac convert <input> [<input> ...] [-o OUTPUT ...] [--output-dir DIR]
              [-j WORKERS] [--barcode-prefix PREFIX ...] [--file-prefix PREFIX ...]
gatac convert --samples SHEET.csv [--output-dir DIR] [-j WORKERS]
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
| `--samples` | — | [Sample sheet](#manual-your-own-names) (CSV) instead of input files: `fragments`, and optionally `output` and `barcode_prefix`, one row per file |
| `--output`, `-o` | `<input>.parquet` (`<input>.tsv.gz` for a `.parquet` input) | Output file(s): one per input, or one with `{sample}` |
| `--output-dir` | same dir as input | Directory for the outputs not named by `-o` |
| `-j`, `--workers` | CPU count | Worker processes for parallel batch conversion |
| `--barcode-prefix` | — | Barcode prefix of the cells, stored in the file footer and changed with [`gatac rename`](rename): one per input, or one with `{sample}`, replaced by each file's [sample name](#automatic-sample); TSV inputs only |
| `--file-prefix` | — | Prefix of the output file names: one per input, or one with `{sample}`; TSV inputs only |

`-o`, `--barcode-prefix` and `--file-prefix` take one value for all files
(with `{sample}`) or one value per input file, in the order of the inputs
(a glob gives its files in sorted order), as their Python counterparts in
{func}`gatac.io.make_parquet_batch` take a string or a list.
| `--overwrite` | off | Replace existing TSV outputs; `.parquet` inputs only |

`--row-group-size` and `--separator` are available from Python
(`make_parquet(row_group_size=..., separator=...)`).

---

## Naming samples and cells

A cell's name is its barcode preceded by the file's **barcode prefix**,
stored in the file footer. Samples that go into one matrix need distinct
cell names, and 10x samples share barcodes, so give each sample its own
prefix. Converting several files without prefixes logs a warning when
their cells share names.

### Automatic: `{sample}`

`{sample}` in `--barcode-prefix` or `--file-prefix` is written as is;
`gatac convert` replaces it, file by file, with the file's sample name:
the parts of the input paths that differ between the files.

**One folder of samples**: the sample is the file name.

```
samples/
├── s1.tsv.gz
└── s2.tsv.gz
```

```bash
gatac convert "samples/*.tsv.gz" --output-dir parquets/ --barcode-prefix "{sample}_"
```

```
parquets/
├── s1.parquet    cells s1_AAACGAAC-1, s1_TTGCATGA-1, ...
└── s2.parquet    cells s2_AAACGAAC-1, s2_TTGCATGA-1, ...
```

**Cell Ranger runs**: every file is `atac_fragments.tsv.gz`, so the sample
is the run directory, and the output files need `--file-prefix` too
(without it, the command stops before converting anything, as both would be
`parquets/atac_fragments.parquet`).

```
runs/
├── A/outs/atac_fragments.tsv.gz
└── B/outs/atac_fragments.tsv.gz
```

```bash
gatac convert "runs/*/outs/atac_fragments.tsv.gz" --output-dir parquets/ \
    --file-prefix "{sample}_" --barcode-prefix "{sample}_"
```

```
parquets/
├── A_atac_fragments.parquet    cells A_AAACGAAC-1, ...
└── B_atac_fragments.parquet    cells B_AAACGAAC-1, ...
```

**Several levels**: when the files differ at more than one level, the
parts are joined with `_`.

```
batch1/
├── A/fragments.tsv.gz
└── B/fragments.tsv.gz
batch2/
└── A/fragments.tsv.gz
```

```bash
gatac convert "batch*/*/fragments.tsv.gz" --output-dir parquets/ \
    --file-prefix "{sample}_" --barcode-prefix "{sample}_"
# → batch1_A_fragments.parquet (cells batch1_A_<barcode>),
#   batch1_B_fragments.parquet, batch2_A_fragments.parquet
```

To see the names before converting, in Python:

```python
import glob
import gatac as ga

ga.io.sample_names(sorted(glob.glob("runs/*/outs/atac_fragments.tsv.gz")))
# ['A', 'B']
```

### Manual: your own names

To choose each file's name and cell prefix yourself, list them in the order
of the inputs, or write them in a sample sheet. Both work the same from the
command line and from Python, and convert the files in parallel.

```
runs/
├── A/outs/atac_fragments.tsv.gz
└── B/outs/atac_fragments.tsv.gz
```

**Lists, one value per input file:**

```bash
gatac convert runs/A/outs/atac_fragments.tsv.gz runs/B/outs/atac_fragments.tsv.gz \
    -o parquets/patient1.parquet parquets/patient2.parquet \
    --barcode-prefix patient1_ patient2_
```

```python
ga.io.make_parquet_batch(
    ["runs/A/outs/atac_fragments.tsv.gz", "runs/B/outs/atac_fragments.tsv.gz"],
    output_paths=["parquets/patient1.parquet", "parquets/patient2.parquet"],
    barcode_prefix=["patient1_", "patient2_"],
)
```

**A sample sheet**, one row per file. `fragments` is required; `output`
and `barcode_prefix` are optional columns, and an empty cell takes the
default (next to the input, or in `--output-dir`; no prefix). Relative
paths are relative to the working directory.

```text
fragments,output,barcode_prefix
runs/A/outs/atac_fragments.tsv.gz,parquets/patient1.parquet,patient1_
runs/B/outs/atac_fragments.tsv.gz,parquets/patient2.parquet,patient2_
```

```bash
gatac convert --samples samples.csv
```

```python
ga.io.make_parquet_batch(samples="samples.csv")  # or a pandas DataFrame
```

Both give:

```
parquets/
├── patient1.parquet    cells patient1_AAACGAAC-1, ...
└── patient2.parquet    cells patient2_AAACGAAC-1, ...
```

**After conversion**, with [`gatac rename`](rename), which rewrites the file
footer only, in milliseconds, so files converted without a prefix, or
with another one, need not be converted again:

```bash
gatac rename parquets/s1.parquet --barcode-prefix patient1_
```

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
columns, in the order of the original file, and the barcodes of the
original file, without the barcode prefix. A `.tsv.gz` output is BGZF-compressed,
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

### Several samples, 8 files at a time

```bash
# Default: one worker per file, up to the CPU count
gatac convert "samples/*.tsv.gz" --output-dir parquets/ -j 8 --barcode-prefix "{sample}_"
```

Naming the samples and their cells is described in
[Naming samples and cells](#naming-samples-and-cells).

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

# Batch, cells named by sample (sampleA_<barcode>, ...); a list gives
# one prefix per file instead
ga.io.make_parquet_batch(
    ["sampleA.tsv.gz", "sampleB.tsv.gz"],
    output_dir="parquets/",
    barcode_prefix="{sample}_",
    workers=4,
)

# Batch from a sample sheet (fragments, output, barcode_prefix)
ga.io.make_parquet_batch(samples="samples.csv")

# Or set a prefix after converting (rewrites the file footer only)
ga.io.set_barcode_prefix("parquets/sampleA.parquet", "A_")

# Back to TSV
ga.io.to_tsv("pbmc_fragments.parquet", "pbmc_roundtrip.tsv.gz")
```
