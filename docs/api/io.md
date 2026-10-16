# Fragment I/O — `gatac.io`

The `gatac.io` namespace converts raw fragment TSV.GZ files to GATAC's Parquet
fragment format, reads them back, and converts them back to TSV. Every GATAC function that reads fragments
goes through `FragmentFile`; the format itself is described
[below](#parquet-fragment-format).

```{eval-rst}
.. currentmodule:: gatac.io

.. autosummary::
   :toctree: generated/
   :nosignatures:

   make_parquet
   make_parquet_batch
   to_tsv
   FragmentFile
   read_fragments_parquet
```

---

## Parquet fragment format

GATAC reads fragments from Parquet files that `make_parquet` (or
`gatac convert`) writes from a `fragments.tsv.gz`. This page describes how
these files store the fragments, and why.

### File layout

A `fragments.tsv.gz` repeats the chromosome and the barcode as text on every
row. On a deep library that is hundreds of millions of copies of a few dozen
chromosome names and a few hundred thousand barcodes. Kept as text, every
batch streamed to the GPU would have to decode these strings and join them
against the cells and chromosomes asked for, and that string work would take
most of the time of the fragment-streaming functions.

GATAC stores each chromosome and barcode as an integer id instead, and stores
the names once, inside the same file:

```{raw} html
:file: ../_static/fragment_layout.svg
```

- **Fragment columns.** `chrom_id`, `start`, `end`, `barcode_id`, `count`,
  one row per fragment, in the order of the input file.
- **Barcode table.** The `bc_name`, `bc_n_unique` and `bc_n_total` columns
  sit next to the fragment columns: row *i* describes barcode id *i*, with its
  full original name, its number of fragments and its read count (the sum of
  `count`). Barcode ids are ordered by decreasing fragment count, so the
  table is filled in the first rows of the file and null everywhere after;
  the null runs take almost no space. Reading the table only touches the
  first row groups.
- **Chromosome names.** They are in the file footer (`gatac.chroms`, a JSON
  list indexed by `chrom_id`), with the format version and the number of
  barcodes.

The barcode table also gives the fragment counts per barcode without
reading the fragments (`FragmentFile.barcode_counts()`).

The file is about 2.6× smaller than the `fragments.tsv.gz` it comes from:

| File, 50M-fragment library | Size |
|---|---|
| `fragments.tsv.gz` (gzip) | 557 MB |
| GATAC Parquet | 211 MB |

The names are stored once, ids and counts are dictionary-encoded, and the
sorted coordinates delta-encode to a few bits before ZSTD compression (see
[Encoding](#encoding)).

```{note}
Ids are per file: chromosomes are numbered in the order they first appear in
the `fragments.tsv.gz`, barcodes by decreasing fragment count. A function
asked for chromosomes or cells by name maps them through that file's own
`gatac.chroms` and barcode table, so the same chromosome can have different
ids in two libraries and the results do not depend on it.
```

### How GATAC reads the file

Every GATAC function that reads fragments goes through
{class}`gatac.io.FragmentFile`. To select cells, it turns the cells you ask
for into a lookup table indexed by barcode id, once per call, and each batch
on the GPU is then a gather into that table; no string column is ever built
on the GPU:

```{raw} html
:file: ../_static/fragment_lookup.svg
```

Measured on a 484M-fragment library (RTX 5070), against the same data with
names in every row:

| Function | Names in every row | Integer ids |
|---|---|---|
| `compute_metrics` | 51 s | 9 s |
| `detect_doublets` | 56 s | 13 s |
| `make_peak_matrix` | 14 s | 6 s |
| `make_tile_matrix` | 49 s | 30 s |

The rest of `make_tile_matrix` is host-side matrix assembly. The whole-file
read behind the gene matrices (`read_fragments_parquet`) builds its
categoricals from the ids and needs about a quarter of the GPU memory.

### Encoding

Each column in the [schematic](#file-layout) gives its encoding and type:
*dict* is Parquet dictionary encoding, *delta* is `DELTA_BINARY_PACKED`, and
*delta strings* is `DELTA_BYTE_ARRAY`.

Fragments are sorted by position in a `fragments.tsv.gz`, so consecutive
`start` and `end` values differ by little and delta-encode to a few bits.
Pages are compressed with ZSTD and written as data page v1, which cuDF's
chunked reader requires. Row groups hold 1,000,000 fragments
(`row_group_size`), the unit GATAC streams in.

### Reading a file outside GATAC

{class}`gatac.io.FragmentFile` gives the fragments back with names. As a
DuckDB view with the usual columns:

```python
import duckdb
import gatac as ga

con = duckdb.connect()
ga.io.FragmentFile("pbmc.parquet").register(con)  # view "fragments"
con.sql("SELECT barcode, count(*) AS n FROM fragments GROUP BY barcode")
```

`FragmentFile(path).iter_arrow()` streams Arrow tables with the same
columns, and {func}`gatac.io.to_tsv` writes the fragments back to a
`fragments.tsv.gz` in file order, BGZF-compressed so `tabix` can index it.
Barcodes keep any prefix added at conversion.

Without GATAC, any Parquet reader gets the ids and the tables:

```python
import json
import pyarrow.parquet as pq

pf = pq.ParquetFile("pbmc.parquet")
meta = pf.metadata.metadata
chroms = json.loads(meta[b"gatac.chroms"])           # chrom_id -> name
n_barcodes = int(meta[b"gatac.n_barcodes"])
barcodes = pf.read(columns=["bc_name"])["bc_name"][:n_barcodes]  # barcode_id -> name
frags = pf.read(columns=["chrom_id", "start", "end", "barcode_id", "count"])
```
