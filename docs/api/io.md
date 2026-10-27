# Fragment I/O — `gatac.io`

The `gatac.io` namespace converts raw fragment TSV.GZ files to GATAC's Parquet
fragment format, reads them back, and converts them back to TSV. Every GATAC function that reads fragments
goes through `FragmentFile`; the format itself is described
[below](#parquet-fragment-format). Every function that reads genome sequence
goes through `Fasta`, and the BED, GTF and cell metrics readers are the ones
behind every GATAC argument taking such a file.

```{eval-rst}
.. currentmodule:: gatac.io

.. autosummary::
   :toctree: generated/
   :nosignatures:

   make_parquet
   make_parquet_batch
   sample_names
   set_barcode_prefix
   to_tsv
   FragmentFile
   read_fragments_parquet
   Fasta
   read_bed
   read_gtf
   read_metrics
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

**Row groups** hold the columns, compressed with ZSTD:

- **Fragment columns.** `chrom_id`, `start`, `length` (`end - start`),
  `barcode_id`, `count`, one row per fragment, in the order of the input
  file. Every reader gives back `end`.
- **Barcode table.** `bc_name`, `bc_n_unique` and `bc_n_total`, next to the
  fragment columns: row *i* holds barcode id *i*'s name in the original
  file, its number of fragments and its read count. Ids go by decreasing
  fragment count, so the table fills the first rows and is null after,
  which costs almost nothing; it gives the counts per barcode without
  reading the fragments (`FragmentFile.barcode_counts()`). It is stored as
  columns, compressed and read only when needed, rather than in the
  footer, which is uncompressed and parsed on every open.

The **footer** holds the file's key-value metadata, small and read on every
open:

- `gatac.format_version`: the version of this layout.
- `gatac.chroms`: the chromosome names, a JSON list indexed by `chrom_id`.
- `gatac.n_barcodes`: the number of barcodes, the rows of the barcode table.
- `gatac.barcode_prefix`: the sample's barcode prefix, empty if none. A
  cell's name is this prefix followed by its `bc_name`. The barcode table
  keeps the names of the original `fragments.tsv.gz`, so the prefix is
  changed by rewriting the footer only, in milliseconds whatever the file's
  size ({func}`gatac.io.set_barcode_prefix`, `gatac rename`), and
  {func}`gatac.io.to_tsv` writes the original file back.

```{important}
Every GATAC function, metrics and matrices included, names cells with
their prefix. A reader of the file outside GATAC must add the prefix
itself to get the same names; see
[Reading a file outside GATAC](#reading-a-file-outside-gatac).
```

The file is about 2.8× smaller than the `fragments.tsv.gz` it comes from.
On PBMC 5k (101M fragments, 477k barcodes), each column of the text file
gzipped on its own compares with the same data in the Parquet file:

| Column | `fragments.tsv.gz` (gzip) | GATAC Parquet |
|---|---|---|
| chromosome | 0.9 MB | < 0.01 MB |
| start | 145.9 MB | 52.8 MB |
| end (as `length`) | 214.1 MB | 114.5 MB |
| barcode | 458.1 MB | 170.0 MB |
| count | 37.1 MB | 31.5 MB |
| **Whole file** | **1,042 MB** | **369 MB** |

The gzip columns add up to 856 MB: gzip loses the other 186 MB to the rows
interleaving the columns, which a column store does not pay. The barcode
names, once per barcode, take 1.9 MB of the 170.0 MB; the rest is the
`barcode_id` of every fragment. `chrom_id` is dictionary-encoded, the sorted
starts delta-encode to a few bits, and lengths, barcode ids and counts are
byte-stream split, all before ZSTD compression (see [Encoding](#encoding)).

```{note}
Ids are per file: chromosomes are numbered in the order they first appear in
the `fragments.tsv.gz`, barcodes by decreasing fragment count. A function
asked for chromosomes or cells by name maps them through that file's own
`gatac.chroms` and barcode table, so the same chromosome can have different
ids in two libraries and the results do not depend on it.
```

### Encoding

| Column | Encoding ([schematic](#file-layout)) | Why it compresses |
|---|---|---|
| `chrom_id` | *dict*: dictionary | a few dozen values |
| `start` | *delta*: `DELTA_BINARY_PACKED` | fragments are sorted by position, so consecutive starts differ by little |
| `length` | *split*: `BYTE_STREAM_SPLIT` | a few hundred bases: cheaper than `end`, and its high bytes are almost always 0 |
| `barcode_id` | *split* | ids go by decreasing fragment count, so the cells' ids are small |
| `count` | *split* | small (duplicate counts), so its high bytes are 0 too |
| `bc_name` | *delta lengths*: `DELTA_LENGTH_BYTE_ARRAY` | string lengths delta-encoded, then the strings back to back |
| `bc_n_unique`, `bc_n_total` | *delta* | the table is sorted by decreasing fragment count |

Byte-stream split stores byte *k* of every value together, so the
near-constant high bytes form runs that ZSTD all but removes. Dictionary
encoding would pack the values into a few bits each, which ZSTD cannot
compress further.

Pages are compressed with ZSTD and written as data page v1, which cuDF's
chunked reader requires. Row groups (`row_group_size`) are the unit GATAC
streams to the GPU.

```{note}
`length` and `count` are uint32 although they fit 16 bits: cuDF (26.02)
fails to decode byte-stream split on uint16, and the two zero bytes take
almost no space. DuckDB reads byte-stream split integers from version 2.0.
```

## How GATAC reads the file

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
| `peak.make_matrix` | 14 s | 6 s |
| `make_tile_matrix` | 49 s | 30 s |

The rest of `make_tile_matrix` is host-side matrix assembly. The whole-file
read behind the gene matrices (`read_fragments_parquet`) builds its
categoricals from the ids and needs about a quarter of the GPU memory.

## Reading a file outside GATAC

{class}`gatac.io.FragmentFile` gives the fragments back with names. As a
DuckDB (2.0 or later) view with the usual columns:

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
Both give the cell names with the file's barcode prefix, except
{func}`gatac.io.to_tsv`, which writes the names of the original file.

Without GATAC, any Parquet reader that decodes byte-stream split
integers (pyarrow, Polars, cuDF, DuckDB 2.0 or later) gets the ids and the
tables:

```python
import json
import pyarrow.compute as pc
import pyarrow.parquet as pq

pf = pq.ParquetFile("pbmc.parquet")
meta = pf.metadata.metadata
chroms = json.loads(meta[b"gatac.chroms"])           # chrom_id -> name
n_barcodes = int(meta[b"gatac.n_barcodes"])
prefix = meta.get(b"gatac.barcode_prefix", b"").decode()
names = pf.read(columns=["bc_name"])["bc_name"][:n_barcodes]
barcodes = pc.binary_join_element_wise(prefix, names, "")  # barcode_id -> cell name
frags = pf.read(columns=["chrom_id", "start", "length", "barcode_id", "count"])
end = pc.add(frags["start"], frags["length"])
```
