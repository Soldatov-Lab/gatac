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

- **Fragment columns.** `chrom_id`, `start`, `length` (`end - start`),
  `barcode_id`, `count`, one row per fragment, in the order of the input
  file. Every reader gives back `end`.
- **Barcode table.** The `bc_name`, `bc_n_unique` and `bc_n_total` columns
  sit next to the fragment columns: row *i* describes barcode id *i*, with its
  full original name, its number of fragments and its read count (the sum of
  `count`). Barcode ids are ordered by decreasing fragment count, so the
  table is filled in the first rows of the file and null everywhere after;
  the null runs take almost no space. Reading the table only touches the
  first row groups. The table is stored as columns rather than in the
  footer because the footer is not compressed, and every Parquet reader
  parses all of it each time it opens the file: with 500,000 barcodes the
  names alone would add over 10 MB to every open. As columns, the names and
  counts are compressed and only read when asked for.
- **Chromosome names.** They are in the file footer (`gatac.chroms`, a JSON
  list indexed by `chrom_id`), with the format version and the number of
  barcodes.

The barcode table also gives the fragment counts per barcode without
reading the fragments (`FragmentFile.barcode_counts()`).

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
| `peak.make_matrix` | 14 s | 6 s |
| `make_tile_matrix` | 49 s | 30 s |

The rest of `make_tile_matrix` is host-side matrix assembly. The whole-file
read behind the gene matrices (`read_fragments_parquet`) builds its
categoricals from the ids and needs about a quarter of the GPU memory.

### Encoding

Each column in the [schematic](#file-layout) gives its encoding and type:
*dict* is Parquet dictionary encoding, *delta* is `DELTA_BINARY_PACKED`,
*split* is `BYTE_STREAM_SPLIT` (byte *k* of every value stored together),
and *delta lengths* is `DELTA_LENGTH_BYTE_ARRAY` (delta-encoded string
lengths, then the strings back to back).

Fragments are sorted by position in a `fragments.tsv.gz`, so consecutive
`start` values differ by little and delta-encode to a few bits. The
fragment length, a few hundred bases, carries the rest of the coordinates:
it costs fewer bits than the `end`, whose consecutive values vary more.
Barcode ids are numbered by decreasing fragment count, so the ids of the
cells, which hold most fragments, are small, and their high bytes almost
always 0; lengths and counts too. Byte-stream split puts those bytes
together, where ZSTD all but removes them, which brings the three columns
within a few percent of their entropy: dictionary encoding packs every id
into as many bits as the largest one needs, and ZSTD cannot recover that
from values that do not start on a byte. The barcode table is sorted by
decreasing fragment count, so its counts delta-encode the same way. Pages
are compressed with ZSTD and written as data page v1, which cuDF's chunked
reader requires; the footer is not compressed. Row groups hold 1,000,000
fragments (`row_group_size`), the unit GATAC streams in.

```{note}
`length` and `count` are uint32 although they fit 16 bits: cuDF (26.02)
fails to decode byte-stream split on uint16, and the two zero bytes take
almost no space. DuckDB reads byte-stream split integers from version 2.0.
```

### Reading a file outside GATAC

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
Barcodes keep any prefix added at conversion.

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
barcodes = pf.read(columns=["bc_name"])["bc_name"][:n_barcodes]  # barcode_id -> name
frags = pf.read(columns=["chrom_id", "start", "length", "barcode_id", "count"])
end = pc.add(frags["start"], frags["length"])
```
