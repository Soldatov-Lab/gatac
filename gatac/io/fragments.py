"""
GATAC fragment Parquet files: the layout, the reader, and the streaming
writer behind :func:`gatac.io.make_parquet`.

Chromosomes and barcodes are integer ids, and the lookup tables travel inside
the same file::

    chrom_id (uint16)  start (uint32)  end (uint32)  barcode_id (uint32)  count (uint16)
    bc_name (string)   bc_n_unique (uint32)          bc_n_total (uint32)

The ``bc_*`` columns are a per-barcode table aligned on the row number: row
``i`` holds barcode id ``i`` (its full original name, its number of fragments
and its read count) and they are null after the last barcode. Barcode ids are
ordered by decreasing fragment count. Chromosome names are in the footer
(``gatac.chroms``), indexed by ``chrom_id``.

Files keep the order of the input fragments and are written with ZSTD and
delta-encoded coordinates. Reading them never builds a string column on the
GPU: barcodes and chromosomes are joined by integer lookups instead of string
merges, which is where most of the time of the fragment-streaming functions
went on deep libraries.
"""
from __future__ import annotations

import json
import logging
import os
import struct
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterator, List, Optional, Sequence, Union

import cudf
import cupy as cp
import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pa_csv
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

FORMAT_VERSION = "1"
FRAGMENT_COLUMNS = ("chrom", "start", "end", "barcode", "count")

NAMED_SCHEMA = pa.schema([
    ("chrom", pa.string()),
    ("start", pa.uint32()),
    ("end", pa.uint32()),
    ("barcode", pa.string()),
    ("count", pa.uint16()),
])
FILE_SCHEMA = pa.schema([
    ("chrom_id", pa.uint16()),
    ("start", pa.uint32()),
    ("end", pa.uint32()),
    ("barcode_id", pa.uint32()),
    ("count", pa.uint16()),
    ("bc_name", pa.string()),
    ("bc_n_unique", pa.uint32()),
    ("bc_n_total", pa.uint32()),
])

# Stored column behind each logical fragment column
_STORED_NAME = {"chrom": "chrom_id", "barcode": "barcode_id"}


class FragmentFile:
    """
    A GATAC fragment Parquet file, as written by :func:`gatac.io.make_parquet`.

    Every GATAC function that reads fragments goes through this class. It
    streams chromosomes and barcodes as integer ids and decodes them to
    names when asked.

    Parameters
    ----------
    path : str or Path
        Fragment Parquet file written by :func:`gatac.io.make_parquet`.

    Attributes
    ----------
    path : Path
    chroms : list of str
        Chromosome names, indexed by ``chrom_id``.
    n_barcodes : int
        Number of barcodes (rows of the barcode table).
    num_row_groups : int

    Examples
    --------
    >>> import gatac as ga
    >>> ff = ga.io.FragmentFile("pbmc.parquet")
    >>> counts = ff.barcode_counts()          # barcode, n_unique, n_total
    >>> for batch in ff.iter_batches(["chrom", "start", "end"],
    ...                              chroms=["chr1", "chr2"],
    ...                              cells=counts["barcode"][:100]):
    ...     ...  # batch: cudf.DataFrame with int chrom / cell indices
    """

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        pf = pq.ParquetFile(self.path)
        self._metadata = pf.metadata
        kv = pf.metadata.metadata or {}
        if b"gatac.format_version" not in kv:
            raise ValueError(
                f"{self.path.name} is not a GATAC fragment file; convert the "
                "fragments.tsv.gz with gatac.io.make_parquet"
            )
        version = kv[b"gatac.format_version"].decode()
        if version != FORMAT_VERSION:
            raise ValueError(
                f"{self.path.name}: fragment format version {version!r} "
                f"is not supported (expected {FORMAT_VERSION!r})"
            )
        self.chroms: List[str] = json.loads(kv[b"gatac.chroms"])
        self.n_barcodes = int(kv[b"gatac.n_barcodes"])
        self._barcodes = None
        self._barcode_index = None

    def __repr__(self) -> str:
        return f"FragmentFile('{self.path}', rows={self.num_rows:,}, barcodes={self.n_barcodes:,})"

    @property
    def num_row_groups(self) -> int:
        return self._metadata.num_row_groups

    @property
    def num_rows(self) -> int:
        """Number of fragments."""
        return self._metadata.num_rows

    @property
    def barcodes(self) -> np.ndarray:
        """
        Barcode names as an object array indexed by barcode id (the stored
        barcode table: ids by decreasing fragment count).
        """
        if self._barcodes is None:
            self._barcodes = self._barcode_table(["bc_name"])["bc_name"].to_numpy(
                zero_copy_only=False
            )
        return self._barcodes

    def barcode_ids(self, barcodes: Sequence[str]) -> np.ndarray:
        """
        Barcode ids of ``barcodes`` (int64), -1 for barcodes absent from the file.
        """
        if self._barcode_index is None:
            self._barcode_index = pd.Index(self.barcodes)
        return self._barcode_index.get_indexer(pd.Index(np.asarray(barcodes, dtype=object)))

    def _barcode_table(self, columns: List[str]) -> pa.Table:
        """The per-barcode table, read from the leading row groups only."""
        row_groups, rows = [], 0
        while rows < self.n_barcodes:
            rows += self._metadata.row_group(len(row_groups)).num_rows
            row_groups.append(len(row_groups))
        table = pq.ParquetFile(self.path).read_row_groups(row_groups, columns=columns)
        return table.slice(0, self.n_barcodes).combine_chunks()

    def barcode_counts(
        self,
        chroms: Optional[Sequence[str]] = None,
        mito_chroms: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """
        Per-barcode fragment counts.

        Parameters
        ----------
        chroms : list of str, optional
            Count only fragments on these chromosomes. None counts all.
        mito_chroms : list of str, optional
            Also return ``n_mito``, the read count (sum of ``count``) on
            these chromosomes, among the counted ones.

        Returns
        -------
        pd.DataFrame
            ``barcode``, ``n_unique`` (fragments), ``n_total`` (sum of
            ``count``) and, with ``mito_chroms``, ``n_mito``. Only barcodes
            with at least one counted fragment are listed, in no particular
            order.
        """
        if chroms is None and mito_chroms is None:
            t = self._barcode_table(["bc_name", "bc_n_unique", "bc_n_total"])
            return pd.DataFrame({
                "barcode": t["bc_name"].to_numpy(zero_copy_only=False),
                "n_unique": t["bc_n_unique"].to_numpy().astype(np.int64),
                "n_total": t["bc_n_total"].to_numpy().astype(np.int64),
            })

        pos = {c: i for i, c in enumerate(self.chroms)}

        def values(names):
            return ", ".join(str(pos[c]) for c in names if c in pos)

        where = ""
        if chroms is not None:
            in_list = values(chroms)
            if not in_list:
                return _empty_counts(mito_chroms is not None)
            where = f"WHERE chrom_id IN ({in_list})"
        mito = ""
        if mito_chroms is not None:
            in_list = values(mito_chroms)
            mito = (f", sum(CASE WHEN chrom_id IN ({in_list}) THEN count ELSE 0 END)::BIGINT AS n_mito"
                    if in_list else ", 0::BIGINT AS n_mito")

        df = duckdb.connect().execute(f"""
            SELECT barcode_id AS barcode, count(*)::BIGINT AS n_unique,
                   sum(count)::BIGINT AS n_total {mito}
            FROM read_parquet('{self.path}') {where}
            GROUP BY barcode_id
        """).fetchdf()
        df["barcode"] = self.barcodes[df["barcode"].to_numpy()]
        return df

    def _row_group_batches(self, row_groups_per_batch: int) -> Iterator[List[int]]:
        n = self.num_row_groups
        for i in range(0, n, row_groups_per_batch):
            yield list(range(i, min(i + row_groups_per_batch, n)))

    def iter_batches(
        self,
        columns: Sequence[str] = ("chrom", "start", "end"),
        row_groups_per_batch: int = 64,
        chroms: Optional[Sequence[str]] = None,
        cells: Optional[Sequence[str]] = None,
    ) -> Iterator[cudf.DataFrame]:
        """
        Stream fragments onto the GPU, a batch of row groups at a time, with
        chromosomes and cells as integer indices.

        Parameters
        ----------
        columns : list of str
            Any of ``chrom``, ``start``, ``end``, ``count``, ``barcode``.
            ``chrom`` is returned as an int32 index into ``chroms`` (into
            :attr:`chroms` when ``chroms`` is None). ``barcode`` is returned
            as the uint32 barcode id, and only when ``cells`` is None.
        row_groups_per_batch : int
            Row groups read per batch (default 64).
        chroms : list of str, optional
            Keep only fragments on these chromosomes.
        cells : list of str, optional
            Keep only fragments of these barcodes, and add ``cell``: the int32
            index of the fragment's barcode in ``cells`` (first occurrence
            when a barcode is listed twice).

        Yields
        ------
        cudf.DataFrame
            One batch. Row order within a batch is not guaranteed.
        """
        columns = list(columns)
        unknown = set(columns) - set(FRAGMENT_COLUMNS)
        if unknown:
            raise ValueError(f"Unknown fragment columns: {sorted(unknown)}")
        if cells is not None and "barcode" in columns:
            raise ValueError("Request either `cells` (cell indices) or the `barcode` column, not both")
        if (chroms is not None and len(chroms) == 0) or (cells is not None and len(cells) == 0):
            return
        if cells is not None:
            cells = np.asarray(cells, dtype=object)
            first = ~pd.Index(cells).duplicated()
        need_chrom = "chrom" in columns or chroms is not None
        need_barcode = "barcode" in columns or cells is not None
        logical = [c for c in FRAGMENT_COLUMNS
                   if (c == "chrom" and need_chrom) or (c == "barcode" and need_barcode)
                   or (c in ("start", "end", "count") and c in columns)]

        read_cols = [_STORED_NAME.get(c, c) for c in logical]
        chrom_lut = cell_lut = None
        if need_chrom:
            if chroms is None:
                chrom_lut = cp.arange(len(self.chroms), dtype=cp.int32)
            else:
                pos = {c: i for i, c in enumerate(chroms)}
                chrom_lut = cp.asarray([pos.get(c, -1) for c in self.chroms], dtype=cp.int32)
        if cells is not None:
            positions = np.flatnonzero(first)
            ids = self.barcode_ids(cells[first])
            found = ids >= 0
            cell_lut = cp.full(self.n_barcodes, -1, dtype=cp.int32)
            cell_lut[cp.asarray(ids[found])] = cp.asarray(positions[found], dtype=cp.int32)

        # Each batch is built in a function so its intermediates are freed
        # before the next read, not held by the suspended generator.
        def batch(rgs):
            df = cudf.read_parquet(self.path, columns=read_cols, row_groups=rgs)
            keep = None
            out = {}
            if need_chrom:
                chrom = chrom_lut[df["chrom_id"].values]
                if chroms is not None:
                    keep = chrom >= 0
                if "chrom" in columns:
                    out["chrom"] = chrom
            for c in ("start", "end", "count"):
                if c in columns:
                    out[c] = df[c].values
            if cells is not None:
                cell = cell_lut[df["barcode_id"].values]
                keep = cell >= 0 if keep is None else keep & (cell >= 0)
                out["cell"] = cell
            elif "barcode" in columns:
                out["barcode"] = df["barcode_id"].values
            del df
            if keep is not None:
                out = {k: v[keep] for k, v in out.items()}
            return cudf.DataFrame(out)

        for rgs in self._row_group_batches(row_groups_per_batch):
            yield batch(rgs)

    def read_categorical(
        self,
        columns: Optional[Sequence[str]] = None,
        low_memory: bool = True,
    ) -> cudf.DataFrame:
        """
        Read the whole file into one GPU DataFrame with the named columns (``chrom``, ``start``, ``end``, ``barcode``, ``count``),
        ``chrom`` and ``barcode`` as categoricals (sorted categories).

        The categoricals are built from the stored integer ids, without
        materialising strings on the GPU.
        """
        columns = list(columns) if columns is not None else list(FRAGMENT_COLUMNS)
        with cudf.option_context("io.parquet.low_memory", low_memory):
            df = cudf.read_parquet(self.path, columns=[_STORED_NAME.get(c, c) for c in columns])
        out = {}
        for c in columns:
            if c == "chrom":
                out[c] = _categorical_from_ids(df.pop("chrom_id").values, np.asarray(self.chroms, dtype=object))
            elif c == "barcode":
                out[c] = _categorical_from_ids(df.pop("barcode_id").values, self.barcodes)
            else:
                out[c] = df.pop(c)
        return cudf.DataFrame(out)

    def register(self, con: duckdb.DuckDBPyConnection, name: str = "fragments") -> None:
        """
        Create a DuckDB view ``name`` with the named columns (``chrom``, ``start``, ``end``, ``barcode``, ``count``).

        The view joins the chromosome and barcode tables, so its row order is
        not the file order.
        """
        con.register(f"{name}_chroms", pa.table({
            "chrom_id": pa.array(np.arange(len(self.chroms)), pa.uint16()),
            "chrom": pa.array(self.chroms, pa.string()),
        }))
        con.register(f"{name}_barcodes", pa.table({
            "barcode_id": pa.array(np.arange(len(self.barcodes)), pa.uint32()),
            "barcode": pa.array(self.barcodes, pa.string()),
        }))
        con.execute(f"""
            CREATE OR REPLACE TEMP VIEW {name} AS
            SELECT c.chrom, f.start, f."end", b.barcode, f.count
            FROM read_parquet('{self.path}') f
            JOIN {name}_chroms c USING (chrom_id)
            JOIN {name}_barcodes b USING (barcode_id)
        """)

    def iter_arrow(self, batch_size: int = 1_000_000) -> Iterator[pa.Table]:
        """
        Stream the file on the CPU as Arrow tables with the named columns (``chrom``, ``start``, ``end``, ``barcode``, ``count``),
        in file order (ids decoded back to names).
        """
        pf = pq.ParquetFile(self.path)
        chrom_names = pa.array(self.chroms, pa.string())
        barcode_names = pa.array(self.barcodes, pa.string())
        for b in pf.iter_batches(batch_size=batch_size, columns=list(FILE_SCHEMA.names[:5])):
            yield pa.table([
                chrom_names.take(b.column("chrom_id")),
                b.column("start"),
                b.column("end"),
                barcode_names.take(b.column("barcode_id")),
                b.column("count"),
            ], schema=NAMED_SCHEMA)

    def to_tsv(
        self,
        output_path: Union[str, Path],
        compresslevel: int = 6,
        overwrite: bool = False,
    ) -> Path:
        """
        Write the fragments back to a fragments TSV, in file order.

        Rows are ``chrom, start, end, barcode, count`` with the stored barcode
        names, including any ``barcode_prefix`` given to
        :func:`gatac.io.make_parquet`. A path ending in ``.gz`` is written as
        BGZF, the blocked gzip of ``bgzip``: any gzip reader opens it, and
        ``tabix -p bed`` indexes it when the fragments are sorted by position,
        as in a Cell Ranger file. Any other path is written uncompressed.

        Parameters
        ----------
        output_path : str or Path
            Output ``.tsv.gz`` (BGZF) or ``.tsv`` file.
        compresslevel : int
            zlib compression level of the BGZF blocks (default 6).
        overwrite : bool
            Replace ``output_path`` if it exists (default False). The default
            output name of :func:`gatac.io.to_tsv` is usually the original
            ``fragments.tsv.gz``, which this protects.

        Returns
        -------
        Path
            Path to the written file.

        Examples
        --------
        >>> import gatac as ga
        >>> ga.io.FragmentFile("pbmc.parquet").to_tsv("pbmc.tsv.gz")
        """
        output_path = Path(output_path)
        if output_path.resolve() == self.path.resolve():
            raise ValueError(f"output_path must differ from the input: {output_path}")
        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"{output_path} exists; pass another output path or overwrite=True"
            )
        options = pa_csv.WriteOptions(include_header=False, delimiter="\t", quoting_style="none")
        # Write next to the target and rename on success: an existing file is
        # only replaced by a complete one
        tmp_path = output_path.with_name(output_path.name + ".tmp")
        try:
            out = (_BgzfWriter(tmp_path, compresslevel) if output_path.suffix == ".gz"
                   else open(tmp_path, "wb"))
            with out:
                for table in self.iter_arrow():
                    sink = pa.BufferOutputStream()
                    pa_csv.write_csv(table, sink, options)
                    out.write(sink.getvalue())
            os.replace(tmp_path, output_path)
        finally:
            tmp_path.unlink(missing_ok=True)
        return output_path


FRAGMENT_DTYPES = {
    'chrom': 'category',
    'start': 'uint32',
    'end': 'uint32',
    'barcode': 'category',
    'count': 'uint16'
}


def read_fragments_parquet(
    input_path: str | Path,
    low_memory: bool = True,
    columns: Optional[List[str]] = None,
) -> cudf.DataFrame:
    """
    Read ATAC fragments from Parquet file optimized for GPU memory.

    Returns the columns of FRAGMENT_DTYPES:
    ['chrom', 'start', 'end', 'barcode', 'count'], with ``chrom`` and
    ``barcode`` as categoricals built from the stored integer ids (see
    :class:`gatac.io.FragmentFile`).

    Examples
    --------
    >>> import gatac as ga
    >>> df = ga.io.read_fragments_parquet("pbmc.parquet")
    >>> df.columns.tolist()
    ['chrom', 'start', 'end', 'barcode', 'count']
    """

    if columns is None:
        columns = list(FRAGMENT_DTYPES.keys())

    df = FragmentFile(input_path).read_categorical(columns, low_memory=low_memory)

    # Ensure dtypes match our expectation to save memory
    for col, dtype in FRAGMENT_DTYPES.items():
        if col in df.columns and df[col].dtype != dtype:
            df[col] = df[col].astype(dtype)

    return df


# Uncompressed bytes per BGZF block, as in htslib: the compressed block must
# fit the 16-bit BSIZE field even when the data does not compress.
_BGZF_BLOCK = 0xFF00
_BGZF_EOF = bytes.fromhex("1f8b08040000000000ff0600424302001b0003000000000000000000")


class _BgzfWriter:
    """Write BGZF: gzip members of at most 64 KiB, compressed on a thread pool."""

    def __init__(self, path: Path, compresslevel: int = 6):
        self._fh = open(path, "wb")
        self._level = compresslevel
        self._pending = b""
        self._pool = ThreadPoolExecutor(min(8, os.cpu_count() or 1))

    def _block(self, data) -> bytes:
        c = zlib.compressobj(self._level, zlib.DEFLATED, -15)
        cdata = c.compress(data) + c.flush()
        # gzip header with the BGZF extra field: "BC", BSIZE = block size - 1
        header = struct.pack("<4BI2BH2BHH", 0x1F, 0x8B, 8, 4, 0, 0, 0xFF, 6,
                             ord("B"), ord("C"), 2, len(cdata) + 25)
        return header + cdata + struct.pack("<II", zlib.crc32(data), len(data))

    def write(self, data) -> None:
        data = memoryview(self._pending + memoryview(data).tobytes())
        full = len(data) - len(data) % _BGZF_BLOCK
        chunks = [data[i:i + _BGZF_BLOCK] for i in range(0, full, _BGZF_BLOCK)]
        for block in self._pool.map(self._block, chunks):
            self._fh.write(block)
        self._pending = data[full:].tobytes()

    def close(self, complete: bool = True) -> None:
        try:
            if complete:
                if self._pending:
                    self._fh.write(self._block(self._pending))
                self._fh.write(_BGZF_EOF)
        finally:
            self._pool.shutdown()
            self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        # No EOF marker after an error, so the file reads as truncated
        self.close(complete=exc_type is None)


def _empty_counts(with_mito: bool) -> pd.DataFrame:
    cols = {"barcode": pd.Series([], dtype=object), "n_unique": pd.Series([], dtype=np.int64),
            "n_total": pd.Series([], dtype=np.int64)}
    if with_mito:
        cols["n_mito"] = pd.Series([], dtype=np.int64)
    return pd.DataFrame(cols)


def _categorical_from_ids(ids: cp.ndarray, names: np.ndarray) -> cudf.Series:
    """
    Categorical Series from integer ids into ``names``: categories are the
    observed names, sorted, as ``astype('category')`` on the strings gives.
    """
    from cudf.core.column import as_column

    # Mark the ids in use; unweighted cp.bincount would take CuPy 14's CUB
    # histogram, which crashes above ~5M bins (raw barcode counts can reach that)
    seen = cp.zeros(len(names), dtype=cp.bool_)
    seen[ids] = True
    present = cp.flatnonzero(seen).get()
    observed = names[present]
    order = pd.Index(observed).argsort()
    remap = np.full(len(names), -1, dtype=np.int32)
    remap[present[order]] = np.arange(len(order), dtype=np.int32)
    codes = cp.asarray(remap)[ids]
    # cuDF has no public constructor for a categorical from codes; this is
    # the internal path pandas-style ``from_codes`` would take.
    col = as_column(codes)._with_type_metadata(
        cudf.CategoricalDtype(categories=list(observed[order]))
    )
    return cudf.Series._from_column(col)


# =============================================================================
# Writer
# =============================================================================

def write_fragments(
    batches: Callable[[], Iterator[pa.Table]],
    output_path: Union[str, Path],
    chroms: Optional[Sequence[str]] = None,
    row_group_size: int = 1_000_000,
    barcode_stats: Optional[pd.DataFrame] = None,
) -> Path:
    """
    Write fragments to a GATAC fragment Parquet file.

    Parameters
    ----------
    batches : callable
        Returns a fresh iterator of Arrow tables with the named columns (``chrom``, ``start``, ``end``, ``barcode``, ``count``),
        in the order to write. Called twice unless ``barcode_stats`` is given
        (once to count fragments per barcode).
    output_path : str or Path
    chroms : list of str, optional
        Chromosome order for ``chrom_id`` (``filter_fragments`` passes the
        input's, to keep its ids). Chromosomes of the data missing from the
        list are appended in order of first appearance. Default: order of
        first appearance.
    row_group_size : int
        Rows per Parquet row group (default 1,000,000).
    barcode_stats : pd.DataFrame, optional
        ``barcode``, ``n_unique``, ``n_total`` of the data
        being written, to skip the counting pass.

    Returns
    -------
    Path
    """
    output_path = Path(output_path)
    # Write next to the target and rename on success: no partial file on error
    tmp_path = output_path.with_name(output_path.name + ".tmp")
    try:
        _write_fragments(batches, tmp_path, chroms, row_group_size, barcode_stats)
        os.replace(tmp_path, output_path)
    finally:
        tmp_path.unlink(missing_ok=True)
    return output_path


def _write_fragments(batches, output_path, chroms, row_group_size, barcode_stats) -> None:
    """Body of :func:`write_fragments`, writing to ``output_path`` directly."""
    if barcode_stats is None:
        barcode_stats = _count_barcodes(batches())
    stats = barcode_stats.sort_values(["n_unique", "barcode"], ascending=[False, True], kind="stable")
    names = pa.array(stats["barcode"].to_numpy(dtype=object), pa.string())
    n_unique = pa.array(stats["n_unique"].to_numpy(), pa.uint32())
    n_total = pa.array(stats["n_total"].to_numpy(), pa.uint32())
    n_bc = len(names)
    bc_index = pd.Index(stats["barcode"].to_numpy(dtype=object))
    chrom_list = list(chroms) if chroms is not None else []
    chrom_pos = {c: i for i, c in enumerate(chrom_list)}
    n_given = len(chrom_list)

    writer = pq.ParquetWriter(
        output_path, FILE_SCHEMA,
        use_dictionary=["chrom_id", "barcode_id", "count"],
        column_encoding={"start": "DELTA_BINARY_PACKED", "end": "DELTA_BINARY_PACKED",
                         "bc_name": "DELTA_BYTE_ARRAY", "bc_n_unique": "DELTA_BINARY_PACKED",
                         "bc_n_total": "DELTA_BINARY_PACKED"},
        # Data page v1: cuDF's chunked (low_memory) reader fails on v2 data pages
        compression="zstd",
        write_page_index=True,
        dictionary_pagesize_limit=64 << 20,
    )
    n = 0
    with writer:
        for chunk in _fixed_size_chunks(batches(), row_group_size):
            k = chunk.num_rows
            chrom_id = _encode(chunk["chrom"], chrom_pos, chrom_list)
            barcode_enc = chunk["barcode"].combine_chunks().dictionary_encode()
            gid = bc_index.get_indexer(barcode_enc.dictionary.to_numpy(zero_copy_only=False))
            if (gid < 0).any():
                raise ValueError("Fragment barcode missing from the barcode statistics")
            barcode_id = gid.astype(np.uint32)[barcode_enc.indices.to_numpy()]
            lo, hi = min(n, n_bc), min(n + k, n_bc)
            pad = k - (hi - lo)
            writer.write_table(pa.table([
                pa.array(chrom_id, pa.uint16()),
                chunk["start"].cast(pa.uint32()),
                chunk["end"].cast(pa.uint32()),
                pa.array(barcode_id, pa.uint32()),
                chunk["count"].cast(pa.uint16()),
                pa.concat_arrays([names.slice(lo, hi - lo), pa.nulls(pad, pa.string())]),
                pa.concat_arrays([n_unique.slice(lo, hi - lo), pa.nulls(pad, pa.uint32())]),
                pa.concat_arrays([n_total.slice(lo, hi - lo), pa.nulls(pad, pa.uint32())]),
            ], schema=FILE_SCHEMA), row_group_size=row_group_size)
            n += k
        if n < n_bc:
            raise ValueError("The barcode statistics list barcodes that have no fragment")
        if len(chrom_list) > 65535:
            raise ValueError("More than 65,535 chromosomes are not supported (chrom_id is uint16)")
        if chroms is not None and len(chrom_list) > n_given:
            logger.warning(
                f"{len(chrom_list) - n_given} chromosome(s) not in `chroms` were appended: "
                f"{chrom_list[n_given:][:5]}"
            )
        writer.add_key_value_metadata({
            "gatac.format_version": FORMAT_VERSION,
            "gatac.chroms": json.dumps(chrom_list),
            "gatac.n_barcodes": str(n_bc),
        })
    logger.debug(f"Wrote {n:,} fragments, {n_bc:,} barcodes")


def _encode(values: pa.ChunkedArray, pos: dict, names: list) -> np.ndarray:
    """Integer ids of ``values`` into ``names``, appending unseen values in order."""
    enc = values.combine_chunks().dictionary_encode()
    for v in enc.dictionary.to_pylist():
        if v not in pos:
            pos[v] = len(names)
            names.append(v)
    lut = np.array([pos[v] for v in enc.dictionary.to_pylist()], dtype=np.int64)
    return lut[enc.indices.to_numpy()]


def _count_barcodes(batches: Iterator[pa.Table]) -> pd.DataFrame:
    """``barcode``, ``n_unique``, ``n_total`` over a stream of fragment tables."""
    parts = []
    for t in batches:
        agg = t.group_by("barcode").aggregate([("count", "count"), ("count", "sum")])
        parts.append(agg.to_pandas())
    if not parts:
        return _empty_counts(False)
    df = pd.concat(parts, ignore_index=True)
    df = df.groupby("barcode", sort=False).sum().reset_index()
    return df.rename(columns={"count_count": "n_unique", "count_sum": "n_total"})


def _fixed_size_chunks(tables: Iterator[pa.Table], size: int) -> Iterator[pa.Table]:
    """Re-slice a stream of tables into tables of exactly ``size`` rows (last may be shorter)."""
    buf, n = [], 0
    for t in tables:
        if t.num_rows == 0:
            continue
        buf.append(t)
        n += t.num_rows
        while n >= size:
            joined = pa.concat_tables(buf)
            yield joined.slice(0, size)
            rest = joined.slice(size)
            buf, n = ([rest], rest.num_rows) if rest.num_rows else ([], 0)
    if n:
        yield pa.concat_tables(buf)
