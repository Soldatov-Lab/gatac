"""
ATAC fragments TSV.GZ to Parquet conversion, and back.
"""

import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Sequence

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from .fragments import NAMED_SCHEMA, FragmentFile, write_fragments

logger = logging.getLogger(__name__)


def make_parquet(
    input_path: str | Path,
    output_path: Optional[str | Path] = None,
    separator: str = '\t',
    barcode_prefix: Optional[str] = None,
    row_group_size: int = 1_000_000,
) -> Path:
    """
    Convert an ATAC fragments TSV.GZ file to GATAC Parquet.

    Chromosomes and barcodes are stored as integer ids, with an in-file
    barcode table (full original names, fragment and read counts per
    barcode); see :class:`gatac.io.FragmentFile`. Files are written with
    ZSTD, delta-encoded starts and byte-stream split lengths, barcode ids
    and counts, keeping the input row order. The
    TSV is parsed once by DuckDB into a temporary Parquet next to the
    output (about the size of the gzipped TSV), which is then streamed to
    the writer, so files larger than RAM are fine.

    Parameters
    ----------
    input_path : str or Path
        A .tsv.gz / .bed.gz (or uncompressed) fragment file with columns
        chrom, start, end, barcode, count.
    output_path : str or Path, optional
        Path for output .parquet file.
        If None, uses input filename with .parquet extension.
    separator : str
        Column separator (default: tab)
    barcode_prefix : str, optional
        Prefix to prepend to barcodes (e.g. "sample1#")
    row_group_size : int
        Number of rows per Parquet row group (default: 1_000_000).
        Larger groups reduce GPU kernel-launch overhead when streaming
        via cudf.read_parquet(row_groups=...).  The default is tuned for
        the gatac batch size of 64 row groups (~64 M rows/batch fits
        comfortably on a 12 GB GPU).

    Returns
    -------
    Path
        Path to the created Parquet file.

    Examples
    --------
    >>> import gatac as ga
    >>> out = ga.io.make_parquet("pbmc.tsv.gz")
    >>> # Or with a sample-specific barcode prefix
    >>> out = ga.io.make_parquet("pbmc.tsv.gz", barcode_prefix="sample1#")
    """
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.with_suffix('').with_suffix('.parquet')
    else:
        output_path = Path(output_path)
    if output_path.resolve() == input_path.resolve():
        raise ValueError(f"output_path must differ from the input: {output_path}")

    logger.info(f"Converting {input_path.name} to Parquet")

    if barcode_prefix:
        barcode_expr = f"'{barcode_prefix}' || column3"
    else:
        barcode_expr = "column3"
    query = f"""
        SELECT
            column0          AS chrom,
            column1::UINTEGER AS start,
            column2::UINTEGER AS "end",
            {barcode_expr}   AS barcode,
            column4::USMALLINT AS count
        FROM read_csv(
            '{input_path}',
            delim='{separator}',
            header=false,
            comment='#',
            ignore_errors=true
        )
        WHERE column0 IS NOT NULL AND column1 IS NOT NULL
          AND column2 IS NOT NULL AND column3 IS NOT NULL
    """

    # The TSV is parsed once, by a parallel COPY to a staging Parquet next to
    # the output; the barcode counts and the writer both read that file. A
    # streamed DuckDB result is produced on one thread, and counting the
    # barcodes first would parse the gzip a second time.
    stage = output_path.with_name(output_path.name + ".stage.parquet")
    try:
        with duckdb.connect() as con:
            con.execute(f"COPY ({query}) TO '{stage}' (FORMAT parquet, ROW_GROUP_SIZE {row_group_size})")
            barcode_stats = con.execute(f"""
                SELECT barcode, count(*)::BIGINT AS n_unique, sum(count)::BIGINT AS n_total
                FROM read_parquet('{stage}') GROUP BY barcode
            """).fetchdf()

        def batches():
            for b in pq.ParquetFile(stage).iter_batches(batch_size=row_group_size):
                yield pa.Table.from_batches([b]).cast(NAMED_SCHEMA)

        write_fragments(
            batches, output_path,
            row_group_size=row_group_size, barcode_stats=barcode_stats,
        )
    finally:
        stage.unlink(missing_ok=True)

    logger.info(f"Created {output_path.name}")
    return output_path


def make_parquet_batch(
    input_paths: list[str | Path],
    output_dir: Optional[str | Path] = None,
    workers: Optional[int] = None,
    separator: str = '\t',
    barcode_prefix: Optional[str | Sequence[str]] = None,
    row_group_size: int = 1_000_000,
    file_prefix: Optional[str | Sequence[str]] = None,
) -> list[Path]:
    """
    Convert multiple ATAC fragment TSV.GZ files to Parquet in parallel.

    Each file is processed in a separate worker process, so DuckDB can
    use all available CPU cores across files simultaneously. Files are
    started largest first, so that the largest do not run alone at the end.

    Parameters
    ----------
    input_paths : list of str or Path
        Paths to input .tsv.gz (or .bed.gz) files.
    output_dir : str or Path, optional
        Directory for output Parquet files.  If None, each output is
        placed in the same directory as its input.
    workers : int, optional
        Number of parallel worker processes.
        Defaults to ``min(len(input_paths), os.cpu_count())``.
    separator : str
        Column separator forwarded to :func:`make_parquet`.
    barcode_prefix : str or list of str, optional
        Prefix forwarded to :func:`make_parquet`: one for all files, or one
        per input file.
    row_group_size : int
        Row-group size forwarded to :func:`make_parquet`.
    file_prefix : str or list of str, optional
        Prepended to each output file name: one for all files, or one per
        input file. Cell Ranger names every sample's file
        ``atac_fragments.tsv.gz``, so with *output_dir* the outputs collide
        unless each gets its own prefix.

    Returns
    -------
    list of Path
        Output Parquet paths in the same order as *input_paths*.

    Raises
    ------
    ValueError
        If two inputs would be written to the same output path.
    Exception
        Re-raises the first worker exception encountered so the caller
        can handle it.

    Examples
    --------
    >>> import gatac as ga
    >>> # Convert multiple samples in parallel
    >>> paths = ga.io.make_parquet_batch(
    ...     ["sampleA.tsv.gz", "sampleB.tsv.gz"],
    ...     output_dir="parquet/",
    ... )
    >>> # With per-sample barcode prefixes (must match input order)
    >>> paths = ga.io.make_parquet_batch(
    ...     ["sampleA.tsv.gz", "sampleB.tsv.gz"],
    ...     barcode_prefix=["A_", "B_"],
    ... )
    >>> # Cell Ranger outputs, all named atac_fragments.tsv.gz: prefix the
    >>> # output files and the barcodes with the sample
    >>> paths = ga.io.make_parquet_batch(
    ...     ["A/outs/atac_fragments.tsv.gz", "B/outs/atac_fragments.tsv.gz"],
    ...     output_dir="parquet/",
    ...     file_prefix=["A_", "B_"],
    ...     barcode_prefix=["A_", "B_"],
    ... )
    """
    input_paths = [Path(p) for p in input_paths]

    def per_file(value, what):
        if isinstance(value, str) or value is None:
            return [value or ""] * len(input_paths)
        values = list(value)
        if len(values) != len(input_paths):
            raise ValueError(f"{len(values)} {what} for {len(input_paths)} input files")
        return values

    prefixes = [p or None for p in per_file(barcode_prefix, "barcode prefixes")]
    file_prefixes = per_file(file_prefix, "file prefixes")

    # Next to each input unless output_dir is given, as make_parquet does
    output_paths = [
        (Path(output_dir) if output_dir is not None else p.parent)
        / (fp + p.with_suffix('').with_suffix('.parquet').name)
        for p, fp in zip(input_paths, file_prefixes)
    ]
    seen: dict[Path, Path] = {}
    for inp, out in zip(input_paths, output_paths):
        key = out.resolve()
        if key in seen:
            raise ValueError(
                f"{seen[key]} and {inp} would both be written to {out}; "
                "give each file its own file_prefix"
            )
        seen[key] = inp
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    n_workers = min(
        len(input_paths),
        workers if workers is not None else (os.cpu_count() or 1),
    )

    logger.info(
        f"Converting {len(input_paths)} file(s) with {n_workers} worker(s)"
    )

    # Largest first, so that the largest files do not run alone at the end
    sizes = [os.path.getsize(p) for p in input_paths]
    order = sorted(range(len(input_paths)), key=lambda i: -sizes[i])

    results: list[Optional[Path]] = [None] * len(input_paths)

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_idx = {
            executor.submit(
                make_parquet,
                input_paths[i],
                output_paths[i],
                separator,
                prefixes[i],
                row_group_size,
            ): i
            for i in order
        }
        for future in as_completed(future_to_idx):
            i = future_to_idx[future]
            results[i] = future.result()  # propagate worker exceptions

    return results  # type: ignore[return-value]


def to_tsv(
    input_path: str | Path,
    output_path: Optional[str | Path] = None,
    compresslevel: int = 6,
    overwrite: bool = False,
) -> Path:
    """
    Convert a GATAC Parquet fragment file back to a fragments TSV.

    Writes ``chrom, start, end, barcode, count`` rows in file order (the
    order of the original ``fragments.tsv.gz``); see
    :meth:`gatac.io.FragmentFile.to_tsv`. A ``.gz`` output is BGZF, so
    ``tabix -p bed`` can index it.

    Parameters
    ----------
    input_path : str or Path
        Fragment Parquet file written by :func:`make_parquet`.
    output_path : str or Path, optional
        Output ``.tsv.gz`` (BGZF) or ``.tsv`` file.
        If None, uses the input filename with a .tsv.gz extension.
    compresslevel : int
        zlib compression level of the BGZF blocks (default 6).
    overwrite : bool
        Replace ``output_path`` if it exists (default False). With the default
        output name this is usually the original ``fragments.tsv.gz``.

    Returns
    -------
    Path
        Path to the written file.

    Examples
    --------
    >>> import gatac as ga
    >>> out = ga.io.to_tsv("pbmc.parquet")  # pbmc.tsv.gz
    """
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.with_suffix(".tsv.gz")
    logger.info(f"Writing {input_path.name} to TSV")
    output_path = FragmentFile(input_path).to_tsv(
        output_path, compresslevel=compresslevel, overwrite=overwrite
    )
    logger.info(f"Created {Path(output_path).name}")
    return output_path
