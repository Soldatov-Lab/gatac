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

SAMPLE = "{sample}"


def _stem(path: Path) -> str:
    """The name of the Parquet file of a fragments file, without ``.parquet``."""
    return path.with_suffix('').with_suffix('.parquet').stem


def sample_names(paths: Sequence[str | Path]) -> list[str]:
    """
    A sample name for each fragments file: the parts of the paths that
    differ between the files, compared from the file name up and joined
    with ``_``; a single file's name without its extensions.

    Examples
    --------
    >>> sample_names(["samples/s1.tsv.gz", "samples/s2.tsv.gz"])
    ['s1', 's2']
    >>> sample_names(["runs/A/outs/atac_fragments.tsv.gz",
    ...               "runs/B/outs/atac_fragments.tsv.gz"])
    ['A', 'B']
    """
    paths = [Path(p).resolve() for p in paths]
    parts = [(_stem(p), *reversed(p.parent.parts)) for p in paths]
    if len(parts) == 1:
        return [parts[0][0]]
    depth = max(len(x) for x in parts)
    parts = [x + ("",) * (depth - len(x)) for x in parts]
    differ = [k for k in range(depth) if len({x[k] for x in parts}) > 1]
    names = ["_".join(x[k] for k in reversed(differ) if x[k]) for x in parts]
    if len(set(names)) != len(names):
        raise ValueError(f"The input paths do not give distinct sample names: {names}")
    return names


def _fill(value: Optional[str], sample: str) -> Optional[str]:
    """``value`` with ``{sample}`` replaced by the sample name."""
    return value.replace(SAMPLE, sample) if value else value


def _read_sample_sheet(samples) -> tuple[list[str], Optional[list[str]], Optional[list[str]]]:
    """Input files, and output paths and barcode prefixes if given, of a sample sheet."""
    import pandas as pd

    if isinstance(samples, pd.DataFrame):
        sheet = samples.fillna("").astype(str)
    else:
        sheet = pd.read_csv(samples, dtype=str, keep_default_na=False)
    sheet.columns = [c.strip() for c in sheet.columns]
    unknown = set(sheet.columns) - {"fragments", "output", "barcode_prefix"}
    if "fragments" not in sheet.columns or unknown:
        raise ValueError(
            "A sample sheet has a fragments column, and optionally output and "
            f"barcode_prefix; got {list(sheet.columns)}"
        )

    def column(name):
        return [v.strip() for v in sheet[name]] if name in sheet.columns else None

    return column("fragments"), column("output"), column("barcode_prefix")


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
        Prefix of the sample's barcodes (e.g. "sample1_"), so that cells keep
        distinct names across samples. ``{sample}`` in it is replaced by the
        input's name without its extensions (``"{sample}_"``). It is stored
        once, in the file footer, and every reader puts it in front of the
        barcode names; the names themselves are stored as in the input,
        which :func:`to_tsv` writes back.
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
    >>> out = ga.io.make_parquet("pbmc.tsv.gz", barcode_prefix="sample1_")
    """
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.with_suffix('').with_suffix('.parquet')
    else:
        output_path = Path(output_path)
    if output_path.resolve() == input_path.resolve():
        raise ValueError(f"output_path must differ from the input: {output_path}")

    barcode_prefix = _fill(barcode_prefix, _stem(input_path))
    logger.info(f"Converting {input_path.name} to Parquet")

    query = f"""
        SELECT
            column0          AS chrom,
            column1::UINTEGER AS start,
            column2::UINTEGER AS "end",
            column3          AS barcode,
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
            barcode_prefix=barcode_prefix,
        )
    finally:
        stage.unlink(missing_ok=True)

    logger.info(f"Created {output_path.name}")
    return output_path


def make_parquet_batch(
    input_paths: Optional[list[str | Path]] = None,
    output_dir: Optional[str | Path] = None,
    workers: Optional[int] = None,
    separator: str = '\t',
    barcode_prefix: Optional[str | Sequence[str]] = None,
    row_group_size: int = 1_000_000,
    file_prefix: Optional[str | Sequence[str]] = None,
    output_paths: Optional[str | Sequence[str | Path]] = None,
    samples: "Optional[str | Path | pd.DataFrame]" = None,
) -> list[Path]:
    """
    Convert multiple ATAC fragment TSV.GZ files to Parquet in parallel.

    Each file is processed in a separate worker process, so DuckDB can
    use all available CPU cores across files simultaneously. Files are
    started largest first, so that the largest do not run alone at the end.

    ``barcode_prefix``, ``file_prefix`` and ``output_paths`` each take one
    value for all files, where ``{sample}`` stands for each file's sample
    name (:func:`sample_names`), or a list with one value per input file,
    in the order of ``input_paths``. A sample sheet (``samples``) gives the
    same per-file values as a table instead.

    Parameters
    ----------
    input_paths : list of str or Path, optional
        Paths to input .tsv.gz (or .bed.gz) files; or give ``samples``.
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
        per input file. ``{sample}`` in it is replaced by each file's sample
        name (:func:`sample_names`): ``"{sample}_"`` names the cells of
        ``samples/s1.tsv.gz`` ``s1_<barcode>``. Files whose cells share
        names are reported, as they cannot be combined into one matrix
        until each has its own prefix (:func:`set_barcode_prefix`).
    row_group_size : int
        Row-group size forwarded to :func:`make_parquet`.
    file_prefix : str or list of str, optional
        Prepended to each output file name: one for all files, or one per
        input file, with ``{sample}`` as in ``barcode_prefix``. Cell Ranger
        names every sample's file ``atac_fragments.tsv.gz``, so with
        *output_dir* the outputs collide unless each gets its own prefix,
        e.g. ``"{sample}_"``.
    output_paths : str or list of str, optional
        The output files themselves, instead of ``output_dir`` and
        ``file_prefix``: one path with ``{sample}`` (e.g.
        ``"parquets/{sample}.parquet"``), or one path per input file.
    samples : str, Path or pandas.DataFrame, optional
        A sample sheet, as a CSV file or a DataFrame, with one row per input
        file: ``fragments`` (the input file), and optionally ``output`` and
        ``barcode_prefix``, which replace ``output_paths`` and
        ``barcode_prefix``. Empty cells take the defaults. Relative paths
        are relative to the working directory.

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
    >>> # Convert multiple samples in parallel, then give each its prefix
    >>> paths = ga.io.make_parquet_batch(
    ...     ["sampleA.tsv.gz", "sampleB.tsv.gz"],
    ...     output_dir="parquet/",
    ... )
    >>> for path, prefix in zip(paths, ["A_", "B_"]):
    ...     ga.io.set_barcode_prefix(path, prefix)
    >>> # Or with per-sample barcode prefixes (must match input order)
    >>> paths = ga.io.make_parquet_batch(
    ...     ["sampleA.tsv.gz", "sampleB.tsv.gz"],
    ...     barcode_prefix=["A_", "B_"],
    ... )
    >>> # Cell Ranger outputs, all named atac_fragments.tsv.gz: prefix the
    >>> # output files and the barcodes with the sample, A and B here
    >>> paths = ga.io.make_parquet_batch(
    ...     ["A/outs/atac_fragments.tsv.gz", "B/outs/atac_fragments.tsv.gz"],
    ...     output_dir="parquet/",
    ...     file_prefix="{sample}_",
    ...     barcode_prefix="{sample}_",
    ... )  # parquet/A_atac_fragments.parquet, cells A_<barcode>, ...
    >>> # Your own names, one per file, as lists or as a sample sheet
    >>> paths = ga.io.make_parquet_batch(
    ...     ["A/outs/atac_fragments.tsv.gz", "B/outs/atac_fragments.tsv.gz"],
    ...     output_paths=["parquet/patient1.parquet", "parquet/patient2.parquet"],
    ...     barcode_prefix=["patient1_", "patient2_"],
    ... )
    >>> paths = ga.io.make_parquet_batch(samples="samples.csv")
    """
    if samples is not None:
        if input_paths is not None:
            raise ValueError("Give input_paths or samples, not both")
        input_paths, sheet_outputs, sheet_prefixes = _read_sample_sheet(samples)
        if sheet_outputs is not None:
            output_paths = sheet_outputs
        if sheet_prefixes is not None:
            barcode_prefix = sheet_prefixes
    if not input_paths:
        raise ValueError("No input files given")
    input_paths = [Path(p) for p in input_paths]

    def per_file(value, what):
        if isinstance(value, (str, Path)) or value is None:
            return [str(value) if value else ""] * len(input_paths)
        values = [str(v) if v else "" for v in value]
        if len(values) != len(input_paths):
            raise ValueError(f"{len(values)} {what} for {len(input_paths)} input files")
        return values

    names = sample_names(input_paths)
    prefixes = [_fill(p, s) or None
                for p, s in zip(per_file(barcode_prefix, "barcode prefixes"), names)]
    file_prefixes = [_fill(p, s) for p, s in zip(per_file(file_prefix, "file prefixes"), names)]
    outputs = [_fill(o, s) for o, s in zip(per_file(output_paths, "output paths"), names)]

    # Next to each input unless output_dir is given, as make_parquet does
    output_paths = [
        Path(o) if o else
        (Path(output_dir) if output_dir is not None else p.parent)
        / (fp + p.with_suffix('').with_suffix('.parquet').name)
        for p, fp, o in zip(input_paths, file_prefixes, outputs)
    ]
    seen: dict[Path, Path] = {}
    for inp, out in zip(input_paths, output_paths):
        key = out.resolve()
        if key in seen:
            raise ValueError(
                f"{seen[key]} and {inp} would both be written to {out}; give each file "
                'its own output, e.g. file_prefix="{sample}_" (--file-prefix) or '
                'output_paths="parquets/{sample}.parquet" (-o)'
            )
        seen[key] = inp
    for out in output_paths:
        out.parent.mkdir(parents=True, exist_ok=True)
    if len(input_paths) == 1:
        return [make_parquet(input_paths[0], output_paths[0], separator, prefixes[0], row_group_size)]

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

    _warn_shared_barcodes(results)
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


def set_barcode_prefix(
    path: str | Path,
    barcode_prefix: str,
    output_path: Optional[str | Path] = None,
) -> Path:
    """
    Set the barcode prefix of a GATAC Parquet fragment file.

    The prefix is stored once, in the file footer, and every reader puts it
    in front of the barcode names, so only the footer is rewritten: the
    file is renamed in place in milliseconds, whatever its size. Use it to
    give each sample's cells distinct names before combining files that
    were converted without prefixes.

    Parameters
    ----------
    path : str or Path
        Fragment Parquet file written by :func:`make_parquet`.
    barcode_prefix : str
        The new prefix, which replaces the file's; ``""`` removes it.
    output_path : str or Path, optional
        Write a copy with the new prefix instead of changing ``path``.

    Returns
    -------
    Path
        The renamed file.

    Examples
    --------
    >>> import gatac as ga
    >>> ga.io.set_barcode_prefix("sampleA.parquet", "A_")
    >>> ga.io.FragmentFile("sampleA.parquet").barcodes[:2]
    array(['A_AAACGAAAGACTCGGT-1', 'A_AAACGAAAGTGAGCTA-1'], dtype=object)
    """
    import shutil

    from ._footer import update_key_value_metadata

    path = Path(path)
    FragmentFile(path)  # a GATAC fragment file
    if output_path is not None:
        output_path = Path(output_path)
        if output_path.resolve() == path.resolve():
            raise ValueError(f"output_path must differ from the input: {output_path}")
        tmp_path = output_path.with_name(output_path.name + ".tmp")
        try:
            shutil.copyfile(path, tmp_path)
            update_key_value_metadata(tmp_path, {"gatac.barcode_prefix": barcode_prefix})
            os.replace(tmp_path, output_path)
        finally:
            tmp_path.unlink(missing_ok=True)
        path = output_path
    else:
        update_key_value_metadata(path, {"gatac.barcode_prefix": barcode_prefix})
    logger.info(f"Barcode prefix of {path.name}: {barcode_prefix!r}")
    return path


def _warn_shared_barcodes(paths: Sequence[Path]) -> None:
    """Warn if files share cell names, which keeps them from being combined."""
    seen: dict[str, str] = {}
    for p in paths:
        names = set(FragmentFile(p).barcodes)
        for other, other_names in seen.items():
            n = len(names & other_names)
            if n:
                logger.warning(
                    f"{other} and {p.name} share {n:,} barcode names; give each sample its own "
                    "prefix before building a matrix of several files (gatac rename)"
                )
                return
        seen[p.name] = names
