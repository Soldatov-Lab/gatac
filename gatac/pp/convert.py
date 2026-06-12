"""
ATAC fragments TSV.GZ to Parquet conversion.
"""

import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import duckdb

logger = logging.getLogger(__name__)


def make_parquet(
    input_path: str | Path,
    output_path: Optional[str | Path] = None,
    separator: str = '\t',
    barcode_prefix: Optional[str] = None,
    row_group_size: int = 1_000_000,
) -> Path:
    """
    Convert ATAC fragments TSV.GZ file to Parquet format.

    Uses DuckDB for parallel decompression and parsing, enabling
    processing of files larger than RAM.

    Parameters
    ----------
    input_path : str or Path
        Path to input .tsv.gz (or .bed.gz) file with ATAC fragments.
        Expected columns: chrom, start, end, barcode, count
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
    >>> out = ga.pp.make_parquet("pbmc.tsv.gz")
    >>> # Or with a sample-specific barcode prefix
    >>> out = ga.pp.make_parquet("pbmc.tsv.gz", barcode_prefix="sample1#")
    """
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.with_suffix('').with_suffix('.parquet')
    else:
        output_path = Path(output_path)

    logger.info(f"Converting {input_path.name} to Parquet")

    if barcode_prefix:
        barcode_expr = f"'{barcode_prefix}' || column3"
    else:
        barcode_expr = "column3"

    con = duckdb.connect()
    con.execute(f"""
        COPY (
            SELECT
                column0          AS chrom,
                column1::UINTEGER AS start,
                column2::UINTEGER AS end,
                {barcode_expr}   AS barcode,
                column4::USMALLINT AS count
            FROM read_csv(
                '{input_path}',
                delim='{separator}',
                header=false,
                comment='#',
                ignore_errors=true
            )
        ) TO '{output_path}'
        (FORMAT PARQUET, COMPRESSION SNAPPY, ROW_GROUP_SIZE {row_group_size})
    """)
    con.close()

    logger.info(f"Created {output_path.name}")
    return output_path


def make_parquet_batch(
    input_paths: list[str | Path],
    output_dir: Optional[str | Path] = None,
    workers: Optional[int] = None,
    separator: str = '\t',
    barcode_prefix: Optional[str] = None,
    row_group_size: int = 1_000_000,
) -> list[Path]:
    """
    Convert multiple ATAC fragment TSV.GZ files to Parquet in parallel.

    Each file is processed in a separate worker process, so DuckDB can
    use all available CPU cores across files simultaneously.

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
    barcode_prefix : str, optional
        Prefix forwarded to :func:`make_parquet`.
    row_group_size : int
        Row-group size forwarded to :func:`make_parquet`.

    Returns
    -------
    list of Path
        Output Parquet paths in the same order as *input_paths*.

    Raises
    ------
    Exception
        Re-raises the first worker exception encountered so the caller
        can handle it.

    Examples
    --------
    >>> import gatac as ga
    >>> # Convert multiple samples in parallel
    >>> paths = ga.pp.make_parquet_batch(
    ...     ["sampleA.tsv.gz", "sampleB.tsv.gz"],
    ...     output_dir="parquet/",
    ... )
    >>> # With per-sample barcode prefixes (must match input order)
    >>> paths = ga.pp.make_parquet_batch(
    ...     ["sampleA.tsv.gz", "sampleB.tsv.gz"],
    ...     barcode_prefix=["A_", "B_"],
    ... )
    """
    input_paths = [Path(p) for p in input_paths]

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_paths: list[Optional[Path]] = [
            output_dir / p.with_suffix('').with_suffix('.parquet').name
            for p in input_paths
        ]
    else:
        output_paths = [None] * len(input_paths)

    n_workers = min(
        len(input_paths),
        workers if workers is not None else (os.cpu_count() or 1),
    )

    logger.info(
        f"Converting {len(input_paths)} file(s) with {n_workers} worker(s)"
    )

    results: list[Optional[Path]] = [None] * len(input_paths)

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_idx = {
            executor.submit(
                make_parquet,
                inp,
                out,
                separator,
                barcode_prefix,
                row_group_size,
            ): i
            for i, (inp, out) in enumerate(zip(input_paths, output_paths))
        }
        for future in as_completed(future_to_idx):
            i = future_to_idx[future]
            results[i] = future.result()  # propagate worker exceptions

    return results  # type: ignore[return-value]
