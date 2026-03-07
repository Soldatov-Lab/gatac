"""
ATAC fragments TSV.GZ to Parquet conversion.
"""

import logging
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
