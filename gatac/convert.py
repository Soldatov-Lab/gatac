"""
ATAC fragments TSV.GZ to Parquet conversion.
"""

import logging
import io
import os
from pathlib import Path
from typing import Optional

import rapidgzip
import polars as pl
import pyarrow.parquet as pq
from tqdm import tqdm

logger = logging.getLogger(__name__)


def make_parquet(
    input_path: str | Path,
    output_path: Optional[str | Path] = None,
    chunk_size: int = 1024 * 1024 * 1024,
    separator: str = '\t',
    progress: bool = False,
) -> Path:
    """
    Convert ATAC fragments TSV.GZ file to Parquet format.

    Uses streaming with rapidgzip for parallel decompression and polars
    for fast CSV parsing, enabling processing of files larger than RAM.

    Parameters
    ----------
    input_path : str or Path
        Path to input .tsv.gz file with ATAC fragments.
        Expected columns: chrom, start, end, barcode, count
    output_path : str or Path, optional
        Path for output .parquet file.
        If None, uses input filename with .parquet extension.
    chunk_size : int
        Chunk size in bytes for streaming (default: 1GB)
    separator : str
        Column separator (default: tab)
    progress : bool
        Whether to show a progress bar (default: False)

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

    # Schema for ATAC fragments
    schema_overrides = {
        'chrom': pl.Categorical,
        'start': pl.UInt32,
        'end': pl.UInt32,
        'barcode': pl.Categorical,
        'count': pl.UInt16
    }
    column_names = ['chrom', 'start', 'end', 'barcode', 'count']

    with rapidgzip.open(str(input_path)) as f:
        pbar = tqdm(unit='B', unit_scale=True, desc=f"Decompressing {input_path.name}") if progress else None
        buffer = b""
        writer = None
        schema = None
        is_first_chunk = True
        chunk_count = 0

        while True:
            # Fill buffer
            while len(buffer) < chunk_size:
                read_size = max(chunk_size - len(buffer), 1024 * 1024)
                chunk = f.read(read_size)
                if not chunk:
                    break
                buffer += chunk
                if pbar is not None:
                    pbar.update(len(chunk))

            if not buffer:
                break

            # Find split point
            last_newline = buffer.rfind(b'\n')
            at_eof = len(buffer) < chunk_size

            if last_newline == -1:
                if at_eof:
                    current_chunk_data = buffer
                    buffer = b""
                else:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        current_chunk_data = buffer
                        buffer = b""
                    else:
                        buffer += chunk
                        continue
            else:
                current_chunk_data = buffer[:last_newline + 1]
                buffer = buffer[last_newline + 1:]

            # Process chunk
            chunk_count += 1
            if is_first_chunk:
                df = pl.read_csv(
                    io.BytesIO(current_chunk_data),
                    separator=separator,
                    has_header=True,
                    new_columns=column_names,
                    schema_overrides=schema_overrides,
                    comment_prefix='#',
                    truncate_ragged_lines=True
                )
                if df.columns != column_names:
                    df = pl.read_csv(
                        io.BytesIO(current_chunk_data),
                        separator=separator,
                        has_header=False,
                        new_columns=column_names,
                        schema_overrides=schema_overrides,
                        comment_prefix='#',
                        truncate_ragged_lines=True
                    )

                schema = df.schema
                arrow_table = df.to_arrow()
                writer = pq.ParquetWriter(str(output_path), arrow_table.schema)
                writer.write_table(arrow_table)
                is_first_chunk = False
                logger.debug(f"Chunk {chunk_count}: {df.height} rows (schema initialized)")
            else:
                df = pl.read_csv(
                    io.BytesIO(current_chunk_data),
                    separator=separator,
                    has_header=False,
                    new_columns=column_names,
                    schema_overrides=schema_overrides,
                    comment_prefix='#',
                    truncate_ragged_lines=True
                )
                if df.height > 0:
                    writer.write_table(df.to_arrow())
                    logger.debug(f"Chunk {chunk_count}: {df.height} rows")

        if pbar is not None:
            pbar.close()

        if writer:
            writer.close()
            logger.info(f"Created {output_path.name}")
        else:
            logger.warning("No data found in input file")

    return output_path
