"""
GPU-accelerated filtering of ATAC fragment parquet files.

This module provides functions to filter fragment data based on cell quality metrics.
"""

import logging
import gc
from pathlib import Path
from typing import Optional, Union, List

import cudf
import cupy as cp
import pandas as pd

logger = logging.getLogger(__name__)


def cleanup_gpu_memory():
    """Force cleanup of GPU memory."""
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


def filter_fragments(
    input_parquet: Union[str, Path, List[Union[str, Path]]],
    output_parquet: Optional[Union[str, Path, List[Union[str, Path]]]] = None,
    metrics: Optional[Union[str, Path, pd.DataFrame, cudf.DataFrame]] = None,
    min_fragments_per_cell: int = 100,
    filter_query: Optional[str] = None,
    barcode_column: str = "barcode",
    barcode_prefix: Optional[str] = None,
    row_groups_per_batch: int = 64,
    chrom_sizes: Optional[Union[dict, object, str]] = None,
) -> Union[Path, List[Path]]:
    """
    Filter ATAC fragment parquet file(s) based on cell quality metrics.

    Uses GPU acceleration and streaming writes for memory-efficient processing
    of large files. Supports both single and multiple input files.

    Parameters
    ----------
    input_parquet : str, Path, or list
        Path to input parquet file(s) containing fragment data.
        Expected columns: 'chrom', 'start', 'end', 'barcode', 'count'
    output_parquet : str, Path, or list, optional
        Path for output filtered parquet file(s). If None, will use
        input name with '_filtered' suffix. For multiple inputs, can be
        None or a list of same length.
    metrics : str, Path, pandas.DataFrame, or cudf.DataFrame, optional
        Cell quality metrics to use for filtering. Can be a path to a
        CSV file, or a pre-loaded DataFrame.
    min_fragments_per_cell : int, default 100
        Minimum number of unique fragments required per cell.
    filter_query : str, optional
        Query string for filtering cells based on metrics.
    barcode_column : str, default "barcode"
        Name of the barcode column.
    barcode_prefix : str, optional
        Prefix to add to barcodes before filtering.
    row_groups_per_batch : int, default 64
        Number of parquet row groups to process per batch.
    chrom_sizes : dict, str, or genome object, optional
        Chromosome sizes for filtering. Can be:
        - String genome name (e.g., 'hg38', 'mm10') - will use built-in genome
        - Dictionary of chromosome names to sizes
        - Genome object with chrom_sizes attribute
        Only fragments on these chromosomes will be counted. If None, all 
        chromosomes are included. This matches SnapATAC2's behavior of 
        excluding non-standard contigs (GL*, KI*).

    Returns
    -------
    output_path : Path or list of Path
        Path(s) to the filtered parquet file(s).

    Examples
    --------
    >>> # Filter by minimum fragment count only
    >>> filter_fragments("fragments.parquet", min_fragments_per_cell=500)

    >>> # Filter using metrics CSV with quality threshold
    >>> filter_fragments(
    ...     "fragments.parquet",
    ...     metrics="metrics.csv",
    ...     filter_query="tsse_score > 5 and n_unique > 1000"
    ... )

    >>> # Filter multiple samples
    >>> filter_fragments(
    ...     ["sample1.parquet", "sample2.parquet"],
    ...     metrics=combined_metrics_df,
    ...     filter_query="tsse_score > 5"
    ... )
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    from .genome import get_chrom_sizes

    # Get valid chromosomes from chrom_sizes
    valid_chroms = None
    if chrom_sizes is not None:
        if isinstance(chrom_sizes, str):
            # String genome name (e.g., 'hg38')
            chrom_sizes_dict = get_chrom_sizes(chrom_sizes)
            valid_chroms = set(chrom_sizes_dict.keys())
        elif hasattr(chrom_sizes, 'chrom_sizes'):
            # Genome object with chrom_sizes attribute
            valid_chroms = set(chrom_sizes.chrom_sizes.keys())
        elif isinstance(chrom_sizes, dict):
            valid_chroms = set(chrom_sizes.keys())
        else:
            raise ValueError("chrom_sizes must be a string (genome name), dict, or object with chrom_sizes attribute")
        logger.info(f"Filtering to {len(valid_chroms)} chromosomes: {sorted(list(valid_chroms))[:5]}...")

    # Handle single vs multiple inputs
    if isinstance(input_parquet, (str, Path)):
        input_paths = [Path(input_parquet)]
        single_input = True
    else:
        input_paths = [Path(p) for p in input_parquet]
        single_input = False

    # Validate inputs exist
    for p in input_paths:
        if not p.exists():
            raise FileNotFoundError(f"Input file not found: {p}")

    # Handle output paths
    if output_parquet is None:
        output_paths = [p.with_name(p.stem + "_filtered.parquet") for p in input_paths]
    elif isinstance(output_parquet, (str, Path)):
        if single_input:
            output_paths = [Path(output_parquet)]
        else:
            raise ValueError("Multiple inputs require multiple output paths or None")
    else:
        output_paths = [Path(p) for p in output_parquet]
        if len(output_paths) != len(input_paths):
            raise ValueError(f"Number of outputs ({len(output_paths)}) must match inputs ({len(input_paths)})")

    # Load metrics if provided
    valid_barcodes = None
    if metrics is not None:
        if isinstance(metrics, (str, Path)):
            logger.info(f"Loading metrics from {metrics}")
            metrics_df = pd.read_csv(metrics)
        elif isinstance(metrics, (pd.DataFrame, cudf.DataFrame)):
            metrics_df = metrics.copy()
        else:
            raise ValueError("metrics must be a path to a CSV file or a pandas/cudf DataFrame")
        
        if barcode_column not in metrics_df.columns:
            raise ValueError(f"Barcode column '{barcode_column}' not found in metrics")
        
        if filter_query:
            logger.info(f"Applying filter: {filter_query}")
            try:
                metrics_df = metrics_df.query(filter_query)
            except Exception as e:
                raise ValueError(f"Invalid filter query '{filter_query}': {e}")
            logger.info(f"Retained {len(metrics_df):,} cells after query filter")
        
        if 'n_unique' in metrics_df.columns:
            metrics_df = metrics_df[metrics_df['n_unique'] >= min_fragments_per_cell]
            logger.info(f"Retained {len(metrics_df):,} cells after min_fragments filter")
        
        if isinstance(metrics_df, cudf.DataFrame):
            valid_barcodes = set(metrics_df[barcode_column].astype(str).to_arrow().to_pylist())
        else:
            valid_barcodes = set(metrics_df[barcode_column].astype(str).values)
        logger.info(f"Total valid barcodes from metrics: {len(valid_barcodes):,}")

    # Process each input file
    for input_path, output_path in zip(input_paths, output_paths):
        logger.info(f"Processing {input_path}")
        _filter_single_file_streaming(
            input_path=input_path,
            output_path=output_path,
            valid_barcodes=valid_barcodes,
            min_fragments_per_cell=min_fragments_per_cell,
            barcode_column=barcode_column,
            barcode_prefix=barcode_prefix,
            row_groups_per_batch=row_groups_per_batch,
            valid_chroms=valid_chroms,
        )
        logger.info(f"Wrote filtered fragments to {output_path}")

    cleanup_gpu_memory()
    
    if single_input:
        return output_paths[0]
    return output_paths


def _filter_single_file_streaming(
    input_path: Path,
    output_path: Path,
    valid_barcodes: Optional[set] = None,
    min_fragments_per_cell: int = 100,
    barcode_column: str = "barcode",
    barcode_prefix: Optional[str] = None,
    row_groups_per_batch: int = 64,
    valid_chroms: Optional[set] = None,
) -> None:
    """
    Filter a single parquet file using streaming writes.

    Parameters
    ----------
    input_path : Path
        Input parquet file path.
    output_path : Path
        Output parquet file path.
    valid_barcodes : set, optional
        Pre-computed set of valid barcodes. If None, will compute from data.
    min_fragments_per_cell : int
        Minimum fragments per cell threshold.
    barcode_column : str
        Name of barcode column.
    barcode_prefix : str, optional
        Prefix to add to barcodes.
    row_groups_per_batch : int
        Batch size for processing.
    valid_chroms : set, optional
        Set of valid chromosome names. Fragments on other chromosomes
        will not be counted toward min_fragments_per_cell threshold.
    """
    import pyarrow.parquet as pq

    # Get row group info
    pq_file = pq.ParquetFile(input_path)
    n_row_groups = pq_file.metadata.num_row_groups
    
    # First pass if no valid_barcodes
    if valid_barcodes is None:
        logger.info("Computing fragment counts per cell")
        barcode_counts = {}
        
        for batch_start in range(0, n_row_groups, row_groups_per_batch):
            batch_end = min(batch_start + row_groups_per_batch, n_row_groups)
            row_groups = list(range(batch_start, batch_end))
            
            # Read chrom column too if we need to filter by chromosome
            if valid_chroms is not None:
                df = cudf.read_parquet(input_path, row_groups=row_groups, columns=[barcode_column, 'chrom'])
                # Filter to valid chromosomes only
                df = df[df['chrom'].isin(list(valid_chroms))]
            else:
                df = cudf.read_parquet(input_path, row_groups=row_groups, columns=[barcode_column])
            
            if barcode_prefix:
                df[barcode_column] = barcode_prefix + df[barcode_column].astype(str)
            
            counts = df.groupby(barcode_column, observed=True).size().to_pandas()
            for barcode, count in counts.items():
                barcode_counts[barcode] = barcode_counts.get(barcode, 0) + count
            
            del df
            cleanup_gpu_memory()
        
        valid_barcodes = {bc for bc, count in barcode_counts.items() 
                         if count >= min_fragments_per_cell}
        logger.info(f"Found {len(valid_barcodes):,} cells with >= {min_fragments_per_cell} fragments")

    if len(valid_barcodes) == 0:
        raise ValueError("No cells passed the filtering criteria")

    valid_bc_series = cudf.Series(list(valid_barcodes))

    # Second pass: filter and stream write
    logger.info("Filtering and writing output (streaming)")
    total_frags_in = 0
    total_frags_out = 0
    writer = None

    try:
        for batch_start in range(0, n_row_groups, row_groups_per_batch):
            batch_end = min(batch_start + row_groups_per_batch, n_row_groups)
            row_groups = list(range(batch_start, batch_end))
            
            df = cudf.read_parquet(input_path, row_groups=row_groups)
            total_frags_in += len(df)
            
            if barcode_prefix:
                df[barcode_column] = barcode_prefix + df[barcode_column].astype(str)
            
            mask = df[barcode_column].isin(valid_bc_series)
            df_filtered = df[mask]
            total_frags_out += len(df_filtered)
            
            if len(df_filtered) > 0:
                # Convert to PyArrow table
                table = df_filtered.to_arrow()
                
                if writer is None:
                    writer = pq.ParquetWriter(output_path, table.schema)
                
                writer.write_table(table)
                del table
            
            del df, df_filtered
            cleanup_gpu_memory()
            
            progress = (batch_end / n_row_groups) * 100
            if batch_end % (row_groups_per_batch * 4) == 0:
                logger.info(f"Progress: {progress:.1f}%")

    finally:
        if writer is not None:
            writer.close()

    logger.info(f"Filtered {total_frags_in:,} -> {total_frags_out:,} fragments "
                f"({100 * total_frags_out / total_frags_in:.1f}% retained)")
