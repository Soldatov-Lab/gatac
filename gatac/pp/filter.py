"""
GPU-accelerated filtering of ATAC fragment parquet files.

This module provides functions to filter fragment data based on cell quality metrics.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Union, List

import cudf
import cupy as cp
import pandas as pd

from ._utils import async_memory_pool, cleanup_gpu_memory
from .._readers import read_metrics
from ..io.fragments import FragmentFile, _write_cells

logger = logging.getLogger(__name__)


def filter_fragments(
    input_parquet: Union[str, Path, List[Union[str, Path]]],
    output_parquet: Optional[Union[str, Path, List[Union[str, Path]]]] = None,
    metrics: Optional[Union[str, Path, pd.DataFrame, cudf.DataFrame]] = None,
    min_fragments_per_cell: int = 100,
    filter_query: Optional[str] = None,
    barcode_column: str = "barcode",
    barcode_prefix: Optional[str | List[str]] = None,
    row_groups_per_batch: int = 64,
    chrom_sizes: Optional[Union[dict, object, str]] = None,
    workers: int = 4,
) -> Union[Path, List[Path]]:
    """
    Filter ATAC fragment parquet file(s) based on cell quality metrics.

    Counts fragments per cell on the GPU and streams the filtered fragments
    to the output, so memory stays bounded on large files. Supports both
    single and multiple input files; several files are filtered at once.

    Parameters
    ----------
    input_parquet : str, Path, or list
        Path to input parquet file(s) containing fragment data.
        Expected columns: 'chrom', 'start', 'end', 'barcode', 'count'
    output_parquet : str, Path, or list, optional
        Path for output filtered parquet file(s). If None, will use
        input name with '_filtered' suffix. For multiple inputs, can be
        None or a list of same length.
    metrics : str, Path, pandas.DataFrame, cudf.DataFrame or polars.DataFrame, optional
        Cell metrics for filtering: the table returned by
        :func:`gatac.pp.compute_metrics`, or a CSV file of it.
    min_fragments_per_cell : int, default 100
        Minimum number of unique fragments required per cell.
    filter_query : str, optional
        Query string for filtering cells based on metrics.
    barcode_column : str, default "barcode"
        Name of the barcode column.
    barcode_prefix : str or list of str, optional
        Barcode prefix of the output, one for all files or one per file
        (default: the input's); see :func:`gatac.io.set_barcode_prefix`.
        Cells are matched to ``metrics`` on the input's names.
    row_groups_per_batch : int, default 64
        Unused; kept for backwards compatibility.
    chrom_sizes : dict, str, or genome object, optional
        Chromosome sizes for filtering. Can be:
        - String genome name (e.g., 'hg38', 'mm10') - will use built-in genome
        - Dictionary of chromosome names to sizes
        - Genome object with chrom_sizes attribute
        Only fragments on these chromosomes will be counted. If None, all 
        chromosomes are included. This matches SnapATAC2's behavior of 
        excluding non-standard contigs (GL*, KI*).
    workers : int, default 4
        Number of files filtered at once. Each holds up to about 2 GB of
        GPU memory.

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
    if barcode_prefix is None or isinstance(barcode_prefix, str):
        prefixes = [barcode_prefix] * len(input_paths)
    else:
        prefixes = list(barcode_prefix)
        if len(prefixes) != len(input_paths):
            raise ValueError(f"Number of barcode prefixes ({len(prefixes)}) must match inputs ({len(input_paths)})")

    # Load metrics if provided
    valid_barcodes = None
    metrics_df = read_metrics(metrics)
    if metrics_df is not None:
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
        
        valid_barcodes = cudf.Series(metrics_df[barcode_column].astype(str).to_numpy(), dtype="str")
        logger.info(f"Total valid barcodes from metrics: {len(valid_barcodes):,}")

    if valid_barcodes is not None and len(valid_barcodes) == 0:
        raise ValueError("No cells passed the filtering criteria")

    # Each file is filtered and written on the GPU (cuDF reads, encodes and
    # compresses it), a few files at a time from threads, so that one file's
    # disk reads and writes overlap another's GPU work. Largest files first,
    # so that they do not run alone at the end.
    order = sorted(range(len(input_paths)), key=lambda i: -input_paths[i].stat().st_size)
    with async_memory_pool(), ThreadPoolExecutor(min(len(input_paths), workers)) as pool:
        futures = [
            pool.submit(
                _filter_single_file_streaming,
                input_path=input_paths[i],
                output_path=output_paths[i],
                valid_barcodes=valid_barcodes,
                min_fragments_per_cell=min_fragments_per_cell,
                barcode_column=barcode_column,
                barcode_prefix=prefixes[i],
                row_groups_per_batch=row_groups_per_batch,
                valid_chroms=valid_chroms,
            )
            for i in order
        ]
        for f in futures:
            f.result()

    cleanup_gpu_memory()
    
    if single_input:
        return output_paths[0]
    return output_paths


def _filter_single_file_streaming(
    input_path: Path,
    output_path: Path,
    valid_barcodes: Optional[cudf.Series] = None,
    min_fragments_per_cell: int = 100,
    barcode_column: str = "barcode",
    barcode_prefix: Optional[str] = None,
    row_groups_per_batch: int = 64,
    valid_chroms: Optional[set] = None,
) -> None:
    """
    Filter a single parquet file using streaming writes.

    The output's per-barcode table is recomputed on the kept fragments, and
    it keeps the input's chromosome ids.

    Parameters
    ----------
    input_path : Path
        Input parquet file path.
    output_path : Path
        Output parquet file path.
    valid_barcodes : cudf.Series, optional
        Barcodes of the cells to keep. If None, cells are selected by
        ``min_fragments_per_cell``.
    min_fragments_per_cell : int
        Minimum fragments per cell threshold.
    barcode_column : str
        Name of the barcode column of the metrics; fragments always use
        ``barcode``.
    barcode_prefix : str, optional
        Barcode prefix of the output; None keeps the input's.
    row_groups_per_batch : int
        Unused; kept for backwards compatibility.
    valid_chroms : set, optional
        Set of valid chromosome names. Fragments on other chromosomes
        will not be counted toward min_fragments_per_cell threshold.
    """
    logger.info(f"Processing {input_path}")
    fragments = FragmentFile(input_path)
    chroms = sorted(valid_chroms) if valid_chroms is not None else None

    # Counts on the kept chromosomes select the cells, and those of the kept
    # cells are the output's barcode table: the writer needs no counting pass.
    # Barcode names are matched on the GPU; only the kept ones reach the host.
    counts = fragments._id_counts(chroms=chroms)
    if valid_barcodes is None:
        keep = (counts["n_unique"] >= min_fragments_per_cell).to_numpy()
    # Cells are matched on the input's names; the output stores them without
    # the prefix, which goes to its footer
    names = fragments._gpu_barcodes().take(cp.asarray(counts["barcode_id"].to_numpy()))
    if valid_barcodes is not None:
        keep = names.isin(valid_barcodes).values_host
    cells = counts[keep].copy()
    stored = names[cp.asarray(keep)].str.slice(len(fragments.barcode_prefix))
    cells["barcode"] = stored.to_arrow().to_numpy(zero_copy_only=False)
    if valid_barcodes is None:
        logger.info(f"Found {len(cells):,} cells with >= {min_fragments_per_cell} fragments")
        if len(cells) == 0:
            raise ValueError(f"No cells passed the filtering criteria in {input_path}")

    prefix = fragments.barcode_prefix if barcode_prefix is None else barcode_prefix
    n_in, n_out = _write_cells(fragments, output_path, cells, chroms=chroms, barcode_prefix=prefix)
    logger.info(f"Filtered {input_path.name}: {n_in:,} -> {n_out:,} fragments "
                f"({100 * n_out / max(n_in, 1):.1f}% retained), {len(cells):,} cells")
