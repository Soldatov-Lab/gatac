"""
Readers shared by every GATAC function that takes a cell metrics table, a
BED file or a gene annotation, so they all accept the same inputs and parse
them the same way.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Mapping, Union

import cudf
import duckdb
import pandas as pd

logger = logging.getLogger(__name__)


def read_metrics(metrics, gpu: bool = False) -> Union[pd.DataFrame, cudf.DataFrame, None]:
    """
    Cell metrics as a DataFrame of its own (a copy of a DataFrame input).

    Parameters
    ----------
    metrics : str, Path, pandas.DataFrame, cudf.DataFrame, polars.DataFrame or None
        A CSV file (as written from :func:`gatac.pp.compute_metrics`) or a
        loaded table.
    gpu : bool
        Return a cuDF DataFrame instead of a pandas one (default False).

    Returns
    -------
    pandas.DataFrame, cudf.DataFrame or None
        None when ``metrics`` is None.

    Raises
    ------
    FileNotFoundError
        If ``metrics`` is a path that does not exist.
    """
    if metrics is None:
        return None
    if isinstance(metrics, (str, os.PathLike)):
        path = Path(metrics)
        if not path.exists():
            raise FileNotFoundError(f"Metrics file not found: {path}")
        logger.info(f"Loading cell metrics from {path}")
        df = pd.read_csv(path)
    elif isinstance(metrics, cudf.DataFrame):
        return metrics.copy() if gpu else metrics.to_pandas()
    elif isinstance(metrics, pd.DataFrame):
        df = metrics.copy()
    elif type(metrics).__module__.startswith("polars"):
        df = metrics.to_pandas()
    else:
        raise TypeError(
            "metrics must be a CSV path or a pandas, cuDF or Polars DataFrame, "
            f"not {type(metrics).__name__}"
        )
    return cudf.from_pandas(df) if gpu else df


_GTF_COLUMNS = {
    "chrom": "VARCHAR", "source": "VARCHAR", "feature": "VARCHAR", "start": "BIGINT",
    "end": "BIGINT", "score": "VARCHAR", "strand": "VARCHAR", "frame": "VARCHAR",
    "attribute": "VARCHAR",
}


def read_gtf(
    path: Union[str, Path],
    feature: str,
    attributes: Mapping[str, str] = {},
) -> pd.DataFrame:
    """
    The rows of one feature type of a GTF or GFF3 file, in file order.

    Parameters
    ----------
    path : str or Path
        GTF, or GFF3 when the file name contains ``.gff``; gzipped files are
        read directly.
    feature : str
        Value of the feature column to keep, e.g. ``"transcript"``.
    attributes : dict
        Output column -> attribute key, e.g. ``{"name": "gene_name"}``: the
        value of ``key "value";`` (GTF) or ``key=value;`` (GFF3), null when
        the key is absent.

    Returns
    -------
    pandas.DataFrame
        ``chrom``, ``start``, ``end`` (as in the file: 1-based, inclusive),
        ``strand`` and the ``attributes`` columns.
    """
    path = Path(path)
    gff3 = ".gff" in path.name.lower()

    def pattern(key):
        return rf"{key}=([^;]+)" if gff3 else rf'{key} "([^"]+)"'

    extract = "".join(
        f', nullif(regexp_extract(attribute, ?, 1), \'\') AS "{col}"' for col in attributes
    )
    columns = ", ".join(f"'{c}': '{t}'" for c, t in _GTF_COLUMNS.items())
    # The columns are given: DuckDB's dialect sniffer fails on the "##" header
    # lines of a GTF. GTF attribute values are quoted mid-field, so no quote
    # character either.
    query = f"""
        SELECT chrom, start, "end", strand{extract}
        FROM read_csv(?, delim='\t', header=false, comment='#', quote='', escape='',
                      auto_detect=false, columns={{{columns}}})
        WHERE feature = ?
    """
    params = [pattern(k) for k in attributes.values()] + [str(path), feature]
    with duckdb.connect() as con:
        con.execute("SET enable_progress_bar = false")
        return con.execute(query, params).df()


def read_bed(path: Union[str, Path]) -> pd.DataFrame:
    """
    The first three columns of a BED file: ``chrom`` (str), ``start``, ``end`` (int64).

    Lines starting with ``#``, ``track`` or ``browser`` are skipped, and
    chromosome names stay strings, so ``1`` matches the chromosome ``"1"``.
    Gzipped files are read directly.
    """
    df = pd.read_csv(
        path, sep="\t", header=None, usecols=[0, 1, 2], names=["chrom", "start", "end"],
        comment="#", dtype=str,
    )
    df = df[~df["chrom"].str.startswith(("track", "browser"))]
    return df.astype({"start": "int64", "end": "int64"}).reset_index(drop=True)
