"""Plotting functions for fragment data."""

from __future__ import annotations

from typing import TYPE_CHECKING

import duckdb
import matplotlib.pyplot as plt
import seaborn as sns

if TYPE_CHECKING:
    import matplotlib.axes


def fragment_size_distribution(
    parquet_path: str,
    *,
    max_size: int = 1000,
    bins: int = 200,
    color: str = "#65da86",
    title: str = "Fragment size distribution",
    ax: "matplotlib.axes.Axes | None" = None,
) -> "matplotlib.axes.Axes":
    """Plot the fragment size distribution from a Parquet fragment file.

    Parameters
    ----------
    parquet_path:
        Path to the Parquet fragment file produced by :func:`gatac.pp.make_parquet`.
    max_size:
        Upper bound on fragment size (bp) to include in the histogram.
    bins:
        Number of histogram bins.
    color:
        Bar colour.
    title:
        Plot title.
    ax:
        Existing :class:`matplotlib.axes.Axes` to draw on.  A new figure is
        created when *None*.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot.

    Examples
    --------
    >>> import gatac as ga
    >>> ax = ga.pl.fragment_size_distribution("pbmc.parquet")
    """
    fragment_sizes = duckdb.sql(
        f"""
        SELECT ("end" - "start") AS fragment_size
        FROM read_parquet('{parquet_path}')
        WHERE ("end" - "start") < {max_size}
        """
    ).df()

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3))

    ax.hist(fragment_sizes["fragment_size"], bins=bins, color=color, edgecolor="none")
    ax.set_xlabel("Fragment size (bp)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    sns.despine(ax=ax)
    plt.tight_layout()

    return ax
