"""Plotting functions for fragment data."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .._cupy_compat import bincount
from ..io.fragments import FragmentFile

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
        Path to the Parquet fragment file produced by :func:`gatac.io.make_parquet`.
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
    # Fragments per size, counted on the GPU one batch at a time. Drawing the
    # sizes that occur, weighted by their counts, gives the same bins as a
    # histogram of every fragment. 16 row groups per batch keep the GPU
    # memory near 1 GB on any library.
    n_per_size = cp.zeros(max_size, dtype=cp.int64)
    for batch in FragmentFile(parquet_path).iter_batches(["start", "end"], row_groups_per_batch=16):
        size = batch["end"].values.astype(cp.int64) - batch["start"].values.astype(cp.int64)
        n_per_size += bincount(size[(size >= 0) & (size < max_size)], minlength=max_size)
    n_per_size = n_per_size.get()
    sizes = np.flatnonzero(n_per_size)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3))

    ax.hist(sizes, bins=bins, weights=n_per_size[sizes], color=color, edgecolor="none")
    ax.set_xlabel("Fragment size (bp)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    sns.despine(ax=ax)
    plt.tight_layout()

    return ax
