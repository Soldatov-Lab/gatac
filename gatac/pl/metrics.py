"""Plotting functions for cell quality metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Union
import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl
    import matplotlib.figure


def qc_metrics(
    metrics: "Union[str, pd.DataFrame, pl.DataFrame]",
    *,
    tsse_threshold: float = 5.0,
    n_unique_threshold: int = 1000,
    bins: int = 80,
    gridsize: int = 60,
    figsize: tuple[float, float] = (14, 4),
) -> "matplotlib.figure.Figure":
    """Plot TSS enrichment and fragment count QC metrics.

    Displays three panels:

    1. TSS enrichment score histogram
    2. log10(unique fragments) histogram
    3. TSSe vs log10(unique fragments) hexbin scatter

    Parameters
    ----------
    metrics:
        Either a path to a CSV file or a :class:`pandas.DataFrame` /
        :class:`polars.DataFrame` produced by :func:`gatac.pp.compute_metrics`.
        Must contain columns ``tsse_score`` and ``n_unique``.
    tsse_threshold:
        Vertical/horizontal line drawn on plots to mark the TSSe cut-off.
    n_unique_threshold:
        Vertical line drawn on plots to mark the unique-fragment cut-off.
    bins:
        Number of histogram bins.
    gridsize:
        Hexbin grid size for the scatter panel.
    figsize:
        Figure size in inches ``(width, height)``.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing all three panels.
    """
    import pandas as pd

    if isinstance(metrics, str):
        if not os.path.isfile(metrics):
            raise FileNotFoundError(f"Metrics file not found: {metrics}")
        df = pd.read_csv(metrics)
    else:
        # Accept polars or cuDF DataFrames as well
        try:
            import polars as pl
            if isinstance(metrics, pl.DataFrame):
                df = metrics.to_pandas()
                metrics = df
        except ImportError:
            pass

        try:
            import cudf
            if isinstance(metrics, cudf.DataFrame):
                df = metrics.to_pandas()
            else:
                df = metrics
        except ImportError:
            df = metrics

    for col in ("tsse_score", "n_unique"):
        if col not in df.columns:
            raise ValueError(f"Expected column '{col}' not found in metrics DataFrame.")

    log_unique = np.log10(np.asarray(df["n_unique"], dtype=float) + 1)
    tsse = np.asarray(df["tsse_score"], dtype=float)
    log_threshold = np.log10(n_unique_threshold)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Panel 1: TSS enrichment score
    axes[0].hist(tsse, bins=bins, color="#2ea44f", edgecolor="none")
    axes[0].axvline(
        tsse_threshold, color="#d62728", linestyle="--",
        label=f"threshold = {tsse_threshold:g}",
    )
    axes[0].set_xlabel("TSS enrichment score")
    axes[0].set_ylabel("Cells")
    axes[0].set_title("TSS enrichment")
    axes[0].legend()

    # Panel 2: Unique fragment count
    axes[1].hist(log_unique, bins=bins, color="#0969da", edgecolor="none")
    axes[1].axvline(
        log_threshold, color="#d62728", linestyle="--",
        label=f"threshold = {n_unique_threshold:,}",
    )
    axes[1].set_xlabel("log10(unique fragments)")
    axes[1].set_title("Fragment count")
    axes[1].legend()

    # Panel 3: TSSe vs log10(n_unique) hexbin
    axes[2].hexbin(log_unique, tsse, gridsize=gridsize, cmap="YlOrRd", mincnt=1)
    axes[2].axvline(log_threshold, color="#d62728", linestyle="--")
    axes[2].axhline(tsse_threshold, color="#d62728", linestyle="--")
    axes[2].set_xlabel("log10(unique fragments)")
    axes[2].set_ylabel("TSS enrichment score")
    axes[2].set_title("Fragment count vs TSSe")

    sns.despine()
    plt.tight_layout()

