import matplotlib.pyplot as plt
import numpy as np


def _motif_label(row):
    return row["name"] if row["name"] else row["motif_id"]


def _find_motif(results, motif):
    """Row index of ``motif`` matched on output_prefix, motif_id, then name."""
    for col in ("output_prefix", "motif_id", "name"):
        hit = np.flatnonzero(results[col].astype(str).to_numpy() == motif)
        if len(hit):
            return hit[0]
    raise KeyError(f"motif {motif!r} not found in output_prefix, motif_id or name")


def bindetect_volcano(res, group1=None, group2=None, label_top=5, ax=None):
    """
    Volcano plot of differential TF binding (TOBIAS BINDetect plot).

    Parameters
    ----------
    res : BINDetectResult
        Output of :func:`gatac.tl.bindetect`.
    group1, group2 : str, optional
        The comparison to plot (default: the first comparison of ``res``).
    label_top : int, default 5
        Number of highlighted motifs labelled on each side, by largest |change|.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.

    Returns
    -------
    matplotlib.axes.Axes

    Examples
    --------
    >>> res = ga.tl.bindetect(tracks, "cisBP_human.meme")
    >>> ga.pl.bindetect_volcano(res, "CD14 Mono", "CD4 Memory")
    """
    if group1 is None or group2 is None:
        group1, group2 = res.comparisons[0]
    if (group1, group2) not in res.comparisons:
        raise ValueError(f"({group1!r}, {group2!r}) is not a comparison of this result: {res.comparisons}")
    base = f"{group1}_{group2}"
    df = res.results
    change = df[f"{base}_change"].to_numpy(float)
    pv = df[f"{base}_pvalue"].to_numpy(float)
    y = -np.log10(np.clip(pv, 1e-300, None))
    hl = df[f"{base}_highlighted"].to_numpy(bool)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    ax.scatter(change[~hl], y[~hl], s=6, color="lightgrey", lw=0, label="not highlighted")
    up, down = hl & (change > 0), hl & (change < 0)
    ax.scatter(change[up], y[up], s=10, color="tab:red", lw=0, label=f"higher in {group1}")
    ax.scatter(change[down], y[down], s=10, color="tab:blue", lw=0, label=f"higher in {group2}")
    for sel in (up, down):
        idx = np.flatnonzero(sel)
        for i in idx[np.argsort(-np.abs(change[idx]))][:label_top]:
            ax.annotate(_motif_label(df.iloc[i]), (change[i], y[i]), fontsize=7,
                        xytext=(2, 2), textcoords="offset points")
    ax.axvline(0, color="black", lw=0.5)
    ax.set_xlabel(f"Differential binding score ({group1} / {group2})")
    ax.set_ylabel("-log10(p-value)")
    ax.legend(fontsize=7, frameon=False)
    ax.grid(False)
    return ax


def footprint_aggregate(tracks, res, motif, groups=None, flank=60, bound_in=None,
                        track="corrected", ax=None):
    """
    Aggregate signal around the binding sites of one motif (TOBIAS PlotAggregate).

    Sites on the minus strand are flipped, so the motif reads 5'→3' left to right.

    Parameters
    ----------
    tracks : FootprintTracks
        Output of :func:`gatac.tl.footprint`; ``track`` must have been kept.
    res : BINDetectResult
        Output of :func:`gatac.tl.bindetect` with ``return_sites=True``.
    motif : str
        Motif ``output_prefix``, ``motif_id`` or ``name``.
    groups : list of str, optional
        Groups to plot (default: all groups of ``res``).
    flank : int, default 60
        Base pairs shown on each side of the site centre.
    bound_in : str, optional
        Only use sites called bound in this group.
    track : str, default "corrected"
        Track to aggregate, e.g. ``"corrected"``, ``"uncorrected"`` or ``"footprint"``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.

    Returns
    -------
    matplotlib.axes.Axes

    Examples
    --------
    >>> ga.pl.footprint_aggregate(tracks, res, "CEBPB", groups=["CD14 Mono", "CD4 Memory"])
    """
    if res.sites is None:
        raise ValueError("res has no site table; run bindetect(..., return_sites=True)")
    if track not in tracks.values:
        raise ValueError(f"track {track!r} was not kept; available: {list(tracks.values)}")
    groups = list(groups or res.groups)
    row = res.results.iloc[_find_motif(res.results, motif)]
    sites = res.sites[res.sites["motif"] == row["output_prefix"]]
    if bound_in is not None:
        sites = sites[sites[f"{bound_in}_bound"] == 1]
    if len(sites) == 0:
        raise ValueError(f"no sites for motif {motif!r}")
    lay = tracks.layout(track)
    offs = np.arange(-flank, flank + 1)
    agg = {g: np.zeros(len(offs)) for g in groups}
    for chrom, sub in sites.groupby("chrom", observed=True):
        mid = (sub["start"].to_numpy() + (sub["end"] - sub["start"]).to_numpy() // 2)
        idx = lay.index(str(chrom), (mid[:, None] + offs[None, :]).ravel()).reshape(len(mid), -1)
        minus = (sub["strand"] == "-").to_numpy()
        idx[minus] = idx[minus, ::-1]
        ok = idx >= 0
        for g in groups:
            v = np.zeros(idx.shape)
            v[ok] = tracks.values[track][g][idx[ok]]
            agg[g] += v.sum(axis=0)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    for g in groups:
        ax.plot(offs, agg[g] / len(sites), label=g, lw=1)
    ax.axvline(0, color="grey", lw=0.5, ls="--")
    ax.set_xlabel("Distance to motif centre (bp)")
    ax.set_ylabel(f"Mean {track} signal")
    ax.set_title(f"{_motif_label(row)} ({len(sites):,} sites)")
    ax.legend(fontsize=7, frameon=False)
    ax.grid(False)
    return ax
