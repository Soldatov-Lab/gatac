"""
AMULET doublet/multiplet detection for single-cell ATAC-seq.

GPU port of the AMULET (Atac-seq MULtiplet Estimation Tool) algorithm of
Thibodeau et al. (2021) to the GATAC data model.

The per-cell Poisson scoring, the row-sum Poisson repeat-inference, and the
BH-FDR correction follow the upstream Python source (`AMULET.py`). The
overlap detection of `FragmentFileOverlapCounter.py` (a per-cell sweep-line)
and the union/cell x region matrix of `AMULET.py` are rewritten as sort +
cumsum passes over all cells of a chromosome at once on the GPU (cupy), and
only the row and column sums of the matrix are ever materialized. The
optional repeat-filter pass is applied at the raw-read level rather than the
overlap level.

The method detects cells whose fragments show an abnormally high number of
overlapping insertions, which is characteristic of doublets or multiplets
(multiple cells captured in the same droplet).

Reference
---------
Thibodeau, A. et al. AMULET: a novel read count-based method for effective
multiplet detection from single-cell ATAC-seq data. Genome Biol 22, 252
(2021). https://doi.org/10.1186/s13059-021-02469-x
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional, Union, List

import cudf
import cupy as cp
import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.stats as stats
import statsmodels.api as sm

from ._utils import cleanup_gpu_memory
from .genome import get_chrom_sizes

logger = logging.getLogger(__name__)

_ROW_GROUPS_PER_BATCH = 64


# ---------------------------------------------------------------------------
# Interval primitives (GPU, sort + cumsum)
# ---------------------------------------------------------------------------

def _check_bits(n_bits: int, what: str):
    if n_bits > 64:
        raise ValueError(f"Cannot pack {what} into 64-bit sort keys ({n_bits} bits needed)")


def _overlap_segments(
    start: cp.ndarray,
    end: cp.ndarray,
    cell: cp.ndarray,
    overlapthresh: int,
) -> tuple:
    """
    Per-cell sweep-line overlap detection for all cells of one chromosome.

    Equivalent to running AMULET's running-sum sweep independently on the
    fragments of every cell: each fragment ``[start, end)`` contributes +1 at
    ``start`` and -1 at ``end``, events at the same position are collapsed,
    and a segment is reported for every maximal run of positions where at
    least ``overlapthresh`` fragments of the cell overlap.

    All events are sorted once by (cell, position). Because the events of
    each cell sum to zero, a global cumsum equals the per-cell running sum.

    Parameters
    ----------
    start, end : cp.ndarray of int64
        Fragment coordinates.
    cell : cp.ndarray of int64
        Cell index of each fragment.
    overlapthresh : int
        Minimum overlap count to report (i.e. expected_overlap + 1).

    Returns
    -------
    seg_cell, seg_start, seg_end : cp.ndarray of int64
    """
    empty = cp.zeros(0, dtype=cp.int64)
    if len(start) < overlapthresh:
        return empty, empty, empty

    pos_bits = max(1, int(end.max()).bit_length())
    cell_bits = max(1, int(cell.max()).bit_length())
    _check_bits(cell_bits + pos_bits + 1, "cell and position")

    # Key: cell | position | is_start. Order within a position is irrelevant
    # since events at the same position are collapsed.
    cell_key = cell.astype(cp.uint64) << np.uint64(pos_bits + 1)
    one = np.uint64(1)
    events = cp.concatenate([
        cell_key | (start.astype(cp.uint64) << one) | one,
        cell_key | (end.astype(cp.uint64) << one),
    ])
    events.sort()

    delta = (events & one).astype(cp.int32) * 2 - 1
    running = cp.cumsum(delta, dtype=cp.int32)
    cell_pos = events >> one
    del events, delta

    # Running sum after the last event of each distinct (cell, position)
    last = cp.empty(len(cell_pos), dtype=bool)
    last[:-1] = cell_pos[1:] != cell_pos[:-1]
    last[-1] = True
    running = running[last]
    cell_pos = cell_pos[last]

    # Every cell ends at 0, so segments never cross a cell boundary.
    inside = running >= overlapthresh
    prev = cp.zeros_like(inside)
    prev[1:] = inside[:-1]
    seg_open = cp.flatnonzero(inside & ~prev)
    seg_close = cp.flatnonzero(~inside & prev)

    pos_mask = np.uint64((1 << pos_bits) - 1)
    seg_cell = (cell_pos[seg_open] >> np.uint64(pos_bits)).astype(cp.int64)
    seg_start = (cell_pos[seg_open] & pos_mask).astype(cp.int64)
    seg_end = (cell_pos[seg_close] & pos_mask).astype(cp.int64)
    return seg_cell, seg_start, seg_end


def _merge_intervals(start: cp.ndarray, end: cp.ndarray) -> tuple:
    """
    Merge closed intervals ``[start, end]`` that overlap or touch.

    Parameters
    ----------
    start, end : cp.ndarray of int64
        Intervals on a single chromosome.

    Returns
    -------
    ids : cp.ndarray of int64
        Merged-region index of each input interval.
    region_start, region_end : cp.ndarray of int64
        Merged regions, sorted by position.
    """
    n = len(start)
    if n == 0:
        empty = cp.zeros(0, dtype=cp.int64)
        return empty, empty, empty

    idx_bits = max(1, (n - 1).bit_length())
    pos_bits = max(1, int(end.max()).bit_length())
    _check_bits(pos_bits + 1 + idx_bits, "position and interval index")

    # Key: position | is_end | interval index. Starts sort before ends at the
    # same position, so touching intervals merge.
    one = np.uint64(1)
    shift = np.uint64(idx_bits)
    idx = cp.arange(n, dtype=cp.uint64)
    events = cp.concatenate([
        ((start.astype(cp.uint64) << one) << shift) | idx,
        (((end.astype(cp.uint64) << one) | one) << shift) | idx,
    ])
    events.sort()

    is_end = ((events >> shift) & one).astype(bool)
    pos = (events >> (shift + one)).astype(cp.int64)
    depth = cp.cumsum(cp.where(is_end, -1, 1), dtype=cp.int64)
    opens = ~is_end & (depth == 1)
    closes = is_end & (depth == 0)
    region = cp.cumsum(opens) - 1

    ids = cp.empty(n, dtype=cp.int64)
    is_start = ~is_end
    ids[(events[is_start] & np.uint64((1 << idx_bits) - 1)).astype(cp.int64)] = region[is_start]
    return ids, pos[opens], pos[closes]


def _hits_regions(
    start: cp.ndarray,
    end: cp.ndarray,
    region_start: cp.ndarray,
    region_end: cp.ndarray,
) -> cp.ndarray:
    """Mask of intervals intersecting any of the sorted, disjoint closed regions."""
    if len(region_start) == 0:
        return cp.zeros(len(start), dtype=bool)
    # The candidate with the largest start <= end also has the largest end.
    idx = cp.searchsorted(region_start, end, side="right") - 1
    return (idx >= 0) & (region_end[cp.maximum(idx, 0)] >= start)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_repeat_regions(repeat_filter: Union[str, Path], chromosomes: List[str]) -> dict:
    """Load a BED file of known repetitive regions, merged per chromosome on the GPU."""
    logger.info(f"Loading repeat regions from {repeat_filter}")
    df = pd.read_csv(repeat_filter, sep="\t", header=None, usecols=[0, 1, 2],
                     names=["chrom", "start", "end"])
    regions = {}
    for chrom, sub in df[df["chrom"].isin(chromosomes)].groupby("chrom"):
        _, r_start, r_end = _merge_intervals(
            cp.asarray(sub["start"].values, dtype=cp.int64),
            cp.asarray(sub["end"].values, dtype=cp.int64),
        )
        regions[chrom] = (r_start, r_end)
    return regions


def _load_fragments(
    fragment_path: Path,
    chromosomes: List[str],
    barcodes: List[str],
    max_insert_size: int,
) -> dict:
    """
    Read candidate-cell fragments into GPU arrays, streaming row groups.

    Returns
    -------
    dict of cp.ndarray: chrom (int32 index into ``chromosomes``),
    cell (int32 index into ``barcodes``), start, end (uint32).
    """
    n_row_groups = pq.ParquetFile(fragment_path).metadata.num_row_groups
    chrom_table = cudf.DataFrame({
        "chrom": chromosomes,
        "chrom_idx": cp.arange(len(chromosomes), dtype=cp.int32),
    })
    barcode_table = cudf.DataFrame({
        "barcode": barcodes,
        "cell": cp.arange(len(barcodes), dtype=cp.int32),
    })

    parts = {"chrom": [], "cell": [], "start": [], "end": []}
    for i in range(0, n_row_groups, _ROW_GROUPS_PER_BATCH):
        df = cudf.read_parquet(
            fragment_path,
            columns=["chrom", "start", "end", "barcode"],
            row_groups=list(range(i, min(i + _ROW_GROUPS_PER_BATCH, n_row_groups))),
        )
        df = df[(df["end"] - df["start"]) <= max_insert_size]
        df = df.merge(chrom_table, on="chrom").merge(barcode_table, on="barcode")
        parts["chrom"].append(df["chrom_idx"].values)
        parts["cell"].append(df["cell"].values)
        parts["start"].append(df["start"].values)
        parts["end"].append(df["end"].values)
        del df

    return {k: cp.concatenate(v) for k, v in parts.items()}


def _find_overlaps(
    fragment_path: Path,
    barcodes: List[str],
    chromosomes: List[str],
    expected_overlap: int,
    max_insert_size: int,
    repeat_filter: Optional[Union[str, Path]],
    min_overlap_bp: int,
) -> tuple:
    """
    Find per-cell overlap segments and assign them to union regions.

    Returns
    -------
    region, cell : cp.ndarray of int64
        One entry per distinct (union region, cell) pair, i.e. the nonzero
        entries of AMULET's binary region x cell matrix.
    n_regions : int
        Number of union regions.
    n_overlaps : int
        Number of overlap segments.
    """
    repeats = _load_repeat_regions(repeat_filter, chromosomes) if repeat_filter else {}

    frags = _load_fragments(fragment_path, chromosomes, barcodes, max_insert_size)
    logger.info(
        f"Scanning {len(frags['start']):,} fragments on {len(chromosomes)} chromosomes "
        f"for cells with >={expected_overlap + 1} overlapping fragments..."
    )

    n_cells = len(barcodes)
    region_parts, cell_parts = [], []
    n_regions = 0
    n_overlaps = 0
    for ci, chrom in enumerate(chromosomes):
        mask = frags["chrom"] == ci
        start = frags["start"][mask].astype(cp.int64)
        end = frags["end"][mask].astype(cp.int64)
        cell = frags["cell"][mask].astype(cp.int64)

        if chrom in repeats:
            keep = ~_hits_regions(start, end, *repeats[chrom])
            start, end, cell = start[keep], end[keep], cell[keep]

        seg_cell, seg_start, seg_end = _overlap_segments(
            start, end, cell, expected_overlap + 1
        )
        if min_overlap_bp > 1:
            keep = (seg_end - seg_start + 1) >= min_overlap_bp
            seg_cell, seg_start, seg_end = seg_cell[keep], seg_start[keep], seg_end[keep]
        if len(seg_start) == 0:
            continue

        ids, region_start, _ = _merge_intervals(seg_start, seg_end)
        pairs = cp.unique(ids * n_cells + seg_cell)
        region_parts.append(pairs // n_cells + n_regions)
        cell_parts.append(pairs % n_cells)
        n_regions += len(region_start)
        n_overlaps += len(seg_start)

    del frags
    if not region_parts:
        empty = cp.zeros(0, dtype=cp.int64)
        return empty, empty, 0, 0
    return cp.concatenate(region_parts), cp.concatenate(cell_parts), n_regions, n_overlaps


# ---------------------------------------------------------------------------
# Poisson tests
# ---------------------------------------------------------------------------

def _infer_repeats(rowsum: np.ndarray, threshold: float) -> np.ndarray:
    """
    Infer repetitive regions via Poisson test on row sums.

    Returns
    -------
    np.ndarray of bool
        True for regions classified as repetitive.
    """
    if len(rowsum) == 0:
        return np.zeros(0, dtype=bool)
    rep_mean = np.mean(rowsum)
    rep_probabilities = stats.poisson.sf(rowsum, rep_mean)
    corrected_rep_probabilities = sm.stats.multipletests(
        rep_probabilities, method="fdr_bh"
    )
    return corrected_rep_probabilities[1] < threshold


def _get_doublets(colsum: np.ndarray, cell_ids: np.ndarray) -> pd.DataFrame:
    """
    Compute per-cell p-value and q-value from column sums via Poisson test.

    Returns
    -------
    pd.DataFrame
        Columns: cell_id, p_value, q_value
    """
    doublet_mean = np.mean(colsum)
    doublet_probabilities = stats.poisson.sf(colsum, doublet_mean)
    # A cell with no overlaps carries no multiplet evidence. Upstream AMULET's
    # sf(0, mean) = 1 - exp(-mean) is ~1 at typical depths, but when almost no
    # cell has an overlap (shallow samples) the mean collapses toward 0, every
    # zero-overlap cell gets p ~= mean, and the whole sample is called doublets.
    doublet_probabilities[colsum == 0] = 1.0
    if doublet_mean < 1:
        logger.warning(
            f"Mean of {doublet_mean:.3g} non-repeat overlaps per cell: too few for "
            f"the AMULET Poisson test to be informative ({int((colsum > 0).sum()):,} "
            f"of {len(colsum):,} cells have any overlap)"
        )
    corrected = sm.stats.multipletests(doublet_probabilities, method="fdr_bh")
    return pd.DataFrame({
        "cell_id": cell_ids,
        "p_value": doublet_probabilities,
        "q_value": corrected[1],
    })


def _no_doublets(cell_ids: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({
        "cell_id": cell_ids,
        "p_value": np.ones(len(cell_ids)),
        "q_value": np.ones(len(cell_ids)),
        "is_doublet": np.zeros(len(cell_ids), dtype=bool),
    })


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_AUTOSOME_RE = re.compile(r"^chr\d+$")


def _filter_autosomes(chrom_sizes: dict) -> dict:
    """Keep only canonical autosomes (chr1..chrN), drop sex chr, mitochondria, and decoys.

    Matches the default behaviour of the original AMULET v1.1 tool
    (``human_autosomes.txt``) and the typical mouse/human convention.
    """
    return {c: sz for c, sz in chrom_sizes.items() if _AUTOSOME_RE.match(c)}


def detect_doublets(
    fragment_path: Union[str, Path],
    chrom_sizes: Union[dict, str],
    barcodes: Optional[List[str]] = None,
    min_fragments: int = 100,
    expected_overlap: int = 2,
    max_insert_size: int = 900,
    q_threshold: float = 0.01,
    q_rep_threshold: float = 0.01,
    repeat_filter: Optional[Union[str, Path]] = None,
    min_overlap_bp: int = 1,
    n_threads: int = 1,
) -> pd.DataFrame:
    """
    AMULET doublet/multiplet detection from a GATAC parquet fragment file.

    Implements the original AMULET Poisson method of Thibodeau et al. (2021):
    cells with an abnormally high number of overlapping fragment insertions
    are flagged as doublets/multiplets. Overlap detection runs on the GPU.

    Parameters
    ----------
    fragment_path : str or Path
        GATAC parquet fragment file (output of ``gatac convert``).
    chrom_sizes : dict or str
        Chromosome sizes dict, or genome name (e.g. 'hg38', 'mm10').
    barcodes : list of str, optional
        Barcodes to test. If None, all barcodes with ``>= min_fragments``
        fragments are used.
    min_fragments : int
        Minimum unique fragments per cell to include (default 100).
    expected_overlap : int
        Expected number of reads overlapping (default 2).
    max_insert_size : int
        Maximum fragment insert size in bp (default 900).
    q_threshold : float
        FDR threshold for doublet calling (default 0.01).
    q_rep_threshold : float
        FDR threshold for inferring repetitive regions (default 0.01).
    repeat_filter : str or Path, optional
        BED file of known repetitive regions. Fragments intersecting them
        are dropped before overlap detection.
    min_overlap_bp : int
        Minimum overlap length in bp to retain (default 1).
    n_threads : int
        Ignored; kept for backwards compatibility. Overlap detection now
        runs on the GPU.

    Returns
    -------
    pd.DataFrame
        Per-cell results with columns: cell_id, p_value, q_value, is_doublet.

    Notes
    -----
    Only canonical autosomes (chr1..chrN) are considered: sex chromosomes,
    mitochondria, and decoy contigs are dropped, matching the default
    behaviour of the original AMULET v1.1 tool (``human_autosomes.txt``).
    AMULET's Poisson model assumes a uniform single-copy background
    signal which is not valid for chrX, chrY, chrM, or unplaced contigs.

    Examples
    --------
    >>> import gatac as ga
    >>> result = ga.pp.detect_doublets("pbmc.parquet", chrom_sizes="hg38")
    >>> result.columns.tolist()
    ['cell_id', 'p_value', 'q_value', 'is_doublet']
    >>> # Filter cells to keep only singlets
    >>> doublets = set(result.loc[result["is_doublet"], "cell_id"])
    >>> keep = adata[~adata.obs_names.isin(doublets)].copy()
    """

    if isinstance(chrom_sizes, str):
        chrom_sizes = get_chrom_sizes(chrom_sizes)

    n_before = len(chrom_sizes)
    chrom_sizes = _filter_autosomes(chrom_sizes)
    n_dropped = n_before - len(chrom_sizes)
    if n_dropped:
        logger.info(
            f"Using {len(chrom_sizes)} autosomes "
            f"(dropped {n_dropped} non-autosome: sex chr / mitochondria / decoys; "
            f"AMULET is designed for autosomes only)"
        )
    if not chrom_sizes:
        raise ValueError(
            "No autosomes found in chrom_sizes. AMULET requires chr1..chrN."
        )

    fragment_path = Path(fragment_path)
    if not fragment_path.exists():
        raise FileNotFoundError(f"Fragment file not found: {fragment_path}")

    if barcodes is None:
        logger.info(f"Auto-detecting barcodes with >={min_fragments} fragments")
        con = duckdb.connect()
        chrom_values = ", ".join(f"'{c}'" for c in chrom_sizes.keys())
        rows = con.execute(f"""
            SELECT barcode, COUNT(*) AS n_frag
            FROM read_parquet('{fragment_path}')
            WHERE chrom IN ({chrom_values})
            GROUP BY barcode
            HAVING COUNT(*) >= {min_fragments}
        """).fetchall()
        con.close()
        barcodes = [r[0] for r in rows]
        logger.info(f"Found {len(barcodes):,} candidate cells")

    if len(barcodes) == 0:
        raise ValueError("No barcodes provided or no cells pass min_fragments threshold")

    cell_ids = np.asarray(barcodes)
    region, cell, n_regions, n_overlaps = _find_overlaps(
        fragment_path=fragment_path,
        barcodes=list(barcodes),
        chromosomes=list(chrom_sizes.keys()),
        expected_overlap=expected_overlap,
        max_insert_size=max_insert_size,
        repeat_filter=repeat_filter,
        min_overlap_bp=min_overlap_bp,
    )

    if n_overlaps == 0:
        result = _no_doublets(cell_ids)
    else:
        logger.info(
            f"Found {n_overlaps:,} overlap segments in {n_regions:,} union regions"
        )
        rowsum = cp.bincount(region, minlength=n_regions).get()

        logger.info("Inferring repetitive regions")
        rep_mask = _infer_repeats(rowsum, q_rep_threshold)
        keep = ~cp.asarray(rep_mask)[region]

        if not bool(keep.any()):
            result = _no_doublets(cell_ids)
        else:
            colsum = cp.bincount(cell[keep], minlength=len(cell_ids)).get()
            result = _get_doublets(colsum, cell_ids)
            result["is_doublet"] = result["q_value"] < q_threshold

    del region, cell
    cleanup_gpu_memory()

    n_doublets = int(result["is_doublet"].sum())
    logger.info(
        f"Detected {n_doublets:,} doublets "
        f"({100 * n_doublets / len(barcodes):.2f}% of {len(barcodes):,} cells)"
    )

    return result
