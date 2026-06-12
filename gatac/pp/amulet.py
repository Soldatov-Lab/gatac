"""
AMULET doublet/multiplet detection for single-cell ATAC-seq.

Direct port of the AMULET (Atac-seq MULtiplet Estimation Tool) algorithm of
Thibodeau et al. (2021) to the GATAC data model.

The overlap-detection sweep-line, the per-cell Poisson scoring, the row-sum
Poisson repeat-inference, and the BH-FDR correction are translated
line-for-line from the upstream Python source (`FragmentFileOverlapCounter.py`
and `AMULET.py`). The data flow has been rewritten to operate on GATAC's
parquet fragment files via DuckDB and to parallelize per-chromosome with a
worker pool; the optional repeat-filter pass is applied at the raw-read level
rather than the overlap level.

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
from typing import Optional, Union, List, Set

import duckdb
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm

from .genome import get_chrom_sizes

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Overlap detection (sweep-line algorithm, ported from AMULET)
# ---------------------------------------------------------------------------

def _get_overlaps(reads: np.ndarray, overlapthresh: int) -> List[List]:
    """
    Sweep-line running-sum algorithm for overlap detection.

    Parameters
    ----------
    reads : np.ndarray, shape (n, 3)
        Reads as [start, end, ...] for a single (chrom, barcode) segment.
    overlapthresh : int
        Minimum overlap count to report (i.e. expected_overlap + 1).

    Returns
    -------
    list of [chr, start, end, minoverlap, maxoverlap, starts_str, ends_str]
    """
    if len(reads) <= overlapthresh - 1:
        return []

    starts = reads[:, 0]
    ends = reads[:, 1]

    overlapindex = np.empty((len(starts) * 2, 2), dtype=np.int64)
    overlapindex[0::2, 0] = starts
    overlapindex[0::2, 1] = 1
    overlapindex[1::2, 0] = ends
    overlapindex[1::2, 1] = -1
    sort_idx = np.argsort(overlapindex[:, 0], kind="mergesort")
    overlapindex = overlapindex[sort_idx]

    runningsum = np.empty(len(overlapindex), dtype=np.int64)
    runningsumpos = np.empty(len(overlapindex), dtype=np.int64)
    runningsum[0] = overlapindex[0, 1]
    runningsumpos[0] = overlapindex[0, 0]
    write_idx = 0
    for i in range(1, len(overlapindex)):
        cursum = runningsum[write_idx] + overlapindex[i, 1]
        if overlapindex[i - 1, 0] == overlapindex[i, 0]:
            runningsum[write_idx] = cursum
        else:
            write_idx += 1
            runningsum[write_idx] = cursum
            runningsumpos[write_idx] = overlapindex[i, 0]
    runningsum = runningsum[:write_idx + 1]
    runningsumpos = runningsumpos[:write_idx + 1]

    rv: List[List] = []
    withinsegment = False
    segmentstart = -1
    minoverlap = -1
    maxoverlap = -1
    for i in range(len(runningsum)):
        if withinsegment:
            if runningsum[i] < overlapthresh:
                rv.append([segmentstart, runningsumpos[i], minoverlap, maxoverlap])
                withinsegment = False
                segmentstart = minoverlap = maxoverlap = -1
            else:
                if runningsum[i] > maxoverlap:
                    maxoverlap = int(runningsum[i])
        else:
            if runningsum[i] >= overlapthresh:
                segmentstart = int(runningsumpos[i])
                minoverlap = maxoverlap = int(runningsum[i])
                withinsegment = True
    if withinsegment:
        rv.append([segmentstart, int(runningsumpos[-1]), minoverlap, maxoverlap])

    return rv


def _format_frag_overlaps(chrom: str, barcode: str, overlaps: List[List]) -> List[str]:
    """Format overlap lines for fragment input (no mapping quality)."""
    lines = []
    for ov in overlaps:
        lines.append(
            f"{chrom}\t{ov[0]}\t{ov[1]}\t{barcode}\t{ov[2]}\t{ov[3]}\t.\t.\t.\n"
        )
    return lines


def _get_union_peaks(overlaps_df: pd.DataFrame) -> np.ndarray:
    """
    Compute the union of overlapping regions across all cells.

    Parameters
    ----------
    overlaps_df : pd.DataFrame
        Must contain columns: chr, start, end.

    Returns
    -------
    np.ndarray, shape (n_regions, 3)
        Union of merged regions as [chr, start, end].
    """
    if len(overlaps_df) == 0:
        return np.zeros((0, 3), dtype=object)

    data = overlaps_df[["chr", "start", "end"]].values
    out = []
    for chrom in np.unique(data[:, 0]):
        chrom_data = data[data[:, 0] == chrom]
        sorted_idx = np.argsort(chrom_data[:, 1], kind="mergesort")
        sorted_data = chrom_data[sorted_idx]
        cur_start, cur_end = sorted_data[0, 1], sorted_data[0, 2]
        for i in range(1, len(sorted_data)):
            nxt_start, nxt_end = sorted_data[i, 1], sorted_data[i, 2]
            if nxt_start > cur_end:
                out.append([chrom, cur_start, cur_end])
                cur_start, cur_end = nxt_start, nxt_end
            else:
                cur_end = max(cur_end, nxt_end)
        out.append([chrom, cur_start, cur_end])
    return np.array(out, dtype=object)


# ---------------------------------------------------------------------------
# Parquet-based overlap finding (DuckDB backend, like FragmentFileOverlapCounter)
# ---------------------------------------------------------------------------

def _load_repeat_regions(repeat_filter: Union[str, Path]) -> Optional[np.ndarray]:
    """Load BED file of known repetitive regions."""
    if not repeat_filter:
        return None
    logger.info(f"Loading repeat regions from {repeat_filter}")
    df = pd.read_csv(repeat_filter, sep="\t", header=None).values[:, 0:3]
    return _get_union_peaks(pd.DataFrame(df, columns=["chr", "start", "end"]))


def _filter_repeats_in_reads(
    reads: np.ndarray,
    sorted_repeats: dict,
    repeat_regions: np.ndarray,
    expected_overlap: int,
) -> np.ndarray:
    """
    For each cell's reads on a chromosome, remove fragments that overlap
    known repetitive regions and recalculate overlap segments.

    Parameters
    ----------
    reads : np.ndarray, shape (n_reads, 2)
        start, end positions sorted by start.
    sorted_repeats : dict
        {chrom: np.ndarray} of repeat region starts (sorted ascending).
    repeat_regions : np.ndarray
        Repeat regions as [chr, start, end].
    expected_overlap : int
        Expected number of reads overlapping.

    Returns
    -------
    np.ndarray, shape (n_filtered_reads, 2)
        Filtered start, end positions.
    """
    if len(reads) == 0:
        return reads

    starts = reads[:, 0]
    ends = reads[:, 1]

    overlap_mask = np.zeros(len(starts), dtype=bool)
    for r_start, r_end in zip(repeat_regions[:, 1], repeat_regions[:, 2]):
        overlap_mask |= ((starts <= r_end) & (ends >= r_start))

    new_starts = starts[~overlap_mask]
    new_ends = ends[~overlap_mask]

    if len(new_starts) == len(starts):
        return reads
    if len(new_starts) <= expected_overlap:
        return np.zeros((0, 2), dtype=reads.dtype)

    counts = np.empty((len(new_starts), 2), dtype=new_starts.dtype)
    counts[:, 0] = new_starts
    counts[:, 1] = 1

    counts2 = np.empty((len(new_ends), 2), dtype=new_ends.dtype)
    counts2[:, 0] = new_ends
    counts2[:, 1] = -1

    combinedcounts = np.concatenate((counts, counts2))
    order = np.argsort(combinedcounts[:, 0], kind="mergesort")
    combinedcounts = combinedcounts[order]

    runningsum = 0
    i = 0
    startoverlap = False
    startoverlapposition = 0
    rv: List[List[int]] = []

    while i < len(combinedcounts):
        runningsum += int(combinedcounts[i, 1])
        j = i + 1
        while j < len(combinedcounts) and combinedcounts[j, 0] == combinedcounts[i, 0]:
            runningsum += int(combinedcounts[j, 1])
            j += 1
        if not startoverlap and runningsum > expected_overlap:
            startoverlap = True
            startoverlapposition = int(combinedcounts[i, 0])
        elif startoverlap and runningsum <= expected_overlap:
            rv.append([startoverlapposition, int(combinedcounts[i, 0])])
            startoverlap = False
        i = j
    if startoverlap:
        rv.append([startoverlapposition, int(combinedcounts[-1, 0])])

    if not rv:
        return np.zeros((0, 2), dtype=reads.dtype)
    return np.array(rv, dtype=reads.dtype)


def _process_chrom_parquet(
    fragment_path: Path,
    chrom: str,
    barcodes: List[str],
    expected_overlap: int,
    max_insert_size: int,
    repeat_regions: Optional[np.ndarray],
    sorted_repeats: Optional[dict],
) -> dict:
    """
    Per-chromosome overlap detection using DuckDB.

    Returns
    -------
    dict with keys: chrom, overlap_lines (list[str]), overlapcounts (dict)
    """
    overlapthresh = expected_overlap + 1
    overlapcounts: dict = {bc: 0 for bc in barcodes}
    overlap_lines: List[str] = []

    con = duckdb.connect()
    bc_df = pd.DataFrame({"barcode": barcodes})
    con.register("barcodes", bc_df)

    rows = con.execute(f"""
        SELECT "start", "end", barcode
        FROM read_parquet('{fragment_path}')
        INNER JOIN barcodes USING (barcode)
        WHERE chrom = '{chrom}'
          AND ("end" - "start") <= {max_insert_size}
        ORDER BY "start"
    """).fetchnumpy()
    con.close()

    if len(rows["start"]) == 0:
        return {"chrom": chrom, "overlap_lines": [], "overlapcounts": overlapcounts}

    starts = rows["start"]
    ends = rows["end"]
    barcodes_arr = rows["barcode"]

    bc_to_reads: dict = {}
    for i in range(len(starts)):
        bc = barcodes_arr[i]
        bc_to_reads.setdefault(bc, []).append((int(starts[i]), int(ends[i])))

    for bc, reads_list in bc_to_reads.items():
        if not reads_list:
            continue
        reads_arr = np.array(reads_list, dtype=np.int64)

        if repeat_regions is not None and len(repeat_regions) > 0:
            filtered_reads = _filter_repeats_in_reads(
                reads_arr, sorted_repeats, repeat_regions, expected_overlap
            )
            if len(filtered_reads) == 0:
                continue
            overlaps = _get_overlaps(filtered_reads, overlapthresh)
        else:
            overlaps = _get_overlaps(reads_arr, overlapthresh)

        overlap_lines.extend(_format_frag_overlaps(chrom, bc, overlaps))
        overlapcounts[bc] += len(overlaps)

    return {"chrom": chrom, "overlap_lines": overlap_lines, "overlapcounts": overlapcounts}


def _find_overlaps_parquet(
    fragment_path: Union[str, Path],
    barcodes: List[str],
    chromosomes: List[str],
    expected_overlap: int = 2,
    max_insert_size: int = 900,
    repeat_filter: Optional[Union[str, Path]] = None,
    n_threads: int = 1,
) -> pd.DataFrame:
    """
    Find per-cell fragment overlaps in a GATAC parquet file.

    Parameters
    ----------
    fragment_path : str or Path
        GATAC parquet fragment file.
    barcodes : list of str
        Barcodes to consider as candidate cells.
    chromosomes : list of str
        Chromosomes to scan.
    expected_overlap : int
        Minimum overlap count to report.
    max_insert_size : int
        Maximum fragment insert size in bp.
    repeat_filter : str or Path, optional
        BED file of known repetitive regions to exclude.
    n_threads : int
        Number of parallel workers (one per chromosome batch).

    Returns
    -------
    pd.DataFrame
        Columns: chr, start, end, cell_id, min_overlap, max_overlap,
        mean_mq, min_mq, max_mq.
    """
    fragment_path = Path(fragment_path)
    if not fragment_path.exists():
        raise FileNotFoundError(f"Fragment file not found: {fragment_path}")

    repeat_regions = _load_repeat_regions(repeat_filter) if repeat_filter else None
    sorted_repeats: Optional[dict] = None
    if repeat_regions is not None and len(repeat_regions) > 0:
        sorted_repeats = {}
        for chrom in np.unique(repeat_regions[:, 0]):
            mask = repeat_regions[:, 0] == chrom
            order = np.argsort(repeat_regions[mask, 1], kind="mergesort")
            sorted_repeats[chrom] = np.column_stack(
                [repeat_regions[mask, 1][order], np.arange(len(repeat_regions))[mask][order]]
            )

    logger.info(
        f"Scanning {len(chromosomes)} chromosomes with {n_threads} worker(s) "
        f"for cells with >={expected_overlap + 1} overlapping fragments..."
    )

    tasks = [
        (fragment_path, chrom, barcodes, expected_overlap, max_insert_size,
         repeat_regions, sorted_repeats)
        for chrom in chromosomes
    ]

    if n_threads > 1:
        from multiprocessing import Pool
        with Pool(min(n_threads, len(tasks))) as pool:
            results = pool.starmap(_process_chrom_parquet, tasks)
    else:
        results = [_process_chrom_parquet(*t) for t in tasks]

    all_lines: List[str] = []
    for r in results:
        all_lines.extend(r["overlap_lines"])

    if not all_lines:
        return pd.DataFrame(
            columns=["chr", "start", "end", "cell_id", "min_overlap", "max_overlap",
                     "mean_mq", "min_mq", "max_mq"]
        )

    header = ["chr", "start", "end", "cell_id", "min_overlap", "max_overlap",
              "mean_mq", "min_mq", "max_mq"]
    from io import StringIO
    overlaps_df = pd.read_csv(StringIO("".join(all_lines)), sep="\t", header=None,
                              names=header)
    return overlaps_df


# ---------------------------------------------------------------------------
# Cell × region matrix and doublet detection
# ---------------------------------------------------------------------------

def _generate_matrix(
    overlaps_df: pd.DataFrame,
    cell_ids: np.ndarray,
    union_overlaps: np.ndarray,
) -> np.ndarray:
    """
    Build a binary cell × region matrix from overlaps and union peaks.

    Parameters
    ----------
    overlaps_df : pd.DataFrame
        Per-cell overlap segments.
    cell_ids : np.ndarray
        Cell IDs (one per column in output matrix).
    union_overlaps : np.ndarray
        Union of all overlap regions.

    Returns
    -------
    np.ndarray, shape (n_regions, n_cells)
        Binary matrix.
    """
    cell_id_to_idx = {cid: i for i, cid in enumerate(cell_ids)}
    matrix = np.zeros((len(union_overlaps), len(cell_ids)), dtype=np.uint8)

    if len(overlaps_df) == 0:
        return matrix

    union_starts = union_overlaps[:, 1]
    union_ends = union_overlaps[:, 2]
    union_chroms = union_overlaps[:, 0]

    overlap_chroms = overlaps_df["chr"].values
    overlap_starts = overlaps_df["start"].values
    overlap_ends = overlaps_df["end"].values
    overlap_cells = overlaps_df["cell_id"].values

    for chrom in np.unique(union_chroms):
        region_mask = union_chroms == chrom
        region_indices = np.where(region_mask)[0]
        region_starts = union_starts[region_mask]
        region_ends = union_ends[region_mask]

        ovl_mask = overlap_chroms == chrom
        ovl_starts = overlap_starts[ovl_mask]
        ovl_ends = overlap_ends[ovl_mask]
        ovl_cells = overlap_cells[ovl_mask]

        for i in range(len(ovl_starts)):
            hits = np.where(
                (region_starts <= ovl_ends[i]) & (region_ends >= ovl_starts[i])
            )[0]
            if len(hits) == 0:
                continue
            cell_idx = cell_id_to_idx.get(ovl_cells[i])
            if cell_idx is None:
                continue
            matrix[region_indices[hits], cell_idx] = 1

    return matrix


def _infer_repeats(
    matrix: np.ndarray,
    union_overlaps: np.ndarray,
    threshold: float,
) -> tuple:
    """
    Infer repetitive regions via Poisson test on row sums.

    Returns
    -------
    rep_regions : np.ndarray
        Regions classified as repetitive.
    non_rep_regions : np.ndarray
        Regions classified as non-repetitive.
    """
    if matrix.shape[0] == 0:
        return np.zeros((0, 3), dtype=object), union_overlaps
    rowsum = np.sum(matrix, axis=1)
    rep_mean = np.mean(rowsum)
    rep_probabilities = stats.poisson.sf(rowsum, rep_mean)
    corrected_rep_probabilities = sm.stats.multipletests(
        rep_probabilities, method="fdr_bh"
    )
    rep_mask = corrected_rep_probabilities[1] < threshold
    rep_regions = union_overlaps[rep_mask]
    non_rep_regions = union_overlaps[~rep_mask]
    return rep_regions, non_rep_regions


def _get_doublets(
    matrix: np.ndarray,
    cell_ids: np.ndarray,
) -> pd.DataFrame:
    """
    Compute per-cell p-value and q-value from column sums via Poisson test.

    Returns
    -------
    pd.DataFrame
        Columns: cell_id, p_value, q_value
    """
    colsum = np.sum(matrix, axis=0)
    doublet_mean = np.mean(colsum)
    doublet_probabilities = stats.poisson.sf(colsum, doublet_mean)
    corrected = sm.stats.multipletests(doublet_probabilities, method="fdr_bh")
    return pd.DataFrame({
        "cell_id": cell_ids,
        "p_value": doublet_probabilities,
        "q_value": corrected[1],
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
    are flagged as doublets/multiplets.

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
        BED file of known repetitive regions.
    min_overlap_bp : int
        Minimum overlap length in bp to retain (default 1).
    n_threads : int
        Parallel workers for overlap detection (default 1).

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

    overlaps_df = _find_overlaps_parquet(
        fragment_path=fragment_path,
        barcodes=barcodes,
        chromosomes=list(chrom_sizes.keys()),
        expected_overlap=expected_overlap,
        max_insert_size=max_insert_size,
        repeat_filter=repeat_filter,
        n_threads=n_threads,
    )

    if len(overlaps_df) > 0 and min_overlap_bp > 1:
        lengths = overlaps_df["end"].values - overlaps_df["start"].values + 1
        overlaps_df = overlaps_df[lengths >= min_overlap_bp].reset_index(drop=True)

    if len(overlaps_df) == 0:
        result = pd.DataFrame({
            "cell_id": barcodes,
            "p_value": np.ones(len(barcodes)),
            "q_value": np.ones(len(barcodes)),
            "is_doublet": np.zeros(len(barcodes), dtype=bool),
        })
    else:
        logger.info(f"Building union of {len(overlaps_df):,} overlap regions")
        union_overlaps = _get_union_peaks(overlaps_df)

        cell_ids = np.asarray(barcodes)
        logger.info(f"Generating {len(union_overlaps):,} x {len(cell_ids):,} matrix")
        matrix = _generate_matrix(overlaps_df, cell_ids, union_overlaps)

        logger.info("Inferring repetitive regions")
        rep_regions, _ = _infer_repeats(matrix, union_overlaps, q_rep_threshold)

        if len(rep_regions) > 0:
            rep_chroms = rep_regions[:, 0]
            rep_starts = rep_regions[:, 1]
            rep_ends = rep_regions[:, 2]
            ovl_chroms = overlaps_df["chr"].values
            ovl_starts = overlaps_df["start"].values
            ovl_ends = overlaps_df["end"].values
            keep = np.ones(len(overlaps_df), dtype=bool)
            for chrom in np.unique(rep_chroms):
                region_mask = rep_chroms == chrom
                r_starts = rep_starts[region_mask]
                r_ends = rep_ends[region_mask]
                ovl_mask = ovl_chroms == chrom
                ovl_idx = np.where(ovl_mask)[0]
                for i in ovl_idx:
                    hits = np.where(
                        (r_starts <= ovl_ends[i]) & (r_ends >= ovl_starts[i])
                    )[0]
                    if len(hits) > 0:
                        keep[i] = False
            rep_filtered_df = overlaps_df[keep].reset_index(drop=True)
        else:
            rep_filtered_df = overlaps_df

        if len(rep_filtered_df) == 0:
            result = pd.DataFrame({
                "cell_id": cell_ids,
                "p_value": np.ones(len(cell_ids)),
                "q_value": np.ones(len(cell_ids)),
                "is_doublet": np.zeros(len(cell_ids), dtype=bool),
            })
        else:
            rep_filtered_union = _get_union_peaks(rep_filtered_df)
            rep_filtered_matrix = _generate_matrix(
                rep_filtered_df, cell_ids, rep_filtered_union
            )
            result = _get_doublets(rep_filtered_matrix, cell_ids)
            result["is_doublet"] = result["q_value"] < q_threshold

    n_doublets = int(result["is_doublet"].sum())
    logger.info(
        f"Detected {n_doublets:,} doublets "
        f"({100 * n_doublets / len(barcodes):.2f}% of {len(barcodes):,} cells)"
    )

    return result
