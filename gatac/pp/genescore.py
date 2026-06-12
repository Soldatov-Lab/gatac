"""
GPU-accelerated ArchR-style gene activity *score* matrix.

This is a faithful GPU port of ArchR's ``addGeneScoreMatrix``
(GreenleafLab/ArchR ``R/MatrixGeneScores.R``). Unlike :func:`gatac.pp.make_gene_matrix`
(a SnapATAC2-style binary paired-insertion count over a fixed regulatory
domain), this computes a *distance-weighted* activity score:

For each chromosome:

1. Bin both Tn5 insertion ends of every fragment into ``tile_size`` tiles and
   build a sparse ``(occupied tile x cell)`` count matrix ``M``, capped at
   ``ceiling`` to limit pileup bias.
2. For each gene, build an extended regulatory window (gene body extended by
   ``gene_upstream``/``gene_downstream``, then further out to
   ``extend_upstream``/``extend_downstream``), optionally clipped so a tile
   cannot contribute across a neighbouring gene (``use_gene_boundaries``).
3. Weight each (gene, tile) pair by ``gene_model(signed_distance)`` times a
   per-gene inverse-width weight (``gene_scale_factor``), giving ``W``.
4. Gene score block = ``W @ M`` (sparse matmul on GPU via cupyx).

Finally each cell column is normalised to ``scale_to`` by its total gene score
and rounded to 3 decimals, matching ArchR exactly.

The core compute (steps 1-4) maps directly onto cuDF group-bys, ``searchsorted``
interval overlaps and a cuSPARSE SpMM, which is why it GPU-accelerates well.
"""
from __future__ import annotations

import gc
import logging
import time
from pathlib import Path
from typing import Callable, List, Literal, Optional, Tuple, Union

import cudf
import cupy as cp
import cupyx.scipy.sparse as cusp
import numpy as np
import pandas as pd
import scipy.sparse as sp

logger = logging.getLogger(__name__)

_INT32_MAX = 2147483647


# =============================================================================
# Gene-model evaluation
# =============================================================================
def _eval_gene_model(model: Union[str, Callable], x: cp.ndarray) -> cp.ndarray:
    """Evaluate the ArchR ``geneModel`` on a CuPy array of signed distances.

    ``model`` is either a callable ``f(x) -> weight`` or an R/Python-style
    expression string in ``x`` (e.g. ``"exp(-abs(x)/5000) + exp(-1)"``).
    Only a small whitelist of math functions is exposed to ``eval``.
    """
    if callable(model):
        return cp.asarray(model(x), dtype=cp.float64)
    # Mirrors ArchR's eval(parse(text = geneModel)); exposes the math names the
    # model can use. Builtins are left intact because cupy's first-time kernel
    # JIT needs __import__ available. The model string is trusted (a parameter),
    # exactly as in ArchR.
    env = {
        "x": x,
        "exp": cp.exp,
        "abs": cp.abs,
        "sqrt": cp.sqrt,
        "log": cp.log,
        "log2": cp.log2,
        "log10": cp.log10,
        "cp": cp,
        "np": cp,
    }
    return cp.asarray(eval(model, env), dtype=cp.float64)  # noqa: S307


# =============================================================================
# Gene annotation -> regulatory regions
# =============================================================================
def load_gene_score_annotation(
    gene_anno: str | Path,
    gene_upstream: int = 5000,
    gene_downstream: int = 0,
    gene_scale_factor: float = 5.0,
    use_tss: bool = False,
    gene_name_key: str = "gene_name",
    gene_id_key: str = "gene_id",
) -> pd.DataFrame:
    """Build ArchR-style per-gene regulatory regions and weights.

    Accepts either a GTF/GFF file or a CSV with columns
    ``symbol, seqnames, start, end, strand`` (the format exported by the
    reproducibility ArchR oracle, so the port can be driven by the exact same
    gene set ArchR used).

    Returns a DataFrame with one row per gene and columns:
    ``chrom, ext_start, ext_end, gene_start, gene_end, strand, name,
    gene_weight`` where ``ext_start/ext_end`` is the extended gene body
    (genomic, 0-based, inclusive end) and ``gene_start/gene_end`` are the
    strand-oriented TSS/TES.
    """
    gene_anno = Path(gene_anno)
    suffix = gene_anno.name.lower()

    if suffix.endswith(".csv") or suffix.endswith(".csv.gz"):
        df = pd.read_csv(gene_anno)
        cols = {c.lower(): c for c in df.columns}
        df = df.rename(columns={
            cols.get("symbol", "symbol"): "name",
            cols.get("seqnames", "seqnames"): "chrom",
        })
        df["chrom"] = df["chrom"].astype(str)
        # genomic start < end; normalise strand to +/-
        g_start = df["start"].astype(np.int64).to_numpy()
        g_end = df["end"].astype(np.int64).to_numpy()
        lo = np.minimum(g_start, g_end)
        hi = np.maximum(g_start, g_end)
        strand = df["strand"].astype(str).to_numpy()
        strand = np.where(np.isin(strand, ["-", "2", "-1"]), "-", "+")
        names = df["name"].astype(str).to_numpy()
        chrom = df["chrom"].astype(str).to_numpy()
    else:
        anno = _load_gtf_genes(gene_anno, gene_name_key, gene_id_key)
        chrom = anno["chrom"].to_numpy()
        lo = anno["start"].to_numpy().astype(np.int64)
        hi = anno["end"].to_numpy().astype(np.int64)
        strand = anno["strand"].to_numpy()
        names = anno["name"].to_numpy()

    plus = strand == "+"

    if use_tss:
        # Gene model anchored on the 1bp TSS, then extended.
        tss = np.where(plus, lo, hi)
        gene_start = tss
        gene_end = tss
        # extend the 1bp TSS
        ext_start = np.where(plus, tss - gene_upstream, tss - gene_downstream)
        ext_end = np.where(plus, tss + gene_downstream, tss + gene_upstream)
    else:
        gene_start = np.where(plus, lo, hi)   # TSS (strand 5')
        gene_end = np.where(plus, hi, lo)     # TES (strand 3')
        ext_start = np.where(plus, lo - gene_upstream, lo - gene_downstream)
        ext_end = np.where(plus, hi + gene_downstream, hi + gene_upstream)

    # ArchR clamps nothing here (seqlengths removed); negative starts handled later.
    width = (ext_end - ext_start + 1).astype(np.float64)
    m = 1.0 / width
    if use_tss:
        gene_weight = np.full(len(width), float(gene_scale_factor))
    else:
        rng = m.max() - m.min()
        if rng == 0:
            gene_weight = np.ones(len(width))
        else:
            gene_weight = 1.0 + m * (gene_scale_factor - 1.0) / rng

    out = pd.DataFrame({
        "chrom": chrom,
        "ext_start": ext_start.astype(np.int64),
        "ext_end": ext_end.astype(np.int64),
        "gene_start": gene_start.astype(np.int64),
        "gene_end": gene_end.astype(np.int64),
        "strand": strand,
        "name": names,
        "gene_weight": gene_weight.astype(np.float64),
    })
    return out


def _load_gtf_genes(gtf_path, gene_name_key, gene_id_key) -> pd.DataFrame:
    """Minimal GTF/GFF gene loader: one (start,end,strand) per gene symbol."""
    import polars as pl

    gtf_path = Path(gtf_path)
    is_gff3 = ".gff" in gtf_path.name.lower()
    cols = ["chrom", "source", "feature", "start", "end",
            "score", "strand", "frame", "attribute"]
    df = pl.read_csv(str(gtf_path), separator="\t", comment_prefix="#",
                     has_header=False, new_columns=cols).select(
        ["chrom", "feature", "start", "end", "strand", "attribute"])
    df = df.filter(pl.col("feature") == "gene")
    if is_gff3:
        df = df.with_columns(
            pl.col("attribute").str.extract(rf"{gene_name_key}=([^;]+)").alias("name"))
    else:
        df = df.with_columns(
            pl.col("attribute").str.extract(rf'{gene_name_key} "([^"]+)"').alias("name"))
    df = df.filter(pl.col("name").is_not_null()).unique(subset=["name"], keep="first")
    pdf = df.select(["chrom", "start", "end", "strand", "name"]).to_pandas()
    pdf["chrom"] = pdf["chrom"].astype(str)
    return pdf


# =============================================================================
# Per-chromosome gene-boundary extended search window
# =============================================================================
def _compute_search_window(
    ext_start: cp.ndarray,   # genomic left of extended gene body, sorted asc
    ext_end: cp.ndarray,     # genomic right of extended gene body
    strand_is_plus: cp.ndarray,
    extend_up: Tuple[int, int],
    extend_down: Tuple[int, int],
    tile_size: int,
    use_gene_boundaries: bool,
) -> Tuple[cp.ndarray, cp.ndarray]:
    """Reproduce ArchR's per-gene ``[s, e]`` overlap window (sorted genes).

    When ``use_gene_boundaries`` is True the window is clipped so it cannot
    extend past a neighbouring gene's extended body (beyond the minimum
    extension), exactly as in ``.addGeneScoreMat``.
    """
    n = ext_start.shape[0]
    up_min, up_max = int(min(extend_up)), int(max(extend_up))
    dn_min, dn_max = int(min(extend_down)), int(max(extend_down))

    pmin_gene = ext_start
    pmax_gene = ext_end

    if not use_gene_boundaries:
        s = pmin_gene - cp.where(strand_is_plus, up_max, dn_max)
        e = pmax_gene + cp.where(strand_is_plus, up_max, dn_max)
        s = cp.maximum(s, 1)
        e = cp.minimum(e, _INT32_MAX)
        return s, e

    # "reverse" = genomic-left extension, "forward" = genomic-right extension.
    # For + strand, left side is upstream; for - strand, left side is downstream.
    p_reverse = cp.where(strand_is_plus, up_max, dn_max)
    p_reverse_min = cp.where(strand_is_plus, up_min, dn_min)
    p_forward = cp.where(strand_is_plus, dn_max, up_max)
    p_forward_min = cp.where(strand_is_plus, dn_min, up_min)

    # Left bound: don't cross previous gene's right edge (+ tile), and keep at
    # least the minimum extension.
    left_neighbor = cp.empty(n, dtype=cp.int64)
    left_neighbor[0] = 1
    left_neighbor[1:] = pmax_gene[:-1] + tile_size
    s = cp.maximum(left_neighbor, pmin_gene - p_reverse)
    s = cp.minimum(pmin_gene - p_reverse_min, s)
    s = cp.maximum(s, 1)

    # Right bound: don't cross next gene's left edge (- tile), keep min extension.
    right_neighbor = cp.empty(n, dtype=cp.int64)
    right_neighbor[:-1] = pmin_gene[1:] - tile_size
    right_neighbor[-1] = pmax_gene[-1] + p_forward[-1]
    e = cp.minimum(right_neighbor, pmax_gene + p_forward)
    e = cp.maximum(pmax_gene + p_forward_min, e)
    e = cp.minimum(e, _INT32_MAX)

    return s, e


# =============================================================================
# Core: gene score matrix on GPU
# =============================================================================
def create_gene_score_matrix_gpu(
    fragments_df: cudf.DataFrame,
    gene_regions: pd.DataFrame,
    geneModel: Union[str, Callable] = "exp(-abs(x)/5000) + exp(-1)",
    extend_upstream: Tuple[int, int] = (1000, 100000),
    extend_downstream: Tuple[int, int] = (1000, 100000),
    tile_size: int = 500,
    ceiling: int = 4,
    scale_to: float = 10000.0,
    use_gene_boundaries: bool = True,
    exclude_chroms: Optional[list] = ("chrY", "chrM"),
    min_fragments_per_cell: int = 100,
    cell_metadata: Optional[cudf.DataFrame] = None,
    filter_query: Optional[str] = None,
    cell_batch_size: Optional[int] = None,
) -> Tuple[sp.csr_matrix, cudf.DataFrame, pd.DataFrame]:
    """Compute the ArchR gene-score matrix. Returns ``(cells x genes)`` CSR."""
    if exclude_chroms is None:
        exclude_chroms = []
    elif isinstance(exclude_chroms, str):
        exclude_chroms = [exclude_chroms]

    gene_regions = gene_regions[~gene_regions["chrom"].isin(list(exclude_chroms))]
    gene_regions = gene_regions.reset_index(drop=True)
    # Global, stable gene ordering (genome-wide). Feature index = this order.
    gene_regions = gene_regions.sort_values(["chrom", "ext_start", "name"]).reset_index(drop=True)
    gene_regions["gene_idx"] = np.arange(len(gene_regions))
    n_genes = len(gene_regions)

    valid_chroms = sorted(set(gene_regions["chrom"]) - set(exclude_chroms))

    # ---- Cell filtering (mirror make_gene_matrix) -----------------------
    frag_chrom_cats = fragments_df["chrom"].dtype.categories.to_arrow().to_pylist()
    fetch_chroms = [c for c in valid_chroms if c in frag_chrom_cats]
    frags_counting = fragments_df[fragments_df["chrom"].isin(fetch_chroms)]

    if cell_metadata is None:
        bc = frags_counting.groupby("barcode", observed=True).agg({"count": ["sum", "size"]})
        bc.columns = ["n_total", "n_unique"]
        bc = bc.reset_index()
        cell_metadata = bc[bc["n_unique"] >= min_fragments_per_cell]
    else:
        if filter_query:
            cell_metadata = cell_metadata.query(filter_query)
        if "n_unique" not in cell_metadata.columns:
            bc = frags_counting.groupby("barcode", observed=True).agg({"count": ["sum", "size"]})
            bc.columns = ["n_total", "n_unique"]
            bc = bc.reset_index()
            cell_metadata = cell_metadata.merge(bc[["barcode", "n_unique"]], on="barcode", how="left")
        cell_metadata = cell_metadata[cell_metadata["n_unique"] >= min_fragments_per_cell]

    valid_barcodes = cell_metadata["barcode"]

    bc_cats = fragments_df["barcode"].dtype.categories.to_arrow().to_pylist()
    fetch_bc = [b for b in valid_barcodes.astype(str).to_arrow().to_pylist() if b in bc_cats]
    fragments_df = fragments_df[
        fragments_df["barcode"].isin(fetch_bc) & fragments_df["chrom"].isin(fetch_chroms)
    ]

    # Fixed cell ordering -> cell_idx
    unique_barcodes = fragments_df["barcode"].unique().reset_index(drop=True)
    barcode_list = unique_barcodes.to_arrow().to_pylist()
    n_cells = len(barcode_list)
    if n_cells == 0:
        raise ValueError("No cells passed filtering for gene score computation")
    barcode_to_idx = cudf.DataFrame({
        "barcode": cudf.Series(barcode_list),
        "cell_idx": cp.arange(n_cells, dtype=cp.int32),
    })
    fragments_df = fragments_df.merge(barcode_to_idx, on="barcode", how="left")

    logger.debug(f"Gene score: {n_cells} cells x {n_genes} genes over {len(valid_chroms)} chroms")

    # ---- Per-chromosome compute ----------------------------------------
    gene_blocks: List[sp.csr_matrix] = []        # each (n_genes_chr x n_cells)
    block_gene_idx: List[np.ndarray] = []         # global gene idx per block row
    total_gs = cp.zeros(n_cells, dtype=cp.float64)  # per-cell total score

    for chrom in valid_chroms:
        if chrom not in frag_chrom_cats:
            continue
        genes_chr = gene_regions[gene_regions["chrom"] == chrom]
        if len(genes_chr) == 0:
            continue
        chrom_frags = fragments_df[fragments_df["chrom"] == chrom]
        if len(chrom_frags) == 0:
            continue

        block, gidx, col_sums = _gene_score_chrom(
            chrom_frags, genes_chr, geneModel,
            extend_upstream, extend_downstream, tile_size, ceiling,
            use_gene_boundaries, n_cells, cell_batch_size,
        )
        if block is not None:
            gene_blocks.append(block)
            block_gene_idx.append(gidx)
            total_gs += col_sums

        del chrom_frags
        cp.get_default_memory_pool().free_all_blocks()

    # ---- Normalise to scale_to per cell & assemble ----------------------
    total_gs_host = cp.asnumpy(total_gs)
    inv = np.zeros_like(total_gs_host)
    nz = total_gs_host > 0
    inv[nz] = scale_to / total_gs_host[nz]

    if gene_blocks:
        full = sp.vstack(gene_blocks, format="csr")          # rows in block order
        row_to_gene = np.concatenate(block_gene_idx)
        # Genes on chromosomes absent/empty in the fragment file (or whole
        # chromosomes with no tile overlap) never produce a block. Pad them as
        # all-zero rows so the matrix always has exactly n_genes rows and stays
        # aligned with gene_metadata.
        seen = np.zeros(n_genes, dtype=bool)
        seen[row_to_gene] = True
        if not seen.all():
            missing_idx = np.flatnonzero(~seen)
            full = sp.vstack(
                [full, sp.csr_matrix((len(missing_idx), n_cells), dtype=full.dtype)],
                format="csr",
            )
            row_to_gene = np.concatenate([row_to_gene, missing_idx])
        # column scaling by per-cell factor, then reorder rows to global gene idx
        full = full.tocsc()
        full = full.multiply(sp.csr_matrix(inv.reshape(1, -1)))  # broadcast over rows
        full = sp.csr_matrix(full)
        full.data = np.round(full.data, 3)
        full.eliminate_zeros()
        # reorder rows so row r corresponds to global gene_idx
        order = np.argsort(row_to_gene)
        full = full[order]
    else:
        full = sp.csr_matrix((n_genes, n_cells), dtype=np.float64)

    # genes x cells -> cells x genes
    matrix = full.T.tocsr().astype(np.float32)

    # ---- Metadata -------------------------------------------------------
    cell_metadata = barcode_to_idx.merge(cell_metadata, on="barcode", how="left")
    cell_metadata = cell_metadata.sort_values("cell_idx").reset_index(drop=True)
    if cell_metadata["barcode"].dtype != "object":
        cell_metadata["barcode"] = cell_metadata["barcode"].astype(str)

    gene_metadata = gene_regions.sort_values("gene_idx").reset_index(drop=True)
    gene_metadata = gene_metadata[
        ["chrom", "gene_start", "gene_end", "strand", "name", "gene_idx"]
    ].rename(columns={"gene_start": "start", "gene_end": "end"})

    return matrix, cell_metadata, gene_metadata


def _gene_score_chrom(
    chrom_frags: cudf.DataFrame,
    genes_chr: pd.DataFrame,
    geneModel,
    extend_upstream,
    extend_downstream,
    tile_size,
    ceiling,
    use_gene_boundaries,
    n_cells,
    cell_batch_size,
) -> Tuple[Optional[sp.csr_matrix], Optional[np.ndarray], cp.ndarray]:
    """Compute one chromosome's (genes_chr x cells) score block (unnormalised)."""
    # --- 1. tile x cell count matrix (both insertion ends, capped) -------
    starts = chrom_frags["start"].astype(cp.int64).values
    ends = chrom_frags["end"].astype(cp.int64).values
    cells = chrom_frags["cell_idx"].astype(cp.int32).values

    left_tiles = (starts // tile_size) * tile_size
    right_tiles = (ends // tile_size) * tile_size
    pos = cp.concatenate([left_tiles, right_tiles])
    cell2 = cp.concatenate([cells, cells])

    uniq_tiles = cp.unique(pos)                       # sorted occupied tile starts
    n_tiles = int(uniq_tiles.shape[0])
    tile_row = cp.searchsorted(uniq_tiles, pos).astype(cp.int32)

    tdf = cudf.DataFrame({"t": tile_row, "c": cell2})
    agg = tdf.groupby(["t", "c"]).size().reset_index(name="x")
    xvals = cp.minimum(agg["x"].values.astype(cp.float64), float(ceiling))
    M = cusp.coo_matrix(
        (xvals, (agg["t"].values, agg["c"].values)),
        shape=(n_tiles, n_cells),
    ).tocsr()
    del tdf, agg, pos, cell2, tile_row, left_tiles, right_tiles, starts, ends, cells

    # --- 2. gene windows (sorted by ext_start within chrom) --------------
    genes_chr = genes_chr.sort_values("ext_start").reset_index(drop=True)
    ext_start = cp.asarray(genes_chr["ext_start"].to_numpy(), dtype=cp.int64)
    ext_end = cp.asarray(genes_chr["ext_end"].to_numpy(), dtype=cp.int64)
    g_tss = cp.asarray(genes_chr["gene_start"].to_numpy(), dtype=cp.int64)
    plus = cp.asarray((genes_chr["strand"] == "+").to_numpy())
    gene_weight = cp.asarray(genes_chr["gene_weight"].to_numpy(), dtype=cp.float64)
    global_idx = genes_chr["gene_idx"].to_numpy()
    n_g = ext_start.shape[0]

    s, e = _compute_search_window(
        ext_start, ext_end, plus,
        extend_upstream, extend_downstream, tile_size, use_gene_boundaries,
    )

    # --- 3. overlap genes <-> tiles via searchsorted ---------------------
    # tile occupies [t, t+tile_size-1]; overlaps [s,e] iff
    #   t <= e  AND  t + tile_size - 1 >= s  =>  t in [s-tile_size+1, e]
    lo = cp.searchsorted(uniq_tiles, s - tile_size + 1, side="left")
    hi = cp.searchsorted(uniq_tiles, e, side="right")
    counts = (hi - lo).astype(cp.int64)
    counts = cp.maximum(counts, 0)
    total_pairs = int(counts.sum())

    if total_pairs == 0:
        return None, None, cp.zeros(n_cells, dtype=cp.float64)

    # Expand per-gene tile ranges without cp.repeat (which rejects array repeats):
    # for each flat position, find its gene via searchsorted on cumulative ends.
    offsets = cp.cumsum(counts)                       # exclusive end per gene
    flat = cp.arange(total_pairs)
    gene_rep = cp.searchsorted(offsets, flat, side="right").astype(cp.int64)
    grp_start = (offsets - counts)[gene_rep]
    within = flat - grp_start
    tile_rep = lo[gene_rep] + within

    # --- 4. signed distance + model weight -------------------------------
    t_start = uniq_tiles[tile_rep]
    t_end = t_start + tile_size - 1
    g_lo = ext_start[gene_rep]      # genomic left of extended gene body
    g_hi = ext_end[gene_rep]        # genomic right
    # IRanges distance: gap between [g_lo,g_hi] and [t_start,t_end], 0 if overlap
    gap = cp.maximum(cp.maximum(g_lo - t_end - 1, t_start - g_hi - 1), 0)
    # sign relative to strand-5' (TSS direction); symmetric models ignore it
    sign = cp.sign(t_start - g_tss[gene_rep])
    sign = cp.where(plus[gene_rep], sign, -sign)
    x = (gap * sign).astype(cp.float64)

    w = _eval_gene_model(geneModel, x)
    w = w * gene_weight[gene_rep]

    W = cusp.coo_matrix(
        (w, (gene_rep.astype(cp.int32), tile_rep.astype(cp.int32))),
        shape=(n_g, n_tiles),
    ).tocsr()
    del gene_rep, tile_rep, within, grp_start, x, w, t_start, t_end, g_lo, g_hi, sign

    # --- 5. gene score block = W @ M  (genes_chr x cells) ----------------
    if cell_batch_size and n_cells > cell_batch_size:
        blocks = []
        for c0 in range(0, n_cells, cell_batch_size):
            c1 = min(c0 + cell_batch_size, n_cells)
            gs = (W @ M[:, c0:c1])
            blocks.append(_cusparse_to_scipy(gs))
        block = sp.hstack(blocks, format="csr")
    else:
        gs = W @ M
        block = _cusparse_to_scipy(gs)

    col_sums = cp.asarray(block.sum(axis=0)).ravel()
    col_sums = cp.asarray(col_sums, dtype=cp.float64)

    del W, M, uniq_tiles, ext_start, ext_end, g_tss, plus, gene_weight, s, e, lo, hi, counts
    cp.get_default_memory_pool().free_all_blocks()

    return block, global_idx, col_sums


def _cusparse_to_scipy(mat) -> sp.csr_matrix:
    """cupyx CSR/COO -> scipy CSR on host."""
    mat = mat.tocsr()
    return sp.csr_matrix(
        (cp.asnumpy(mat.data), cp.asnumpy(mat.indices), cp.asnumpy(mat.indptr)),
        shape=mat.shape,
    )


# =============================================================================
# AnnData export + top-level driver
# =============================================================================
def gene_score_to_anndata(matrix, cell_metadata, gene_metadata):
    import scanpy as sc

    obs = cell_metadata.to_pandas() if hasattr(cell_metadata, "to_pandas") else cell_metadata
    obs.index = obs["barcode"].astype(str).values
    var = gene_metadata.copy()
    var.index = var["name"].astype(str).values
    if var.index.duplicated().any():
        dup = var.index.duplicated(keep=False)
        var.loc[dup, "name2"] = var.loc[dup, "name"] + "_" + var.loc[dup, "gene_idx"].astype(str)
        var.index = var["name2"].fillna(pd.Series(var.index, index=var.index)).values
    adata = sc.AnnData(X=matrix, obs=obs, var=var)
    return adata


def make_gene_score_matrix(
    input_parquet: str | Path,
    gene_anno: str | Path,
    output_path: Optional[str | Path] = None,
    gene_model: Union[str, Callable] = "exp(-abs(x)/5000) + exp(-1)",
    tile_size: int = 500,
    extend_upstream: Tuple[int, int] = (1000, 100000),
    extend_downstream: Tuple[int, int] = (1000, 100000),
    gene_upstream: int = 5000,
    gene_downstream: int = 0,
    use_gene_boundaries: bool = True,
    use_tss: bool = False,
    ceiling: int = 4,
    gene_scale_factor: float = 5.0,
    scale_to: float = 10000.0,
    exclude_chroms: Optional[list] = ("chrY", "chrM"),
    min_fragments_per_cell: int = 100,
    metrics: Optional[str | Path | "cudf.DataFrame"] = None,
    filter_query: Optional[str] = None,
    barcode_prefix: Optional[str] = None,
    low_memory: bool = False,
    cell_batch_size: Optional[int] = None,
    gene_name_key: str = "gene_name",
    gene_id_key: str = "gene_id",
) -> "sc.AnnData":
    """GPU-accelerated ArchR-style gene activity *score* matrix.

    Faithful port of ArchR ``addGeneScoreMatrix``. See module docstring for the
    algorithm. Parameter defaults match ArchR's defaults.

    Parameters
    ----------
    input_parquet
        ATAC fragments parquet (columns: chrom, start, end, barcode, count).
    gene_anno
        GTF/GFF gene annotation, or a CSV with columns
        ``symbol, seqnames, start, end, strand``.
    gene_model
        ArchR ``geneModel``: an expression string in ``x`` (signed distance to
        TSS), or a Python callable ``f(x)->weight``.
    tile_size, ceiling, scale_to, gene_scale_factor
        ArchR tiling / capping / normalisation / gene-width-weight parameters.
    extend_upstream, extend_downstream
        ``(min, max)`` bp extension used for the regulatory search window.
    gene_upstream, gene_downstream
        bp the gene body is grown before the model is applied.
    use_gene_boundaries
        Clip windows so tiles cannot contribute across a neighbouring gene.
    use_tss
        Build the model on the 1bp TSS rather than the gene body.

    Returns
    -------
    AnnData of shape ``(cells, genes)`` with normalised gene scores.
    """
    from .process import read_fragments_parquet
    import scanpy as sc

    input_parquet = Path(input_parquet)
    if output_path is None:
        output_path = input_parquet.with_suffix("").with_name(
            input_parquet.stem + "_gene_score_matrix.h5ad")
    else:
        output_path = Path(output_path)

    gene_regions = load_gene_score_annotation(
        gene_anno,
        gene_upstream=gene_upstream,
        gene_downstream=gene_downstream,
        gene_scale_factor=gene_scale_factor,
        use_tss=use_tss,
        gene_name_key=gene_name_key,
        gene_id_key=gene_id_key,
    )

    cell_metadata_input = None
    if metrics is not None:
        if isinstance(metrics, cudf.DataFrame):
            cell_metadata_input = metrics
        else:
            mp = Path(metrics)
            if mp.exists():
                cell_metadata_input = cudf.read_csv(str(mp))
            else:
                logger.warning(f"Metrics file {mp} not found; proceeding without it.")

    def _cleanup():
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()

    def _run(use_low_mem):
        df = read_fragments_parquet(input_parquet, low_memory=use_low_mem)
        df = df.sort_values("barcode")
        _cleanup()
        return create_gene_score_matrix_gpu(
            fragments_df=df,
            gene_regions=gene_regions,
            geneModel=gene_model,
            extend_upstream=extend_upstream,
            extend_downstream=extend_downstream,
            tile_size=tile_size,
            ceiling=ceiling,
            scale_to=scale_to,
            use_gene_boundaries=use_gene_boundaries,
            exclude_chroms=list(exclude_chroms) if exclude_chroms else [],
            min_fragments_per_cell=min_fragments_per_cell,
            cell_metadata=cell_metadata_input,
            filter_query=filter_query,
            cell_batch_size=cell_batch_size,
        )

    start = time.perf_counter()
    try:
        matrix, cell_metadata, gene_metadata = _run(low_memory)
    except (MemoryError, RuntimeError) as exc:
        msg = str(exc).lower()
        if any(k in msg for k in ("out of memory", "bad_alloc", "cudaerrormemoryallocation")) and not low_memory:
            logger.warning(f"CUDA OOM; retrying with low_memory=True: {exc}")
            _cleanup()
            matrix, cell_metadata, gene_metadata = _run(True)
        else:
            raise

    adata = gene_score_to_anndata(matrix, cell_metadata, gene_metadata)
    if barcode_prefix:
        adata.obs_names = [f"{barcode_prefix}{b}" for b in adata.obs_names]

    adata.write_h5ad(str(output_path))
    logger.info(
        f"Created {output_path.name}: {adata.shape[0]:,} cells x {adata.shape[1]:,} genes "
        f"({time.perf_counter() - start:.1f}s)")
    return adata
