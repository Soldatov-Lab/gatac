"""
GPU-accelerated Latent Semantic Indexing (TF-IDF + truncated SVD) for
ATAC-seq matrices.

A port of ``ArchR:::.computeLSI`` / ``ArchR:::.projectLSI`` (ArchR 1.0.3),
accelerated with CuPy. The three TF-IDF variants are ArchR's ``LSIMethod``
1/2/3, with ``tf = x / depth``, ``df`` = per-feature sum over *training* cells
and ``n`` = number of training cells:

===========  ===================  =================================
``method``   ArchR name           value on each nonzero
===========  ===================  =================================
1            ``tf-logidf``        ``tf * log(1 + n/df)``
2 (default)  ``log(tf-idf)``      ``log(tf * (n/df) * scale_to + 1)``
3            ``logtf-logidf``     ``log(tf + 1) * log(1 + n/df)``
===========  ===================  =================================

``method=2`` is the default because it is what ``addIterativeLSI`` uses (only
the internal ``.computeLSI`` defaults to 1), and it is also Signac's
``RunTFIDF(method = 1)``.

Notes on fidelity
-----------------
* **Depth follows ``binarize``.** ArchR binarizes *before* ``colSums``, so the
  per-cell depth is the number of nonzero features when ``binarize=True`` and
  the row sum of counts when ``binarize=False``. GATAC tile matrices are
  ``uint16`` unless built with ``count_strategy="binarize"``, so conflating the
  two silently mis-normalises a default matrix.
* **The embedding is ``V @ diag(d)``** — singular vectors scaled by singular
  values — matching ArchR's ``matSVD``.
* **Projection re-computes depth for the cells being projected** and reuses
  only the training ``row_sums``/``n_train`` for the IDF, plus the *fitted*
  ``binarize`` and feature subset. See :func:`project_lsi`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import cupy as cp
import cupyx.scipy.sparse as cusp
import cupyx.scipy.sparse.linalg as cusla
import numpy as np
import scipy.sparse as sp

from ._gpu import ChunkedMatrix, _to_gpu_csr

logger = logging.getLogger(__name__)

__all__ = ["LSIModel"]

#: ArchR's ``LSIMethod`` string aliases.
_METHOD_ALIASES = {
    "tf-logidf": 1,
    "log(tf-idf)": 2,
    "logtf-logidf": 3,
}


# ---------------------------------------------------------------------------
# TF-IDF kernel — one warp per cell
# ---------------------------------------------------------------------------
#
# A warp per row rather than a thread per row: within-row reads are then
# coalesced and the per-cell depth reduction is a __shfl_down_sync tree.
# Measured 1 ms vs 9 ms for the thread-per-row shape on a 20 M-nonzero matrix.
#
# ``idf`` arrives pre-transformed — log(1 + n/df) for methods 1 and 3, bare
# n/df for method 2 — so the log over the feature axis is paid n_features
# times instead of nnz times.
#
# ``inv_depth`` is supplied by the caller, never derived here, because
# projection must use the *new* cells' depth while taking the IDF from
# training.
_tfidf_kernel = cp.RawKernel(
    r"""
    extern "C" __global__
    void tfidf(float* data, const int* indices, const int* indptr,
               const float* idf, const float* inv_depth,
               const float scale_to, const int method,
               const int binarize, const int n_rows) {
        int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
        int lane = threadIdx.x & 31;
        if (warp >= n_rows) return;

        int s = indptr[warp], e = indptr[warp + 1];
        float inv = inv_depth[warp];

        for (int j = s + lane; j < e; j += 32) {
            float x  = binarize ? 1.0f : data[j];
            float tf = x * inv;
            float w  = idf[indices[j]];
            float v;
            if      (method == 1) v = tf * w;                        // tf-logidf
            else if (method == 2) v = log1pf(tf * w * scale_to);     // log(tf-idf)
            else                  v = log1pf(tf) * w;               // logtf-logidf
            data[j] = v;
        }
    }
    """,
    "tfidf",
)


def _launch_warp_per_row(kernel, n_rows: int, *args) -> None:
    """Launch *kernel* with one warp per row."""
    block = 256
    grid = (int(n_rows) * 32 + block - 1) // block
    kernel((grid,), (block,), args)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


@dataclass
class LSIModel:
    """
    A fitted LSI, carrying exactly what is needed to project new cells.

    ``idf`` is deliberately *not* stored: it is a pure function of
    ``row_sums``, ``n_train`` and ``method`` (and ArchR likewise recomputes it
    in ``.projectLSI``), so deriving it in one place removes a field that could
    contradict the settings beside it.
    """

    embedding: np.ndarray            #: (n_cells, n_comps) V @ diag(d)
    singular_values: np.ndarray      #: (n_comps,)
    feature_loadings: np.ndarray     #: (n_kept, n_comps) — retained features, `idx` order
    idx: np.ndarray                  #: retained feature indices, into the fitted feature set
    row_sums: np.ndarray             #: training per-feature sums over `idx` (ArchR rowSm)
    n_train: int                     #: training cell count (ArchR nCol)
    method: int
    scale_to: float
    binarize: bool
    random_state: int = 0
    depth_cor: np.ndarray | None = None      #: |r| with log10 depth, per component
    dims_dropped: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    n_held_out: int = 0
    n_zero_depth_projected: int = 0


# ---------------------------------------------------------------------------
# Pieces
# ---------------------------------------------------------------------------


def _resolve_method(method: int | str) -> int:
    """Map ArchR's ``LSIMethod`` value or string alias onto 1/2/3."""
    if isinstance(method, str):
        key = method.strip().lower()
        if key not in _METHOD_ALIASES:
            raise ValueError(
                f"Unknown LSI method {method!r}. Use 1, 2, 3 or one of "
                f"{sorted(_METHOD_ALIASES)}."
            )
        return _METHOD_ALIASES[key]
    if method not in (1, 2, 3):
        raise ValueError(f"method must be 1, 2 or 3 (got {method!r}).")
    return int(method)


def _cell_depth(X: cusp.csr_matrix, binarize: bool) -> cp.ndarray:
    """
    Per-cell depth, honouring *binarize*, as ArchR's ``colSm``.

    ArchR binarizes before ``Matrix::colSums``, so with ``binarize=True`` the
    depth is the number of nonzero features and with ``binarize=False`` it is
    the sum of the counts. Using nnz for a count-valued matrix would divide a
    count numerator by an nnz denominator, and the resulting ``tf`` would not
    sum to 1 per cell.
    """
    if binarize:
        return cp.diff(X.indptr).astype(cp.float32)
    return cp.asarray(X.sum(axis=1), dtype=cp.float32).ravel()


def _feature_sums(X: cusp.csr_matrix, binarize: bool) -> cp.ndarray:
    """Per-feature sums over the rows of *X*, honouring *binarize* (ArchR ``rowSm``)."""
    n_features = X.shape[1]
    if binarize:
        # In a valid CSR each (row, col) pair appears once, so a bincount over
        # the column indices is the document frequency — no binary copy needed.
        return cp.bincount(
            X.indices.astype(cp.intp), minlength=n_features
        ).astype(cp.float32)
    return cp.asarray(X.sum(axis=0), dtype=cp.float32).ravel()


def _idf_from_row_sums(
    row_sums: cp.ndarray, n_train: int, method: int
) -> cp.ndarray:
    """
    IDF vector, pre-transformed for the kernel.

    Methods 1 and 3 fold ``log(1 + n/df)`` in here so the log is paid once per
    feature rather than once per nonzero; method 2 needs the bare ratio because
    its log wraps the whole product.
    """
    df = cp.maximum(row_sums, cp.float32(1.0))
    ratio = cp.float32(n_train) / df
    if method == 2:
        return ratio.astype(cp.float32)
    return cp.log1p(ratio).astype(cp.float32)


def tfidf_gpu(
    X: cusp.csr_matrix,
    *,
    method: int = 2,
    scale_to: float = 1e4,
    binarize: bool = True,
    idf: cp.ndarray | None = None,
    inv_depth: cp.ndarray | None = None,
    row_sums: cp.ndarray | None = None,
    n_train: int | None = None,
) -> tuple[cp.ndarray, cp.ndarray]:
    """
    Apply TF-IDF **in place** to a cells × features GPU CSR matrix.

    Parameters
    ----------
    X
        Cells × features CSR on device. ``X.data`` is converted to float32 in
        place; the index arrays are untouched.
    method
        ArchR ``LSIMethod`` 1/2/3.
    scale_to
        ArchR's ``scaleTo``.
    binarize
        Treat every nonzero as 1, in both the depth and the numerator.
    idf
        Pre-computed IDF (already log-transformed for methods 1 and 3). When
        ``None`` it is derived from *row_sums* and *n_train*, which then default
        to this matrix's own values.
    inv_depth
        Reciprocal per-cell depth. When ``None`` it is computed from *X*
        honouring *binarize*. Projection passes the new cells' own depth.
    row_sums, n_train
        Training-side IDF inputs; only used when *idf* is ``None``.

    Returns
    -------
    tuple[cp.ndarray, cp.ndarray]
        ``(idf, inv_depth)`` actually used, for reuse by a projection.
    """
    method = _resolve_method(method)
    n_cells = X.shape[0]

    X.data = X.data.astype(cp.float32, copy=False)
    indices32 = X.indices.astype(cp.int32, copy=False)
    indptr32 = X.indptr.astype(cp.int32, copy=False)

    if inv_depth is None:
        depth = _cell_depth(X, binarize)
        # ArchR: colSm[colSm == 0] <- 1, so an empty row stays all-zero
        depth = cp.where(depth == 0, cp.float32(1.0), depth)
        inv_depth = (cp.float32(1.0) / depth).astype(cp.float32)
    else:
        inv_depth = inv_depth.astype(cp.float32, copy=False)

    if idf is None:
        if row_sums is None:
            row_sums = _feature_sums(X, binarize)
        if n_train is None:
            n_train = n_cells
        idf = _idf_from_row_sums(row_sums, n_train, method)
    else:
        idf = idf.astype(cp.float32, copy=False)

    _launch_warp_per_row(
        _tfidf_kernel, n_cells,
        X.data, indices32, indptr32, idf, inv_depth,
        np.float32(scale_to), np.int32(method),
        np.int32(bool(binarize)), np.int32(n_cells),
    )
    return idf, inv_depth


def _svd(
    X,
    n_comps: int,
    *,
    tol: float = 1e-5,
    ncv: int | None = None,
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    """
    Truncated SVD of a sparse matrix or LinearOperator, descending.

    ``cupyx``'s ``svds`` runs Lanczos on the implicit Gram operator (the
    smaller of ``X Xᵀ`` / ``Xᵀ X``), which for an ATAC matrix is cells × cells.
    Defaults differ deliberately from CuPy's: ``tol=1e-5`` is irlba's default —
    hence ArchR's and Signac's behaviour — where CuPy uses machine precision,
    and the wider Krylov basis measured 29–47 % faster at 200k–500k cells with
    no loss of accuracy.

    Randomized SVD is not offered: against a float64 ARPACK reference it lost
    the trailing components on both CPU and GPU.
    """
    n_rows, n_cols = X.shape
    max_k = min(n_rows, n_cols) - 1
    if n_comps > max_k:
        raise ValueError(
            f"n_comps={n_comps} must be < min(n_cells, n_features) = "
            f"{min(n_rows, n_cols)}; the solver requires k < min(m, n)."
        )
    if ncv is None:
        ncv = min(max_k, max(4 * n_comps + 1, n_comps + 60))

    u, s, vt = cusla.svds(X, k=n_comps, tol=tol, ncv=ncv)
    order = cp.argsort(s)[::-1]
    return u[:, order], s[order], vt[order]


def drop_depth_correlated(
    embedding: np.ndarray,
    depth: np.ndarray,
    cutoff: float | None = 0.75,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Flag components correlated with sequencing depth (ArchR's ``corCutOff``).

    *depth* should be **total fragments per cell** — ``obs["n_unique"]``, the
    analogue of ArchR's ``nFrags`` — not the nonzero count of whichever feature
    submatrix was used; using the latter drops components ArchR keeps.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(keep_mask, abs_correlation)``, both length ``n_comps``.
    """
    d = np.log10(np.asarray(depth, dtype=np.float64) + 1.0)
    d_sd = d.std()
    e = np.asarray(embedding, dtype=np.float64)
    e_sd = e.std(axis=0)

    if d_sd == 0:
        # constant depth — correlation is undefined, nothing to drop
        r = np.zeros(e.shape[1])
    else:
        dz = (d - d.mean()) / d_sd
        with np.errstate(invalid="ignore", divide="ignore"):
            ez = (e - e.mean(axis=0)) / np.where(e_sd == 0, 1.0, e_sd)
        r = np.abs((ez * dz[:, None]).mean(axis=0))
        r[e_sd == 0] = 0.0

    keep = np.ones(e.shape[1], dtype=bool) if cutoff is None else (r <= cutoff)
    return keep, r


# ---------------------------------------------------------------------------
# Fit / project
# ---------------------------------------------------------------------------


def compute_lsi(
    X,
    n_comps: int = 30,
    *,
    method: int | str = 2,
    scale_to: float = 1e4,
    binarize: bool = True,
    outlier_quantiles: tuple[float, float] | None = None,
    depth: np.ndarray | None = None,
    depth_cor_cutoff: float | None = None,
    scale_by_sv: bool = True,
    tol: float = 1e-5,
    ncv: int | None = None,
    random_state: int = 0,
    chunk_size: int | None = None,
) -> LSIModel:
    """
    Fit LSI on a cells × features matrix, following ``ArchR:::.computeLSI``.

    Parameters
    ----------
    X
        Cells × features sparse matrix (scipy or cupyx). Not modified: the
        device copy is what gets TF-IDF'd in place.
    n_comps
        Number of components.
    method
        ArchR ``LSIMethod`` 1/2/3, or a string alias.
    scale_to
        ArchR's ``scaleTo``.
    binarize
        Treat nonzeros as 1. Also determines what "depth" means (see module
        docstring).
    outlier_quantiles
        Depth quantiles whose tails are **held out** of the fit and projected
        back in afterwards, as ArchR does with ``c(0.02, 0.98)``. ``None``
        fits every cell.
    depth
        Total fragments per cell, for the depth-correlation filter only. When
        ``None`` the matrix's own per-cell depth is used, with a warning.
    depth_cor_cutoff
        Drop components whose \\|r\\| with ``log10(depth)`` exceeds this.
        ``None`` keeps every component but still reports the correlations.
    scale_by_sv
        Return ``V @ diag(d)`` (ArchR's ``matSVD``) rather than ``V``.
    tol, ncv
        Solver controls; see :func:`_svd`.
    random_state
        Seed for the Lanczos starting vector.
    chunk_size
        Stream the matrix from host memory in row chunks of this size instead
        of holding it in VRAM.

    Returns
    -------
    LSIModel
        With ``embedding`` covering **all** input rows: fitted cells from the
        SVD, held-out cells projected.
    """
    method = _resolve_method(method)
    n_cells, n_features = X.shape

    # ---- depth, honouring binarize ---------------------------------------
    X_gpu_full = _to_gpu_csr(X) if chunk_size is None else None
    if X_gpu_full is not None:
        depth_all = _cell_depth(X_gpu_full, binarize)
    else:
        Xh = X.tocsr()
        depth_all = cp.asarray(
            np.diff(Xh.indptr).astype(np.float32)
            if binarize
            else np.asarray(Xh.sum(axis=1)).ravel().astype(np.float32)
        )
    depth_np = cp.asnumpy(depth_all)

    # ---- cell bookkeeping -------------------------------------------------
    n_empty = int((depth_np == 0).sum())
    if n_empty:
        raise ValueError(
            f"{n_empty} cell(s) have no features in the selected set. "
            "Filter empty cells before running LSI, e.g.:\n"
            "    sc.pp.filter_cells(adata, min_counts=1)\n"
            "or widen the feature selection."
        )

    train_mask = np.ones(n_cells, dtype=bool)
    if outlier_quantiles is not None:
        lo, hi = np.quantile(depth_np, sorted(outlier_quantiles))
        held = (depth_np <= lo) | (depth_np >= hi)
        if held.all() or (n_cells - int(held.sum())) < n_comps + 1:
            # Near-constant depth makes lo == hi, so the condition catches
            # every cell. Fitting nothing is worse than fitting everything.
            logger.warning(
                "Depth-outlier hold-out would leave "
                f"{n_cells - int(held.sum())} training cell(s) for "
                f"n_comps={n_comps}; fitting all cells instead. This happens "
                "when per-cell depth is near-constant."
            )
        else:
            train_mask = ~held
    n_train = int(train_mask.sum())
    n_held_out = n_cells - n_train
    if n_held_out:
        logger.info(
            f"Holding out {n_held_out} depth-outlier cell(s); "
            f"fitting on {n_train}."
        )

    # ---- training submatrix ----------------------------------------------
    if X_gpu_full is not None:
        X_train = X_gpu_full if n_held_out == 0 else X_gpu_full[cp.asarray(train_mask)]
    else:
        X_train = X.tocsr() if n_held_out == 0 else X.tocsr()[train_mask]

    # ---- retained features (ArchR: rowSm > 0 on training cells) ----------
    if X_gpu_full is not None:
        row_sums_all = _feature_sums(X_train, binarize)
        keep_feat = row_sums_all > 0
        idx = cp.asnumpy(cp.where(keep_feat)[0])
        row_sums = row_sums_all[keep_feat]
    else:
        Xt = X_train
        rs = (
            np.diff(Xt.tocsc().indptr).astype(np.float32)
            if binarize
            else np.asarray(Xt.sum(axis=0)).ravel().astype(np.float32)
        )
        idx = np.where(rs > 0)[0]
        row_sums = cp.asarray(rs[idx])
    n_kept = len(idx)
    if n_kept < n_features:
        logger.info(
            f"Dropping {n_features - n_kept} feature(s) with zero training "
            f"signal; {n_kept} retained."
        )

    if n_comps >= min(n_train, n_kept):
        raise ValueError(
            f"n_comps={n_comps} must be < min(n_train, n_features_kept) = "
            f"{min(n_train, n_kept)} (n_train={n_train}, kept={n_kept})."
        )

    # ---- TF-IDF + SVD -----------------------------------------------------
    idf = _idf_from_row_sums(row_sums, n_train, method)

    if X_gpu_full is not None:
        X_fit = X_train[:, cp.asarray(idx)] if n_kept < n_features else X_train
        X_fit = _to_gpu_csr(X_fit).copy()
        inv_depth_train = (
            cp.float32(1.0) / _cell_depth(X_fit, binarize).clip(1.0)
        ).astype(cp.float32)
        tfidf_gpu(
            X_fit, method=method, scale_to=scale_to, binarize=binarize,
            idf=idf, inv_depth=inv_depth_train,
        )
        u, s, vt = _svd(X_fit, n_comps, tol=tol, ncv=ncv)
        del X_fit
    else:
        Xf = X_train[:, idx].tocsr() if n_kept < n_features else X_train.tocsr()
        Xf = _tfidf_host_chunked(
            Xf, idf=idf, method=method, scale_to=scale_to,
            binarize=binarize, chunk_size=chunk_size,
        )
        store, op = ChunkedMatrix(Xf, chunk_size), None
        op = store.as_operator()
        u, s, vt = _svd(op, n_comps, tol=tol, ncv=ncv)
        del store, op, Xf

    cp.get_default_memory_pool().free_all_blocks()

    emb_train = (u * s) if scale_by_sv else u
    loadings = cp.asnumpy(vt.T)          # (n_kept, n_comps)
    s_np = cp.asnumpy(s)

    model = LSIModel(
        embedding=np.empty((n_cells, n_comps), dtype=np.float32),
        singular_values=s_np,
        feature_loadings=loadings.astype(np.float32),
        idx=np.asarray(idx, dtype=np.int64),
        row_sums=cp.asnumpy(row_sums).astype(np.float32),
        n_train=n_train,
        method=method,
        scale_to=float(scale_to),
        binarize=bool(binarize),
        random_state=random_state,
        n_held_out=n_held_out,
    )
    model.embedding[train_mask] = cp.asnumpy(emb_train).astype(np.float32)
    del u, s, vt, emb_train
    cp.get_default_memory_pool().free_all_blocks()

    # ---- reinsert held-out cells by projection ---------------------------
    if n_held_out:
        X_out = (
            X_gpu_full[cp.asarray(~train_mask)]
            if X_gpu_full is not None
            else X.tocsr()[~train_mask]
        )
        proj, n_zero = _project(model, X_out, already_subset=False)
        model.embedding[~train_mask] = proj
        model.n_zero_depth_projected = n_zero
        del X_out

    del X_gpu_full
    cp.get_default_memory_pool().free_all_blocks()

    # ---- depth-correlation filter ----------------------------------------
    if depth is None:
        logger.warning(
            "No `depth` supplied for the depth-correlation filter; falling "
            "back to per-cell depth of the selected features. ArchR uses total "
            "fragments per cell (obs['n_unique'])."
        )
        depth_for_cor = depth_np
    else:
        depth_for_cor = np.asarray(depth)

    keep, r = drop_depth_correlated(
        model.embedding, depth_for_cor, cutoff=depth_cor_cutoff
    )
    model.depth_cor = r
    model.dims_dropped = np.where(~keep)[0]
    if model.dims_dropped.size:
        logger.info(
            f"Dropping {model.dims_dropped.size} component(s) correlated with "
            f"depth (|r| > {depth_cor_cutoff}): "
            f"{model.dims_dropped.tolist()}"
        )
        model.embedding = model.embedding[:, keep]

    # A flat trailing spectrum means the surplus components are not supported
    # by the data — and they are what makes the solver slow.
    if len(s_np) > 5:
        tail = s_np[-5:]
        if float(tail[0] / tail[-1]) < 1.02:
            logger.info(
                "The trailing singular values are nearly identical "
                f"({tail[0]:.3g} → {tail[-1]:.3g}); n_comps={n_comps} likely "
                "exceeds the rank this data supports. A smaller n_comps is "
                "both faster and no less informative."
            )

    return model


def _tfidf_host_chunked(
    X: sp.csr_matrix,
    *,
    idf: cp.ndarray,
    method: int,
    scale_to: float,
    binarize: bool,
    chunk_size: int,
) -> sp.csr_matrix:
    """TF-IDF a host CSR matrix in row chunks, writing back in place."""
    X = X.tocsr()
    if X.dtype != np.float32:
        X = sp.csr_matrix(
            (X.data.astype(np.float32), X.indices, X.indptr), shape=X.shape
        )
    n_cells = X.shape[0]
    for start in range(0, n_cells, chunk_size):
        end = min(start + chunk_size, n_cells)
        d0, d1 = int(X.indptr[start]), int(X.indptr[end])
        n_rows = end - start
        d_data = cp.asarray(X.data[d0:d1])
        d_indices = cp.asarray(X.indices[d0:d1].astype(np.int32))
        d_indptr = cp.asarray((X.indptr[start : end + 1] - d0).astype(np.int32))

        depth = (
            cp.diff(d_indptr).astype(cp.float32)
            if binarize
            else cp.asarray(
                np.add.reduceat(
                    X.data[d0:d1], (X.indptr[start:end] - d0), dtype=np.float32
                )
            )
        )
        inv_depth = (cp.float32(1.0) / cp.maximum(depth, 1.0)).astype(cp.float32)

        _launch_warp_per_row(
            _tfidf_kernel, n_rows,
            d_data, d_indices, d_indptr, idf, inv_depth,
            np.float32(scale_to), np.int32(method),
            np.int32(bool(binarize)), np.int32(n_rows),
        )
        X.data[d0:d1] = cp.asnumpy(d_data)
        del d_data, d_indices, d_indptr, depth, inv_depth
        cp.get_default_memory_pool().free_all_blocks()
    return X


def _project(
    model: LSIModel, X_new, already_subset: bool = False
) -> tuple[np.ndarray, int]:
    """Core of :func:`project_lsi`; returns ``(embedding, n_zero_depth)``."""
    Xg = _to_gpu_csr(X_new)

    # 1. subset to the retained features
    if not already_subset and Xg.shape[1] != len(model.idx):
        Xg = _to_gpu_csr(Xg[:, cp.asarray(model.idx)])
    Xg = Xg.copy()

    # 2/3. the fitted binarize setting, then depth recomputed on THESE cells
    depth = _cell_depth(Xg, model.binarize)
    n_zero = int(cp.sum(depth == 0))
    inv_depth = (
        cp.float32(1.0) / cp.where(depth == 0, cp.float32(1.0), depth)
    ).astype(cp.float32)

    # 4/5. TF from the new cells' depth, IDF from the training row sums
    idf = _idf_from_row_sums(
        cp.asarray(model.row_sums), model.n_train, model.method
    )
    tfidf_gpu(
        Xg, method=model.method, scale_to=model.scale_to,
        binarize=model.binarize, idf=idf, inv_depth=inv_depth,
    )

    emb = Xg.dot(cp.asarray(model.feature_loadings))
    out = cp.asnumpy(emb).astype(np.float32)
    del Xg, emb, idf, inv_depth, depth
    cp.get_default_memory_pool().free_all_blocks()
    return out, n_zero


def project_lsi(model: LSIModel, X_new) -> np.ndarray:
    """
    Project new cells onto a fitted LSI, following ``ArchR:::.projectLSI``.

    The order is not interchangeable:

    1. subset to the retained features (``model.idx``),
    2. apply the **fitted** ``binarize`` setting,
    3. **recompute** per-cell depth on the new cells,
    4. TF from that depth,
    5. IDF from the training ``row_sums`` and ``n_train``,

    then ``emb = X_new @ loadings``. A cell whose nonzeros all fell in features
    dropped at fit time has zero depth after step 1 and projects to an all-zero
    row, as in ArchR (``colSm[colSm == 0] <- 1``), rather than to NaN.

    Parameters
    ----------
    model
        A fitted :class:`LSIModel`.
    X_new
        Cells × features matrix, either over the model's original feature set
        or already subset to ``model.idx``.

    Returns
    -------
    np.ndarray
        ``(n_new_cells, n_comps)`` embedding, in the model's component order
        *before* any depth-correlation filtering.
    """
    emb, n_zero = _project(model, X_new)
    if n_zero:
        logger.warning(
            f"{n_zero} projected cell(s) have no signal in the model's "
            "retained features; their embedding rows are zero."
        )
    return emb
