"""
Graph clustering for :func:`gatac.tl.iterative_lsi`.

ArchR clusters cells between LSI rounds by handing the embedding to
``Seurat::FindNeighbors`` → ``Seurat::FindClusters``, then applying two
post-processing steps of its own. This module reproduces that on GPU.

The pipeline, with the source each step is transcribed from:

1. **kNN** — ``k = 20`` euclidean neighbours, Seurat's ``k.param`` default.
2. **SNN** — Seurat's ``ComputeSNN``: with ``A`` the binary kNN adjacency
   (self included, as the neighbour search returns it), ``shared = A @ Aᵀ`` and
   the Jaccard weight is ``shared / (2k - shared)``; edges at or below
   ``prune.SNN = 1/15`` are dropped.
3. **Community detection** at ``resolution = 2`` (ArchR's ``clusterParams``).
   Leiden by default rather than ArchR's Louvain — see *Known differences* for
   the measurement behind that.
4. **Small-cluster reassignment** — ``ArchR::addClusters``::

       tabClust <- table(clust); clustAssign <- which(tabClust < nOutlier)
       knni  <- .computeKNN(matDR[-idxi, ], matDR[idxi, ], knnAssign)
       clust[idxi] <- majority vote over clust[-idxi][knni]

   Clusters smaller than ``n_outlier = 5`` are dissolved and their cells
   reassigned by a ``knn_assign = 10`` majority vote among the *remaining*
   cells.
5. **maxClusters merge** — also ``addClusters``::

       meanDR <- per-cluster centroids
       hc <- hclust(dist(meanDR)); ct <- cutree(hc, maxClusters)

   A *merge* of centroids under complete linkage, not a resolution search.
   ``max_clusters = 6`` for iterative LSI.
6. **Relabelling by dendrogram order**, which ArchR does last. Cosmetic — it
   renames rather than repartitions — but it makes labels reproducible.

Known differences from Seurat/ArchR
-----------------------------------
* **Leiden by default, not Louvain.** ArchR clusters with Louvain via Seurat's
  ``FindClusters(algorithm = 1)``. Measured against ``ArchR::addClusters`` on a
  real GATAC embedding (5,184 cells, 30 dims), cuGraph's Leiden reproduced
  ArchR's partition better than cuGraph's Louvain did — ARI 0.928 against
  0.900, both finding 6 clusters. Leiden also guarantees well-connected
  communities, which Louvain does not, and is the only one of the two cuGraph
  lets us seed. Pass ``flavor="louvain"`` to match ArchR's algorithm choice
  rather than its output.
* **``resolution`` is not on Seurat's scale.** On a small dense graph, cuGraph
  returns all-singleton communities at resolution 2 where Seurat's Louvain
  partitions normally. ``snn_cluster`` detects that and backs the resolution
  off, warning; it is not a silent substitution, but it does mean the effective
  resolution can differ from the requested one on small inputs.
* **No ``n.start`` restarts.** Seurat runs Louvain 10 times and keeps the best
  modularity; cuGraph exposes no equivalent, so a single run is used.
* **Approximate vs exact kNN.** Seurat's default ``nn.method="annoy"`` is
  approximate; cuML's brute-force search here is exact, and they disagree on
  about 5 % of neighbours. That is not a rounding detail — it is the single
  largest source of disagreement with ``addIterativeLSI``, larger than
  Leiden-vs-Louvain. Pass ``knn="annoy"`` to match ArchR's search; the ``knn``
  parameter of :func:`snn_cluster` carries the measurements.
* **The embedding is quantised** to 2 decimals before the graph is built (see
  the ``quantize`` parameter for the measurements behind that choice). It was
  introduced to damp a nondeterministic upstream eigensolver; ``lsi``'s
  ``deterministic=True`` path now removes that source of variation at the root,
  so quantisation is kept for its (separately measured) agreement with ArchR
  rather than as a reproducibility crutch. ArchR needs no such step, running on
  a deterministic CPU solver, though its clustering is no more stable across
  *seeds* than this is across runs.
* **Self-loops dropped** before community detection. Seurat's SNN carries a
  unit diagonal, and cuGraph rejects self-loops in places. This is not a
  mismatch with Seurat: its Louvain (``src/RModularityOptimizer.cpp``) reads
  only the strict lower triangle of the graph, so the diagonal never reaches
  the optimiser there either. Checked rather than assumed — given the same
  neighbour lists, :func:`snn_graph` and ``Seurat::FindNeighbors`` (5.5.0)
  produced graphs with the same 432,912 edges and the same partition at every
  Leiden seed tried.
"""

from __future__ import annotations

import logging

import cupy as cp
import cupyx.scipy.sparse as cusp
import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["snn_cluster", "snn_graph"]

_CUGRAPH_HINT = (
    "Graph clustering requires cuGraph, which is not installed in this "
    "environment.\n"
    "Install it with the matching CUDA extra, e.g.\n"
    "    uv sync --extra cuda12\n"
    "or pass your own clusterer via `cluster_fn=...` (it receives the z-scored "
    "embedding and must return integer labels)."
)


def _import_cugraph():
    """
    Import cuGraph on first use.

    Imported lazily — and never at module scope — so that ``import gatac``
    works in an environment without cuGraph, and so the failure surfaces only
    when the default clusterer is actually reached.
    """
    try:
        import cugraph  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(_CUGRAPH_HINT) from exc
    return cugraph


#: Seurat's ``FindNeighbors`` default tree count, and the value the
#: measurements in :func:`snn_cluster` were taken at.
_ANNOY_TREES = 50

_ANNOY_HINT = (
    "knn='annoy' needs the `annoy` package, which is not installed.\n"
    "Install it with\n"
    "    uv add annoy\n"
    "or use the default exact search, knn='exact'."
)


def _knn_exact(X: np.ndarray, k: int) -> np.ndarray:
    """Exact euclidean kNN indices, ``(n, k)``, self included."""
    from cuml.neighbors import NearestNeighbors  # noqa: PLC0415

    nn = NearestNeighbors(n_neighbors=k, metric="euclidean", output_type="cupy")
    Xg = cp.asarray(X, dtype=cp.float32)
    nn.fit(Xg)
    _, idx = nn.kneighbors(Xg)
    return cp.asnumpy(idx).astype(np.int32)


def _knn_annoy(X: np.ndarray, k: int, *, random_state: int,
               n_trees: int = _ANNOY_TREES) -> np.ndarray:
    """
    Annoy's approximate kNN — the search ``Seurat::FindNeighbors`` uses.

    Not a performance choice: this runs on the CPU and is slower than the
    exact GPU search. It exists because ArchR's iterative LSI is measurably a
    function of *which* approximation Annoy makes; see the ``knn`` parameter of
    :func:`snn_cluster`.

    Built single-threaded and from an explicit seed, so the index — and
    therefore the neighbour lists — are reproducible. Trees are random, so this
    does not reproduce ``RcppAnnoy``'s own lists (measured 95 % neighbour
    overlap with them, the same as the exact search has); what carries over is
    the error *structure*, which is what the downstream agreement depends on.
    """
    try:
        from annoy import AnnoyIndex  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(_ANNOY_HINT) from exc

    Xh = np.ascontiguousarray(np.asarray(X, dtype=np.float32))
    n, d = Xh.shape
    index = AnnoyIndex(d, "euclidean")
    index.set_seed(int(random_state))
    for i in range(n):
        index.add_item(i, Xh[i])
    index.build(n_trees, n_jobs=1)
    idx = np.empty((n, k), dtype=np.int32)
    for i in range(n):
        idx[i] = index.get_nns_by_item(i, k, search_k=-1)
    return idx


def _knn_indices(X: np.ndarray, k: int, *, knn: str = "exact",
                 random_state: int = 0) -> np.ndarray:
    """Euclidean kNN indices, ``(n, k)``, self included; see *knn* choices."""
    if knn == "exact":
        return _knn_exact(X, k)
    if knn == "annoy":
        return _knn_annoy(X, k, random_state=random_state)
    raise ValueError(f"knn must be 'exact' or 'annoy' (got {knn!r}).")


def snn_graph(
    embedding: np.ndarray,
    *,
    k: int = 20,
    prune: float = 1.0 / 15.0,
    knn: str = "exact",
    random_state: int = 0,
) -> cusp.csr_matrix:
    """
    Shared-nearest-neighbour graph, following ``Seurat::ComputeSNN``.

    With ``A`` the binary kNN adjacency including self, the edge weight is the
    Jaccard index ``shared / (2k - shared)``, and edges at or below *prune* are
    dropped. The diagonal is removed: it is a constant self-loop that shifts
    modularity without affecting the partition, and cuGraph rejects self-loops
    in places.

    Parameters
    ----------
    embedding
        ``n_cells x n_dims``, z-scored by the caller.
    k
        Neighbours per cell (Seurat's ``k.param``).
    prune
        Jaccard weight at or below which edges are dropped (``prune.SNN``).
    knn
        ``"exact"`` (cuML, GPU) or ``"annoy"`` (CPU, Seurat's search). See the
        ``knn`` parameter of :func:`snn_cluster`.
    random_state
        Seeds the Annoy index; ignored by the exact search.

    Returns
    -------
    cupyx.scipy.sparse.csr_matrix
        Symmetric weighted graph, ``n_cells x n_cells``.
    """
    n = embedding.shape[0]
    idx = _knn_indices(embedding, k, knn=knn, random_state=random_state)

    rows = cp.repeat(cp.arange(n, dtype=cp.int32), k)
    cols = cp.asarray(idx.ravel())
    A = cusp.csr_matrix(
        (cp.ones(rows.size, dtype=cp.float32), (rows, cols)), shape=(n, n)
    )
    del rows, cols

    shared = (A @ A.T).tocoo()
    del A
    cp.get_default_memory_pool().free_all_blocks()

    # Jaccard, then prune. Seurat keeps weights strictly greater than `prune`.
    w = shared.data / (cp.float32(2 * k) - shared.data)
    keep = (w > cp.float32(prune)) & (shared.row != shared.col)
    graph = cusp.csr_matrix(
        (w[keep], (shared.row[keep], shared.col[keep])), shape=(n, n)
    )
    logger.info(
        f"SNN graph: {n:,} cells, k={k}, {graph.nnz:,} edges after pruning "
        f"at {prune:.4g}"
    )
    del shared, w, keep
    cp.get_default_memory_pool().free_all_blocks()
    return graph


def _communities(graph: cusp.csr_matrix, resolution: float, flavor: str,
             random_state: int) -> np.ndarray:
    """Community detection on the SNN graph via cuGraph."""
    cugraph = _import_cugraph()
    import cudf  # noqa: PLC0415

    coo = graph.tocoo()
    # upper triangle only; cuGraph symmetrises an undirected graph itself
    upper = coo.row < coo.col
    df = cudf.DataFrame(
        {
            "src": coo.row[upper].astype(cp.int32),
            "dst": coo.col[upper].astype(cp.int32),
            "weight": coo.data[upper].astype(cp.float32),
        }
    )
    del coo, upper

    G = cugraph.Graph(directed=False)
    G.from_cudf_edgelist(
        df, source="src", destination="dst", edge_attr="weight",
        symmetrize=True,
    )
    del df

    if flavor == "leiden":
        parts, _ = cugraph.leiden(G, resolution=resolution,
                                  random_state=random_state)
    elif flavor == "louvain":
        parts, _ = cugraph.louvain(G, resolution=resolution)
    else:
        raise ValueError(f"flavor must be 'louvain' or 'leiden' (got {flavor!r}).")

    n = graph.shape[0]
    labels = np.full(n, -1, dtype=np.int64)
    vertex = parts["vertex"].to_numpy()
    partition = parts["partition"].to_numpy()
    labels[vertex] = partition
    # cells pruned into isolation get their own singleton community
    orphan = np.where(labels < 0)[0]
    if orphan.size:
        labels[orphan] = partition.max() + 1 + np.arange(orphan.size)
        logger.info(
            f"{orphan.size} cell(s) had no surviving SNN edge and were left as "
            "singletons; the small-cluster pass will reassign them."
        )
    return labels


def _reassign_small_clusters(
    embedding: np.ndarray,
    labels: np.ndarray,
    *,
    n_outlier: int,
    knn_assign: int,
) -> np.ndarray:
    """
    Dissolve clusters below *n_outlier* cells (``ArchR::addClusters``).

    Each orphaned cell takes the majority label among its *knn_assign* nearest
    neighbours **excluding** the dissolved cluster's own cells, matching
    ArchR's ``matDR[-idxi, ]`` candidate pool.
    """
    from cuml.neighbors import NearestNeighbors  # noqa: PLC0415

    labels = labels.copy()
    uniq, counts = np.unique(labels, return_counts=True)
    small = uniq[counts < n_outlier]
    if small.size == 0:
        return labels

    logger.info(
        f"Reassigning {int(np.isin(labels, small).sum())} cell(s) from "
        f"{small.size} cluster(s) smaller than {n_outlier}."
    )
    Xg = cp.asarray(embedding, dtype=cp.float32)
    for cl in small:
        idxi = np.where(labels == cl)[0]
        if idxi.size == 0:                     # already absorbed
            continue
        pool = np.where(labels != cl)[0]
        if pool.size == 0:
            continue
        k = int(min(knn_assign, pool.size))
        nn = NearestNeighbors(n_neighbors=k, metric="euclidean",
                              output_type="cupy")
        nn.fit(Xg[cp.asarray(pool)])
        _, nbr = nn.kneighbors(Xg[cp.asarray(idxi)])
        nbr = cp.asnumpy(nbr)
        cand = labels[pool][nbr]               # (len(idxi), k)
        for row, cell in enumerate(idxi):
            vals, cnt = np.unique(cand[row], return_counts=True)
            labels[cell] = vals[np.argmax(cnt)]
    del Xg
    cp.get_default_memory_pool().free_all_blocks()
    return labels


def _merge_to_max_clusters(
    embedding: np.ndarray, labels: np.ndarray, max_clusters: int
) -> np.ndarray:
    """
    Merge down to *max_clusters* by clustering the cluster centroids.

    ``ArchR::addClusters`` uses ``hclust(dist(meanDR))`` then
    ``cutree(hc, maxClusters)`` — R's ``hclust`` defaults to complete linkage
    and ``dist`` to euclidean. This is a merge of existing clusters, not a
    resolution search.
    """
    from scipy.cluster.hierarchy import fcluster, linkage  # noqa: PLC0415

    uniq = np.unique(labels)
    if uniq.size <= max_clusters:
        return labels
    centroids = np.vstack(
        [embedding[labels == cl].mean(axis=0) for cl in uniq]
    )
    Z = linkage(centroids, method="complete", metric="euclidean")
    merged = fcluster(Z, t=max_clusters, criterion="maxclust")
    logger.info(
        f"Merging {uniq.size} cluster(s) down to {len(np.unique(merged))} "
        f"(max_clusters={max_clusters})."
    )
    mapping = dict(zip(uniq, merged))
    return np.asarray([mapping[c] for c in labels])


def _relabel_by_dendrogram(
    embedding: np.ndarray, labels: np.ndarray
) -> np.ndarray:
    """
    Renumber clusters 0..k-1 in dendrogram order, as ArchR does last.

    Purely cosmetic — it renames, never repartitions — but it makes the label
    numbering a deterministic function of the data rather than of Louvain's
    internal ordering.
    """
    from scipy.cluster.hierarchy import dendrogram, linkage  # noqa: PLC0415

    uniq = np.unique(labels)
    if uniq.size < 2:
        return np.zeros_like(labels)
    centroids = np.vstack([embedding[labels == cl].mean(axis=0) for cl in uniq])
    Z = linkage(centroids, method="complete", metric="euclidean")
    order = dendrogram(Z, no_plot=True)["leaves"]
    mapping = {uniq[old]: new for new, old in enumerate(order)}
    return np.asarray([mapping[c] for c in labels], dtype=np.int64)


def snn_cluster(
    embedding: np.ndarray,
    *,
    k: int = 20,
    resolution: float = 2.0,
    prune: float = 1.0 / 15.0,
    max_clusters: int | None = 6,
    n_outlier: int = 5,
    knn_assign: int = 10,
    flavor: str = "leiden",
    knn: str = "exact",
    quantize: int | None = 2,
    max_resolution_retries: int = 3,
    random_state: int = 0,
) -> np.ndarray:
    """
    Cluster cells the way ArchR's iterative LSI does.

    See the module docstring for the provenance of each step and for the known
    differences from Seurat/ArchR.

    Parameters
    ----------
    embedding
        ``n_cells x n_dims`` embedding; z-score it per dimension first, as
        ArchR does before clustering.
    k
        Neighbours per cell (Seurat's ``k.param``).
    resolution
        Louvain/Leiden resolution (ArchR's ``clusterParams$resolution``).
    prune
        SNN Jaccard weight at or below which edges are dropped
        (``prune.SNN``).
    max_clusters
        Merge down to at most this many clusters. ``None`` disables the merge.
    n_outlier
        Clusters smaller than this are dissolved and their cells reassigned.
    knn_assign
        Neighbours used for the reassignment majority vote.
    flavor
        ``"leiden"`` (default) or ``"louvain"``. ArchR reaches Louvain through
        Seurat, but measured against ``addClusters`` on a real embedding Leiden
        agreed *better* (ARI 0.928 vs 0.900) — and cuGraph lets us seed Leiden
        while its Louvain takes no seed, so Leiden is also the reproducible
        choice. See *Known differences*.
    knn
        Neighbour search: ``"exact"`` (default, cuML on the GPU) or
        ``"annoy"`` (CPU, the approximate search ``Seurat::FindNeighbors``
        uses and therefore the one behind every ArchR ``addClusters`` result).

        This is an ArchR-compatibility switch, not a speed one — ``"annoy"``
        is the slower, less accurate search. It is offered because ArchR's
        iterative LSI turns out to depend on *which* approximation its
        neighbour search makes. Measured on the 4,437-cell PBMC oracle, with
        everything else held fixed and each arm scored by the feature set the
        *next* LSI round selects (ArchR's own seed-to-seed Jaccard, the bar, is
        0.927):

        =============================  =============  =============
        neighbour search               J vs seed 1    J vs seed 2
        =============================  =============  =============
        exact (cuML)                   0.889          0.856
        exact (Seurat's RANN)          0.861          0.831
        pynndescent (approximate)      0.892          0.889
        exact, 5 % random perturbation 0.896-0.906    0.860-0.874
        Seurat's own Annoy lists       0.955          0.909
        **annoy (this option)**        **0.957-0.960**  **0.911-0.914**
        =============================  =============  =============

        Note what the control rows say: perturbing the exact lists at Annoy's
        own error rate does *not* reproduce the gain, and a different
        approximate search does not either — so this is Annoy's particular
        error structure, not approximation in general, and not agreement with
        one ArchR draw (both ArchR seeds improve). Accuracy is not the axis:
        raising Annoy's ``n_trees`` to 200 moves it back towards the exact
        search *and* back down to 0.906.

        The cost is that the search leaves the GPU: 0.43 s here against ~0.01 s
        exact, scaling with cell count. Hence the default stays ``"exact"`` and
        this is opt-in, per run and per call.
    quantize
        Round the embedding to this many decimals before building the graph.
        ``None`` disables it.

        The motivation: ``cupyx``'s eigensolver is not run-to-run
        deterministic — reduction order varies, leaving ~2e-6 relative noise in
        the embedding regardless of ``random_state`` or ``tol`` — and a kNN
        graph is chaotic with respect to that. Measured on a real tile matrix,
        it moved 8 of ~507,000 edges, which Leiden amplified into a different
        partition, which then selected different variable features.

        Rounding to a grid coarser than the noise **reduces** that sensitivity
        but does not eliminate it, because cells sitting near a rounding
        boundary still flip. Measured over five identical runs on a 5,184-cell
        embedding (median pairwise ARI, and ARI against
        ``ArchR::addClusters`` on the same input):

        =============  ==================  ==============
        ``quantize``   run-to-run ARI      ARI vs ArchR
        =============  ==================  ==============
        ``None``       0.69                0.928
        4              0.63                0.961
        3              —                   0.952
        **2**          **0.99**            **0.952**
        1              —                   0.901
        =============  ==================  ==============

        Hence the default of 2: it damps the run-to-run variance by an order of
        magnitude *and* agrees with ArchR slightly better than the unrounded
        embedding, while 1 decimal starts to lose real signal.

        Treat this as variance reduction, not a determinism guarantee. No
        setting reached bitwise identity: the partition at ``resolution=2`` has
        several near-equal optima on that data, and ArchR is unstable in the
        same way across *seeds* (its own seed-to-seed feature Jaccard was
        0.54). If you need a strictly reproducible partition, pass a
        deterministic clusterer via ``cluster_fn``.
    max_resolution_retries
        How many times to halve *resolution* if the partition comes back
        degenerate — more than half the cells in clusters below *n_outlier*.
        cuGraph's resolution does not map onto Seurat's on every graph; see the
        note in the source. ``0`` disables the retry and raises instead.
    random_state
        Seed. Consumed by Leiden and by the Annoy index; cuGraph's Louvain
        accepts none.

    Returns
    -------
    np.ndarray
        Integer labels ``0..n_clusters-1``, one per row of *embedding*, in
        dendrogram order.

    Examples
    --------
    >>> from gatac.tl._clustering import snn_cluster
    >>> z = (emb - emb.mean(0)) / emb.std(0)
    >>> labels = snn_cluster(z, resolution=2.0, max_clusters=6)
    """
    if knn not in ("exact", "annoy"):
        raise ValueError(f"knn must be 'exact' or 'annoy' (got {knn!r}).")
    emb = np.ascontiguousarray(np.asarray(embedding, dtype=np.float32))
    if emb.ndim != 2:
        raise ValueError(f"embedding must be 2-D, got shape {emb.shape}.")
    if quantize is not None:
        # see the `quantize` docstring: this is what makes the discrete output
        # reproducible despite a nondeterministic eigensolver upstream
        emb = np.ascontiguousarray(np.round(emb, quantize).astype(np.float32))
    n = emb.shape[0]
    if n <= k:
        raise ValueError(
            f"Need more cells than neighbours: n_cells={n}, k={k}."
        )

    graph = snn_graph(emb, k=k, prune=prune, knn=knn, random_state=random_state)

    # cuGraph's `resolution` is not interchangeable with Seurat's on every
    # graph. On a small dense SNN graph (measured: 643 cells, 264k edges) both
    # its Louvain and its Leiden return *every node as its own community* at
    # resolution 2 -- no merge improves modularity -- where Seurat's Louvain at
    # the same nominal resolution finds a real partition. Left alone, the
    # small-cluster pass below would then dissolve all of them and cascade into
    # a single cluster, which looks like a plausible answer and is not one.
    # So detect the degenerate case and back the resolution off, saying so.
    res = float(resolution)
    for attempt in range(max_resolution_retries + 1):
        labels = _communities(graph, res, flavor, random_state)
        sizes = np.bincount(labels)
        frac_small = float(sizes[sizes < n_outlier].sum()) / n
        if frac_small <= 0.5 or attempt == max_resolution_retries:
            break
        logger.warning(
            f"{flavor} at resolution {res:g} put "
            f"{frac_small:.0%} of cells in clusters smaller than {n_outlier} "
            f"({len(sizes)} communities for {n} cells) — a degenerate "
            "partition for this graph. Retrying at half that resolution."
        )
        res /= 2.0
    if frac_small > 0.5:
        raise ValueError(
            f"{flavor} could not find a non-degenerate partition of this graph: "
            f"at resolution {res:g}, {frac_small:.0%} of cells are in clusters "
            f"smaller than {n_outlier}. The graph may be too small or too dense "
            f"(n_cells={n}, k={k}, edges={graph.nnz}). Try a lower `resolution` "
            "or a smaller `k`, or supply your own clusterer via `cluster_fn`."
        )
    if res != resolution:
        logger.warning(
            f"Using resolution {res:g} rather than the requested "
            f"{resolution:g}; see the warning above."
        )
    del graph
    cp.get_default_memory_pool().free_all_blocks()
    logger.info(f"{flavor.capitalize()} found {len(np.unique(labels))} cluster(s).")

    labels = _reassign_small_clusters(
        emb, labels, n_outlier=n_outlier, knn_assign=knn_assign
    )
    if max_clusters is not None:
        labels = _merge_to_max_clusters(emb, labels, max_clusters)
    labels = _relabel_by_dendrogram(emb, labels)
    logger.info(f"Clustering complete: {len(np.unique(labels))} cluster(s).")
    return labels
