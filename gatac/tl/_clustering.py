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
* **No ``n.start`` restarts.** Seurat runs Louvain 10 times and keeps the best
  modularity; cuGraph exposes no equivalent, so a single run is used.
* **Approximate vs exact kNN.** Seurat's default ``nn.method="annoy"`` is
  approximate; cuML's brute-force search here is exact, so a handful of
  neighbours can differ near ties.
* **Self-loops dropped** before Louvain. Seurat's SNN carries a unit diagonal;
  it shifts modularity by a constant and cuGraph rejects self-loops in places.
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


def _knn_indices(X: np.ndarray, k: int) -> np.ndarray:
    """Exact euclidean kNN indices, ``(n, k)``, self included."""
    from cuml.neighbors import NearestNeighbors  # noqa: PLC0415

    nn = NearestNeighbors(n_neighbors=k, metric="euclidean", output_type="cupy")
    Xg = cp.asarray(X, dtype=cp.float32)
    nn.fit(Xg)
    _, idx = nn.kneighbors(Xg)
    return cp.asnumpy(idx).astype(np.int32)


def snn_graph(
    embedding: np.ndarray,
    *,
    k: int = 20,
    prune: float = 1.0 / 15.0,
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

    Returns
    -------
    cupyx.scipy.sparse.csr_matrix
        Symmetric weighted graph, ``n_cells x n_cells``.
    """
    n = embedding.shape[0]
    idx = _knn_indices(embedding, k)

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
    random_state
        Seed. Consumed by Leiden; cuGraph's Louvain accepts none.

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
    emb = np.ascontiguousarray(np.asarray(embedding, dtype=np.float32))
    if emb.ndim != 2:
        raise ValueError(f"embedding must be 2-D, got shape {emb.shape}.")
    n = emb.shape[0]
    if n <= k:
        raise ValueError(
            f"Need more cells than neighbours: n_cells={n}, k={k}."
        )

    graph = snn_graph(emb, k=k, prune=prune)
    labels = _communities(graph, resolution, flavor, random_state)
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
