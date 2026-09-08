"""
Graph clustering for :func:`gatac.tl.iterative_lsi`.

ArchR's iterative LSI clusters cells between LSI rounds via
``Seurat::FindNeighbors`` → ``Seurat::FindClusters`` plus two post-processing
steps. Reproducing that on GPU is the subject of a later phase; this module
currently provides only the dependency plumbing, so that importing
:mod:`gatac` never requires cuGraph and the error a user sees when the default
clusterer is unavailable is actionable.

The target algorithm, for reference:

1. kNN, ``k.param = 20``, euclidean, on the z-scored depth-filtered embedding.
2. SNN graph: ``shared = A @ Aᵀ`` for binary kNN adjacency ``A``; Jaccard
   weight ``w = shared / (2k - shared)``; prune edges with ``w < 1/15``.
3. Louvain at ``resolution = 2``.
4. Clusters with ``< n_outlier`` (5) cells are dissolved and their cells
   reassigned by a ``knn_assign`` (10) nearest-neighbour majority vote.
5. If ``n_clusters > max_clusters`` (6), hierarchically cluster the cluster
   centroids and cut to ``max_clusters`` — a *merge*, not a resolution search.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["snn_louvain"]

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


def snn_louvain(
    embedding: np.ndarray,
    *,
    k: int = 20,
    resolution: float = 2.0,
    prune: float = 1.0 / 15.0,
    max_clusters: int | None = 6,
    n_outlier: int = 5,
    knn_assign: int = 10,
    flavor: str = "louvain",
    random_state: int = 0,
) -> np.ndarray:
    """
    Cluster cells the way ArchR's iterative LSI does. **Not yet implemented.**

    Parameters
    ----------
    embedding
        ``n_cells x n_dims`` embedding; z-scored per dimension by the caller.
    k
        Neighbours per cell (Seurat's ``k.param``).
    resolution
        Louvain/Leiden resolution (ArchR's ``clusterParams$resolution``).
    prune
        SNN Jaccard weight below which edges are dropped (Seurat's
        ``prune.SNN``).
    max_clusters
        Merge down to at most this many clusters via centroid hierarchical
        clustering. ``None`` disables the merge.
    n_outlier
        Clusters smaller than this are dissolved and reassigned.
    knn_assign
        Neighbours used for the reassignment majority vote.
    flavor
        ``"louvain"`` or ``"leiden"``.
    random_state
        Seed.

    Returns
    -------
    np.ndarray
        Integer cluster labels, one per row of *embedding*.
    """
    raise NotImplementedError(
        "gatac.tl._clustering.snn_louvain is not implemented yet "
        "(planned as phase 3 of the LSI work; see LSI_PLAN.md). "
        "Pass `cluster_fn=...` to supply your own clusterer in the meantime."
    )
