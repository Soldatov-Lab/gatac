from __future__ import annotations

from typing import Optional

import numpy as np


def _as_2d_bias_array(bias_values: np.ndarray) -> np.ndarray:
    """Normalize bias values to a 2D float array."""
    bias_array = np.asarray(bias_values, dtype=np.float64)

    if bias_array.ndim == 1:
        bias_array = bias_array[:, None]
    elif bias_array.ndim != 2:
        raise ValueError("Bias values must be a 1D or 2D array.")

    if bias_array.shape[0] == 0:
        raise ValueError("Bias values must contain at least one item.")

    return bias_array


def compute_bias_knn_indices(
    bias_values: np.ndarray,
    *,
    n_neighbors: int,
    exclude_self: bool = True,
) -> np.ndarray:
    """Return k-nearest neighbors in bias space."""
    bias_array = _as_2d_bias_array(bias_values)

    if n_neighbors < 1:
        raise ValueError("'n_neighbors' must be at least 1.")

    n_items = bias_array.shape[0]
    if exclude_self and n_items < 2:
        raise ValueError("Need at least two items to exclude self from bias matching.")

    query_neighbors = min(n_items, n_neighbors + int(exclude_self))

    _, indices = _query_bias_knn(
        bias_array,
        bias_array,
        n_neighbors=query_neighbors,
        require_cuml=True,
    )

    indices = np.asarray(indices, dtype=np.int32)
    if exclude_self:
        indices = indices[:, 1:]

    if indices.shape[1] == 0:
        raise ValueError("No bias-matched neighbors available after excluding self.")

    if indices.shape[1] < n_neighbors:
        repeats = int(np.ceil(n_neighbors / indices.shape[1]))
        indices = np.tile(indices, (1, repeats))

    return indices[:, :n_neighbors]


def normalize_bias_matrix(bias_values: np.ndarray) -> np.ndarray:
    """Whiten a bias matrix using the chromVAR/scPrinter transform."""
    bias_array = _as_2d_bias_array(bias_values)

    if bias_array.shape[1] == 1:
        std = float(np.std(bias_array[:, 0], ddof=1)) if bias_array.shape[0] > 1 else 0.0
        if not np.isfinite(std) or std == 0:
            return np.zeros_like(bias_array)
        return bias_array / std

    mat = bias_array.T
    cov = np.cov(mat)
    jitter = 0.0
    eye = np.eye(cov.shape[0], dtype=np.float64)

    for _ in range(6):
        try:
            chol_cov = np.linalg.cholesky(cov + jitter * eye)
            return np.linalg.solve(chol_cov, mat).T
        except np.linalg.LinAlgError:
            jitter = 1e-10 if jitter == 0 else jitter * 10

    raise np.linalg.LinAlgError("Could not normalize bias matrix with Cholesky decomposition.")


def _query_bias_knn(
    query_bias: np.ndarray,
    reference_bias: np.ndarray,
    *,
    n_neighbors: int,
    require_cuml: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Query nearest neighbors in bias space using cuML or a SciPy fallback."""
    query_array = _as_2d_bias_array(query_bias)
    reference_array = _as_2d_bias_array(reference_bias)

    if query_array.shape[1] != reference_array.shape[1]:
        raise ValueError(
            "Query and reference bias arrays must have the same number of features."
        )
    if n_neighbors < 1:
        raise ValueError("'n_neighbors' must be at least 1.")

    n_neighbors = min(n_neighbors, reference_array.shape[0])

    try:
        from cuml.neighbors import NearestNeighbors

        knn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
        knn.fit(reference_array)
        distances, indices = knn.kneighbors(query_array)
        distances = np.asarray(distances, dtype=np.float64)
        indices = np.asarray(indices, dtype=np.int32)
    except ImportError as exc:
        if require_cuml:
            raise ImportError(
                "cuML is required for k-NN bias matching. Install via: "
                "pip install cuml-cu12 (or the appropriate CUDA build)."
            ) from exc

        from scipy.spatial import cKDTree

        tree = cKDTree(reference_array)
        distances, indices = tree.query(query_array, k=n_neighbors)
        distances = np.asarray(distances, dtype=np.float64)
        indices = np.asarray(indices, dtype=np.int32)

        if n_neighbors == 1:
            distances = distances[:, None]
            indices = indices[:, None]

    return distances, indices


def sample_bias_matched_indices(
    target_bias: np.ndarray,
    candidate_bias: np.ndarray,
    *,
    n_samples: Optional[int] = None,
    n_bins: int = 50,
    replace: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Sample candidate indices whose bias distribution matches the target."""
    target_array = _as_2d_bias_array(target_bias)
    candidate_array = _as_2d_bias_array(candidate_bias)

    if target_array.shape[1] != candidate_array.shape[1]:
        raise ValueError(
            "Target and candidate bias arrays must have the same number of features."
        )
    if n_bins < 1:
        raise ValueError("'n_bins' must be at least 1.")

    sample_count = target_array.shape[0] if n_samples is None else int(n_samples)
    if sample_count < 1:
        raise ValueError("'n_samples' must be at least 1.")
    if not replace and sample_count > candidate_array.shape[0]:
        raise ValueError(
            "Cannot sample more unique bias-matched items than there are candidates."
        )

    rng = np.random.default_rng() if rng is None else rng

    sampled_target_indices = rng.choice(
        target_array.shape[0],
        size=sample_count,
        replace=target_array.shape[0] < sample_count,
    )
    sampled_targets = target_array[sampled_target_indices]

    bandwidth = max(
        np.linalg.norm(np.ptp(candidate_array, axis=0)) / max(1, n_bins),
        1e-6,
    )

    knn_neighbors = min(
        candidate_array.shape[0],
        max(64, min(2048, 8 * n_bins)),
    )
    knn_distances, knn_indices = _query_bias_knn(
        sampled_targets,
        candidate_array,
        n_neighbors=knn_neighbors,
        require_cuml=False,
    )

    sampled_indices = np.empty(sample_count, dtype=np.int32)
    candidate_index_array = np.arange(candidate_array.shape[0], dtype=np.int32)
    used_mask = np.zeros(candidate_array.shape[0], dtype=bool)

    for i, bias_vector in enumerate(sampled_targets):
        row_distances = knn_distances[i]
        row_indices = knn_indices[i]

        order = np.argsort(row_indices)
        row_indices = row_indices[order]
        row_distances = row_distances[order]

        weights = np.exp(-0.5 * (row_distances / bandwidth) ** 2)

        if not replace:
            weights = np.where(used_mask[row_indices], 0.0, weights)

        if not np.isfinite(weights).all() or weights.sum() == 0:
            if replace:
                sampled_indices[i] = int(row_indices[np.argmin(row_distances)])
            else:
                available_indices = candidate_index_array[~used_mask]
                if available_indices.size == 0:
                    raise ValueError("No unused candidate items remain for unique sampling.")

                # Rare fallback: exact search only when the local k-NN set is exhausted.
                available_distances = np.linalg.norm(
                    candidate_array[available_indices] - bias_vector,
                    axis=1,
                )
                best_available = available_indices[np.argmin(available_distances)]
                sampled_indices[i] = int(best_available)
                used_mask[sampled_indices[i]] = True
            continue

        weights /= weights.sum()
        local_choice = int(rng.choice(np.arange(len(row_indices), dtype=np.int32), p=weights))
        sampled_indices[i] = int(row_indices[local_choice])
        if not replace:
            used_mask[sampled_indices[i]] = True

    return sampled_indices