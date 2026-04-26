"""
Core DTW backend — pure NumPy.

Function-level API shared by all backends:

    compute_distance_matrix(x, y, metric)
    compute_accumulated_cost(distance_matrix, band=None)
    find_path(accumulated_cost)
    dtw(x, y, metric, band, return_path)

Both ``core`` and ``accel`` backends expose the same callables so that callers
can swap implementations transparently.
"""

from typing import List, Optional, Tuple

import numpy as np

BACKEND_NAME = "core"
SUPPORTED_METRICS = ("euclidean", "manhattan", "cosine")


def compute_distance_matrix(
    x: np.ndarray,
    y: np.ndarray,
    metric: str = "euclidean",
) -> np.ndarray:
    """Pairwise distance matrix between rows of ``x`` (N x D) and ``y`` (M x D)."""
    if metric not in SUPPORTED_METRICS:
        raise ValueError(
            f"Unsupported distance metric: {metric}. Supported: {SUPPORTED_METRICS}"
        )

    x = np.ascontiguousarray(x, dtype=np.float32)
    y = np.ascontiguousarray(y, dtype=np.float32)

    if metric == "euclidean":
        # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x·y  — memory-friendly vs broadcasting
        x_sq = np.sum(x * x, axis=1, keepdims=True)         # (N, 1)
        y_sq = np.sum(y * y, axis=1, keepdims=True).T       # (1, M)
        cross = x @ y.T                                     # (N, M)
        dist_sq = np.maximum(x_sq + y_sq - 2.0 * cross, 0.0)
        return np.sqrt(dist_sq, dtype=np.float32)

    if metric == "manhattan":
        return np.abs(x[:, None, :] - y[None, :, :]).sum(axis=2).astype(np.float32)

    # cosine
    x_norm = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
    y_norm = y / (np.linalg.norm(y, axis=1, keepdims=True) + 1e-8)
    return (1.0 - x_norm @ y_norm.T).astype(np.float32)


def compute_accumulated_cost(
    distance_matrix: np.ndarray,
    band: Optional[int] = None,
) -> np.ndarray:
    """
    Accumulated cost matrix.

    Args:
        distance_matrix: (N, M) pairwise distances.
        band: Sakoe-Chiba band radius. ``None`` means unconstrained.
    """
    N, M = distance_matrix.shape
    acc = np.full((N, M), np.inf, dtype=np.float32)
    acc[0, 0] = distance_matrix[0, 0]

    if band is None:
        for i in range(1, N):
            acc[i, 0] = acc[i - 1, 0] + distance_matrix[i, 0]
        for j in range(1, M):
            acc[0, j] = acc[0, j - 1] + distance_matrix[0, j]
        for i in range(1, N):
            for j in range(1, M):
                acc[i, j] = distance_matrix[i, j] + min(
                    acc[i - 1, j], acc[i, j - 1], acc[i - 1, j - 1]
                )
        return acc

    # Banded variant
    for i in range(N):
        j_start = max(0, i - band)
        j_end = min(M, i + band + 1)
        for j in range(j_start, j_end):
            if i == 0 and j == 0:
                continue
            best = np.inf
            if i > 0:
                v = acc[i - 1, j]
                if v < best:
                    best = v
            if j > 0:
                v = acc[i, j - 1]
                if v < best:
                    best = v
            if i > 0 and j > 0:
                v = acc[i - 1, j - 1]
                if v < best:
                    best = v
            acc[i, j] = distance_matrix[i, j] + best
    return acc


def find_path(accumulated_cost: np.ndarray) -> List[Tuple[int, int]]:
    """Backtrack the optimal alignment path from the accumulated cost matrix."""
    N, M = accumulated_cost.shape
    i, j = N - 1, M - 1
    path = [(i, j)]
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            up = accumulated_cost[i - 1, j]
            left = accumulated_cost[i, j - 1]
            diag = accumulated_cost[i - 1, j - 1]
            best = min(up, left, diag)
            if best == diag:
                i -= 1
                j -= 1
            elif best == up:
                i -= 1
            else:
                j -= 1
        path.append((i, j))
    path.reverse()
    return path


def dtw(
    x: np.ndarray,
    y: np.ndarray,
    metric: str = "euclidean",
    band: Optional[int] = None,
    return_path: bool = False,
) -> Tuple[float, Optional[List[Tuple[int, int]]], Optional[np.ndarray]]:
    """
    Compute DTW distance between two sequences.

    Returns ``(distance, path_or_None, accumulated_cost_or_None)``. Path and
    accumulated cost are returned only when ``return_path=True``.
    """
    dist_mat = compute_distance_matrix(x, y, metric=metric)
    acc = compute_accumulated_cost(dist_mat, band=band)
    distance = float(acc[-1, -1])
    if return_path:
        return distance, find_path(acc), acc
    return distance, None, None
