"""
Accel DTW backend — numba JIT for inner DP loops, BLAS-backed cdist for the
pairwise distance matrix.

Exposes the same function-level API as ``backends.core.dtw`` so callers can
swap implementations without code changes.
"""

from typing import List, Optional, Tuple

import numpy as np
from numba import njit
from scipy.spatial.distance import cdist

BACKEND_NAME = "accel"
SUPPORTED_METRICS = ("euclidean", "manhattan", "cosine")


def compute_distance_matrix(
    x: np.ndarray,
    y: np.ndarray,
    metric: str = "euclidean",
) -> np.ndarray:
    if metric not in SUPPORTED_METRICS:
        raise ValueError(
            f"Unsupported distance metric: {metric}. Supported: {SUPPORTED_METRICS}"
        )
    x = np.ascontiguousarray(x, dtype=np.float32)
    y = np.ascontiguousarray(y, dtype=np.float32)

    if metric == "euclidean":
        return cdist(x, y, metric="euclidean").astype(np.float32)
    if metric == "manhattan":
        return cdist(x, y, metric="cityblock").astype(np.float32)

    # cosine — guarded against zero-norm rows (matches core behaviour)
    x_norm = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
    y_norm = y / (np.linalg.norm(y, axis=1, keepdims=True) + 1e-8)
    return (1.0 - x_norm @ y_norm.T).astype(np.float32)


@njit(cache=True)
def _accum_full(dm: np.ndarray) -> np.ndarray:
    N, M = dm.shape
    acc = np.empty((N, M), dtype=np.float32)
    acc[0, 0] = dm[0, 0]
    for i in range(1, N):
        acc[i, 0] = acc[i - 1, 0] + dm[i, 0]
    for j in range(1, M):
        acc[0, j] = acc[0, j - 1] + dm[0, j]
    for i in range(1, N):
        for j in range(1, M):
            a = acc[i - 1, j]
            b = acc[i, j - 1]
            c = acc[i - 1, j - 1]
            best = a if a < b else b
            if c < best:
                best = c
            acc[i, j] = dm[i, j] + best
    return acc


@njit(cache=True)
def _accum_banded(dm: np.ndarray, band: int) -> np.ndarray:
    N, M = dm.shape
    INF = np.float32(np.inf)
    acc = np.full((N, M), INF, dtype=np.float32)
    acc[0, 0] = dm[0, 0]
    for i in range(N):
        j_start = i - band
        if j_start < 0:
            j_start = 0
        j_end = i + band + 1
        if j_end > M:
            j_end = M
        for j in range(j_start, j_end):
            if i == 0 and j == 0:
                continue
            best = INF
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
            acc[i, j] = dm[i, j] + best
    return acc


def compute_accumulated_cost(
    distance_matrix: np.ndarray,
    band: Optional[int] = None,
) -> np.ndarray:
    dm = np.ascontiguousarray(distance_matrix, dtype=np.float32)
    if band is None:
        return _accum_full(dm)
    return _accum_banded(dm, int(band))


@njit(cache=True)
def _find_path_jit(acc: np.ndarray):
    N, M = acc.shape
    max_len = N + M
    path_i = np.empty(max_len, dtype=np.int64)
    path_j = np.empty(max_len, dtype=np.int64)
    i = N - 1
    j = M - 1
    path_i[0] = i
    path_j[0] = j
    p = 1
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            up = acc[i - 1, j]
            left = acc[i, j - 1]
            diag = acc[i - 1, j - 1]
            best = diag
            if up < best:
                best = up
            if left < best:
                best = left
            # tie-break order matches core: diag > up > left
            if best == diag:
                i -= 1
                j -= 1
            elif best == up:
                i -= 1
            else:
                j -= 1
        path_i[p] = i
        path_j[p] = j
        p += 1
    return path_i[:p], path_j[:p]


def find_path(accumulated_cost: np.ndarray) -> List[Tuple[int, int]]:
    acc = np.ascontiguousarray(accumulated_cost, dtype=np.float32)
    pi, pj = _find_path_jit(acc)
    # core API returns List[Tuple[int, int]] in start-to-end order
    return [(int(i), int(j)) for i, j in zip(pi[::-1], pj[::-1])]


def dtw(
    x: np.ndarray,
    y: np.ndarray,
    metric: str = "euclidean",
    band: Optional[int] = None,
    return_path: bool = False,
) -> Tuple[float, Optional[List[Tuple[int, int]]], Optional[np.ndarray]]:
    dm = compute_distance_matrix(x, y, metric=metric)
    acc = compute_accumulated_cost(dm, band=band)
    distance = float(acc[-1, -1])
    if return_path:
        return distance, find_path(acc), acc
    return distance, None, None
