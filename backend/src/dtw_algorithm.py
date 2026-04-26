"""
DTW Algorithm Module — backend-aware facade.

Public classes (``DTWAlgorithm``, ``ConstrainedDTW``, ``FastDTW``) preserve their
historical NumPy-only API for backwards compatibility, but delegate the heavy
lifting to a swappable backend selected via ``backends.get_backend``.

Default backend is ``"auto"`` (prefers accel when numba is installed).
"""

from typing import List, Optional, Tuple

import numpy as np

from backends import BackendName, get_backend


class DTWAlgorithm:
    """
    Dynamic Time Warping with pluggable backend.

    Args:
        distance_metric: ``'euclidean' | 'manhattan' | 'cosine'``.
        backend: ``'auto' | 'core' | 'accel'``.
        band: Sakoe-Chiba band radius. ``None`` (default) is unconstrained;
            an integer constrains the warping path so that ``|i-j| <= band``,
            which speeds up large sequences and suppresses pathological warps.
            Choose roughly 10–20% of the longer sequence; too small a band
            can leave the end-cell unreachable (``inf``).
    """

    def __init__(
        self,
        distance_metric: str = "euclidean",
        backend: BackendName = "auto",
        band: Optional[int] = None,
    ):
        self._backend_name: BackendName = backend
        self._backend = get_backend(backend)
        self.supported_metrics = list(self._backend.SUPPORTED_METRICS)
        if distance_metric not in self.supported_metrics:
            raise ValueError(
                f"Unsupported distance metric: {distance_metric}. "
                f"Supported: {self.supported_metrics}"
            )
        if band is not None and band < 0:
            raise ValueError(f"band must be non-negative, got {band}")
        self.distance_metric = distance_metric
        self.band = band

    @property
    def backend_name(self) -> str:
        return self._backend.BACKEND_NAME

    def compute_distance_matrix(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self._backend.compute_distance_matrix(x, y, metric=self.distance_metric)

    def compute_accumulated_cost(
        self,
        distance_matrix: np.ndarray,
        band: Optional[int] = None,
    ) -> np.ndarray:
        # explicit band overrides the instance default
        effective_band = band if band is not None else self.band
        return self._backend.compute_accumulated_cost(
            distance_matrix, band=effective_band
        )

    def find_path(self, accumulated_cost: np.ndarray) -> List[Tuple[int, int]]:
        return self._backend.find_path(accumulated_cost)

    def compute_dtw(
        self,
        x: np.ndarray,
        y: np.ndarray,
        return_path: bool = False,
        band: Optional[int] = None,
    ) -> Tuple[float, Optional[List[Tuple[int, int]]], Optional[np.ndarray]]:
        effective_band = band if band is not None else self.band
        return self._backend.dtw(
            x, y,
            metric=self.distance_metric,
            band=effective_band,
            return_path=return_path,
        )

    def compute_dtw_normalized(
        self, x: np.ndarray, y: np.ndarray, band: Optional[int] = None
    ) -> float:
        distance, path, _ = self.compute_dtw(x, y, return_path=True, band=band)
        return distance / len(path)


class ConstrainedDTW:
    """DTW with a Sakoe-Chiba band constraint."""

    def __init__(
        self,
        window_size: int = 10,
        distance_metric: str = "euclidean",
        backend: BackendName = "auto",
    ):
        self.window_size = window_size
        self._dtw = DTWAlgorithm(distance_metric=distance_metric, backend=backend)

    def compute_constrained_dtw(self, x: np.ndarray, y: np.ndarray) -> float:
        distance, _, _ = self._dtw._backend.dtw(
            x, y,
            metric=self._dtw.distance_metric,
            band=self.window_size,
            return_path=False,
        )
        return distance


class FastDTW:
    """
    Approximate DTW with multi-resolution refinement.

    The previous placeholder implementation simply fell back to full DTW; this
    version does a coarse-to-fine pass that constrains the fine search to a
    band around the upsampled coarse path, which is the spirit of FastDTW.
    """

    def __init__(self, radius: int = 1, distance_metric: str = "euclidean",
                 backend: BackendName = "auto"):
        self.radius = max(1, int(radius))
        self._dtw = DTWAlgorithm(distance_metric=distance_metric, backend=backend)

    def compute_fastdtw(self, x: np.ndarray, y: np.ndarray) -> float:
        N, M = len(x), len(y)
        # Use full DTW for tiny inputs where downsampling is meaningless.
        if N < 20 or M < 20:
            distance, _, _ = self._dtw.compute_dtw(x, y)
            return distance

        # Coarse pass on 2x downsampled sequences to estimate warping skew.
        x_lo = x[::2]
        y_lo = y[::2]
        _, coarse_path, _ = self._dtw.compute_dtw(x_lo, y_lo, return_path=True)

        # Pick a band that covers the upsampled coarse warp plus the requested
        # search radius. Using a global band keeps the call backend-agnostic.
        max_skew = max((abs(2 * i - 2 * j) for i, j in coarse_path), default=0)
        band = max(max_skew, abs(N - M)) + self.radius
        distance, _, _ = self._dtw._backend.dtw(
            x, y,
            metric=self._dtw.distance_metric,
            band=band,
            return_path=False,
        )
        return distance


if __name__ == "__main__":
    print("DTW Algorithm Module — backend facade")
    dtw = DTWAlgorithm()
    print(f"Active backend: {dtw.backend_name}")
