"""
DTW Backends

- ``core`` : pure-NumPy reference implementation (no extra dependencies).
- ``accel``: optional numba/torch accelerated implementation.

Use :func:`get_backend` to select an implementation. The accel backend is
imported lazily so that environments without numba/torch can still use core.
"""

from typing import Literal

BackendName = Literal["core", "accel", "auto"]

_AVAILABLE = {"core": True, "accel": None}


def is_accel_available() -> bool:
    """Return True iff the accel backend is fully importable (numba + module)."""
    if _AVAILABLE["accel"] is None:
        try:
            import numba  # noqa: F401
            from .accel import dtw as _accel_dtw  # noqa: F401
            _AVAILABLE["accel"] = True
        except ImportError:
            _AVAILABLE["accel"] = False
    return _AVAILABLE["accel"]


def get_backend(name: BackendName = "auto"):
    """
    Return the requested DTW backend module.

    Args:
        name: ``"core"``, ``"accel"``, or ``"auto"`` (prefer accel if available).
    """
    if name == "auto":
        name = "accel" if is_accel_available() else "core"

    if name == "core":
        from .core import dtw as backend
        return backend
    if name == "accel":
        if not is_accel_available():
            raise ImportError(
                "accel backend requested but numba is not installed. "
                "Install via: pip install -r requirements-accel.txt"
            )
        from .accel import dtw as backend
        return backend
    raise ValueError(f"Unknown backend: {name!r}")
