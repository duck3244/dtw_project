"""Backend-level DTW tests: equivalence, determinism, band, validation."""

import numpy as np
import pytest

from backends import get_backend, is_accel_available

requires_accel = pytest.mark.skipif(
    not is_accel_available(), reason="accel backend (numba) not installed"
)


@pytest.fixture(scope="module")
def core():
    return get_backend("core")


@pytest.fixture(scope="module")
def accel():
    if not is_accel_available():
        pytest.skip("accel backend (numba) not installed")
    return get_backend("accel")


@pytest.fixture
def seqs():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((40, 13)).astype(np.float32)
    y = rng.standard_normal((50, 13)).astype(np.float32)
    return x, y


# ---- core baseline ----------------------------------------------------------

def test_core_deterministic(core, seqs):
    x, y = seqs
    d1, p1, _ = core.dtw(x, y, return_path=True)
    d2, p2, _ = core.dtw(x, y, return_path=True)
    assert d1 == d2
    assert p1 == p2


def test_core_path_endpoints(core, seqs):
    x, y = seqs
    _, path, _ = core.dtw(x, y, return_path=True)
    assert path[0] == (0, 0)
    assert path[-1] == (len(x) - 1, len(y) - 1)


def test_core_unsupported_metric(core, seqs):
    x, y = seqs
    with pytest.raises(ValueError):
        core.compute_distance_matrix(x, y, metric="bogus")


# ---- band semantics ---------------------------------------------------------

def test_large_band_matches_unconstrained(core, seqs):
    x, y = seqs  # |N - M| = 10
    d_full, _, _ = core.dtw(x, y, band=None)
    d_band, _, _ = core.dtw(x, y, band=100)
    assert d_full == d_band


def test_band_too_small_yields_inf(core, seqs):
    x, y = seqs  # |N - M| = 10
    d, _, _ = core.dtw(x, y, band=5)
    assert np.isinf(d)


# ---- core ≡ accel equivalence ----------------------------------------------

@requires_accel
@pytest.mark.parametrize("metric", ["euclidean", "manhattan", "cosine"])
@pytest.mark.parametrize("band", [None, 15, 30])
def test_core_accel_equivalence(core, accel, seqs, metric, band):
    x, y = seqs
    d_c, p_c, _ = core.dtw(x, y, metric=metric, band=band, return_path=True)
    d_a, p_a, _ = accel.dtw(x, y, metric=metric, band=band, return_path=True)
    if np.isinf(d_c) and np.isinf(d_a):
        return  # 양쪽 동일하게 도달 불가 — band 너무 작음
    assert np.isclose(d_c, d_a, rtol=1e-4, atol=1e-3)
    assert p_c == p_a


@requires_accel
@pytest.mark.parametrize(
    "shape",
    [(20, 20), (60, 80), (150, 200)],
)
def test_core_accel_distance_matrix_close(core, accel, shape):
    rng = np.random.default_rng(1)
    x = rng.standard_normal((shape[0], 13)).astype(np.float32)
    y = rng.standard_normal((shape[1], 13)).astype(np.float32)
    dm_c = core.compute_distance_matrix(x, y, metric="euclidean")
    dm_a = accel.compute_distance_matrix(x, y, metric="euclidean")
    assert np.allclose(dm_c, dm_a, rtol=1e-4, atol=1e-3)


# ---- self-distance & symmetry sanity ----------------------------------------

def test_self_distance_is_minimal(core):
    rng = np.random.default_rng(2)
    x = rng.standard_normal((30, 13)).astype(np.float32)
    y = rng.standard_normal((30, 13)).astype(np.float32)
    d_self, _, _ = core.dtw(x, x)
    d_other, _, _ = core.dtw(x, y)
    assert d_self <= d_other
