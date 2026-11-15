"""Tests for Compression-Lib codecs and best_method selection."""

import numpy as np
import pytest

from compression_lib import (
    best_method,
    compression_ratio,
    estimated_storage_bytes,
    fourier_basis,
    fourier_compress,
    hybrid_compress,
    quantize,
    svd_compress,
)


def _smooth_matrix(n=32):
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return (np.sin(t)[None, :] + 0.5 * np.cos(2 * t)[None, :]) * np.ones((n, 1))


def test_svd_compress_low_rank_exact():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(8, 3)) @ rng.normal(size=(3, 12))
    approx = svd_compress(a, rank=3)
    assert np.allclose(approx, a, atol=1e-8)


def test_quantize_bounds_and_error():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(16, 16))
    q = quantize(a, bits=8)
    assert q.min() >= a.min() - 1e-9 and q.max() <= a.max() + 1e-9
    assert np.linalg.norm(a - q) / np.linalg.norm(a) < 0.05


def test_fourier_compress_smooth_exact():
    a = _smooth_matrix()
    approx = fourier_compress(a, num_coeffs=4)
    assert np.linalg.norm(a - approx) / np.linalg.norm(a) < 1e-6


def test_hybrid_compress_runs():
    a = _smooth_matrix()
    assert hybrid_compress(a).shape == a.shape


def test_estimated_storage_and_ratio():
    a = np.ones((32, 32))
    assert estimated_storage_bytes(a, "quantize8") == 32 * 32
    assert compression_ratio(a, "quantize8") == pytest.approx(8.0)


def test_best_method_picks_within_budget():
    a = _smooth_matrix()
    pick = best_method(a, max_error=0.01)
    assert pick is not None and pick["relative_error"] <= 0.01


def test_best_method_none_for_random():
    rng = np.random.default_rng(2)
    a = rng.normal(size=(32, 32))
    assert best_method(a, max_error=1e-6) is None


def test_validation():
    with pytest.raises(ValueError):
        svd_compress(np.ones((4, 4)), rank=0)
    with pytest.raises(ValueError):
        quantize(np.ones((4, 4)), bits=32)
    with pytest.raises(ValueError):
        estimated_storage_bytes(np.ones((4, 4)), "nope")
