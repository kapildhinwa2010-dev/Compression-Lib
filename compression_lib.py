# Compression-Lib - Reusable compression-codec library
# Part of the Kapildhinwa2010-dev Toolkit.

"""Compression-Lib - reusable matrix compression codecs.

Four codecs on 2-D float matrices: SVD low-rank, uniform quantization,
Fourier-basis fitting, and a hybrid (SVD then quantize). This library
exposes the codecs plus best_method(), which picks the smallest codec
that meets a relative-error budget.

Honest limitation: none of these codecs beats plain float32 storage for
random matrices - real gains only appear on low-rank or smooth data.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "fourier_basis",
    "svd_compress",
    "quantize",
    "fourier_compress",
    "hybrid_compress",
    "estimated_storage_bytes",
    "compression_ratio",
    "best_method",
]


def fourier_basis(n_points: int, num_coeffs: int = 4) -> np.ndarray:
    """Design matrix [1, sin t, cos t, sin 2t, cos 2t, ...] over one period."""
    if n_points < 1 or num_coeffs < 1:
        raise ValueError("n_points and num_coeffs must be >= 1")
    t = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    cols = [np.ones(n_points)]
    for k in range(1, num_coeffs + 1):
        cols.append(np.sin(k * t))
        cols.append(np.cos(k * t))
    return np.stack(cols, axis=1)


def _check(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.size == 0:
        raise ValueError("matrix must be non-empty and 2-D")
    return matrix


def svd_compress(matrix: np.ndarray, rank: int = 4) -> np.ndarray:
    matrix = _check(matrix)
    if rank < 1:
        raise ValueError("rank must be >= 1")
    u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    rank = min(rank, s.size)
    return (u[:, :rank] * s[:rank]) @ vt[:rank, :]


def quantize(matrix: np.ndarray, bits: int = 8) -> np.ndarray:
    matrix = _check(matrix)
    if bits < 1 or bits > 16:
        raise ValueError("bits must be in [1, 16]")
    lo, hi = float(matrix.min()), float(matrix.max())
    levels = 2 ** bits - 1
    scale = (hi - lo) / levels if hi > lo else 1.0
    codes = np.clip(np.round((matrix - lo) / scale), 0, levels)
    return codes * scale + lo


def fourier_compress(matrix: np.ndarray, num_coeffs: int = 4) -> np.ndarray:
    matrix = _check(matrix)
    if num_coeffs < 1:
        raise ValueError("num_coeffs must be >= 1")
    basis = fourier_basis(matrix.shape[1], num_coeffs)
    coeffs, _, _, _ = np.linalg.lstsq(basis, matrix.T, rcond=None)
    return coeffs.T @ basis.T


def hybrid_compress(matrix: np.ndarray, rank: int = 4, bits: int = 8) -> np.ndarray:
    """SVD low-rank followed by quantization."""
    return quantize(svd_compress(matrix, rank=rank), bits=bits)


def _method_map(matrix: np.ndarray) -> dict:
    return {
        "svd_rank4": svd_compress(matrix, rank=4),
        "quantize8": quantize(matrix, bits=8),
        "fourier4": fourier_compress(matrix, num_coeffs=4),
        "hybrid": hybrid_compress(matrix, rank=4, bits=8),
    }


def estimated_storage_bytes(matrix: np.ndarray, method: str) -> int:
    """Rough stored-size estimate for one codec.

    svd and fourier coefficients are counted as float64 (8 bytes each);
    quantize8 and hybrid store one byte per value.
    """
    matrix = _check(matrix)
    m, n = matrix.shape
    counts = {
        "svd_rank4": 8 * 4 * (m + n + 1),
        "quantize8": m * n,
        "fourier4": 8 * (2 * 4 + 1) * n,
        "hybrid": 4 * (m + n + 1),
    }
    if method not in counts:
        raise ValueError("unknown method: {}".format(method))
    return int(counts[method])


def compression_ratio(matrix: np.ndarray, method: str) -> float:
    """original float64 bytes / estimated stored bytes (>= 1 means smaller)."""
    matrix = _check(matrix)
    original = matrix.size * 8
    return original / max(1, estimated_storage_bytes(matrix, method))


def best_method(matrix: np.ndarray, max_error: float = 0.1) -> dict:
    """Smallest codec whose relative error is within max_error.

    Returns a dict with method, relative_error, estimated_storage_bytes and
    compression_ratio, or None when no codec meets the budget (typical for
    random matrices).
    """
    matrix = _check(matrix)
    if max_error < 0:
        raise ValueError("max_error must be >= 0")
    norm = float(np.linalg.norm(matrix))
    best = None
    for name, approx in _method_map(matrix).items():
        rel = float(np.linalg.norm(matrix - approx)) / norm if norm else 0.0
        if rel > max_error:
            continue
        size = estimated_storage_bytes(matrix, name)
        if best is None or size < best["estimated_storage_bytes"]:
            best = {
                "method": name,
                "relative_error": rel,
                "estimated_storage_bytes": size,
                "compression_ratio": matrix.size * 8 / max(1, size),
            }
    return best
