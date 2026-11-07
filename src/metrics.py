"""Depth quality metrics and repeated timing helpers.

PSNR uses a fixed 10.0 depth-unit data range by default. SSIM is a deterministic
7x7, sigma=1.5 Gaussian-window implementation with population covariance and
validity-normalized local statistics, independent of optional libraries.
"""

from __future__ import annotations

import math
import time
from typing import Callable

import numpy as np

from .edge_detection import edge_precision_recall_f1

DEFAULT_DEPTH_DATA_RANGE = 10.0
SSIM_WINDOW_SIZE = 7
SSIM_SIGMA = 1.5


def _pair(reference: np.ndarray, prediction: np.ndarray,
          valid_mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a, b = np.asarray(reference, dtype=np.float64), np.asarray(prediction, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("reference and prediction must be matching 2D arrays")
    valid = np.isfinite(a) & np.isfinite(b)
    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.shape != a.shape:
            raise ValueError("valid_mask must match image dimensions")
        valid &= mask
    if not valid.any():
        raise ValueError("no valid overlapping pixels")
    return a, b, valid


def rmse(reference: np.ndarray, prediction: np.ndarray, *, valid_mask=None) -> float:
    a, b, valid = _pair(reference, prediction, valid_mask)
    return float(np.sqrt(np.mean((a[valid] - b[valid]) ** 2)))


def mae(reference: np.ndarray, prediction: np.ndarray, *, valid_mask=None) -> float:
    a, b, valid = _pair(reference, prediction, valid_mask)
    return float(np.mean(np.abs(a[valid] - b[valid])))


def psnr(reference: np.ndarray, prediction: np.ndarray, *, data_range: float = DEFAULT_DEPTH_DATA_RANGE,
         valid_mask=None) -> float:
    if not np.isfinite(data_range) or data_range <= 0:
        raise ValueError("data_range must be a positive finite value")
    mse = rmse(reference, prediction, valid_mask=valid_mask) ** 2
    return math.inf if mse == 0 else float(10 * math.log10(data_range ** 2 / mse))


def _gaussian_blur(array: np.ndarray) -> np.ndarray:
    radius = SSIM_WINDOW_SIZE // 2
    coords = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(coords ** 2) / (2 * SSIM_SIGMA ** 2))
    kernel /= kernel.sum()
    result = array.astype(np.float64)
    # Fixed symmetric padding and separable convolution for reproducibility.
    for axis in (0, 1):
        pads = [(0, 0), (0, 0)]
        pads[axis] = (radius, radius)
        padded = np.pad(result, pads, mode="symmetric")
        result = sum(kernel[i] * np.take(padded, np.arange(result.shape[axis]) + i, axis=axis)
                     for i in range(SSIM_WINDOW_SIZE))
    return result


def ssim(reference: np.ndarray, prediction: np.ndarray, *, data_range: float = DEFAULT_DEPTH_DATA_RANGE,
         valid_mask=None) -> float:
    """Masked local SSIM with fixed 7x7 Gaussian (sigma 1.5), C1/C2 per Wang et al."""
    if not np.isfinite(data_range) or data_range <= 0:
        raise ValueError("data_range must be a positive finite value")
    a, b, valid = _pair(reference, prediction, valid_mask)
    weight = valid.astype(np.float64)
    # Normalize each local statistic by valid support so holes do not act as zero depth.
    support = np.maximum(_gaussian_blur(weight), 1e-12)
    def local(value):
        return _gaussian_blur(np.where(valid, value, 0.0)) / support
    mux, muy = local(a), local(b)
    vx = np.maximum(local(a * a) - mux * mux, 0.0)
    vy = np.maximum(local(b * b) - muy * muy, 0.0)
    covariance = local(a * b) - mux * muy
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    score = ((2 * mux * muy + c1) * (2 * covariance + c2)) / (
        (mux * mux + muy * muy + c1) * (vx + vy + c2)
    )
    return float(np.mean(score[valid]))


def processing_time(function: Callable, *args, warmup_runs: int = 1,
                    timed_runs: int = 5, **kwargs) -> tuple[object, dict[str, object]]:
    """Warm up, time repeatedly, and return result plus raw and median seconds."""
    if warmup_runs < 0 or timed_runs < 1:
        raise ValueError("warmup_runs must be >= 0 and timed_runs must be >= 1")
    result = None
    for _ in range(warmup_runs):
        result = function(*args, **kwargs)
    samples = []
    for _ in range(timed_runs):
        start = time.perf_counter()
        result = function(*args, **kwargs)
        samples.append(time.perf_counter() - start)
    return result, {"median_seconds": float(np.median(samples)), "samples_seconds": samples,
                    "warmup_runs": warmup_runs, "timed_runs": timed_runs}


def evaluate(reference: np.ndarray, prediction: np.ndarray, *,
             data_range: float = DEFAULT_DEPTH_DATA_RANGE, valid_mask=None) -> dict[str, float]:
    a, b, valid = _pair(reference, prediction, valid_mask)
    edges = edge_precision_recall_f1(a, b, valid_mask=valid)
    return {"rmse": rmse(a, b, valid_mask=valid), "mae": mae(a, b, valid_mask=valid),
            "psnr": psnr(a, b, data_range=data_range, valid_mask=valid),
            "ssim": ssim(a, b, data_range=data_range, valid_mask=valid), **edges}
