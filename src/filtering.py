"""RGB-guided edge-aware filters for dense depth refinement."""

from __future__ import annotations

import numpy as np

from .preprocessing import validate_rgb


def _box_filter(array: np.ndarray, radius: int) -> np.ndarray:
    """Mean filter with a pure NumPy fallback when OpenCV is unavailable."""
    try:
        import cv2
        return cv2.boxFilter(array.astype(np.float32), -1, (2*radius+1, 2*radius+1), normalize=True, borderType=cv2.BORDER_REFLECT)
    except ImportError:
        pad = radius
        p = np.pad(array.astype(np.float64), ((pad,pad),(pad,pad)), mode="reflect")
        integral = np.pad(p, ((1,0),(1,0))).cumsum(0).cumsum(1)
        k = 2*radius+1
        sums = (integral[k:k+array.shape[0], k:k+array.shape[1]]
                - integral[:array.shape[0], k:k+array.shape[1]]
                - integral[k:k+array.shape[0], :array.shape[1]]
                + integral[:array.shape[0], :array.shape[1]])
        return (sums / (k*k)).astype(np.float32)


def _rgb_gray(rgb: np.ndarray) -> np.ndarray:
    image = validate_rgb(rgb).astype(np.float32) / 255.0
    return image @ np.array([0.299, 0.587, 0.114], dtype=np.float32)


def joint_bilateral_filter(depth: np.ndarray, rgb: np.ndarray, *, radius: int = 5, sigma_color: float = 0.1, sigma_space: float = 3.0) -> np.ndarray:
    """Joint bilateral filter whose range weights come from RGB and whose spatial weights come from distance."""
    z = np.asarray(depth, dtype=np.float32)
    guide = validate_rgb(rgb).astype(np.float32) / 255.0
    if z.ndim != 2 or z.shape != guide.shape[:2]:
        raise ValueError("depth and RGB dimensions must match")
    if radius < 1 or sigma_color <= 0 or sigma_space <= 0:
        raise ValueError("radius and sigmas must be positive")
    valid = np.isfinite(z)
    if not valid.any():
        raise ValueError("depth has no valid values")
    filled = np.where(valid, z, 0.0)
    h, w = z.shape
    weighted = np.zeros_like(z, dtype=np.float64)
    norm = np.zeros_like(z, dtype=np.float64)
    for dy in range(-radius, radius+1):
        for dx in range(-radius, radius+1):
            ys, yd = slice(max(0, -dy), min(h, h-dy)), slice(max(0, dy), min(h, h+dy))
            xs, xd = slice(max(0, -dx), min(w, w-dx)), slice(max(0, dx), min(w, w+dx))
            spatial = np.exp(-(dx*dx+dy*dy)/(2*sigma_space*sigma_space))
            delta = guide[yd, xd] - guide[ys, xs]
            color = np.exp(-np.sum(delta*delta, axis=2)/(2*sigma_color*sigma_color))
            weight = spatial * color * valid[ys, xs]
            weighted[yd, xd] += weight * filled[ys, xs]
            norm[yd, xd] += weight
    result = np.divide(weighted, norm, out=np.full_like(weighted, np.nan), where=norm > 0).astype(np.float32)
    result[~np.isfinite(result)] = z[~np.isfinite(result)]
    return result


def guided_filter(depth: np.ndarray, rgb: np.ndarray, *, radius: int = 8, epsilon: float = 1e-3) -> np.ndarray:
    """Guided filter using grayscale RGB as guide; uses only box filtering primitives."""
    z = np.asarray(depth, dtype=np.float32)
    guide = _rgb_gray(rgb)
    if z.ndim != 2 or z.shape != guide.shape:
        raise ValueError("depth and RGB dimensions must match")
    if radius < 1 or epsilon <= 0:
        raise ValueError("radius and epsilon must be positive")
    valid = np.isfinite(z)
    if not valid.any():
        raise ValueError("depth has no valid values")
    # Fill holes from local valid averages before computing local linear models.
    mean_valid = _box_filter(valid.astype(np.float32), radius)
    filled = np.where(valid, z, 0.0)
    base = _box_filter(filled, radius) / np.maximum(mean_valid, 1e-8)
    mean_i = _box_filter(guide, radius)
    mean_ii = _box_filter(guide*guide, radius)
    mean_iz = _box_filter(guide*filled, radius) / np.maximum(mean_valid, 1e-8)
    cov = mean_iz - mean_i*base
    var = mean_ii - mean_i*mean_i
    a = cov / (var + epsilon)
    b = base - a*mean_i
    mean_a = _box_filter(a, radius)
    mean_b = _box_filter(b, radius)
    result = mean_a*guide + mean_b
    return result.astype(np.float32)
