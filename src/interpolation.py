"""Depth upsampling methods with scale factors 2, 4 and 8."""

from __future__ import annotations

import numpy as np

_METHODS = {"nearest": 0, "bilinear": 1, "bicubic": 2}


def _resize_numpy(image: np.ndarray, scale: int, method: str) -> np.ndarray:
    """NumPy fallback using OpenCV-style half-pixel coordinates."""
    h, w = image.shape
    oh, ow = h * scale, w * scale
    y = np.clip((np.arange(oh) + 0.5) / scale - 0.5, 0, h - 1)
    x = np.clip((np.arange(ow) + 0.5) / scale - 0.5, 0, w - 1)
    if method == "nearest":
        return image[np.rint(y).astype(int)[:, None], np.rint(x).astype(int)[None, :]]
    if method == "bicubic":
        def cubic_weight(distance):
            distance = np.abs(distance)
            return np.where(distance <= 1, 1.5*distance**3 - 2.5*distance**2 + 1,
                            np.where(distance < 2, -0.5*distance**3 + 2.5*distance**2 - 4*distance + 2, 0))
        out = np.zeros((oh, ow), dtype=np.float64)
        for oy in range(-1, 3):
            iy = np.clip(np.floor(y).astype(int) + oy, 0, h-1)
            wy = cubic_weight(y - (np.floor(y) + oy))
            for ox in range(-1, 3):
                ix = np.clip(np.floor(x).astype(int) + ox, 0, w-1)
                wx = cubic_weight(x - (np.floor(x) + ox))
                out += (wy[:, None] * wx[None, :]) * image[iy[:, None], ix[None, :]]
        return out
    y0, x0 = np.floor(y).astype(int), np.floor(x).astype(int)
    y1, x1 = np.minimum(y0 + 1, h - 1), np.minimum(x0 + 1, w - 1)
    wy, wx = y - y0, x - x0
    return ((1-wy[:, None]) * ((1-wx[None, :])*image[y0[:, None], x0[None, :]] + wx[None, :]*image[y0[:, None], x1[None, :]]) + wy[:, None] * ((1-wx[None, :])*image[y1[:, None], x0[None, :]] + wx[None, :]*image[y1[:, None], x1[None, :]]))


def upsample_depth(depth: np.ndarray, scale: int = 2, method: str = "bilinear") -> np.ndarray:
    """Upsample a 2D depth map to ``(H*scale, W*scale)``."""
    if scale not in (2, 4, 8):
        raise ValueError("scale must be one of 2, 4, or 8")
    key = method.lower().replace("-", "_")
    key = {"nearest_neighbor": "nearest", "nearest_neighbour": "nearest", "linear": "bilinear", "cubic": "bicubic"}.get(key, key)
    if key not in _METHODS:
        raise ValueError(f"Unknown interpolation method: {method}")
    src = np.asarray(depth, dtype=np.float32)
    if src.ndim != 2 or min(src.shape) < 1:
        raise ValueError("depth must be a non-empty 2D array")
    if not np.isfinite(src).all():
        raise ValueError("depth must be finite; handle missing values before interpolation")
    try:
        import cv2
    except ImportError:
        return _resize_numpy(src, scale, key).astype(np.float32)
    modes = {"nearest": cv2.INTER_NEAREST, "bilinear": cv2.INTER_LINEAR, "bicubic": cv2.INTER_CUBIC}
    return cv2.resize(src, (src.shape[1] * scale, src.shape[0] * scale), interpolation=modes[key]).astype(np.float32)


def nearest_neighbour(depth: np.ndarray, scale: int = 2) -> np.ndarray:
    return upsample_depth(depth, scale, "nearest")


def bilinear(depth: np.ndarray, scale: int = 2) -> np.ndarray:
    return upsample_depth(depth, scale, "bilinear")


def bicubic(depth: np.ndarray, scale: int = 2) -> np.ndarray:
    return upsample_depth(depth, scale, "bicubic")
