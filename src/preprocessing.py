"""Input loading and validation helpers for RGB guided depth refinement."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np

PathLike = Union[str, Path]


def area_downsample(depth: np.ndarray, scale: int) -> np.ndarray:
    """Average non-overlapping scale-square cells, ignoring invalid samples."""
    array = np.asarray(depth, dtype=np.float32)
    if array.ndim != 2 or scale not in (2, 4, 8):
        raise ValueError("depth must be 2D and scale must be 2, 4, or 8")
    height, width = array.shape
    if height % scale or width % scale:
        raise ValueError("depth dimensions must be divisible by scale")
    blocks = array.reshape(height//scale, scale, width//scale, scale).transpose(0, 2, 1, 3)
    valid = np.isfinite(blocks)
    counts = valid.sum(axis=(2, 3))
    totals = np.where(valid, blocks, 0).sum(axis=(2, 3))
    return np.divide(totals, counts, out=np.full(totals.shape, np.nan, np.float32), where=counts > 0)


def validate_rgb(rgb: np.ndarray) -> np.ndarray:
    """Validate an HxWx3 RGB image and return a contiguous array."""
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] != 3 or not image.shape[0] or not image.shape[1]:
        raise ValueError("RGB image must have shape (height, width, 3).")
    if image.dtype.kind not in "uif" or not np.isfinite(image).all():
        raise ValueError("RGB image must contain finite numeric values.")
    if image.dtype.kind == "f" and (image.min() < 0 or image.max() > 255):
        raise ValueError("Floating-point RGB values must be in [0, 255].")
    return np.ascontiguousarray(image.astype(np.uint8) if image.dtype.kind == "f" else image)


def validate_depth(depth: np.ndarray, expected_shape: tuple[int, int] | None = None) -> np.ndarray:
    """Return depth as float32, converting NaN/Inf and nonpositive values to NaN."""
    array = np.asarray(depth)
    if array.ndim == 3 and array.shape[2] == 1:
        array = array[..., 0]
    if array.ndim != 2 or not array.shape[0] or not array.shape[1]:
        raise ValueError("Depth map must be a non-empty 2D array.")
    if array.dtype.kind not in "uif":
        raise ValueError("Depth map must have a numeric data type.")
    if expected_shape is not None and array.shape != expected_shape:
        raise ValueError(f"Depth shape {array.shape} does not match RGB shape {expected_shape}.")
    result = array.astype(np.float32, copy=True)
    result[(~np.isfinite(result)) | (result <= 0)] = np.nan
    if not np.isfinite(result).any():
        raise ValueError("Depth map contains no valid positive values.")
    return result


def load_rgb(source: PathLike | np.ndarray) -> np.ndarray:
    """Load RGB from a path or validate an already-loaded array."""
    if isinstance(source, np.ndarray):
        return validate_rgb(source)
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        from PIL import Image
        with Image.open(path) as image:
            return validate_rgb(np.asarray(image.convert("RGB")))
    except ImportError:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("Loading image paths requires Pillow or OpenCV.") from exc
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not decode RGB image: {path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_depth(source: PathLike | np.ndarray) -> np.ndarray:
    """Load depth from NPY or an image file (single channel or grayscale)."""
    if isinstance(source, np.ndarray):
        return validate_depth(source)
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".npy":
        return validate_depth(np.load(path, allow_pickle=False))
    try:
        from PIL import Image
        with Image.open(path) as image:
            values = np.asarray(image)
    except ImportError:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("Loading image paths requires Pillow or OpenCV.") from exc
        values = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if values is None:
            raise ValueError(f"Could not decode depth image: {path}")
    if values.ndim == 3:
        if values.shape[2] == 1:
            values = values[..., 0]
        else:
            raise ValueError("Depth image must be single-channel.")
    return validate_depth(values)


def normalize_depth(depth: np.ndarray, *, invalid_value: float = 0.0) -> np.ndarray:
    """Min-max normalize valid values to [0, 1]; constant maps become zeros."""
    array = validate_depth(depth)
    valid = np.isfinite(array)
    low, high = float(array[valid].min()), float(array[valid].max())
    result = np.full(array.shape, invalid_value, dtype=np.float32)
    if high > low:
        result[valid] = (array[valid] - low) / (high - low)
    return result


def fill_invalid_depth(depth: np.ndarray) -> np.ndarray:
    """Iteratively fill invalid pixels from valid 8-neighbours.

    Each iteration reads the accumulated result. Remaining pixels (for example,
    an entirely invalid component) are initialized to the valid-value median.
    """
    array = validate_depth(depth)
    valid = np.isfinite(array)
    if valid.all():
        return array
    fill_value = float(np.median(array[valid]))
    result = np.where(valid, array, fill_value).astype(np.float32)
    height, width = array.shape
    for _ in range(max(array.shape)):
        missing = ~valid
        if not missing.any():
            break
        sums = np.zeros(array.shape, dtype=np.float64)
        counts = np.zeros(array.shape, dtype=np.uint8)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                y0, y1 = max(0, -dy), min(height, height-dy)
                x0, x1 = max(0, -dx), min(width, width-dx)
                src_y0, src_y1 = y0+dy, y1+dy
                src_x0, src_x1 = x0+dx, x1+dx
                neighbour_valid = valid[src_y0:src_y1, src_x0:src_x1]
                sums[y0:y1, x0:x1] += np.where(
                    neighbour_valid, result[src_y0:src_y1, src_x0:src_x1], 0.0
                )
                counts[y0:y1, x0:x1] += neighbour_valid
        can_fill = missing & (counts > 0)
        result[can_fill] = (sums[can_fill] / counts[can_fill]).astype(np.float32)
        valid[can_fill] = True
    return result
