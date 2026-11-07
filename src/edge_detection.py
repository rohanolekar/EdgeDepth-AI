"""Deterministic edge metrics for depth maps.

Depth edges use central-difference gradient magnitude with a fixed absolute
threshold (depth units per pixel). Precision and recall use symmetric square
matching tolerance: unmatched predicted edges are false positives.
"""

from __future__ import annotations

import numpy as np

from .preprocessing import validate_rgb

EDGE_THRESHOLD = 0.10
EDGE_TOLERANCE_PIXELS = 1


def detect_rgb_edges(rgb: np.ndarray, *, threshold: float = 40.0) -> np.ndarray:
    """Detect RGB edges with Canny (low=threshold, high=2*threshold)."""
    image = validate_rgb(rgb)
    try:
        import cv2
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return cv2.Canny(gray, int(threshold), int(2 * threshold)) > 0
    except ImportError:
        gray = image.astype(np.float32).mean(axis=2)
        gy, gx = np.gradient(gray)
        return np.hypot(gx, gy) >= threshold


def depth_edges(depth: np.ndarray, *, threshold: float = EDGE_THRESHOLD,
                valid_mask: np.ndarray | None = None) -> np.ndarray:
    """Threshold central-difference gradient magnitude at a fixed depth-unit threshold."""
    z = np.asarray(depth, dtype=np.float64)
    if z.ndim != 2 or min(z.shape) < 2:
        raise ValueError("depth must be a 2D array with both dimensions >= 2")
    valid = np.isfinite(z)
    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.shape != z.shape:
            raise ValueError("valid_mask must match depth dimensions")
        valid &= mask
    # Fill invalid pixels for gradient calculation only; exclude their edges.
    fill = float(np.median(z[valid])) if valid.any() else 0.0
    safe = np.where(valid, z, fill)
    gy, gx = np.gradient(safe)
    return (np.hypot(gx, gy) >= threshold) & valid


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    padded = np.pad(mask, radius, mode="constant")
    result = np.zeros(mask.shape, dtype=bool)
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            result |= padded[dy:dy+mask.shape[0], dx:dx+mask.shape[1]]
    return result


def edge_precision_recall_f1(
    reference_depth: np.ndarray,
    predicted_depth: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
    threshold: float = EDGE_THRESHOLD,
    tolerance: int = EDGE_TOLERANCE_PIXELS,
) -> dict[str, float]:
    """Return tolerance-matched edge precision, recall and F1.

    Predicted edges within ``tolerance`` of a valid reference edge are true
    positives; unmatched predictions count as false positives. Matching is
    symmetric and one-to-many (standard tolerant edge-map evaluation).
    """
    ref = np.asarray(reference_depth)
    pred = np.asarray(predicted_depth)
    if ref.shape != pred.shape or ref.ndim != 2:
        raise ValueError("depth maps must be matching 2D arrays")
    if tolerance < 0 or threshold < 0:
        raise ValueError("threshold and tolerance must be non-negative")
    valid = np.isfinite(ref) & np.isfinite(pred)
    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.shape != ref.shape:
            raise ValueError("valid_mask must match depth dimensions")
        valid &= mask
    ref_edges = depth_edges(ref, threshold=threshold, valid_mask=valid)
    pred_edges = depth_edges(pred, threshold=threshold, valid_mask=valid)
    n_ref, n_pred = int(ref_edges.sum()), int(pred_edges.sum())
    matched_pred = pred_edges & _dilate(ref_edges, tolerance)
    matched_ref = ref_edges & _dilate(pred_edges, tolerance)
    precision = float(matched_pred.sum() / n_pred) if n_pred else (1.0 if n_ref == 0 else 0.0)
    recall = float(matched_ref.sum() / n_ref) if n_ref else (1.0 if n_pred == 0 else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"edge_precision": precision, "edge_recall": recall, "edge_f1": f1}


def edge_preservation(reference_depth: np.ndarray, predicted_depth: np.ndarray, *, tolerance: int = EDGE_TOLERANCE_PIXELS) -> float:
    """Compatibility wrapper returning edge recall; use the full PR/F1 metric for evaluation."""
    return edge_precision_recall_f1(reference_depth, predicted_depth, tolerance=tolerance)["edge_recall"]
