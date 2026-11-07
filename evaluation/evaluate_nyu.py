"""Zero-shot real-world NYU validation for the pretrained depth-refinement CNN.

This evaluator uses only NYU validation images. It does not train or fine-tune
the model. Depth arrays are in metres; model inputs and outputs are normalized
by the checkpoint's 10 metre training range.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from models.depth_refinement_cnn import DepthRefinementCNN  # noqa: F401 (checkpoint architecture)
from src.filtering import joint_bilateral_filter
from src.inference import load_model, refine_batch
from src.interpolation import upsample_depth
from src.metrics import evaluate
from src.preprocessing import area_downsample, fill_invalid_depth, load_depth, load_rgb

ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "datasets" / "nyu-depth" / "nyu-depth"
CHECKPOINT = ROOT / "models" / "checkpoints" / "depth_refinement_v2_best.pt"
RESULTS = ROOT / "evaluation" / "results" / "nyu"
HEIGHT = WIDTH = 128
SCALE = 4
DEPTH_RANGE_METRES = 10.0


def _matching_pairs(dataset_root: Path, split: str) -> list[tuple[str, Path, Path]]:
    """Find matching .jpg/.npy files in one NYU split, keyed by filename stem."""
    if split not in {"train", "val"}:
        raise ValueError("split must be 'train' or 'val'")
    rgb_dir, depth_dir = dataset_root / "images" / split, dataset_root / "depth" / split
    if not rgb_dir.is_dir() or not depth_dir.is_dir():
        raise FileNotFoundError(f"Expected validation folders: {rgb_dir} and {depth_dir}")
    rgb_files = {p.stem: p for p in rgb_dir.glob("*.jpg")}
    depth_files = {p.stem: p for p in depth_dir.glob("*.npy")}
    rgb_only, depth_only = sorted(rgb_files.keys() - depth_files.keys()), sorted(depth_files.keys() - rgb_files.keys())
    if rgb_only or depth_only:
        raise ValueError(f"Validation filenames do not match (RGB only: {len(rgb_only)}, depth only: {len(depth_only)})")
    pairs = [(stem, rgb_files[stem], depth_files[stem]) for stem in sorted(rgb_files)]
    if not pairs:
        raise ValueError(f"No matching validation pairs found in {rgb_dir} and {depth_dir}")
    return pairs


def _matching_validation_pairs(dataset_root: Path) -> list[tuple[str, Path, Path]]:
    """Backward-compatible validation-pair helper used by the zero-shot evaluator."""
    return _matching_pairs(dataset_root, "val")


def _resize_rgb(rgb: np.ndarray) -> np.ndarray:
    """Resize RGB to the fixed CNN input size using area resampling."""
    try:
        import cv2
        return cv2.resize(rgb, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
    except ImportError:
        from PIL import Image
        return np.asarray(Image.fromarray(rgb).resize((WIDTH, HEIGHT), Image.Resampling.BOX))


def _resize_depth_and_validity(depth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Resize metric depth while preventing NaN/zero holes from bleeding in."""
    valid = np.isfinite(depth) & (depth > 0)
    if not valid.any():
        raise ValueError("Depth map has no valid positive samples")
    try:
        import cv2
        weights = cv2.resize(valid.astype(np.float32), (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        totals = cv2.resize(np.where(valid, depth, 0).astype(np.float32),
                            (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
    except ImportError:
        # Pillow's floating-point BOX resize provides a deterministic area-style fallback.
        from PIL import Image
        weights = np.asarray(Image.fromarray(valid.astype(np.float32), mode="F").resize(
            (WIDTH, HEIGHT), Image.Resampling.BOX), dtype=np.float32)
        totals = np.asarray(Image.fromarray(np.where(valid, depth, 0).astype(np.float32), mode="F").resize(
            (WIDTH, HEIGHT), Image.Resampling.BOX), dtype=np.float32)
    resized_valid = weights > 1e-6
    resized = np.divide(totals, weights, out=np.full((HEIGHT, WIDTH), np.nan, np.float32), where=resized_valid)
    if not resized_valid.any():
        raise ValueError("Depth map has no valid values after resizing")
    return resized.astype(np.float32), resized_valid


def _make_comparison(rgb: np.ndarray, ground_truth: np.ndarray, outputs: dict[str, np.ndarray], path: Path) -> None:
    """Save a consistent-range visual comparison for one validation sample."""
    from PIL import Image, ImageDraw

    panels = [("RGB", rgb), ("Ground truth (m)", ground_truth),
              ("Bicubic (m)", outputs["bicubic"]),
              ("Joint bilateral (m)", outputs["joint_bilateral"]),
              ("CNN v2 (m)", outputs["cnn_v2"]),
              ("CNN absolute error (m)", np.abs(outputs["cnn_v2"] - ground_truth))]
    label_height = 22
    canvas = Image.new("RGB", (WIDTH * len(panels), HEIGHT + label_height), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (label, values) in enumerate(panels):
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        if values.ndim == 3:
            tile = Image.fromarray(values.astype(np.uint8), mode="RGB")
        elif "error" in label.lower():
            severity = np.clip(values / DEPTH_RANGE_METRES, 0, 1)
            heat = np.zeros((*values.shape, 3), dtype=np.uint8)
            heat[..., 0] = (severity * 255).astype(np.uint8)
            heat[..., 1] = ((1 - severity) * 80).astype(np.uint8)
            tile = Image.fromarray(heat)
        else:
            gray = (np.clip(values / DEPTH_RANGE_METRES, 0, 1) * 255).astype(np.uint8)
            tile = Image.fromarray(gray, mode="L").convert("RGB")
        x = index * WIDTH
        canvas.paste(tile, (x, label_height))
        draw.text((x + 3, 3), label, fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _json_safe(value):
    """Replace non-finite metric values with JSON null."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


@torch.inference_mode()
def evaluate_nyu(*, samples: int = 100, device: str | None = None,
                 output_prefix: str = "nyu_v2", dataset_root: str | Path = DATASET_ROOT,
                 checkpoint: str | Path = CHECKPOINT) -> dict:
    """Evaluate the checkpoint on a deterministic subset of NYU validation pairs."""
    if samples < 1:
        raise ValueError("samples must be positive")
    if not output_prefix or Path(output_prefix).name != output_prefix:
        raise ValueError("output_prefix must be a non-empty filename prefix")
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model, checkpoint_info = load_model(checkpoint, device=selected_device)
    depth_range = float(checkpoint_info.get("config", {}).get("depth_range", DEPTH_RANGE_METRES))
    if not np.isfinite(depth_range) or depth_range <= 0:
        raise ValueError("Checkpoint depth_range must be positive and finite")

    pairs = _matching_validation_pairs(Path(dataset_root))
    # Fixed seed and sorted source filenames make subset selection reproducible.
    rng = np.random.default_rng(1234)
    selected_indices = np.sort(rng.choice(len(pairs), size=min(samples, len(pairs)), replace=False))
    chosen = [pairs[int(i)] for i in selected_indices]
    scene_metrics: list[dict[str, dict[str, float]]] = []
    outputs_for_visual: dict[str, np.ndarray] | None = None
    first_rgb = first_gt = None
    for name, rgb_path, depth_path in chosen:
        rgb = _resize_rgb(load_rgb(rgb_path))
        raw_depth = load_depth(depth_path)  # NYU files store float16 metres; loader preserves units.
        gt, gt_valid = _resize_depth_and_validity(raw_depth)
        if rgb.shape[:2] != gt.shape:
            raise ValueError(f"Resized RGB/depth shape mismatch for {name}")

        # Area averaging creates genuine 4x sparse depth observations; NaNs are ignored.
        lowres = area_downsample(gt, SCALE)
        lowres_valid = np.isfinite(lowres)
        if not lowres_valid.any():
            raise ValueError(f"No valid low-resolution observations for {name}")
        filled_lowres = fill_invalid_depth(lowres)
        bicubic_depth = upsample_depth(filled_lowres, SCALE, "bicubic")[:HEIGHT, :WIDTH]
        bicubic_depth = np.clip(bicubic_depth, 0.0, depth_range)
        guide = np.ascontiguousarray(rgb)
        joint_bilateral = joint_bilateral_filter(bicubic_depth, guide)
        joint_bilateral = np.clip(joint_bilateral, 0.0, depth_range)

        # CNN receives the same bicubic initial estimate in normalized metres.
        rgb_tensor = torch.from_numpy(guide.transpose(2, 0, 1).copy()).float().unsqueeze(0).to(selected_device) / 255.0
        initial_tensor = torch.from_numpy(np.clip(bicubic_depth / depth_range, 0, 1)[None, None]).float().to(selected_device)
        lr_mask_hr = upsample_depth(lowres_valid.astype(np.float32), SCALE, "nearest")[:HEIGHT, :WIDTH]
        mask_tensor = torch.from_numpy((lr_mask_hr > .5)[None, None]).float().to(selected_device)
        prediction_normalized = refine_batch(model, rgb_tensor, initial_tensor, mask_tensor)
        prediction_metres = (prediction_normalized.squeeze(0).squeeze(0).cpu().numpy() * depth_range).astype(np.float32)
        prediction_metres = np.clip(prediction_metres, 0.0, depth_range)

        candidates = {"bicubic": bicubic_depth, "joint_bilateral": joint_bilateral,
                      "cnn_v2": prediction_metres}
        scene_metrics.append({method: evaluate(gt, pred, data_range=depth_range, valid_mask=gt_valid)
                              for method, pred in candidates.items()})
        if outputs_for_visual is None:
            outputs_for_visual, first_rgb, first_gt = candidates, guide, gt

    metric_names = scene_metrics[0]["bicubic"].keys()
    macro = {method: {metric: float(np.mean([scene[method][metric] for scene in scene_metrics]))
                      for metric in metric_names} for method in ("bicubic", "joint_bilateral", "cnn_v2")}
    RESULTS.mkdir(parents=True, exist_ok=True)
    visual_path = RESULTS / f"{output_prefix}_comparison.png"
    metrics_path = RESULTS / f"{output_prefix}_metrics.json"
    _make_comparison(first_rgb, first_gt, outputs_for_visual, visual_path)
    report = {
        "dataset": "NYU Depth validation; zero-shot real-world validation; no NYU training or fine-tuning",
        "split": "validation",
        "checkpoint": str(Path(checkpoint)),
        "device": str(selected_device),
        "gpu_name": torch.cuda.get_device_name(selected_device) if selected_device.type == "cuda" else None,
        "sample_count": len(chosen),
        "requested_samples": samples,
        "depth_units": "metres",
        "resolution": [HEIGHT, WIDTH],
        "scale": SCALE,
        "depth_range_metres": depth_range,
        "methods": ["bicubic", "joint_bilateral", "cnn_v2"],
        "metric_definitions": {
            "rmse_mae": "computed over finite positive reference depth pixels only",
            "psnr": f"fixed data range of {depth_range} metres",
            "ssim": "project fixed 7x7 Gaussian SSIM implementation, sigma 1.5, masked local statistics",
            "edges": "project depth-gradient threshold (0.10 metres/pixel), 1-pixel tolerance, precision/recall/F1",
            "bicubic": "4x area-downsampled observed depth, invalid samples filled from valid neighbours, bicubic upsampling",
            "joint_bilateral": "RGB-guided filter applied to the same bicubic initial depth",
            "cnn_v2": "normalized CNN output multiplied once by checkpoint depth_range to obtain metres",
        },
        "validation_filenames": [name for name, _, _ in chosen],
        "metrics_macro_average": macro,
        "visual_output": str(visual_path),
    }
    metrics_path.write_text(json.dumps(_json_safe(report), indent=2, allow_nan=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=100, help="number of validation images (default: 100)")
    parser.add_argument("--device", choices=("cpu", "cuda"), help="inference device; defaults to CUDA when available")
    parser.add_argument("--output-prefix", default="nyu_v2", help="prefix for metrics JSON and comparison PNG")
    args = parser.parse_args()
    report = evaluate_nyu(samples=args.samples, device=args.device, output_prefix=args.output_prefix)
    print(json.dumps(_json_safe(report["metrics_macro_average"]), indent=2, allow_nan=False))
    print(f"Metrics: {RESULTS / (args.output_prefix + '_metrics.json')}")
    print(f"Comparison: {RESULTS / (args.output_prefix + '_comparison.png')}")


if __name__ == "__main__":
    main()
