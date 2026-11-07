"""Compare bicubic, joint bilateral, v2, and fine-tuned v3 on NYU validation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from evaluation.evaluate_nyu import (
    CHECKPOINT as V2_CHECKPOINT,
    DATASET_ROOT,
    DEPTH_RANGE_METRES,
    HEIGHT,
    RESULTS,
    SCALE,
    WIDTH,
    _matching_validation_pairs,
    _resize_depth_and_validity,
    _resize_rgb,
)
from src.filtering import joint_bilateral_filter
from src.inference import load_model, refine_batch
from src.interpolation import upsample_depth
from src.metrics import evaluate
from src.preprocessing import area_downsample, fill_invalid_depth, load_depth, load_rgb

ROOT = Path(__file__).resolve().parents[1]
V3_CHECKPOINT = ROOT / "models" / "checkpoints" / "depth_refinement_v3_best.pt"
METRIC_KEYS = ("rmse", "mae", "psnr", "ssim", "edge_precision", "edge_recall", "edge_f1")
METHODS = ("bicubic", "joint_bilateral", "cnn_v2", "cnn_v3")


def _comparison_png(rgb: np.ndarray, gt: np.ndarray, outputs: dict[str, np.ndarray], path: Path) -> None:
    from PIL import Image, ImageDraw

    panels = [("RGB", rgb, True), ("Ground truth (m)", gt, False),
              ("Bicubic (m)", outputs["bicubic"], False),
              ("Joint bilateral (m)", outputs["joint_bilateral"], False),
              ("CNN v2 (m)", outputs["cnn_v2"], False),
              ("CNN v3 (m)", outputs["cnn_v3"], False),
              ("V2 abs error (m)", np.abs(outputs["cnn_v2"] - gt), False),
              ("V3 abs error (m)", np.abs(outputs["cnn_v3"] - gt), False)]
    tile_h, tile_w, label_h = HEIGHT, WIDTH, 22
    columns, rows = 4, 2
    canvas = Image.new("RGB", (columns * tile_w, rows * (tile_h + label_h)), "#0b1729")
    draw = ImageDraw.Draw(canvas)
    for index, (label, values, is_rgb) in enumerate(panels):
        if is_rgb:
            tile = Image.fromarray(values.astype(np.uint8), mode="RGB")
        else:
            finite = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
            scale = 10.0 if "error" not in label.lower() else 2.0
            gray = (np.clip(finite / scale, 0, 1) * 255).astype(np.uint8)
            tile = Image.fromarray(gray, mode="L").convert("RGB")
        x = (index % columns) * tile_w
        y = (index // columns) * (tile_h + label_h)
        canvas.paste(tile, (x, y + label_h))
        draw.text((x + 4, y + 4), label, fill="#e8f2fc")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_outputs(report: dict, output_prefix: str, visual_rgb: np.ndarray,
                   visual_gt: np.ndarray, visual_outputs: dict[str, np.ndarray]) -> dict[str, Path]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": RESULTS / f"{output_prefix}_metrics.json",
        "csv": RESULTS / f"{output_prefix}_comparison.csv",
        "png": RESULTS / f"{output_prefix}_comparison.png",
        "report": RESULTS / f"{output_prefix}_report.md",
    }
    paths["json"].write_text(json.dumps(_json_safe(report), indent=2, allow_nan=False), encoding="utf-8")
    with paths["csv"].open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("metric", *METHODS))
        writer.writeheader()
        for metric in METRIC_KEYS:
            writer.writerow({"metric": metric, **{method: report["metrics_macro_average"][method][metric]
                                                  for method in METHODS}})
    _comparison_png(visual_rgb, visual_gt, visual_outputs, paths["png"])

    scores = report["metrics_macro_average"]
    comparison = all((scores["cnn_v3"]["rmse"] < scores["cnn_v2"]["rmse"],
                      scores["cnn_v3"]["mae"] < scores["cnn_v2"]["mae"],
                      scores["cnn_v3"]["psnr"] > scores["cnn_v2"]["psnr"],
                      scores["cnn_v3"]["ssim"] > scores["cnn_v2"]["ssim"],
                      scores["cnn_v3"]["edge_f1"] >= scores["cnn_v2"]["edge_f1"]))
    higher_is_better = {"rmse": False, "mae": False, "psnr": True, "ssim": True,
                        "edge_precision": True, "edge_recall": True, "edge_f1": True}
    deltas = {metric: float(scores["cnn_v3"][metric] - scores["cnn_v2"][metric])
              for metric in METRIC_KEYS}
    changes = {metric: {"v2": scores["cnn_v2"][metric], "v3": scores["cnn_v3"][metric],
                        "delta_v3_minus_v2": deltas[metric],
                        "direction": ("unchanged" if deltas[metric] == 0 else
                                      "improved" if ((deltas[metric] > 0) == higher_is_better[metric])
                                      else "regressed")}
               for metric in METRIC_KEYS}
    report_lines = [
        "# NYU Depth V2 v2 vs v3 comparison",
        "",
        f"Zero-shot v2 vs NYU-fine-tuned v3 on the same {report['sample_count']} deterministic validation frames.",
        "",
        "| Method | RMSE (m) | MAE (m) | PSNR (dB) | SSIM | Edge P | Edge R | Edge F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {"bicubic": "Bicubic", "joint_bilateral": "Joint bilateral", "cnn_v2": "CNN v2", "cnn_v3": "CNN v3"}
    for method in METHODS:
        values = scores[method]
        report_lines.append("| " + labels[method] + " | " + " | ".join(f"{values[k]:.6f}" for k in METRIC_KEYS) + " |")
    report_lines.extend(["", "## V3 change from v2", "",
                         "| Metric | V2 | V3 | V3 − V2 | Result |",
                         "|---|---:|---:|---:|---|"])
    for metric in METRIC_KEYS:
        change = changes[metric]
        report_lines.append(f"| {metric} | {change['v2']:.6f} | {change['v3']:.6f} | "
                            f"{change['delta_v3_minus_v2']:+.6f} | {change['direction']} |")
    report_lines.extend(["", "## Selection decision", "",
                         ("V3 meets the defined selection rule: lower RMSE and MAE, higher PSNR and SSIM, "
                          "and edge F1 no lower than v2." if comparison else
                          "V3 does not meet the defined selection rule. Keep v2 as the application model; "
                          "review the metric trade-offs before any further tuning."),
                         "", "Selection rule: v3 must improve all four depth-quality metrics and maintain or improve edge F1.",
                         "", "Artifacts are generated from the measured run; no scores are adjusted."])
    paths["report"].write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    report["v2_to_v3"] = changes
    report["selection"] = {"v3_meets_rule": comparison,
                           "rule": "RMSE/MAE lower, PSNR/SSIM higher, edge F1 >= v2"}
    # Persist the selection decision in the JSON as well.
    paths["json"].write_text(json.dumps(_json_safe(report), indent=2, allow_nan=False), encoding="utf-8")
    return paths


@torch.inference_mode()
def evaluate_comparison(*, samples: int = 100, device: str | None = None,
                        output_prefix: str = "nyu_v2_v3",
                        dataset_root: str | Path = DATASET_ROOT,
                        v2_checkpoint: str | Path = V2_CHECKPOINT,
                        v3_checkpoint: str | Path = V3_CHECKPOINT) -> dict:
    if samples < 1:
        raise ValueError("samples must be positive")
    if not output_prefix or Path(output_prefix).name != output_prefix:
        raise ValueError("output_prefix must be a simple non-empty filename prefix")
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model_v2, info_v2 = load_model(v2_checkpoint, device=selected_device)
    model_v3, info_v3 = load_model(v3_checkpoint, device=selected_device)
    depth_range_v2 = float(info_v2.get("config", {}).get("depth_range", DEPTH_RANGE_METRES))
    depth_range_v3 = float(info_v3.get("config", {}).get("depth_range", depth_range_v2))
    if not np.isclose(depth_range_v2, depth_range_v3):
        raise ValueError("v2 and v3 checkpoints use different depth ranges")
    depth_range = depth_range_v2

    pairs = _matching_validation_pairs(Path(dataset_root))
    rng = np.random.default_rng(1234)
    indices = np.sort(rng.choice(len(pairs), size=min(samples, len(pairs)), replace=False))
    chosen = [pairs[int(i)] for i in indices]
    metrics_by_method = {method: [] for method in METHODS}
    visual_rgb = visual_gt = visual_outputs = None
    for name, rgb_path, depth_path in chosen:
        rgb = _resize_rgb(load_rgb(rgb_path))
        raw_depth = load_depth(depth_path)
        gt, gt_valid = _resize_depth_and_validity(raw_depth)
        if rgb.shape[:2] != gt.shape:
            raise ValueError(f"Resized RGB/depth shape mismatch for {name}")
        lowres = area_downsample(gt, SCALE)
        low_valid = np.isfinite(lowres) & (lowres > 0)
        if not low_valid.any():
            raise ValueError(f"No valid low-resolution observations for {name}")
        filled_lowres = fill_invalid_depth(lowres)
        bicubic = np.clip(upsample_depth(filled_lowres, SCALE, "bicubic")[:HEIGHT, :WIDTH], 0, depth_range)
        joint_bilateral = np.clip(joint_bilateral_filter(bicubic, np.ascontiguousarray(rgb)), 0, depth_range)

        rgb_tensor = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float()[None].to(selected_device) / 255.0
        initial_tensor = torch.from_numpy(np.clip(bicubic / depth_range, 0, 1)[None, None]).float().to(selected_device)
        high_mask = upsample_depth(low_valid.astype(np.float32), SCALE, "nearest")[:HEIGHT, :WIDTH]
        mask_tensor = torch.from_numpy((high_mask > .5)[None, None]).float().to(selected_device)
        outputs = {
            "bicubic": bicubic,
            "joint_bilateral": joint_bilateral,
        }
        for key, model in (("cnn_v2", model_v2), ("cnn_v3", model_v3)):
            normalized = refine_batch(model, rgb_tensor, initial_tensor, mask_tensor)
            metres = normalized.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32) * depth_range
            outputs[key] = np.clip(metres, 0, depth_range)

        for method, prediction in outputs.items():
            metrics_by_method[method].append(evaluate(gt, prediction, data_range=depth_range,
                                                       valid_mask=gt_valid))
        if visual_outputs is None:
            visual_rgb, visual_gt, visual_outputs = rgb, gt, outputs

    macro = {method: {metric: float(np.mean([scene[metric] for scene in metrics_by_method[method]]))
                      for metric in METRIC_KEYS} for method in METHODS}
    report = {
        "dataset": "NYU Depth V2 validation; v3 supervised fine-tuning used train split only",
        "split": "validation",
        "sample_count": len(chosen),
        "requested_samples": samples,
        "validation_filenames": [name for name, _, _ in chosen],
        "device": str(selected_device),
        "gpu_name": torch.cuda.get_device_name(selected_device) if selected_device.type == "cuda" else None,
        "v2_checkpoint": str(Path(v2_checkpoint)),
        "v3_checkpoint": str(Path(v3_checkpoint)),
        "depth_units": "metres",
        "depth_range_metres": depth_range,
        "resolution": [HEIGHT, WIDTH],
        "scale": SCALE,
        "selection_seed": 1234,
        "methods": list(METHODS),
        "metric_definitions": {
            "rmse_mae": "computed over finite positive ground-truth pixels only",
            "psnr": f"fixed data range of {depth_range} metres",
            "ssim": "project fixed 7x7 Gaussian SSIM implementation, sigma 1.5, masked local statistics",
            "edges": "project depth-gradient threshold 0.10 metres/pixel, 1-pixel tolerance, edge precision/recall/F1",
            "preprocessing": "same 128x128 RGB/depth, 4x area downsampling, filled observations, bicubic CNN initial, and masks for all methods",
        },
        "metrics_macro_average": macro,
    }
    output_paths = _write_outputs(report, output_prefix, visual_rgb, visual_gt, visual_outputs)
    report["outputs"] = {key: str(value) for key, value in output_paths.items()}
    output_paths["json"].write_text(json.dumps(_json_safe(report), indent=2, allow_nan=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--output-prefix", default="nyu_v2_v3")
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--v2-checkpoint", type=Path, default=V2_CHECKPOINT)
    parser.add_argument("--v3-checkpoint", type=Path, default=V3_CHECKPOINT)
    args = parser.parse_args()
    result = evaluate_comparison(samples=args.samples, device=args.device,
                                 output_prefix=args.output_prefix, dataset_root=args.dataset_root,
                                 v2_checkpoint=args.v2_checkpoint, v3_checkpoint=args.v3_checkpoint)
    print(json.dumps(result["metrics_macro_average"], indent=2, allow_nan=False))
    print(json.dumps(result["outputs"], indent=2))


if __name__ == "__main__":
    main()
