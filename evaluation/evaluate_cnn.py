"""Evaluate a trained refinement checkpoint against classical baselines.

All methods receive each identical synthetic test scene, guide and observed
low-resolution depth. This evaluates synthetic data only; it is not real-world
dataset validation.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from models.depth_refinement_cnn import DepthRefinementCNN
from src.dataset import SyntheticDataConfig, SyntheticDepthDataset
from src.filtering import guided_filter, joint_bilateral_filter
from src.inference import load_model
from src.interpolation import upsample_depth
from src.metrics import DEFAULT_DEPTH_DATA_RANGE, evaluate

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "sample_data" / "results"
CHECKPOINT = ROOT / "models" / "checkpoints" / "depth_refinement_best.pt"


@torch.inference_mode()
def _time_model(model, sample, device, warmup_runs, timed_runs):
    rgb = sample["rgb"].unsqueeze(0).to(device)
    initial = sample["initial_depth"].unsqueeze(0).to(device)
    mask = sample["validity_mask"].unsqueeze(0).to(device)
    def call():
        return model.refine(rgb, initial, mask)
    for _ in range(warmup_runs):
        call()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    samples = []
    result = None
    for _ in range(timed_runs):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        result = call()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        samples.append(time.perf_counter()-start)
    return result.squeeze(0).squeeze(0).detach().cpu().numpy(), {
        "median_seconds": float(statistics.median(samples)), "samples_seconds": samples,
        "warmup_runs": warmup_runs, "timed_runs": timed_runs}


def _cnn_depth_to_physical(normalized_depth: np.ndarray, depth_range: float) -> np.ndarray:
    """Convert normalized CNN output to the physical units used by the baselines."""
    if not np.isfinite(depth_range) or depth_range <= 0:
        raise ValueError("depth_range must be a positive finite value")
    return np.asarray(normalized_depth, dtype=np.float32) * float(depth_range)


def _visualize(sample, outputs, max_depth, error_range, path):
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError("Pillow is required to save comparison images") from exc
    rgb = (sample["rgb"].numpy().transpose(1, 2, 0)*255).clip(0, 255).astype(np.uint8)
    gt = sample["ground_truth"].squeeze(0).numpy()*max_depth
    low = sample["lowres_depth"].squeeze(0).numpy()*max_depth
    scale = gt.shape[0]//low.shape[0]
    low = upsample_depth(low, scale, "nearest")[:gt.shape[0], :gt.shape[1]]
    images = [("RGB guidance", rgb), ("Ground truth", gt), ("Low-res observation", low),
              ("Initial bilinear", outputs["bilinear"]),
              ("Joint bilateral", outputs["joint_bilateral"]), ("CNN refined", outputs["cnn"]),
              ("Abs error: joint", np.abs(outputs["joint_bilateral"]-gt)),
              ("Abs error: CNN", np.abs(outputs["cnn"]-gt))]
    tile_h, tile_w = gt.shape
    label_h = 22
    canvas = Image.new("RGB", (tile_w*len(images), tile_h+label_h), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (label, data) in enumerate(images):
        if data.ndim == 3:
            image = Image.fromarray(data)
        elif label.startswith("Abs error"):
            values = np.clip(data/error_range, 0, 1)
            rgb_err = np.zeros((*data.shape, 3), dtype=np.uint8)
            rgb_err[..., 0] = (255*values).astype(np.uint8)
            rgb_err[..., 1] = (255*(1-values)*.25).astype(np.uint8)
            image = Image.fromarray(rgb_err)
        else:
            gray = (np.clip(data/max_depth, 0, 1)*255).astype(np.uint8)
            image = Image.fromarray(gray, "L").convert("RGB")
        x = i*tile_w
        canvas.paste(image, (x, label_h))
        draw.text((x+3, 3), label, fill="black")
    canvas.save(path)


def evaluate_checkpoint(checkpoint=CHECKPOINT, *, test_samples=8, height=64, width=64,
                        scale=2, seed=1234, noise_std=.03, missing_rate=.05,
                        warmup_runs=1, timed_runs=5, device=None, output_prefix="cnn"):
    if test_samples < 1 or warmup_runs < 0 or timed_runs < 1:
        raise ValueError("test_samples/timed_runs must be positive and warmups non-negative")
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, info = load_model(checkpoint, device=selected_device)
    config = SyntheticDataConfig(samples=test_samples, height=height, width=width, scale=scale,
                                 seed=seed, noise_std=noise_std, missing_rate=missing_rate)
    dataset = SyntheticDepthDataset(config, split="test")
    per_scene, inference_times, first_outputs, first_sample = [], [], None, None
    depth_range = float(info.get("config", {}).get("depth_range", DEFAULT_DEPTH_DATA_RANGE))
    max_depth = depth_range
    for index in range(len(dataset)):
        sample = dataset[index]
        rgb = sample["rgb"].numpy().transpose(1, 2, 0)
        initial = sample["initial_depth"].squeeze(0).numpy()*max_depth
        lowres = sample["lowres_depth"].squeeze(0).numpy()*max_depth
        lowres_mask = sample["lowres_validity"].squeeze(0).numpy() > .5
        gt = sample["ground_truth"].squeeze(0).numpy()*max_depth
        valid = sample["valid_mask"].squeeze(0).numpy().astype(bool)
        filled_lowres = lowres.copy()
        if not lowres_mask.all():
            # Dataset interpolation already used invalid-depth filling; expose the same filled input.
            filled_lowres = np.where(lowres_mask, lowres, np.nan)
            from src.preprocessing import fill_invalid_depth
            filled_lowres = fill_invalid_depth(filled_lowres)
        interp = {method: upsample_depth(filled_lowres, scale, method)[:height, :width]
                  for method in ("nearest", "bilinear", "bicubic")}
        rgb_u8 = np.rint(rgb*255).clip(0, 255).astype(np.uint8)
        jbf = joint_bilateral_filter(interp["bilinear"], rgb_u8)
        guided = guided_filter(interp["bilinear"], rgb_u8)
        cnn_normalized, timing = _time_model(model, sample, selected_device, warmup_runs, timed_runs)
        cnn_physical = _cnn_depth_to_physical(cnn_normalized, depth_range)
        inference_times.append(timing)
        outputs = {**interp, "joint_bilateral": jbf, "guided": guided, "cnn": cnn_physical}
        per_scene.append({name: evaluate(gt, pred, data_range=depth_range, valid_mask=valid)
                          for name, pred in outputs.items()})
        if index == 0:
            first_outputs, first_sample = outputs, sample
    names = per_scene[0].keys()
    metrics = {name: {metric: float(np.mean([scene[name][metric] for scene in per_scene]))
                      for metric in per_scene[0][name]} for name in names}
    cnn_inference = {"median_seconds_per_image": float(np.median([t["median_seconds"] for t in inference_times])),
                     "scene_timings": inference_times, "warmup_runs": warmup_runs,
                     "timed_runs": timed_runs, "device": str(selected_device),
                     "measurement_scope": "model forward only; excludes host/device transfers and preprocessing"}
    RESULTS.mkdir(parents=True, exist_ok=True)
    visual_path = RESULTS / f"{output_prefix}_comparison.png"
    _visualize(first_sample, first_outputs, max_depth, max_depth*.25, visual_path)
    report = {"dataset": "procedural synthetic test scenes; no real-world validation",
              "configuration": {"test_samples": test_samples, "height": height, "width": width,
                  "scale": scale, "seed": seed, "noise_std": noise_std,
                  "missing_rate": missing_rate, "depth_range": depth_range,
                  "split": "test (separate deterministic scene seed stream)"},
              "checkpoint": str(checkpoint), "device": str(selected_device),
              "gpu_name": torch.cuda.get_device_name(selected_device) if selected_device.type == "cuda" else None,
              "inference_timing": cnn_inference, "metrics_macro_average": metrics,
              "visual_output": str(visual_path)}
    (RESULTS / f"{output_prefix}_metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False, default=str), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--test-samples", type=int, default=8)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--scale", type=int, choices=(2, 4, 8), default=2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--noise-std", type=float, default=.03)
    parser.add_argument("--missing-rate", type=float, default=.05)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--output-prefix", default="cnn")
    args = parser.parse_args()
    report = evaluate_checkpoint(args.checkpoint, test_samples=args.test_samples, height=args.height,
        width=args.width, scale=args.scale, seed=args.seed, noise_std=args.noise_std,
        missing_rate=args.missing_rate, warmup_runs=args.warmups, timed_runs=args.runs, device=args.device,
        output_prefix=args.output_prefix)
    print(json.dumps(report["metrics_macro_average"], indent=2, default=str))


if __name__ == "__main__":
    main()
