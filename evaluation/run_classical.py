"""Run a reproducible RGB-guided depth upsampling evaluation.

Synthetic depth is generated at high resolution, then averaged over non-overlapping
scale-by-scale cells (area downsampling). Sensor noise and missing samples are
optional. User inputs are an aligned full-resolution RGB guide and depth reference.

Example: python -m evaluation.run_classical --scale 4 --noise-std 0.03 --missing-rate 0.02
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.edge_detection import EDGE_THRESHOLD, EDGE_TOLERANCE_PIXELS
from src.filtering import guided_filter, joint_bilateral_filter
from src.interpolation import upsample_depth
from src.metrics import DEFAULT_DEPTH_DATA_RANGE, evaluate, processing_time
from src.preprocessing import area_downsample, fill_invalid_depth, load_depth, load_rgb

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "sample_data" / "results"
METHODS = ("nearest", "bilinear", "bicubic", "joint_bilateral", "guided")


def make_synthetic_benchmark(scale: int = 2, *, noise_std: float = 0.0,
                             missing_rate: float = 0.0, seed: int = 17) -> dict:
    """Generate a deterministic, independent high-resolution scene and sensor input."""
    if scale not in (2, 4, 8) or noise_std < 0 or not 0 <= missing_rate < 1:
        raise ValueError("invalid scale, noise_std, or missing_rate")
    height, width = 96, 128
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    xn, yn = x / width, y / height

    # Continuous background variation plus independently defined foreground objects.
    gt = (3.4 + 0.38*np.sin(2*np.pi*xn*1.3) + 0.27*np.cos(2*np.pi*yn*1.7)
          + 0.16*np.sin(2*np.pi*(xn+yn)*2.1)).astype(np.float32)
    ellipse = ((x-40)**2 / 24**2 + (y-48)**2 / 31**2) <= 1
    rectangle = (x >= 73) & (x < 112) & (y >= 17) & (y < 74)
    disk = (x-93)**2 + (y-77)**2 <= 13**2
    gt[ellipse] = 2.0 + 0.12*np.sin(xn[ellipse]*22) + 0.08*np.cos(yn[ellipse]*18)
    gt[rectangle] = 5.6 + 0.18*np.sin(yn[rectangle]*21) + 0.10*np.cos(xn[rectangle]*25)
    gt[disk] = 2.65 + 0.08*np.sin(xn[disk]*30 + yn[disk]*8)
    gt = np.clip(gt, 0.5, 8.0).astype(np.float32)

    rgb = np.empty((height, width, 3), dtype=np.float32)
    rgb[:] = (75, 125, 65)
    rgb[ellipse] = (185, 65, 45)
    rgb[rectangle] = (45, 95, 205)
    rgb[disk] = (230, 190, 55)
    texture = (5*np.sin(xn*60) + 4*np.cos(yn*47))[..., None]
    rgb = np.clip(rgb + texture, 0, 255).astype(np.uint8)

    lowres = area_downsample(gt, scale)
    rng = np.random.default_rng(seed)
    if noise_std:
        lowres = lowres + rng.normal(0, noise_std, lowres.shape).astype(np.float32)
    if missing_rate:
        lowres[rng.random(lowres.shape) < missing_rate] = np.nan
    validity = np.isfinite(gt)
    return {"rgb": rgb, "lowres_depth": lowres, "ground_truth": gt,
            "valid_mask": validity,
            "configuration": {"source": "deterministic_synthetic", "height": height,
                "width": width, "scale": scale, "downsampling": "non-overlapping area mean",
                "noise_std_depth_units": noise_std, "missing_rate": missing_rate,
                "random_seed": seed, "ground_truth_valid_fraction": float(validity.mean())}}


def _upsample_crop(lowres: np.ndarray, scale: int, method: str, shape: tuple[int, int]) -> np.ndarray:
    result = upsample_depth(lowres, scale, method)
    return result[:shape[0], :shape[1]]


def _save_panel(rgb, gt, outputs, metrics, valid_mask):
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError("Saving visual results requires Pillow in the active environment.") from exc
    tiles = [("RGB guide", rgb)]
    tiles += [("Ground truth", gt)]
    tiles += [(name, outputs[name]) for name in METHODS]
    h, w = gt.shape
    label_h = 44
    canvas = Image.new("RGB", (w*len(tiles), h+label_h), "white")
    draw = ImageDraw.Draw(canvas)
    lo, hi = float(np.nanmin(gt[valid_mask])), float(np.nanmax(gt[valid_mask]))
    for i, (name, data) in enumerate(tiles):
        x = i*w
        draw.text((x+3, 2), name, fill="black")
        if i >= 2:
            values = metrics[name]
            draw.text((x+3, 17), f"RMSE {values['rmse']:.3f}  F1 {values['edge_f1']:.3f}", fill="black")
        if data.ndim == 3:
            image = Image.fromarray(data.astype(np.uint8))
        else:
            scaled = np.clip((np.asarray(data)-lo)/(hi-lo)*255, 0, 255).astype(np.uint8)
            image = Image.fromarray(scaled, "L").convert("RGB")
            if name == "Ground truth" and not valid_mask.all():
                pixels = np.asarray(image).copy()
                pixels[~valid_mask] = (255, 0, 255)
                image = Image.fromarray(pixels)
        canvas.paste(image, (x, label_h))
    canvas.save(RESULTS / "synthetic_comparison.png")


def run(rgb_source=None, depth_source=None, *, scale=2, noise_std=0.0,
        missing_rate=0.0, seed=17, warmup_runs=1, timed_runs=5,
        depth_range=DEFAULT_DEPTH_DATA_RANGE):
    if (rgb_source is None) != (depth_source is None):
        raise ValueError("Provide both --rgb and --depth, or neither for synthetic evaluation.")
    if rgb_source is None:
        benchmark = make_synthetic_benchmark(scale, noise_std=noise_std,
                                             missing_rate=missing_rate, seed=seed)
        rgb, lowres = benchmark["rgb"], benchmark["lowres_depth"]
        gt, valid_mask = benchmark["ground_truth"], benchmark["valid_mask"]
        config = benchmark["configuration"]
    else:
        rgb = load_rgb(rgb_source)
        gt = load_depth(depth_source)
        if gt.shape != rgb.shape[:2]:
            raise ValueError("RGB guide and high-resolution depth reference must have matching dimensions.")
        valid_mask = np.isfinite(gt)
        if gt.shape[0] % scale or gt.shape[1] % scale:
            raise ValueError("reference dimensions must be divisible by scale")
        lowres = area_downsample(gt, scale)
        rng = np.random.default_rng(seed)
        if noise_std:
            lowres = lowres + rng.normal(0, noise_std, lowres.shape).astype(np.float32)
        if missing_rate:
            lowres[rng.random(lowres.shape) < missing_rate] = np.nan
        config = {"source": "user", "rgb_path": str(rgb_source), "depth_path": str(depth_source),
                  "height": int(gt.shape[0]), "width": int(gt.shape[1]), "scale": scale,
                  "downsampling": "non-overlapping area mean", "noise_std_depth_units": noise_std,
                  "missing_rate": missing_rate, "random_seed": seed,
                  "ground_truth_valid_fraction": float(valid_mask.mean())}
    # Fill only the low-resolution observation; keep GT and its original mask intact.
    lowres_filled = fill_invalid_depth(lowres)
    guide = rgb
    shape = gt.shape
    outputs = {name: _upsample_crop(lowres_filled, scale, name, shape)
               for name in ("nearest", "bilinear", "bicubic")}
    outputs["joint_bilateral"] = joint_bilateral_filter(outputs["bilinear"], guide)
    outputs["guided"] = guided_filter(outputs["bilinear"], guide)

    timings = {"interpolation_only": {}, "complete_filtering_pipeline": {}}
    for name in ("nearest", "bilinear", "bicubic"):
        _, timing = processing_time(_upsample_crop, lowres_filled, scale, name, shape,
                                    warmup_runs=warmup_runs, timed_runs=timed_runs)
        timings["interpolation_only"][name] = timing
    for name, filter_function in (("joint_bilateral", joint_bilateral_filter),
                                  ("guided", guided_filter)):
        def pipeline(filter_function=filter_function):
            interpolation = _upsample_crop(lowres_filled, scale, "bilinear", shape)
            return filter_function(interpolation, guide)
        _, timing = processing_time(pipeline, warmup_runs=warmup_runs, timed_runs=timed_runs)
        timings["complete_filtering_pipeline"][name] = timing

    scores = {name: evaluate(gt, output, data_range=depth_range, valid_mask=valid_mask)
              for name, output in outputs.items()}
    RESULTS.mkdir(parents=True, exist_ok=True)
    definitions = {"rmse_mae": "computed over originally valid finite ground-truth pixels",
        "psnr": f"10*log10(depth_range^2/MSE), fixed depth_range={depth_range}",
        "ssim": "masked local Gaussian SSIM, 7x7 window, sigma=1.5, population covariance",
        "edges": f"central-difference gradient magnitude threshold={EDGE_THRESHOLD} depth units/pixel; symmetric matching tolerance={EDGE_TOLERANCE_PIXELS} pixel; unmatched predictions are false positives",
        "downsampling": "non-overlapping area mean over scale x scale cells; sensor noise and missing samples applied afterward"}
    report = {"configuration": config, "metric_definitions": definitions,
              "timing_configuration": {"warmup_runs": warmup_runs, "timed_runs": timed_runs,
                  "reported_statistic": "median", "timing_categories": list(timings)},
              "timings": timings, "metrics": scores}
    def safe(value):
        if isinstance(value, dict): return {k: safe(v) for k, v in value.items()}
        if isinstance(value, list): return [safe(v) for v in value]
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            return "Infinity" if value > 0 else "-Infinity" if value < 0 else None
        if isinstance(value, np.integer): return int(value)
        return value
    with (RESULTS / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(safe(report), file, indent=2, allow_nan=False)
    _save_panel(rgb, gt, outputs, scores, valid_mask)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb", type=Path)
    parser.add_argument("--depth", type=Path, help="full-resolution reference depth (Numpy .npy recommended)")
    parser.add_argument("--scale", type=int, choices=(2, 4, 8), default=2)
    parser.add_argument("--noise-std", type=float, default=0.0, help="Gaussian sensor noise in depth units")
    parser.add_argument("--missing-rate", type=float, default=0.0, help="fraction of low-resolution observations set invalid")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--depth-range", type=float, default=DEFAULT_DEPTH_DATA_RANGE,
                        help="fixed depth range used for PSNR and SSIM (default 10.0 units)")
    args = parser.parse_args()
    report = run(args.rgb, args.depth, scale=args.scale, noise_std=args.noise_std,
                 missing_rate=args.missing_rate, seed=args.seed, warmup_runs=args.warmups,
                 timed_runs=args.runs, depth_range=args.depth_range)
    print(json.dumps(report["metrics"], indent=2, allow_nan=False, default=str))


if __name__ == "__main__":
    main()
