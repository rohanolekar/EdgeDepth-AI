"""Reproducible procedural RGB/depth scenes for synthetic model training."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from src.interpolation import upsample_depth
from src.preprocessing import area_downsample, fill_invalid_depth


@dataclass(frozen=True)
class SyntheticDataConfig:
    samples: int = 32
    height: int = 64
    width: int = 64
    scale: int = 2
    noise_std: float = 0.03
    missing_rate: float = 0.05
    seed: int = 1234
    max_depth: float = 10.0


def _generate_scene(height: int, width: int, rng: np.random.Generator,
                    max_depth: float) -> tuple[np.ndarray, np.ndarray]:
    """Make one varied scene with distinct depth planes and aligned RGB objects."""
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    xn, yn = xx / max(width-1, 1), yy / max(height-1, 1)
    background = rng.uniform(4.0, 7.2)
    depth = background + 0.35*np.sin(xn*rng.uniform(4, 12) + rng.uniform(0, 6))
    depth += 0.25*np.cos(yn*rng.uniform(4, 12) + rng.uniform(0, 6))
    rgb = np.empty((height, width, 3), dtype=np.float32)
    rgb[:] = rng.integers(35, 200, size=3)
    object_count = int(rng.integers(2, 6))
    for index in range(object_count):
        cx, cy = rng.uniform(.15, .85), rng.uniform(.15, .85)
        rx, ry = rng.uniform(.08, .23), rng.uniform(.08, .25)
        kind = index % 3
        if kind == 0:
            mask = ((xn-cx)/rx)**2 + ((yn-cy)/ry)**2 <= 1
        elif kind == 1:
            # Slanted rectangle produces a non-axis-aligned discontinuity.
            u = xn-cx + .32*(yn-cy)
            v = yn-cy - .12*(xn-cx)
            mask = (np.abs(u) <= rx) & (np.abs(v) <= ry*.55)
        else:
            # Irregular object from a radial multi-frequency contour.
            theta = np.arctan2(yn-cy, xn-cx)
            radius = np.sqrt(((xn-cx)/rx)**2 + ((yn-cy)/ry)**2)
            contour = 1 + .12*np.sin(3*theta+rng.uniform(0, 6)) + .07*np.cos(5*theta+rng.uniform(0, 6))
            mask = radius <= contour
        plane = rng.uniform(1.0, 8.5)
        depth[mask] = plane + rng.uniform(-.18, .18)*np.sin(xn[mask]*rng.uniform(5, 20))
        rgb[mask] = rng.integers(25, 240, size=3)

    # Add a thin structure and deterministic texture that is not identical to depth.
    intercept = rng.uniform(.2, .8)
    slope = rng.uniform(-.45, .45)
    line = np.abs(yn-(slope*xn+intercept)) < max(1.0/height, .018)
    depth[line] = rng.uniform(1.0, 8.5)
    rgb[line] = rng.integers(15, 245, size=3)
    texture = rng.normal(0, 5, size=(height, width, 1))
    texture += (8*np.sin(xn*rng.uniform(15, 55)))[..., None]
    rgb = np.clip(rgb + texture, 0, 255).astype(np.uint8)
    depth = np.clip(depth, .25, max_depth*.95).astype(np.float32)
    return rgb, depth


class SyntheticDepthDataset(Dataset):
    """On-demand synthetic scenes with deterministic per-index random streams."""

    def __init__(self, config: SyntheticDataConfig, *, split: str = "train") -> None:
        if split not in {"train", "validation", "test"}:
            raise ValueError("split must be train, validation, or test")
        if config.samples < 1 or config.height < 16 or config.width < 16:
            raise ValueError("samples must be positive and dimensions at least 16")
        if config.scale not in (2, 4, 8) or config.height % config.scale or config.width % config.scale:
            raise ValueError("dimensions must be divisible by scale 2, 4, or 8")
        if config.max_depth <= 0 or config.noise_std < 0 or not 0 <= config.missing_rate < 1:
            raise ValueError("invalid depth/noise/missing-rate settings")
        self.config, self.split = config, split
        self.split_offset = {"train": 0, "validation": 1_000_000, "test": 2_000_000}[split]

    def __len__(self) -> int:
        return self.config.samples

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        cfg = self.config
        rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, self.split_offset, index]))
        rgb, gt = _generate_scene(cfg.height, cfg.width, rng, cfg.max_depth)
        lowres = area_downsample(gt, cfg.scale)
        complete_lowres = lowres.copy()
        if cfg.noise_std:
            lowres += rng.normal(0, cfg.noise_std, lowres.shape).astype(np.float32)
        if cfg.missing_rate:
            lowres[rng.random(lowres.shape) < cfg.missing_rate] = np.nan
        if not np.isfinite(lowres).any():
            lowres.flat[0] = complete_lowres.flat[0]
        lowres_mask = np.isfinite(lowres)
        # Retain the observation mask; fill only values consumed by interpolation.
        filled = fill_invalid_depth(lowres)
        initial = upsample_depth(filled, cfg.scale, "bilinear")
        mask_hr = upsample_depth(lowres_mask.astype(np.float32), cfg.scale, "nearest")
        scale = float(cfg.max_depth)
        rgb_t = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float() / 255.0
        initial_t = torch.from_numpy(np.clip(initial / scale, 0, 1)[None]).float()
        mask_t = torch.from_numpy((mask_hr > .5)[None]).float()
        lr_mask_t = torch.from_numpy(lowres_mask[None].copy()).float()
        gt_t = torch.from_numpy((gt / scale)[None]).float()
        lr_t = torch.from_numpy((filled / scale)[None]).float()
        valid_t = torch.ones_like(gt_t, dtype=torch.bool)
        inputs = torch.cat((rgb_t, initial_t, mask_t), dim=0)
        return {"inputs": inputs, "rgb": rgb_t, "initial_depth": initial_t,
                "validity_mask": mask_t, "ground_truth": gt_t,
                "valid_mask": valid_t, "lowres_depth": lr_t,
                "lowres_validity": lr_mask_t}


def make_synthetic_splits(*, train_samples=32, validation_samples=8, test_samples=8,
                          seed=1234, **config_kwargs):
    """Create disjoint deterministic train/validation/test scene streams."""
    base = dict(config_kwargs)
    return {
        "train": SyntheticDepthDataset(SyntheticDataConfig(samples=train_samples, seed=seed, **base), split="train"),
        "validation": SyntheticDepthDataset(SyntheticDataConfig(samples=validation_samples, seed=seed, **base), split="validation"),
        "test": SyntheticDepthDataset(SyntheticDataConfig(samples=test_samples, seed=seed, **base), split="test"),
    }
