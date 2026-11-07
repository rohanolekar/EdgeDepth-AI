"""NYU Depth V2 dataset adapter matching the project's inference preprocessing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from evaluation.evaluate_nyu import (
    DATASET_ROOT,
    SCALE,
    _matching_pairs,
    _resize_depth_and_validity,
    _resize_rgb,
)
from src.interpolation import upsample_depth
from src.preprocessing import area_downsample, fill_invalid_depth, load_depth, load_rgb


class NYUDepthDataset(Dataset):
    """Paired NYU frames prepared for the same 4x CNN input used at evaluation.

    Ground truth and inputs use normalized 0–1 depth at ``depth_range`` metres.
    Invalid ground-truth pixels are zero-filled only for tensor safety and remain
    excluded through ``valid_mask``. Invalid observed samples are filled for
    interpolation but their validity is retained in the mask channel.
    """

    def __init__(self, *, split: str, dataset_root: str | Path = DATASET_ROOT,
                 depth_range: float = 10.0, augment: bool = False) -> None:
        if split not in {"train", "val"}:
            raise ValueError("split must be 'train' or 'val'")
        if not np.isfinite(depth_range) or depth_range <= 0:
            raise ValueError("depth_range must be positive and finite")
        self.pairs = _matching_pairs(Path(dataset_root), split)
        self.split = split
        self.depth_range = float(depth_range)
        self.augment = bool(augment and split == "train")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        name, rgb_path, depth_path = self.pairs[index]
        rgb = _resize_rgb(load_rgb(rgb_path))
        raw_depth = load_depth(depth_path)
        gt_metres, gt_valid = _resize_depth_and_validity(raw_depth)
        if rgb.shape[:2] != gt_metres.shape:
            raise ValueError(f"Resized RGB/depth shape mismatch for {name}")

        lowres = area_downsample(gt_metres, SCALE)
        low_valid = np.isfinite(lowres) & (lowres > 0)
        if not low_valid.any():
            raise ValueError(f"No valid low-resolution observations for {name}")
        filled_lowres = fill_invalid_depth(lowres)
        # Keep the low-resolution validity separate, and match evaluator's
        # bicubic initial depth for the model's normalized depth channel.
        initial_metres = np.clip(upsample_depth(filled_lowres, SCALE, "bicubic"), 0, self.depth_range)
        validity_hr = upsample_depth(low_valid.astype(np.float32), SCALE, "nearest") > .5

        rgb_t = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float() / 255.0
        initial_t = torch.from_numpy((initial_metres / self.depth_range)[None]).float()
        validity_t = torch.from_numpy(validity_hr[None].astype(np.float32))
        gt_t = torch.from_numpy(np.where(gt_valid, gt_metres / self.depth_range, 0.0)[None].astype(np.float32))
        valid_t = torch.from_numpy(gt_valid[None].astype(bool))

        if self.augment:
            rgb_t, initial_t, validity_t, gt_t, valid_t = self._augment(
                rgb_t, initial_t, validity_t, gt_t, valid_t
            )

        return {
            "name": name,
            "rgb": rgb_t,
            "initial_depth": initial_t,
            "validity_mask": validity_t,
            "ground_truth": gt_t,
            "valid_mask": valid_t,
        }

    @staticmethod
    def _augment(*tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Apply synchronized, distribution-safe geometric augmentation."""
        result = list(tensors)
        if torch.rand(()) < 0.5:
            result = [torch.flip(item, dims=(-1,)) for item in result]
        if torch.rand(()) < 0.5:
            result = [torch.flip(item, dims=(-2,)) for item in result]
        # NYU frames are square after resize, so quarter turns preserve dimensions.
        turns = int(torch.randint(0, 4, ()).item())
        if turns:
            result = [torch.rot90(item, turns, dims=(-2, -1)) for item in result]
        return tuple(result)
