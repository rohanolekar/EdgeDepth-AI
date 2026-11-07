"""Compact residual U-Net for RGB-guided depth refinement."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels), nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels), nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class DepthRefinementCNN(nn.Module):
    """Five-channel compact U-Net predicting a bounded depth residual.

    RGB and depth inputs are normalized to [0, 1]. The forward call returns a
    normalized residual; ``refine`` adds it to the initial depth and clamps the
    final prediction to the physical normalized interval [0, 1].
    """

    input_channels = 5

    def __init__(self, base_channels: int = 16, max_residual: float = 0.25) -> None:
        super().__init__()
        if base_channels < 8 or base_channels % 8:
            raise ValueError("base_channels must be >= 8 and divisible by 8")
        if not 0 < max_residual <= 1:
            raise ValueError("max_residual must be in (0, 1]")
        c1, c2, c3 = base_channels, 2*base_channels, 4*base_channels
        self.max_residual = float(max_residual)
        self.enc1 = ConvBlock(5, c1)
        self.enc2 = ConvBlock(c1, c2)
        self.bottleneck = ConvBlock(c2, c3)
        self.pool = nn.MaxPool2d(2)
        self.dec2 = ConvBlock(c3+c2, c2)
        self.dec1 = ConvBlock(c2+c1, c1)
        self.residual_head = nn.Conv2d(c1, 1, 1)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    @staticmethod
    def _validate(rgb: torch.Tensor, depth: torch.Tensor, mask: torch.Tensor) -> None:
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError("rgb must have shape (N, 3, H, W)")
        expected = (rgb.shape[0], 1, rgb.shape[2], rgb.shape[3])
        if tuple(depth.shape) != expected or tuple(mask.shape) != expected:
            raise ValueError("depth and validity mask must each have shape (N, 1, H, W)")
        if min(rgb.shape[-2:]) < 4:
            raise ValueError("spatial dimensions must be at least 4 pixels")

    def forward(self, rgb: torch.Tensor, initial_depth: torch.Tensor,
                validity_mask: torch.Tensor) -> torch.Tensor:
        """Return bounded normalized residual with shape (N, 1, H, W)."""
        self._validate(rgb, initial_depth, validity_mask)
        x = torch.cat((rgb, initial_depth, validity_mask.to(dtype=rgb.dtype)), dim=1)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        mid = self.bottleneck(self.pool(e2))
        d2 = F.interpolate(mid, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat((d2, e2), dim=1))
        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat((d1, e1), dim=1))
        return torch.tanh(self.residual_head(d1)) * self.max_residual

    def refine(self, rgb: torch.Tensor, initial_depth: torch.Tensor,
               validity_mask: torch.Tensor) -> torch.Tensor:
        """Return constrained normalized depth in [0, 1]."""
        residual = self(rgb, initial_depth, validity_mask)
        return torch.clamp(initial_depth + residual, min=0.0, max=1.0)
