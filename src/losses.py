"""Masked depth reconstruction and gradient losses."""

from __future__ import annotations

import torch
from torch import nn


class DepthReconstructionLoss(nn.Module):
    """Valid-pixel L1 plus edge-aware depth-gradient L1 in normalized units."""

    def __init__(self, gradient_weight: float = 0.25) -> None:
        super().__init__()
        if gradient_weight < 0:
            raise ValueError("gradient_weight must be non-negative")
        self.gradient_weight = float(gradient_weight)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor,
                valid_mask: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if prediction.shape != target.shape or prediction.ndim != 4 or prediction.shape[1] != 1:
            raise ValueError("prediction and target must have matching (N, 1, H, W) shapes")
        if valid_mask.shape != target.shape:
            raise ValueError("valid_mask must match target shape")
        valid = valid_mask.bool() & torch.isfinite(target) & torch.isfinite(prediction)
        if not valid.any():
            zero = prediction.sum() * 0.0
            return zero, {"l1": zero.detach(), "gradient": zero.detach()}
        safe_prediction = torch.where(valid, prediction, torch.zeros_like(prediction))
        safe_target = torch.where(valid, target, torch.zeros_like(target))
        l1 = torch.abs(safe_prediction[valid] - safe_target[valid]).mean()
        dx_p = safe_prediction[..., :, 1:] - safe_prediction[..., :, :-1]
        dx_t = safe_target[..., :, 1:] - safe_target[..., :, :-1]
        dy_p = safe_prediction[..., 1:, :] - safe_prediction[..., :-1, :]
        dy_t = safe_target[..., 1:, :] - safe_target[..., :-1, :]
        valid_dx = valid[..., :, 1:] & valid[..., :, :-1]
        valid_dy = valid[..., 1:, :] & valid[..., :-1, :]
        gradient_terms = []
        if valid_dx.any():
            gradient_terms.append(torch.abs(dx_p[valid_dx] - dx_t[valid_dx]).mean())
        if valid_dy.any():
            gradient_terms.append(torch.abs(dy_p[valid_dy] - dy_t[valid_dy]).mean())
        gradient = torch.stack(gradient_terms).mean() if gradient_terms else l1 * 0.0
        total = l1 + self.gradient_weight * gradient
        return total, {"l1": l1.detach(), "gradient": gradient.detach()}
