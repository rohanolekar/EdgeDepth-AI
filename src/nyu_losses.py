"""Masked reconstruction, edge-gradient, and structural losses for NYU tuning."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class NYURefinementLoss(nn.Module):
    """Normalized depth L1 plus edge-weighted gradients and masked local SSIM.

    Weights are fixed before the run: L1 1.0, gradient 0.25, structural 0.05.
    The local SSIM term uses a 7x7 box window as a mild structure regularizer;
    reported evaluation SSIM continues to use the project's fixed Gaussian metric.
    """

    def __init__(self, gradient_weight: float = 0.25, structural_weight: float = 0.05,
                 window_size: int = 7) -> None:
        super().__init__()
        if gradient_weight < 0 or structural_weight < 0 or window_size < 3 or window_size % 2 == 0:
            raise ValueError("loss weights must be non-negative and window_size odd and >= 3")
        self.gradient_weight = float(gradient_weight)
        self.structural_weight = float(structural_weight)
        self.window_size = int(window_size)

    def _structural_loss(self, prediction: torch.Tensor, target: torch.Tensor,
                         valid: torch.Tensor) -> torch.Tensor:
        kernel = self.window_size
        padding = kernel // 2
        weight = valid.to(dtype=prediction.dtype)

        def local_mean(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            support = F.avg_pool2d(weight, kernel, stride=1, padding=padding, count_include_pad=True)
            mean = F.avg_pool2d(value * weight, kernel, stride=1, padding=padding,
                                count_include_pad=True) / support.clamp_min(1e-8)
            return mean, support

        mean_p, support = local_mean(prediction)
        mean_t, _ = local_mean(target)
        mean_p2, _ = local_mean(prediction.square())
        mean_t2, _ = local_mean(target.square())
        mean_pt, _ = local_mean(prediction * target)
        var_p = (mean_p2 - mean_p.square()).clamp_min(0)
        var_t = (mean_t2 - mean_t.square()).clamp_min(0)
        covariance = mean_pt - mean_p * mean_t
        c1, c2 = 0.01**2, 0.03**2
        score = ((2 * mean_p * mean_t + c1) * (2 * covariance + c2)) / (
            (mean_p.square() + mean_t.square() + c1) * (var_p + var_t + c2)
        ).clamp_min(1e-8)
        centers = valid & (support > 1e-8)
        if not centers.any():
            return prediction.sum() * 0.0
        return ((1.0 - score[centers]) * 0.5).mean()

    def forward(self, prediction: torch.Tensor, target: torch.Tensor,
                valid_mask: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if prediction.shape != target.shape or prediction.ndim != 4 or prediction.shape[1] != 1:
            raise ValueError("prediction and target must have matching (N, 1, H, W) shapes")
        if valid_mask.shape != target.shape:
            raise ValueError("valid_mask must match target shape")
        valid = valid_mask.bool() & torch.isfinite(target) & torch.isfinite(prediction)
        if not valid.any():
            zero = prediction.sum() * 0.0
            return zero, {"l1": zero.detach(), "gradient": zero.detach(), "structural": zero.detach()}

        safe_prediction = torch.where(valid, prediction, torch.zeros_like(prediction))
        safe_target = torch.where(valid, target, torch.zeros_like(target))
        l1 = (safe_prediction[valid] - safe_target[valid]).abs().mean()

        dx_p, dx_t = safe_prediction[..., :, 1:] - safe_prediction[..., :, :-1], safe_target[..., :, 1:] - safe_target[..., :, :-1]
        dy_p, dy_t = safe_prediction[..., 1:, :] - safe_prediction[..., :-1, :], safe_target[..., 1:, :] - safe_target[..., :-1, :]
        valid_dx = valid[..., :, 1:] & valid[..., :, :-1]
        valid_dy = valid[..., 1:, :] & valid[..., :-1, :]
        grad_terms = []
        if valid_dx.any():
            edge_weight = 1.0 + 4.0 * dx_t.abs()
            grad_terms.append((edge_weight[valid_dx] * (dx_p - dx_t).abs()[valid_dx]).sum() /
                              edge_weight[valid_dx].sum().clamp_min(1e-8))
        if valid_dy.any():
            edge_weight = 1.0 + 4.0 * dy_t.abs()
            grad_terms.append((edge_weight[valid_dy] * (dy_p - dy_t).abs()[valid_dy]).sum() /
                              edge_weight[valid_dy].sum().clamp_min(1e-8))
        gradient = torch.stack(grad_terms).mean() if grad_terms else l1 * 0.0
        structural = self._structural_loss(safe_prediction, safe_target, valid)
        total = l1 + self.gradient_weight * gradient + self.structural_weight * structural
        return total, {"l1": l1.detach(), "gradient": gradient.detach(),
                       "structural": structural.detach()}
