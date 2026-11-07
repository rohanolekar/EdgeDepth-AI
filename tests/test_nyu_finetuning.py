"""Tests for the NYU fine-tuning loss and synchronized augmentation."""

import torch

from src.nyu_dataset import NYUDepthDataset
from src.nyu_losses import NYURefinementLoss


def test_nyu_loss_masks_invalid_reference_pixels_and_backpropagates():
    target = torch.tensor([[[[0.2, 0.3], [0.4, 0.5]]]])
    target[0, 0, 0, 1] = float("nan")
    valid = torch.tensor([[[[True, False], [True, True]]]])
    prediction = torch.tensor([[[[0.1, 10.0], [0.3, 0.7]]]], requires_grad=True)
    loss, parts = NYURefinementLoss()(prediction, target, valid)
    assert torch.isfinite(loss)
    assert set(parts) == {"l1", "gradient", "structural"}
    loss.backward()
    assert prediction.grad is not None
    assert prediction.grad[0, 0, 0, 1] == 0


def test_nyu_augmentation_keeps_rgb_depth_target_and_mask_aligned():
    aligned = torch.arange(64, dtype=torch.float32).reshape(1, 8, 8)
    transformed = NYUDepthDataset._augment(aligned, aligned.clone(), aligned.clone(),
                                           aligned.clone(), aligned.clone())
    for item in transformed[1:]:
        torch.testing.assert_close(item, transformed[0])
