"""Model, synthetic dataset, loss, and checkpoint tests."""

from pathlib import Path

import numpy as np
import pytest
import torch

from models.depth_refinement_cnn import DepthRefinementCNN
from evaluation.evaluate_cnn import _cnn_depth_to_physical, _time_model
from src.dataset import SyntheticDataConfig, SyntheticDepthDataset, make_synthetic_splits
from src.inference import load_model, refine_batch, save_checkpoint
from src.losses import DepthReconstructionLoss
from src.metrics import evaluate


def _inputs(batch=2, height=32, width=40):
    return (torch.rand(batch, 3, height, width), torch.rand(batch, 1, height, width),
            torch.ones(batch, 1, height, width))


def test_model_shapes_residual_and_depth_constraints_cpu():
    model = DepthRefinementCNN(base_channels=8)
    rgb, initial, mask = _inputs()
    residual = model(rgb, initial, mask)
    prediction = model.refine(rgb, initial, mask)
    cpu_inference = refine_batch(model.eval(), rgb, initial, mask)
    assert residual.shape == (2, 1, 32, 40)
    assert torch.all(residual.abs() <= model.max_residual)
    assert prediction.min() >= 0 and prediction.max() <= 1
    torch.testing.assert_close(prediction, initial)
    torch.testing.assert_close(cpu_inference, initial)
    assert not cpu_inference.requires_grad


def test_timed_inference_disables_gradients_and_returns_numpy():
    model = DepthRefinementCNN(base_channels=8).eval()
    sample = {"rgb": torch.rand(3, 16, 16),
              "initial_depth": torch.rand(1, 16, 16),
              "validity_mask": torch.ones(1, 16, 16)}
    grad_states = []
    model.register_forward_hook(lambda _module, _inputs, _output: grad_states.append(torch.is_grad_enabled()))
    prediction, timing = _time_model(model, sample, torch.device("cpu"), warmup_runs=1, timed_runs=2)
    assert isinstance(prediction, np.ndarray)
    assert prediction.shape == (16, 16)
    assert grad_states == [False, False, False]
    assert timing["warmup_runs"] == 1
    assert timing["timed_runs"] == 2
    assert len(timing["samples_seconds"]) == 2
    assert all(parameter.grad is None for parameter in model.parameters())


def test_normalized_cnn_prediction_is_converted_before_metric_calculation():
    normalized_prediction = np.array([[.1, .2], [.3, .4]], dtype=np.float32)
    physical_ground_truth = normalized_prediction * 10.0
    physical_prediction = _cnn_depth_to_physical(normalized_prediction, depth_range=10.0)
    np.testing.assert_allclose(physical_prediction, physical_ground_truth)
    assert np.mean(np.abs(normalized_prediction-physical_ground_truth)) > 1.0
    scores = evaluate(physical_ground_truth, physical_prediction, data_range=10.0)
    assert scores["rmse"] == 0
    assert scores["mae"] == 0
    assert scores["psnr"] == float("inf")
    assert scores["ssim"] == pytest.approx(1.0)
    assert scores["edge_precision"] == pytest.approx(1.0)
    assert scores["edge_recall"] == pytest.approx(1.0)
    assert scores["edge_f1"] == pytest.approx(1.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_forward_inference():
    device = torch.device("cuda")
    model = DepthRefinementCNN(base_channels=8).to(device).eval()
    rgb, initial, mask = (x.to(device) for x in _inputs(batch=1, height=32, width=32))
    with torch.inference_mode():
        output = model.refine(rgb, initial, mask)
    assert output.device.type == "cuda"
    assert output.shape == (1, 1, 32, 32)


def test_dataset_sample_validity_and_reproducible_disjoint_splits():
    cfg = SyntheticDataConfig(samples=2, height=32, width=32, scale=2, seed=90)
    train_a = SyntheticDepthDataset(cfg, split="train")
    train_b = SyntheticDepthDataset(cfg, split="train")
    val = SyntheticDepthDataset(cfg, split="validation")
    sample = train_a[0]
    assert sample["inputs"].shape == (5, 32, 32)
    assert torch.isfinite(sample["inputs"]).all()
    assert torch.isfinite(sample["ground_truth"]).all()
    torch.testing.assert_close(sample["ground_truth"], train_b[0]["ground_truth"])
    assert not torch.equal(sample["ground_truth"], val[0]["ground_truth"])
    splits_a = make_synthetic_splits(train_samples=1, validation_samples=1, test_samples=1,
                                     seed=5, height=32, width=32)
    splits_b = make_synthetic_splits(train_samples=1, validation_samples=1, test_samples=1,
                                     seed=5, height=32, width=32)
    for name in splits_a:
        torch.testing.assert_close(splits_a[name][0]["ground_truth"], splits_b[name][0]["ground_truth"])


def test_loss_masks_invalid_pixels_and_backpropagates():
    prediction = torch.tensor([[[[1.0, 100.0], [1.0, 1.0]]]], requires_grad=True)
    target = torch.tensor([[[[0.0, float("nan")], [0.0, 0.0]]]])
    valid = torch.tensor([[[[True, False], [True, True]]]])
    loss, parts = DepthReconstructionLoss()(prediction, target, valid)
    assert torch.isfinite(loss)
    assert parts["l1"].item() == pytest.approx(1.0)
    loss.backward()
    assert prediction.grad is not None
    assert prediction.grad[0, 0, 0, 1] == 0


def test_checkpoint_save_and_load(tmp_path):
    model = DepthRefinementCNN(base_channels=8)
    path = tmp_path / "model.pt"
    save_checkpoint(path, model, epoch=2, validation_loss=.1, config={"depth_range": 10.0})
    loaded, metadata = load_model(path)
    assert metadata["epoch"] == 2
    for expected, actual in zip(model.parameters(), loaded.parameters()):
        torch.testing.assert_close(expected, actual)


def test_v3_checkpoint_loads_and_runs_cpu_inference():
    project_root = Path(__file__).resolve().parents[1]
    checkpoint_path = project_root / "models" / "checkpoints" / "depth_refinement_v3_best.pt"
    assert checkpoint_path.is_file(), "the v3 checkpoint must be present in the project"
    model, metadata = load_model(checkpoint_path, device="cpu")
    v2_path = project_root / "models" / "checkpoints" / "depth_refinement_v2_best.pt"
    _, v2_metadata = load_model(v2_path, device="cpu")
    assert model.input_channels == 5
    assert metadata["model"] == v2_metadata["model"]
    assert metadata["config"]["depth_range"] == 10.0
    assert metadata["config"]["dataset"] == "NYU Depth V2"
    rgb, initial, mask = _inputs(batch=1, height=32, width=32)
    output = refine_batch(model, rgb, initial, mask)
    assert output.shape == (1, 1, 32, 32)
    assert torch.isfinite(output).all()
    assert output.min() >= 0 and output.max() <= 1
    assert not output.requires_grad
