"""Regression tests for the classical depth benchmark and evaluation metrics."""

import numpy as np
import pytest

from evaluation.run_classical import area_downsample, make_synthetic_benchmark
from src.edge_detection import edge_precision_recall_f1
from src.filtering import guided_filter, joint_bilateral_filter
from src.interpolation import upsample_depth
from src.metrics import evaluate, mae, processing_time, psnr, rmse, ssim
from src.preprocessing import fill_invalid_depth, normalize_depth, validate_depth, validate_rgb


def test_synthetic_ground_truth_is_independent_and_area_downsampled():
    bench = make_synthetic_benchmark(scale=2)
    gt, low = bench["ground_truth"], bench["lowres_depth"]
    assert gt.shape == (96, 128)
    assert low.shape == (48, 64)
    np.testing.assert_allclose(low, area_downsample(gt, 2))
    # High-resolution geometry has sub-block depth variation; it is not a tiled LR image.
    assert np.unique(gt).size > 100
    nearest = upsample_depth(low, 2, "nearest")
    assert rmse(gt, nearest, valid_mask=bench["valid_mask"]) > 0.01


def test_configurable_sensor_noise_and_missing_samples_do_not_change_gt():
    clean = make_synthetic_benchmark(4, seed=2)
    noisy = make_synthetic_benchmark(4, noise_std=0.2, missing_rate=0.2, seed=2)
    np.testing.assert_array_equal(clean["ground_truth"], noisy["ground_truth"])
    assert not np.array_equal(clean["lowres_depth"], noisy["lowres_depth"])
    assert np.isnan(noisy["lowres_depth"]).any()


def test_edge_precision_recall_f1_penalizes_false_positive_edges():
    ref = np.ones((20, 30), dtype=np.float32)
    ref[:, 10:] = 2.0
    pred = ref.copy()
    pred[:, 20:] += 1.0
    scores = edge_precision_recall_f1(ref, pred, threshold=0.1, tolerance=0)
    assert scores["edge_recall"] == pytest.approx(1.0)
    assert scores["edge_precision"] < 1.0
    assert scores["edge_f1"] < 1.0


def test_iterative_nan_fill_reads_accumulated_updates():
    source = np.array([[1.0, np.nan, np.nan, np.nan, 9.0]], dtype=np.float32)
    result = fill_invalid_depth(source)
    assert np.isfinite(result).all()
    assert result[0, 2] == pytest.approx(5.0)


def test_invalid_ground_truth_pixels_are_excluded_from_every_metric():
    gt = np.array([[1.0, np.nan], [2.0, 3.0]], dtype=np.float32)
    pred = np.array([[1.0, 999.0], [2.0, 3.0]], dtype=np.float32)
    valid = np.isfinite(gt)
    scores = evaluate(gt, pred, valid_mask=valid, data_range=10.0)
    assert scores["rmse"] == 0
    assert scores["mae"] == 0
    assert scores["psnr"] == float("inf")
    assert scores["ssim"] == pytest.approx(1.0)


def test_psnr_fixed_depth_range_and_ssim_deterministic():
    ref = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]], dtype=np.float32)
    pred = ref + 0.5
    assert psnr(ref, pred, data_range=10.0) == pytest.approx(10*np.log10(400))
    assert ssim(ref, pred, data_range=10.0) == ssim(ref, pred, data_range=10.0)
    with pytest.raises(ValueError):
        psnr(ref, pred, data_range=0)


def test_repeated_timing_reports_configured_runs_and_median():
    calls = []
    def work(value):
        calls.append(value)
        return value + len(calls)
    result, timing = processing_time(work, 3, warmup_runs=2, timed_runs=5)
    assert len(calls) == 7
    assert timing["warmup_runs"] == 2
    assert timing["timed_runs"] == 5
    assert len(timing["samples_seconds"]) == 5
    assert timing["median_seconds"] == float(np.median(timing["samples_seconds"]))
    assert result == 10


def test_interpolation_and_filters_share_shape_and_validate():
    depth = np.arange(48, dtype=np.float32).reshape(6, 8) / 10 + 1
    rgb = np.zeros((12, 16, 3), dtype=np.uint8)
    rgb[:, 8:] = 220
    for method in ("nearest", "bilinear", "bicubic"):
        assert upsample_depth(depth, 2, method).shape == (12, 16)
    initial = upsample_depth(depth, 2, "bilinear")
    assert joint_bilateral_filter(initial, rgb, radius=2).shape == initial.shape
    assert guided_filter(initial, rgb, radius=2).shape == initial.shape


def test_preprocessing_validation_and_metrics_known_errors():
    rgb = np.zeros((4, 6, 3), dtype=np.uint8)
    assert validate_rgb(rgb).shape == rgb.shape
    depth = np.array([[1.0, np.nan], [2.0, 3.0]], dtype=np.float32)
    checked = validate_depth(depth)
    assert np.isnan(checked[0, 1])
    assert normalize_depth(checked).max() == 1
    assert rmse(np.ones((2, 2)), np.ones((2, 2))) == 0
    assert mae(np.ones((2, 2)), np.ones((2, 2))) == 0
