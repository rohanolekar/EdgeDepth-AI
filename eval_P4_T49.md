1RUA24CSE0372,1RUA24CSE0360,1RUA25CSE7014,1RUA25CSE7004

# EdgeDepth AI — P4 Evaluation Report

**Team ID:** T49  
**Team name:** Meta  
**Problem:** P4 — Edge-Preserving Depth Map Upsampling and Refinement

## Metric order and definitions

No separate organizer metric-order specification was found in the repository. This report follows the project README and implementation order: **RMSE, MAE, PSNR, SSIM, edge precision, edge recall, edge F1**.

- RMSE and MAE use overlapping finite predictions and finite reference pixels, excluding invalid reference pixels.
- PSNR uses the fixed depth range 10.0 (metres for NYU; synthetic depth units for the procedural benchmark).
- SSIM uses the project's fixed 7×7 Gaussian implementation with sigma 1.5 and masked local statistics.
- Depth edges use central-difference gradient magnitude threshold 0.10 depth units per pixel and symmetric one-pixel matching tolerance. Unmatched predicted edges count against precision.
- Values below are macro averages from the saved JSON evaluation reports. Metrics are rounded to four decimal places for display.

## Synthetic CNN v2 comparison

The source artifact is [`sample_data/results/cnn_v2_metrics.json`](sample_data/results/cnn_v2_metrics.json), generated from **96 held-out procedural synthetic test scenes**. Each scene is 128×128 at 4× scale. Low-resolution observations were formed by area downsampling, then adding Gaussian noise with standard deviation 0.03 and missing samples at rate 0.05. All methods in that run share the same per-scene RGB, low-resolution observation, and ground truth. The split uses a separate deterministic test seed stream from training and validation. These values are synthetic depth units and are not real-world measurements.

| Method | RMSE | MAE | PSNR (dB) | SSIM | Edge precision | Edge recall | Edge F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Nearest | 0.4104 | 0.1378 | 28.1875 | 0.8835 | 0.6866 | 0.9698 | 0.8034 |
| Bilinear | 0.3768 | 0.1491 | 28.9410 | 0.9023 | 0.6581 | 0.9084 | 0.7597 |
| Bicubic | 0.3499 | 0.1434 | 29.5793 | 0.9042 | 0.6176 | 0.9339 | 0.7402 |
| Joint bilateral | 0.2913 | 0.1324 | 31.2303 | 0.9525 | 0.7257 | 0.9605 | 0.8197 |
| Guided | 0.4563 | 0.2424 | 27.3285 | 0.8994 | 0.5249 | 0.8987 | 0.6499 |
| CNN v2 | 0.1369 | 0.0386 | 37.8840 | 0.9876 | 0.9619 | 0.9909 | 0.9760 |

The synthetic evaluator reports CNN inference at **0.0025464 seconds per image** (about 2.546 ms): CUDA model-forward timing only, excluding host/device transfers and preprocessing. It used 3 warm-up runs and 10 timed runs per scene; the reported aggregate is the median of the per-scene medians. Hardware was an NVIDIA GeForce RTX 4050 Laptop GPU. This timing is not an end-to-end app latency measurement.

## NYU Depth V2 zero-shot validation

The source artifact is [`evaluation/results/nyu/nyu_v2_metrics.json`](evaluation/results/nyu/nyu_v2_metrics.json). This is **100 NYU Depth V2 validation samples**, resized to 128×128 and evaluated at 4× scale, with depth in metres and a fixed 10-metre data range. The CNN was not trained or fine-tuned on NYU. These real-world validation results are separate from both synthetic training and synthetic testing.

| Method | RMSE (m) | MAE (m) | PSNR (dB) | SSIM | Edge precision | Edge recall | Edge F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Bicubic | 0.0941 | 0.0453 | 42.0615 | 0.9684 | 0.9250 | 0.5777 | 0.6905 |
| Joint bilateral | 0.1002 | 0.0504 | 41.4661 | 0.9690 | 0.9156 | 0.6503 | 0.7509 |
| CNN v2 | 0.1509 | 0.0722 | 37.2308 | 0.9482 | 0.7738 | 0.8439 | 0.8001 |

On this NYU subset, CNN v2 has the highest edge recall and edge F1 among the three methods. Bicubic has the lowest RMSE and MAE and the highest PSNR. Joint bilateral has the highest SSIM. CNN v2 does **not** lead every metric. The NYU JSON contains no inference timing, so no NYU latency is reported.

## NYU Depth V2 fine-tuning and v2/v3 comparison

The source artifacts are [`evaluation/results/nyu/nyu_v2_v3_metrics.json`](evaluation/results/nyu/nyu_v2_v3_metrics.json), [`evaluation/results/nyu/nyu_v2_v3_comparison.csv`](evaluation/results/nyu/nyu_v2_v3_comparison.csv), and [`evaluation/results/nyu/nyu_v2_v3_comparison.png`](evaluation/results/nyu/nyu_v2_v3_comparison.png). The comparison uses the same 100 deterministic NYU validation samples, 128×128 resolution, 4× scale, validity masks, preprocessing, and metric definitions for bicubic, joint bilateral, v2, and v3. **This is validation evaluation, not an independent test-set evaluation.** The v3 checkpoint was selected using loss on the NYU validation split.

| Method | RMSE (m) | MAE (m) | PSNR (dB) | SSIM | Edge precision | Edge recall | Edge F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Bicubic | 0.0941 | 0.0453 | 42.0615 | 0.9684 | 0.9250 | 0.5777 | 0.6905 |
| Joint bilateral | 0.1002 | 0.0504 | 41.4661 | 0.9690 | 0.9156 | 0.6503 | 0.7509 |
| CNN v2, zero-shot | 0.1509 | 0.0722 | 37.2308 | 0.9482 | 0.7738 | 0.8439 | 0.8001 |
| CNN v3, NYU-fine-tuned | 0.0671 | 0.0317 | 45.1046 | 0.9825 | 0.9618 | 0.8230 | 0.8853 |

V3 improves RMSE, MAE, PSNR, SSIM, edge precision, and edge F1 over v2 in this subset. Edge recall falls from 0.8439 to 0.8230 (−0.0210); v3's higher precision yields a higher F1. The saved comparison selection rule is met, but validation checkpoint selection means this comparison is not independent evidence of test-set generalization.

The v3 fine-tune initialized from v2 and used 795 NYU training pairs for gradient updates; 654 validation pairs were used for loss-based checkpoint selection only. It ran for 30 epochs on CUDA using the NVIDIA GeForce RTX 4050 Laptop GPU, with AdamW at `1e-5`, batch size 8, gradient clipping at 1.0, and synchronized flips/90-degree rotations. Its masked loss combined normalized L1 (1.0), edge-weighted gradient (0.25), and masked local structural consistency (0.05). Best validation loss was `0.0043759312` at epoch 30; duration was `1,754.55` seconds and peak allocated GPU memory was `240,031,744` bytes. The checkpoint is [`models/checkpoints/depth_refinement_v3_best.pt`](models/checkpoints/depth_refinement_v3_best.pt), with history at [`models/checkpoints/training_history_v3.json`](models/checkpoints/training_history_v3.json).

## Training record

The checkpoint metadata and [`models/checkpoints/training_history_v2.json`](models/checkpoints/training_history_v2.json) record training on 512 procedural scenes, with 96 separate validation scenes, 128×128 images, 4× scale, batch size 8, and 20 epochs. The model used five channels (RGB, upsampled depth, and validity mask), depth normalization range 10.0, and masked L1 plus 0.25× masked depth-gradient L1 loss.

- Best validation loss: `0.0046876` at epoch 20.
- Training duration: `84.2273` seconds.
- Peak GPU allocation: `236,436,992` bytes (about `225.5 MiB`).
- Device: CUDA on NVIDIA GeForce RTX 4050 Laptop GPU.
- Checkpoint: [`models/checkpoints/depth_refinement_v2_best.pt`](models/checkpoints/depth_refinement_v2_best.pt).

## Scope and limitations

Synthetic training and held-out synthetic testing assess behavior on procedural scenes only. NYU validation is a separate zero-shot evaluation on 100 validation samples and is not NYU training or fine-tuning. No physical sensor testing is claimed. These evaluations do not establish performance across all real-world sensors, settings, or scenes. The Streamlit app's optional per-upload metrics are not included in either table; no single-image demo metrics are mixed into these dataset-level results.
