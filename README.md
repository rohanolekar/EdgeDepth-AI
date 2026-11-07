# EdgeDepth AI

EdgeDepth AI is a computer vision hackathon project for edge-preserving depth map upsampling and refinement (problem P4). It currently contains classical interpolation/filtering baselines and a lightweight residual CNN prototype.

## Phase 2 CNN

`models/depth_refinement_cnn.py` implements a compact three-level U-Net. It concatenates five channels: normalized RGB (3), normalized initial depth (1), and a nearest-upsampled depth-validity mask (1). The initial interpolation is bilinear for the synthetic v2 path and bicubic for NYU-fine-tuned v3. The network predicts a bounded one-channel residual, initialized to zero. `refine()` adds that residual to the initial depth and clamps normalized output to `[0, 1]`; training and inference use a configurable physical depth range (10 metres for NYU checkpoints).

The training objective combines valid-pixel L1 depth error with a masked depth-gradient L1 loss. Invalid ground-truth pixels do not contribute. The model uses GroupNorm and SiLU and avoids a dependency on torchvision.

## Synthetic data

`src/dataset.py` creates deterministic procedural scenes with variable background surfaces, multiple rectangular, elliptical and irregular objects, thin structures, texture, depth variation and sharp discontinuities. Independent high-resolution RGB and depth maps are produced per sample. Low-resolution depth is generated with non-overlapping area averaging, followed by configurable Gaussian noise and missing samples. Train, validation and test data use separate deterministic seed streams.

This dataset is synthetic. Synthetic metrics describe performance on procedural scenes and do not establish performance on real sensors, real scenes, or public depth datasets. Saved synthetic and NYU results are reported separately in the evaluation documentation.

## Environment

Use Python 3.12 and an installed PyTorch build. A CUDA-enabled PyTorch installation uses the GPU when `torch.cuda.is_available()`; training and evaluation fall back to CPU otherwise. The model is intentionally small for development on an NVIDIA RTX 4050 Laptop GPU with 6 GB VRAM. Larger images, batches or channel widths increase memory use.

Install the packages listed in `requirements.txt`. For GPU use, select the PyTorch wheel matching the installed CUDA driver/runtime using the official PyTorch installation selector; a generic CPU-only wheel will not enable CUDA.

## Train

The quick default uses three epochs, 32 training scenes, 8 validation scenes, 64x64 images and batch size 4:

```powershell
python -m src.train
```

Example longer run:

```powershell
python -m src.train --epochs 40 --train-samples 512 --validation-samples 96 --batch-size 8 --height 128 --width 128 --scale 4 --patience 7
```

The trainer selects the best checkpoint by validation loss and saves it to `models/checkpoints/depth_refinement_best.pt`, with a companion `training_history.json`. Training and validation are independent scene streams; test data is not used for checkpoint selection.

## Evaluate

Evaluate the held-out synthetic test stream with:

```powershell
python -m evaluation.evaluate_cnn
```

The evaluator compares nearest, bilinear, bicubic, joint bilateral, guided filtering and CNN refinement on the same test scenes and observations. It reports RMSE, MAE, fixed-range PSNR, SSIM, edge precision/recall/F1, and separate repeated CNN inference timings. It writes machine-readable metrics to `sample_data/results/cnn_metrics.json` and a visual panel to `sample_data/results/cnn_comparison.png`.

## Streamlit app

Run the interactive RGB-guided depth refinement interface locally with:

```powershell
python -m streamlit run app.py
```

Upload an RGB JPG/PNG, a low-resolution, single-channel `.npy` depth map in metres, and optionally aligned ground truth. CNN v3, fine-tuned on NYU Depth V2 training data, is the default; CNN v2 remains selectable as a fallback. The dashboard compares RGB guidance, input depth, ground truth when supplied, bicubic upsampling, and the selected CNN. With ground truth, it computes masked RMSE, MAE, PSNR, SSIM, and edge precision/recall/F1 using the project metrics. Refined depth and a comparison visualization can be downloaded. Model loading is lazy, CPU inference is supported, and NYU dataset files are not required at app startup. Stored benchmarks are clearly separate from live-upload metrics.

For Streamlit Community Cloud, select Python 3.12 in the app's Advanced settings to match the project's tested Python and PyTorch environment. The Cloud Python default can change independently of this repository.

### NYU Depth V2 fine-tuning results

CNN v3 was initialized from v2 and fine-tuned with the 795 NYU training pairs. The 654 NYU validation pairs were used for validation-loss checkpoint selection. This comparison uses 100 deterministic samples from that same validation split at 128×128 and 4× scale. It is a validation evaluation, **not an independent test-set evaluation**. Full precision values are in `evaluation/results/nyu/nyu_v2_v3_metrics.json`; the original v2 zero-shot report remains in `evaluation/results/nyu/nyu_v2_metrics.json`.

| Method | RMSE (m) | MAE (m) | PSNR (dB) | SSIM | Edge precision | Edge recall | Edge F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Bicubic | 0.0941 | 0.0453 | 42.0615 | 0.9684 | 0.9250 | 0.5777 | 0.6905 |
| Joint bilateral | 0.1002 | 0.0504 | 41.4661 | 0.9690 | 0.9156 | 0.6503 | 0.7509 |
| CNN v2, zero-shot | 0.1509 | 0.0722 | 37.2308 | 0.9482 | 0.7738 | 0.8439 | 0.8001 |
| CNN v3, fine-tuned | 0.0671 | 0.0317 | 45.1046 | 0.9825 | 0.9618 | 0.8230 | 0.8853 |

V3 improves RMSE, MAE, PSNR, SSIM, edge precision, and edge F1 on this comparison. Edge recall decreases from 0.8439 to 0.8230. Because v3 checkpoint selection used the NYU validation split, these scores are not independent test-set results and do not establish generalization beyond this subset.

The v3 run used AdamW (`1e-5` learning rate, `1e-4` weight decay), batch size 8, synchronized flips/quarter-turns, gradient clipping at 1.0, a plateau scheduler, and early stopping. Its loss weights were masked L1 1.0, edge-weighted gradient 0.25, and masked local structural consistency 0.05. Best validation loss was `0.0043759312` at epoch 30; training took `1,754.55` seconds on an NVIDIA RTX 4050 Laptop GPU. The saved checkpoint and history are `models/checkpoints/depth_refinement_v3_best.pt` and `models/checkpoints/training_history_v3.json`.

To repeat fine-tuning and compare on the fixed 100-sample validation subset:

```powershell
python -m src.finetune_nyu --epochs 30 --batch-size 8 --learning-rate 1e-5 --patience 5
python -m evaluation.evaluate_nyu_comparison --samples 100
```

## Project documentation

- [System architecture](docs/system_architecture.md)
- [Design document](docs/design_document.md)
- [Ablation and failure analysis](docs/ablation_and_failure_analysis.md)
- [P4 evaluation report](eval_P4_T49.md)

Tests:

```powershell
python -m pytest -q
```

## Classical phase

The original pipeline remains available in `src/preprocessing.py`, `src/interpolation.py`, `src/filtering.py`, `src/edge_detection.py` and `src/metrics.py`. The synthetic classical evaluation is run with `python -m evaluation.run_classical`.

## Limitations

- Training data is procedurally generated and does not model all real sensor artifacts or scene distributions.
- The default model, epoch count and dataset are rapid-development settings, not tuned production settings.
- The model uses a fixed normalized depth range; deployment data outside the configured range needs an explicit range and preprocessing choice.
- CNN results must be measured on held-out scenes and, before real-world claims, on representative real datasets.
- Classical filters are CPU implementations; runtime comparisons depend on image size, hardware and timing configuration.
