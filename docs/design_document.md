# EdgeDepth AI Design Document

## Problem statement

Depth sensors and depth maps are often available at lower spatial resolution than RGB images. Upsampling depth with generic interpolation smooths values across object boundaries and can blur thin structures. EdgeDepth AI studies RGB-guided upsampling and refinement while retaining classical methods as baselines.

## Objectives

- Accept a color guidance image and a sparse or low-resolution depth map.
- Produce a dense depth map at a selected integer scale while keeping invalid sensor samples distinguishable from measured values.
- Compare classical interpolation and filtering with a lightweight CNN.
- Report genuine measurements on separate synthetic test scenes and on an explicitly identified NYU Depth V2 validation subset.
- Provide a local Streamlit interface that can run on CUDA or CPU.

## Proposed solution

The project has two related flows. The classical evaluation runs nearest, bilinear, and bicubic interpolation, joint bilateral filtering, and guided filtering against the same synthetic high-resolution reference. The learned flow supplies RGB, an upsampled depth estimate, and a validity mask to the same five-channel residual U-Net. The Streamlit app defaults to NYU-fine-tuned v3 and keeps v2 selectable as a fallback.

CNN v2 was trained on procedural synthetic scenes and evaluated on held-out synthetic scenes; its NYU result is zero-shot. CNN v3 starts from v2 and is fine-tuned on NYU Depth V2 training images. NYU validation images are used only for loss-based checkpoint selection and evaluation; they do not contribute gradient updates.

## System architecture

See [System Architecture](system_architecture.md) for the Mermaid diagram and detailed data flow. At a high level, RGB and depth are validated, the depth validity mask is retained, interpolation creates dense depth inputs, and the five-channel tensor is passed to the existing PyTorch model. CUDA is used when available; otherwise inference runs on CPU.

## Dataset and preprocessing

### Synthetic data

`src/dataset.py` creates procedural RGB images and independent depth references with multiple depth planes, irregular and rectangular objects, thin structures, texture, and sharp discontinuities. For each reference, low-resolution observations are made by non-overlapping area averaging, followed by configurable Gaussian noise and missing samples. Train, validation, and test use separate deterministic seed streams.

The v2 run used 512 training scenes and 96 validation scenes at 128×128, scale 4, noise standard deviation 0.03, and missing-depth rate 0.05. The CNN evaluation used 96 separate deterministic test scenes with the same spatial, scale, noise, and missing-rate settings. The synthetic depth range is normalized by 10.0; these synthetic depth units are not metres.

### NYU Depth V2 zero-shot validation

`evaluation/evaluate_nyu.py` evaluates only `images/val` and `depth/val`. It selects 100 validation pairs deterministically, resizes RGB and validity-weighted metric depth to 128×128, forms 4× low-resolution observations with area downsampling, fills invalid low-resolution samples for interpolation, and compares bicubic, RGB-guided joint bilateral, and CNN v2. The selected depth values and reported RMSE/MAE are in metres. This is zero-shot real-world validation; no NYU samples were used for training or fine-tuning.

### NYU Depth V2 supervised fine-tuning

`src/nyu_dataset.py` forms aligned 128×128 RGB/reference pairs and 4× area-downsampled depth observations from the NYU training split. RGB, initial depth, observation mask, target, and target-validity mask receive synchronized flips and quarter-turn rotations. The 795 training pairs are the only samples used for gradient updates. The 654 validation pairs are used for early stopping and best-checkpoint selection; consequently, the reported 100-sample comparison from that split is validation evaluation, not an independent test-set evaluation. `evaluation/evaluate_nyu_comparison.py` compares v2 and v3 on the same deterministic 100 frames and includes the unchanged bicubic and joint bilateral baselines.

### Invalid values and normalization

Finite positive depth samples are valid. Nonfinite and nonpositive depth values are invalid. Filling is performed on a working copy for interpolation; ground-truth validity is preserved for scoring. Both checkpoints use a 10 metre range and the same five-channel architecture. Depth is normalized once at input and model output is converted to metres once for reporting.

## Classical interpolation methods

- **Nearest-neighbour:** copies the closest low-resolution sample and preserves its value, with block-like output.
- **Bilinear:** computes a weighted average of nearby samples and yields a smoother baseline.
- **Bicubic:** uses a cubic neighbourhood and often appears smoother, but can overshoot around discontinuities.
- **Joint bilateral filtering:** filters depth with spatial weights and RGB-derived range weights to align smoothing with color boundaries.
- **Guided filtering:** uses local linear models guided by grayscale RGB and can preserve some edges while smoothing within regions.

The synthetic evaluator's standalone joint bilateral and guided baselines are applied to bilinearly upsampled depth. The NYU evaluator applies joint bilateral filtering to bicubic depth. This difference is retained in the evaluation scripts and is called out in the ablation analysis.

## Residual U-Net architecture

`models/depth_refinement_cnn.py` implements a compact fully convolutional encoder-decoder. Its five input channels are RGB (3), normalized initial depth (1), and an upsampled validity mask (1). With the v2 configuration, the feature widths are 16, 32, and 64 channels. Two-convolution blocks use GroupNorm and SiLU; max pooling forms the encoder and bilinear feature resizing with skip connections forms the decoder. A 1×1 head predicts a normalized residual, bounded by `tanh` and the configured maximum residual. The head is initialized to zero. `refine()` adds the residual once and clamps normalized output to `[0, 1]`.

The synthetic v2 training/evaluation path uses bilinear initial depth. NYU fine-tuning/evaluation uses bicubic initial depth. The Streamlit app uses bicubic input for v3 to match its NYU training and evaluation path; selecting v2 retains the earlier bilinear app path.

## Training procedure and loss function

The v2 training run used PyTorch `Dataset`/`DataLoader`, batch size 8, AdamW with learning rate 0.001 and weight decay `1e-4`, 20 requested epochs, deterministic seed 1234, and validation-based best-checkpoint selection. The best model was saved at epoch 20. Training used the NVIDIA GeForce RTX 4050 Laptop GPU with CUDA.

The loss is the valid-pixel L1 reconstruction term plus 0.25 times the L1 depth-gradient term. Gradient differences are included only for adjacent pairs where both reference pixels are valid. The v2 experiment retained the v1 gradient weight so the increased training data and training duration could be assessed without changing the loss weighting.

Recorded v2 training facts from `models/checkpoints/training_history_v2.json`:

- Best validation loss: `0.0046876183` at epoch 20.
- Training duration: `84.2273` seconds.
- Peak allocated GPU memory: `236,436,992` bytes (about `225.5 MiB`).
- Checkpoint: `models/checkpoints/depth_refinement_v2_best.pt`.

The v3 fine-tune initialized from the v2 checkpoint and used AdamW (`1e-5`, weight decay `1e-4`), batch size 8, gradient clipping at 1.0, a `ReduceLROnPlateau` schedule, and early stopping. Its normalized loss is masked L1 (1.0), edge-weighted gradient (0.25), and masked local structural consistency (0.05). The saved run completed 30 epochs on an RTX 4050 Laptop GPU; best validation loss was `0.0043759312` at epoch 30, duration `1,754.55` seconds, peak allocation `240,031,744` bytes. History is in `models/checkpoints/training_history_v3.json`.

## Inference pipeline

The Streamlit app accepts RGB JPG/PNG and a low-resolution depth `.npy` in metres. It checks that depth is numeric and single-channel, identifies invalid values, fills invalid values only for interpolation, and constructs a nearest-upsampled validity mask. The default v3 checkpoint receives normalized bicubic initial depth; selectable v2 retains the app's bilinear input. Both use `load_model` and `refine_batch`, with CUDA when available and CPU fallback. Optional aligned ground truth enables masked RMSE, MAE, PSNR, SSIM, and edge precision/recall/F1. No live metrics are shown without ground truth.

## Evaluation metrics

The repository contains no separate organizer metric-order specification. The reports follow the order already used in the README and metric functions:

1. RMSE
2. MAE
3. PSNR
4. SSIM
5. Edge precision
6. Edge recall
7. Edge F1

RMSE and MAE use overlapping finite predictions and finite reference pixels, excluding invalid reference pixels. PSNR uses a fixed range of 10 depth units (metres for NYU). SSIM uses the project's fixed 7×7 Gaussian window with sigma 1.5, population covariance, and validity-normalized local statistics. Depth edges use central-difference gradient magnitude threshold 0.10 depth units per pixel and symmetric one-pixel tolerance; unmatched predicted edges count against precision. The exact definitions are implemented in `src/metrics.py` and `src/edge_detection.py`.

The synthetic CNN evaluator reports macro-average per-scene metrics over 96 synthetic test scenes. NYU metrics are macro averages over 100 validation images. These datasets and reports are separate and must not be combined into one score.

## Hardware and software requirements

- Python 3.12 is the project development environment.
- PyTorch is required for CNN inference and training.
- CUDA is optional. The recorded v2 run used CUDA on an NVIDIA GeForce RTX 4050 Laptop GPU; inference also supports CPU.
- Streamlit, NumPy, Pillow, and OpenCV headless are listed in `requirements.txt` for the app and project pipeline.
- NYU dataset files are only needed when running the NYU evaluation script. They are not loaded when the Streamlit app starts.

Run the web app from the project root with `python -m streamlit run app.py`.

## Limitations and future improvements

- Training scenes are procedural and do not cover the full variety of real RGB/depth sensors, materials, lighting, or missing-data patterns.
- NYU comparison is a 100-image validation subset at 128×128 and scale 4. V3 checkpoint selection used the validation split, so this is not an independent test-set evaluation and does not establish broad real-world generalization.
- On the recorded subset, v3 improves six metrics versus v2, but edge recall falls from 0.8439 to 0.8230. Broader held-out NYU testing is needed before generalization claims.
- The fixed 10-metre normalization and clipping limit depth outside that range.
- RGB boundaries do not always align with depth boundaries, and similar RGB colors can conceal depth changes.
- Larger and more varied real sensor datasets, explicit sensor noise modeling, calibration of edge thresholds, broader held-out evaluation, and the untested combinations in the ablation plan remain future work.
