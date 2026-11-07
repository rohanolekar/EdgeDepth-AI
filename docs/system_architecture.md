# EdgeDepth AI System Architecture

## Architecture diagram

```mermaid
flowchart TD
    subgraph Inputs
        RGB[RGB image JPG or PNG]
        LR[Low-resolution depth NPY]
        GT[Optional aligned ground-truth depth NPY]
    end

    UI[Streamlit EdgeDepth AI interface]
    RGB --> UI
    LR --> UI
    GT -. optional .-> UI

    UI --> RgbPrep[Decode RGB, resize to output grid, convert to RGB float in 0 to 1]
    UI --> DepthPrep[Validate metric depth; mark nonfinite and nonpositive samples invalid]
    DepthPrep --> Mask[Validity mask]
    DepthPrep --> Fill[Fill invalid samples for interpolation only]
    Fill --> Base[Bicubic upsampling for baseline display]
    Fill --> Initial[Checkpoint-compatible interpolation and normalize by depth range]
    Mask --> MaskUp[Nearest-neighbour mask upsampling]

    RgbPrep --> Rgb3[RGB guidance: 3 channels]
    Initial --> Depth1[Initial depth: 1 channel]
    MaskUp --> Mask1[Validity: 1 channel]
    Rgb3 --> Cat[Concatenate five channels]
    Depth1 --> Cat
    Mask1 --> Cat

    Cat --> Net[Residual U-Net CNN in PyTorch: v3 default, v2 fallback]
    Net --> Residual[Bounded normalized depth residual]
    Depth1 --> Add[Add residual once; clamp to normalized 0 to 1]
    Residual --> Add
    Add --> Refined[Refined depth in metres]

    subgraph Runtime[Inference runtime]
        Device{CUDA available?}
        Device -->|Yes| CUDA[PyTorch CUDA inference]
        Device -->|No| CPU[PyTorch CPU inference]
    end
    Net -. executes on .-> Device
    Base --> UI
    Refined --> UI

    GT --> Eval[Optional masked evaluation]
    Base --> Eval
    Refined --> Eval
    Eval --> Metrics[RMSE, MAE, PSNR, SSIM, edge precision, recall, and F1]
    Metrics --> UI
```

## Data flow

1. The Streamlit interface accepts an RGB image and a low-resolution, single-channel `.npy` depth array. A high-resolution ground-truth depth array is optional and is only used to calculate metrics.
2. RGB is decoded in channel order RGB and converted to float values in `[0, 1]`. The output grid is the depth grid multiplied by the selected 2×, 4×, or 8× scale. Large inputs are bounded to a maximum output side of 512 pixels.
3. Depth is interpreted in metres. Finite positive samples are valid; NaN, infinite, and nonpositive samples are marked invalid. Invalid input samples are filled only to make interpolation possible; the validity mask retains where the sensor supplied data.
4. The app produces a bicubic upsampled baseline for display. CNN v3 uses bicubic initial depth, matching its NYU fine-tuning and evaluation path; the selectable v2 fallback retains the app's earlier bilinear input. Each initial depth is normalized by the loaded checkpoint's saved range (10 metres). The low-resolution validity mask is upsampled with nearest-neighbour interpolation.
5. RGB, initial depth, and validity mask are concatenated into five channels: RGB (3), initial depth (1), and validity (1). The loaded residual U-Net predicts a bounded normalized residual. The residual is added once to the initial depth and the final normalized depth is clamped to `[0, 1]` before conversion back to metres.
6. Inference runs on CUDA when PyTorch reports it available and otherwise on CPU. The checkpoint is loaded only after the user requests inference. NYU data is not required for app startup.
7. If ground truth is supplied, only finite positive reference pixels contribute to the app's RMSE, MAE, PSNR, SSIM, and edge precision, recall, and F1.

The five-channel model architecture is unchanged between v2 and v3. V2 was trained on procedural scenes and evaluated zero-shot on NYU. V3 was initialized from v2 and fine-tuned using NYU training data only; the NYU validation split selected its checkpoint. Its 100-sample comparison is a validation evaluation, not an independent test-set result. The standalone evaluators and app use the same loader and normalized output convention, converting normalized depth to metres once using the checkpoint range.
