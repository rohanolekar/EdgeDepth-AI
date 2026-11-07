# NYU Depth V2 v2 vs v3 comparison

Zero-shot v2 vs NYU-fine-tuned v3 on the same 100 deterministic validation frames. V3 checkpoint selection used the NYU validation split, so this is validation evaluation, not independent test-set evaluation.

| Method | RMSE (m) | MAE (m) | PSNR (dB) | SSIM | Edge P | Edge R | Edge F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Bicubic | 0.094144 | 0.045271 | 42.061498 | 0.968375 | 0.925029 | 0.577687 | 0.690471 |
| Joint bilateral | 0.100174 | 0.050362 | 41.466090 | 0.969044 | 0.915590 | 0.650257 | 0.750892 |
| CNN v2 | 0.150928 | 0.072194 | 37.230764 | 0.948231 | 0.773750 | 0.843928 | 0.800076 |
| CNN v3 | 0.067083 | 0.031712 | 45.104587 | 0.982538 | 0.961774 | 0.822978 | 0.885324 |

## V3 change from v2

| Metric | V2 | V3 | V3 − V2 | Result |
|---|---:|---:|---:|---|
| rmse | 0.150928 | 0.067083 | -0.083845 | improved |
| mae | 0.072194 | 0.031712 | -0.040482 | improved |
| psnr | 37.230764 | 45.104587 | +7.873823 | improved |
| ssim | 0.948231 | 0.982538 | +0.034306 | improved |
| edge_precision | 0.773750 | 0.961774 | +0.188023 | improved |
| edge_recall | 0.843928 | 0.822978 | -0.020950 | regressed |
| edge_f1 | 0.800076 | 0.885324 | +0.085249 | improved |

## Selection decision

V3 meets the defined selection rule: lower RMSE and MAE, higher PSNR and SSIM, and edge F1 no lower than v2.

Selection rule: v3 must improve all four depth-quality metrics and maintain or improve edge F1.

Artifacts are generated from the measured run; no scores are adjusted.
