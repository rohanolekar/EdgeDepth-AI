# Ablation Plan and Failure Analysis

## Proposed ablation configurations and completion status

The configurations below describe method compositions. Standalone methods with a different starting interpolation do not count as evaluating a proposed composition.

| ID | Proposed configuration | Status in existing artifacts | Evidence and clarification |
|---|---|---|---|
| A | Bicubic only | Evaluated | Present as a method in the 96-scene synthetic report and the 100-image NYU report. |
| B | Bicubic + guided filter | Not evaluated as this composition | The synthetic report evaluates guided filtering on a bilinearly upsampled map. The NYU report has no guided-filter result. |
| C | Bicubic + joint bilateral filter | Evaluated on NYU only | The NYU evaluator applies the RGB-guided joint bilateral filter to bicubic depth. In the synthetic evaluator, joint bilateral filtering is applied to bilinear depth instead. |
| D | Bicubic + CNN | Evaluated on NYU only | The NYU evaluator supplies bicubic depth as the CNN initial estimate. Synthetic CNN v2 evaluation supplies the dataset's bilinear initial estimate. |
| E | Bicubic + guided filter + CNN | Not evaluated | No result artifact runs guided filtering followed by CNN refinement. |

No scores are assigned to unevaluated configurations. The existing synthetic report contains useful single-stage comparisons, but its bilinear starts mean they cannot be substituted for the bicubic compositions listed above. The only synthetic evidence for A is bicubic by itself.

## NYU supervised fine-tuning comparison

V3 keeps the v2 five-channel residual U-Net architecture and changes the training data/objective: it is initialized from v2 and fine-tuned on the 795 NYU training pairs with masked depth, edge-weighted gradient, and low-weight structural losses. The 654 NYU validation pairs select the checkpoint. Evaluation uses the same 100 deterministic validation frames and preprocessing for all methods. Because the validation split was used for checkpoint selection, this comparison is **not an independent test-set evaluation**.

| Method | RMSE (m) | MAE (m) | PSNR (dB) | SSIM | Edge P | Edge R | Edge F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Bicubic | 0.0941 | 0.0453 | 42.0615 | 0.9684 | 0.9250 | 0.5777 | 0.6905 |
| Joint bilateral | 0.1002 | 0.0504 | 41.4661 | 0.9690 | 0.9156 | 0.6503 | 0.7509 |
| CNN v2, zero-shot | 0.1509 | 0.0722 | 37.2308 | 0.9482 | 0.7738 | 0.8439 | 0.8001 |
| CNN v3, fine-tuned | 0.0671 | 0.0317 | 45.1046 | 0.9825 | 0.9618 | 0.8230 | 0.8853 |

V3 improves the four depth-quality metrics, edge precision, and edge F1 over v2, while edge recall regresses by about 0.021. Precision rises enough that F1 improves. This is a measured trade-off, not evidence that v3 improves every edge metric. The saved output and full configuration are in `evaluation/results/nyu/nyu_v2_v3_metrics.json` and `evaluation/results/nyu/nyu_v2_v3_report.md`.

## Failure cases and experiments needed

These are risks inferred from the pipeline and task, not claims that a controlled failure study has already measured them.

### Large missing depth regions

Iterative neighbor filling can propagate nearby values into holes, but very large holes have no local measurement support. The CNN receives the validity mask, yet must infer depth from RGB and context; that inference can be ambiguous.

**Future experiment:** create held-out masks with increasing hole area and varied hole shapes. Report valid-pixel metrics inside and outside holes separately, plus edge F1 at hole boundaries. Compare the current fill, a nearest-valid fill, and a learned or confidence-aware fill.

### Thin structures

Area downsampling can remove structures thinner than the low-resolution cell size. Upsampling and filtering may then fail to reconstruct their depth even when the RGB guide retains the structure.

**Future experiment:** vary structure width and orientation against the 4× sampling grid. Measure depth error on the structure mask, boundary displacement, and edge recall. Include cases where the RGB structure is visible and cases where it has low contrast.

### Similar RGB colors with different depths

Joint bilateral filtering relies on color differences to guide depth weights. If two surfaces have similar colors but a real depth discontinuity, the filter may smooth across that boundary. CNN behavior may also depend on patterns learned from the synthetic color/depth correlations.

**Future experiment:** generate paired scenes with identical RGB around a boundary but different depth jumps, then sweep color contrast independently of depth contrast. Compare A–E using fixed scene inputs and masks.

### Sensor noise

The v2 synthetic evaluation uses Gaussian noise standard deviation 0.03 and a 0.05 missing rate. This does not cover all real sensor noise, quantization, structured holes, multipath artifacts, or depth-dependent uncertainty.

**Future experiment:** sweep noise levels and distributions, depth-dependent noise, quantization, and structured missing patterns. Keep a clean reference fixed, report each condition independently, and document whether the noise settings match a target sensor.

### Abrupt depth boundaries

Bicubic interpolation can overshoot near a sharp transition. A color boundary can be displaced relative to the depth edge, and the CNN's residual is bounded to the configured normalized range, which can limit correction size.

**Future experiment:** control depth jump magnitude, edge angle, RGB/depth edge alignment, and local texture. Report boundary localization error and per-edge precision/recall alongside RMSE/MAE.

## Evaluation protocol for the planned ablation

For a valid composition comparison, every method must use the same RGB guide, low-resolution observations, high-resolution reference, validity mask, resolution, and scale. Run A–E on both the synthetic held-out test stream and a declared NYU validation subset where the method is applicable. Save per-scene outputs and configuration metadata. Keep training data, synthetic test scenes, and NYU validation separate. Do not choose thresholds or hyperparameters based on the evaluated test references.
