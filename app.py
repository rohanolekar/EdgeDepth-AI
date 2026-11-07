"""Streamlit research dashboard for EdgeDepth AI."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageOps

from src.inference import load_model, refine_batch
from src.interpolation import upsample_depth
from src.metrics import evaluate
from src.preprocessing import fill_invalid_depth, load_depth, load_rgb

ROOT = Path(__file__).resolve().parent
CHECKPOINTS = {
    "CNN v3 · NYU fine-tuned (default)": {
        "version": "v3",
        "path": ROOT / "models" / "checkpoints" / "depth_refinement_v3_best.pt",
        "initial_method": "bicubic",
    },
    "CNN v2 · synthetic fallback": {
        "version": "v2",
        "path": ROOT / "models" / "checkpoints" / "depth_refinement_v2_best.pt",
        "initial_method": "bilinear",
    },
}
NYU_RESULTS_V2 = ROOT / "evaluation" / "results" / "nyu" / "nyu_v2_metrics.json"
NYU_RESULTS_V2_V3 = ROOT / "evaluation" / "results" / "nyu" / "nyu_v2_v3_metrics.json"
DEMO_RGB = ROOT / "sample_data" / "demo" / "nyu_0000.jpg"
DEMO_GT = ROOT / "sample_data" / "demo" / "nyu_0000.npy"
DEMO_LOW_DEPTH = ROOT / "sample_data" / "demo" / "nyu_0000_lr.npy"
MAX_OUTPUT_SIDE = 512
MAX_RGB_SIDE = 2048
SCALE_CHOICES = (2, 4, 8)
GITHUB_URL = "https://github.com/rohanolekar/EdgeDepth-AI"

st.set_page_config(page_title="EdgeDepth AI | Depth Refinement", page_icon="🌊", layout="wide")

st.html("""<style>
    .block-container {max-width: 1500px; padding-top: 1.4rem; padding-bottom: 2.5rem;}
    .hero {padding: 1.35rem 1.55rem; border: 1px solid rgba(78,163,210,.25);
        border-radius: 18px; background: linear-gradient(125deg,#0c1c32,#102b49 62%,#123b57);}
    .hero h1 {margin: .15rem 0 .3rem; letter-spacing: -.035em;}
    .eyebrow {color:#68d8ed; font-size:.76rem; font-weight:700; letter-spacing:.13em; text-transform:uppercase;}
    .muted {color:#aabbd0;}
    .section-title {margin:1.2rem 0 .25rem; font-size:1.32rem; font-weight:700;}
    @media (max-width: 700px) { .block-container {padding-left:1rem;padding-right:1rem;} }
</style>""")


@st.cache_resource(show_spinner=False)
def _load_cached_model(checkpoint_path: str, device_name: str):
    """Load the existing checkpoint once per app process and selected device."""
    return load_model(checkpoint_path, device=device_name)


def _decode_rgb(upload) -> tuple[np.ndarray, tuple[int, int]]:
    """Decode an RGB upload, bounding server-side memory for large images."""
    source = upload if isinstance(upload, (str, Path)) else io.BytesIO(upload.getvalue())
    with Image.open(source) as image:
        original_size = image.size
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((MAX_RGB_SIDE, MAX_RGB_SIDE), Image.Resampling.LANCZOS)
        return load_rgb(np.asarray(image)), original_size


def _decode_depth(upload) -> np.ndarray:
    """Read a numeric single-channel NPY depth map, preserving invalid values."""
    try:
        source = upload if isinstance(upload, (str, Path)) else io.BytesIO(upload.getvalue())
        values = np.load(source, allow_pickle=False)
    except (ValueError, OSError, EOFError) as exc:
        raise ValueError("Could not read this file as a numeric .npy depth map.") from exc
    return load_depth(values)


def _resize_rgb(rgb: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    try:
        import cv2
        resized = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
    except ImportError:
        resized = np.asarray(Image.fromarray(rgb).resize((width, height), Image.Resampling.LANCZOS))
    return np.ascontiguousarray(resized.astype(np.uint8))


def _resize_depth(depth: np.ndarray, size: tuple[int, int], mode: str) -> np.ndarray:
    width, height = size
    try:
        import cv2
        interpolation = {"nearest": cv2.INTER_NEAREST, "bilinear": cv2.INTER_LINEAR,
                         "bicubic": cv2.INTER_CUBIC}[mode]
        return cv2.resize(depth.astype(np.float32), (width, height), interpolation=interpolation).astype(np.float32)
    except ImportError:
        resampling = {"nearest": Image.Resampling.NEAREST, "bilinear": Image.Resampling.BILINEAR,
                      "bicubic": Image.Resampling.BICUBIC}[mode]
        return np.asarray(Image.fromarray(depth.astype(np.float32), mode="F").resize((width, height), resampling),
                          dtype=np.float32)


def _resize_masked_depth(depth: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Resize GT while preventing invalid samples from contaminating valid ones."""
    valid = np.isfinite(depth) & (depth > 0)
    if not valid.any():
        raise ValueError("Ground-truth depth contains no valid positive samples.")
    try:
        import cv2
        width, height = size
        downsample = width <= depth.shape[1] and height <= depth.shape[0]
        mode = cv2.INTER_AREA if downsample else cv2.INTER_LINEAR
        support = cv2.resize(valid.astype(np.float32), (width, height), interpolation=mode)
        total = cv2.resize(np.where(valid, depth, 0).astype(np.float32), (width, height), interpolation=mode)
    except ImportError:
        width, height = size
        support = np.asarray(Image.fromarray(valid.astype(np.float32), mode="F").resize(
            (width, height), Image.Resampling.BOX), dtype=np.float32)
        total = np.asarray(Image.fromarray(np.where(valid, depth, 0).astype(np.float32), mode="F").resize(
            (width, height), Image.Resampling.BOX), dtype=np.float32)
    resized_valid = support > 1e-6
    resized = np.divide(total, support, out=np.full(support.shape, np.nan, np.float32), where=resized_valid)
    if not resized_valid.any():
        raise ValueError("Ground-truth depth has no valid pixels at the output resolution.")
    return resized.astype(np.float32), resized_valid


def _depth_visual(depth: np.ndarray | Image.Image, depth_range: float) -> Image.Image:
    # A depth visual that has already been colorized may be passed here by UI
    # composition code. Preserve its existing colors instead of colorizing RGB.
    if isinstance(depth, Image.Image) and depth.mode in ("RGB", "RGBA"):
        return depth.convert("RGB")
    if isinstance(depth, Image.Image) and depth.mode == "L":
        depth = np.asarray(depth)
    else:
        depth = np.asarray(depth)
    if depth.ndim != 2:
        raise ValueError("Depth visualization expects a single-channel depth map or an RGB/RGBA visualization.")
    finite = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    gray = (np.clip(finite / depth_range, 0, 1) * 255).astype(np.uint8)
    # ImageOps.colorize requires mode L; make that mode explicit.
    luminance = Image.fromarray(gray, mode="L")
    return ImageOps.colorize(luminance, black="#10243A", white="#55D8E8")


def _prepare_inputs(rgb: np.ndarray, low_depth: np.ndarray, scale: int, depth_range: float,
                    *, initial_method: str = "bilinear"):
    """Prepare model tensors using the trained pipeline's physical and normalized units."""
    if scale not in SCALE_CHOICES:
        raise ValueError("Select an upsampling scale of 2x, 4x, or 8x.")
    if initial_method not in ("bilinear", "bicubic"):
        raise ValueError("CNN initial depth method must be bilinear or bicubic.")
    if min(rgb.shape[:2]) < 16:
        raise ValueError("RGB image must be at least 16 pixels wide and high.")
    rgb_aspect, depth_aspect = rgb.shape[1] / rgb.shape[0], low_depth.shape[1] / low_depth.shape[0]
    if abs(rgb_aspect / depth_aspect - 1.0) > 0.10:
        raise ValueError("RGB and depth maps must have matching aspect ratios (within 10%).")
    valid = np.isfinite(low_depth) & (low_depth > 0)
    if not valid.any():
        raise ValueError("Depth map has no valid positive measurements.")
    if np.any(low_depth[valid] > depth_range * 1.5):
        raise ValueError(f"Depth values appear outside the checkpoint's metre range (0–{depth_range:g} m).")

    min_low_side, max_low_side = max(4, int(np.ceil(16 / scale))), MAX_OUTPUT_SIDE // scale
    shrink = min(1.0, max_low_side / max(low_depth.shape))
    out_h = max(min_low_side, int(round(low_depth.shape[0] * shrink)))
    out_w = max(min_low_side, int(round(low_depth.shape[1] * shrink)))
    target_size = (out_w, out_h)
    if (out_h, out_w) != low_depth.shape:
        try:
            import cv2
            support = cv2.resize(valid.astype(np.float32), target_size, interpolation=cv2.INTER_AREA)
            total = cv2.resize(np.where(valid, low_depth, 0).astype(np.float32), target_size,
                               interpolation=cv2.INTER_AREA)
        except ImportError:
            support = np.asarray(Image.fromarray(valid.astype(np.float32), mode="F").resize(
                target_size, Image.Resampling.BOX), dtype=np.float32)
            total = np.asarray(Image.fromarray(np.where(valid, low_depth, 0).astype(np.float32), mode="F").resize(
                target_size, Image.Resampling.BOX), dtype=np.float32)
        resized_valid = _resize_depth(valid.astype(np.float32), target_size, "nearest") > .5
        depth_resized = np.divide(total, support, out=np.full(support.shape, np.nan, np.float32), where=support > 1e-6)
        depth_resized[~resized_valid] = np.nan
        valid_resized = resized_valid
    else:
        depth_resized, valid_resized = low_depth.astype(np.float32, copy=True), valid
    if not valid_resized.any():
        raise ValueError("No valid depth samples remain after resizing.")

    # Fill only the model input. Keep the observation validity for its fifth channel.
    filled = np.clip(fill_invalid_depth(np.where(valid_resized, depth_resized, np.nan)), 0, depth_range)
    bicubic = np.clip(upsample_depth(filled, scale, "bicubic"), 0, depth_range)
    initial = np.clip(upsample_depth(filled, scale, initial_method), 0, depth_range)
    mask_high = upsample_depth(valid_resized.astype(np.float32), scale, "nearest") > .5
    height, width = bicubic.shape
    rgb_resized = _resize_rgb(rgb, (width, height))
    rgb_tensor = torch.from_numpy(rgb_resized.transpose(2, 0, 1).copy()).float().unsqueeze(0) / 255.0
    depth_tensor = torch.from_numpy((initial / depth_range)[None, None].astype(np.float32))
    mask_tensor = torch.from_numpy(mask_high.astype(np.float32)[None, None])
    return rgb_resized, bicubic, depth_resized, rgb_tensor, depth_tensor, mask_tensor


def _as_rgb_panel(image: np.ndarray | Image.Image) -> Image.Image:
    """Normalize RGB, grayscale L, and RGBA image panels to RGB for PNG composition."""
    if isinstance(image, Image.Image):
        if image.mode == "L":
            return image.convert("RGB")
        if image.mode == "RGBA":
            return image.convert("RGB")
        return image.convert("RGB")
    array = np.asarray(image)
    if array.ndim == 2:
        return Image.fromarray(array.astype(np.uint8), mode="L").convert("RGB")
    if array.ndim == 3 and array.shape[2] == 4:
        return Image.fromarray(array.astype(np.uint8), mode="RGBA").convert("RGB")
    if array.ndim == 3 and array.shape[2] == 3:
        return Image.fromarray(array.astype(np.uint8), mode="RGB")
    raise ValueError("Comparison panels must be RGB, grayscale, or RGBA images.")


def _visualization_png(rgb, low_depth, bicubic, cnn, depth_range, model_label="CNN v3") -> bytes:
    # ``low_depth`` may already be the colored PIL image displayed by the app.
    # Convert it to RGB directly instead of passing it through colorize again.
    low_panel = low_depth if isinstance(low_depth, Image.Image) else _depth_visual(low_depth, depth_range)
    panels = [_as_rgb_panel(rgb), _as_rgb_panel(low_panel),
              _as_rgb_panel(_depth_visual(bicubic, depth_range)),
              _as_rgb_panel(_depth_visual(cnn, depth_range))]
    width = max(panel.width for panel in panels)
    height = max(panel.height for panel in panels)
    labels = ("RGB guidance", "Input depth", "Bicubic", model_label)
    canvas = Image.new("RGB", (width * 2, height * 2 + 100), "#0b1729")
    from PIL import ImageDraw
    draw = ImageDraw.Draw(canvas)
    for index, (panel, label) in enumerate(zip(panels, labels)):
        x, y = (index % 2) * width, (index // 2) * (height + 50) + 40
        panel.thumbnail((width, height), Image.Resampling.LANCZOS)
        canvas.paste(panel, (x, y))
        draw.text((x + 8, y - 25), label, fill="#e7f5ff")
    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def _npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, array.astype(np.float32), allow_pickle=False)
    return buffer.getvalue()


def _show_upload_status(label: str, upload, parser, *, depth: bool = False):
    if upload is None:
        st.caption(f"{label}: waiting for upload")
        return None
    try:
        value = parser(upload)
        shape = value[0].shape if isinstance(value, tuple) else value.shape
        suffix = " · invalid values will be masked" if depth and not np.isfinite(value).all() else ""
        filename = upload.name if hasattr(upload, "name") else Path(upload).name
        st.success(f"{filename} · {shape[1]} × {shape[0]} px{suffix}")
        return value
    except Exception as exc:
        st.error(f"{label}: {exc}")
        return None


def _show_benchmark():
    st.markdown('<div class="section-title">NYU Depth V2 validation benchmark · 100 samples</div>', unsafe_allow_html=True)
    st.caption("128 × 128 · 4× scale · metres. This is validation evaluation, not an independent test-set result. V3 checkpoint selection used the NYU validation split. Neither table contains live-upload metrics.")
    try:
        v2_data = json.loads(NYU_RESULTS_V2.read_text(encoding="utf-8"))
        comparison_data = json.loads(NYU_RESULTS_V2_V3.read_text(encoding="utf-8"))
        v2_metrics = v2_data["metrics_macro_average"]
        v3_metrics = comparison_data["metrics_macro_average"]["cnn_v3"]
        names = {"bicubic": "Bicubic", "joint_bilateral": "Joint bilateral",
                 "cnn_v2": "CNN v2 · zero-shot", "cnn_v3": "CNN v3 · fine-tuned"}
        rows = []
        for key, name in names.items():
            values = v3_metrics if key == "cnn_v3" else v2_metrics[key]
            rows.append({"Method": name, "RMSE (m)": values["rmse"], "MAE (m)": values["mae"],
                         "PSNR (dB)": values["psnr"], "SSIM": values["ssim"],
                         "Edge precision": values["edge_precision"], "Edge recall": values["edge_recall"],
                         "Edge F1": values["edge_f1"]})
        st.dataframe(rows, hide_index=True, width="stretch")
        st.caption("The bicubic, joint bilateral, and v2 rows are read from the original v2 benchmark artifact, which remains unchanged. V3 is read from the separate v2/v3 comparison artifact.")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        st.info(f"The saved NYU benchmark summary is unavailable: {exc}")


# ── Header ──────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.html(f"""<div class="hero">
    <div class="eyebrow">Computer vision · Problem P4 · Team T49</div>
    <h1>EdgeDepth AI</h1>
    <div style="font-size:1.25rem;font-weight:650">AI-Powered Depth Map Refinement</div>
    <p class="muted">Upsample sparse depth observations with RGB guidance, then refine scene structure and boundaries using a lightweight residual U-Net.</p>
</div>""")
badge_cols = st.columns(3)
badge_cols[0].markdown("**Model**  \nResidual U-Net CNN")
badge_cols[1].markdown(f"**Device**  \n{'CUDA · ' + torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU fallback'}")
badge_cols[2].markdown("**Scale**  \nChoose 2×, 4× or 8× in the workspace")

st.markdown('<div class="section-title">Input workspace</div>', unsafe_allow_html=True)
st.caption("Upload an RGB guide and a matching low-resolution depth map. Ground truth is optional and used only for measured evaluation.")
demo_actions = st.columns([1, 1, 3])
with demo_actions[0]:
    if st.button("Load Demo Sample", type="secondary", width="stretch"):
        st.session_state["demo_sample_active"] = True
with demo_actions[1]:
    if st.button("Use Custom Uploads", width="stretch"):
        st.session_state["demo_sample_active"] = False
demo_mode = st.session_state.get("demo_sample_active", False)
if demo_mode:
    st.success("Demo Sample mode is active · NYU Depth V2 sample nyu_0000 with ground truth.")

up_cols = st.columns(3)
with up_cols[0]:
    rgb_upload = st.file_uploader("RGB image · JPG / PNG", type=("jpg", "jpeg", "png"), key="rgb_upload")
with up_cols[1]:
    depth_upload = st.file_uploader("Low-resolution depth · .npy", type=("npy",), key="depth_upload")
with up_cols[2]:
    gt_upload = st.file_uploader("Optional ground truth · .npy", type=("npy",), key="gt_upload")

if demo_mode:
    demo_paths = (DEMO_RGB, DEMO_LOW_DEPTH, DEMO_GT)
    missing_demo = [path.name for path in demo_paths if not path.is_file()]
    if missing_demo:
        st.error("Demo sample files are missing: " + ", ".join(missing_demo) +
                 ". Ensure the three files in sample_data/demo/ are included in the deployment.")
        rgb_source = low_depth_source = gt_source = None
    else:
        rgb_source, low_depth_source, gt_source = DEMO_RGB, DEMO_LOW_DEPTH, DEMO_GT
else:
    rgb_source, low_depth_source, gt_source = rgb_upload, depth_upload, gt_upload

scale_col, model_col, action_col = st.columns([1, 2, 2])
with scale_col:
    scale = st.select_slider("Upsampling scale", options=SCALE_CHOICES, value=4,
                             format_func=lambda value: f"{value}×")
with model_col:
    selected_model_label = st.selectbox("Refinement model", options=tuple(CHECKPOINTS), index=0)
model_config = CHECKPOINTS[selected_model_label]
model_version = model_config["version"]
cnn_label = f"CNN {model_version}"
checkpoint_path = model_config["path"]
with action_col:
    st.write("")
    active_inputs_ready = rgb_source is not None and low_depth_source is not None
    run = st.button("Run Depth Refinement", type="primary", disabled=not active_inputs_ready,
                    width="stretch")

status_cols = st.columns(3)
with status_cols[0]:
    rgb_parsed = _show_upload_status("RGB image", rgb_source, _decode_rgb)
with status_cols[1]:
    depth_parsed = _show_upload_status("Depth map", low_depth_source, _decode_depth, depth=True)
with status_cols[2]:
    gt_parsed = _show_upload_status("Ground truth", gt_source, _decode_depth, depth=True) if gt_source else None

if run:
    try:
        if rgb_parsed is None or depth_parsed is None:
            raise ValueError("Resolve the invalid RGB or depth upload before running inference.")
        rgb_array, rgb_original_size = rgb_parsed
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"The selected {cnn_label} checkpoint is not included in this deployment.")

        with st.status("Running depth refinement", expanded=True) as progress:
            st.write(f"Loading the cached {cnn_label} checkpoint…")
            model, checkpoint_info = _load_cached_model(str(checkpoint_path), str(device))
            depth_range = float(checkpoint_info.get("config", {}).get("depth_range", 10.0))
            if not np.isfinite(depth_range) or depth_range <= 0:
                raise ValueError("The checkpoint contains an invalid depth normalization range.")

            st.write("Preprocessing inputs and computing the bicubic baseline…")
            rgb_used, bicubic, low_resized, rgb_tensor, depth_tensor, mask_tensor = _prepare_inputs(
                rgb_array, depth_parsed, scale, depth_range,
                initial_method=model_config["initial_method"])
            st.write(f"Running {cnn_label} refinement…")
            refined_normalized = refine_batch(model, rgb_tensor.to(device), depth_tensor.to(device), mask_tensor.to(device))
            refined_metres = refined_normalized.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32) * depth_range
            refined_metres = np.clip(refined_metres, 0, depth_range)

            # Metrics are computed only when the user supplied a valid GT map.
            gt_metrics = None
            gt_resized = None
            if gt_source is not None:
                if gt_parsed is None:
                    raise ValueError("The ground-truth file is invalid; remove it or provide a valid depth map.")
                gt_aspect = gt_parsed.shape[1] / gt_parsed.shape[0]
                output_aspect = bicubic.shape[1] / bicubic.shape[0]
                if abs(gt_aspect / output_aspect - 1.0) > .10:
                    raise ValueError("Ground-truth depth must align with the RGB image aspect ratio (within 10%).")
                st.write("Evaluating both methods against valid ground-truth pixels…")
                gt, gt_valid = _resize_masked_depth(gt_parsed, (bicubic.shape[1], bicubic.shape[0]))
                gt_metrics = {
                    "Bicubic": evaluate(gt, bicubic, data_range=depth_range, valid_mask=gt_valid),
                    cnn_label: evaluate(gt, refined_metres, data_range=depth_range, valid_mask=gt_valid),
                }
            progress.update(label="Refinement complete", state="complete", expanded=False)

        st.markdown('<div class="section-title">Visual results</div>', unsafe_allow_html=True)
        st.caption(f"Output: {bicubic.shape[1]} × {bicubic.shape[0]} px · {scale}× · {device.type.upper()} · Depth range 0–{depth_range:g} m. Click a panel to inspect it at full size.")
        low_visual = _depth_visual(low_resized, depth_range).resize(
            (bicubic.shape[1], bicubic.shape[0]), Image.Resampling.NEAREST)
        panels = [("RGB guidance", rgb_used, "RGB", "Color guidance supplied to the CNN."),
                  ("Input low-resolution depth", low_visual, "Depth measurements",
                   "Low-resolution observations; invalid samples are shown dark.")]
        if gt_resized is not None:
            panels.append(("Ground-truth depth", _depth_visual(gt_resized, depth_range),
                           "Ground-truth depth in metres", "High-resolution reference depth."))
        panels.extend([
            ("Bicubic upsampled depth", _depth_visual(bicubic, depth_range), "Bicubic estimate",
             "Classical bicubic upsampling baseline."),
            (f"{cnn_label}-refined depth", _depth_visual(refined_metres, depth_range), "CNN refined depth",
             f"{cnn_label} refinement in metres."),
        ])
        for start in range(0, len(panels), 2):
            panel_cols = st.columns(2)
            for col, (title, image, alt, explanation) in zip(panel_cols, panels[start:start + 2]):
                with col, st.container(border=True):
                    st.markdown(f"**{title}**")
                    st.image(image, caption=explanation, alt=alt, width="stretch")

        if gt_metrics is not None:
            st.markdown('<div class="section-title">Measured comparison</div>', unsafe_allow_html=True)
            st.caption("Computed from this upload using the project's fixed-range metric implementations. Only finite positive ground-truth pixels are scored; units are metres unless marked otherwise.")
            metric_rows = []
            labels = (("RMSE", "m"), ("MAE", "m"), ("PSNR", "dB"), ("SSIM", ""),
                      ("Edge precision", ""), ("Edge recall", ""), ("Edge F1", ""))
            keys = ("rmse", "mae", "psnr", "ssim", "edge_precision", "edge_recall", "edge_f1")
            for (label, unit), key in zip(labels, keys):
                metric_rows.append({"Metric": label,
                                    "Bicubic": f"{gt_metrics['Bicubic'][key]:.4f}{(' ' + unit) if unit else ''}",
                                    cnn_label: f"{gt_metrics[cnn_label][key]:.4f}{(' ' + unit) if unit else ''}"})
            st.table(metric_rows)
        else:
            st.info("Quantitative evaluation is unavailable: upload aligned high-resolution ground truth to calculate metrics.")

        st.markdown('<div class="section-title">Downloads</div>', unsafe_allow_html=True)
        download_cols = st.columns(2)
        download_cols[0].download_button("Download refined depth (.npy)", data=_npy_bytes(refined_metres),
                                         file_name="edgedepth_refined_metres.npy", mime="application/octet-stream",
                                         width="stretch")
        vis_png = _visualization_png(rgb_used, low_visual, bicubic, refined_metres,
                                     depth_range, model_label=cnn_label)
        download_cols[1].download_button("Download comparison image (.png)", data=vis_png,
                                         file_name="edgedepth_comparison.png", mime="image/png", width="stretch")
        if rgb_original_size != (rgb_array.shape[1], rgb_array.shape[0]):
            st.caption(f"Large RGB upload was resized for processing from {rgb_original_size[0]} × {rgb_original_size[1]} to {rgb_array.shape[1]} × {rgb_array.shape[0]} px to limit memory use.")

    except FileNotFoundError as exc:
        st.error(str(exc))
    except (ValueError, OSError) as exc:
        st.error(str(exc))
    except Exception:
        st.error("Inference failed. Check that the inputs are valid and retry.")

st.markdown('<div class="section-title">Model information</div>', unsafe_allow_html=True)
with st.container(border=True):
    info_cols = st.columns(2)
    with info_cols[0]:
        st.markdown("**Five-channel input**")
        st.write("RGB guidance (3 channels) + normalized initial depth (1 channel) + observed-depth validity mask (1 channel). The network predicts a bounded residual added to the initial depth. V3 uses bicubic initial depth to match NYU fine-tuning; v2 retains the app's bilinear input path.")
        st.caption("Default checkpoint: `depth_refinement_v3_best.pt` · v2 remains available as a selectable synthetic-training fallback.")
    with info_cols[1]:
        st.markdown("**Training and validation scope**")
        st.write("CNN v3 was initialized from v2 and fine-tuned on NYU Depth V2 training data. Checkpoint selection used the NYU validation split. CNN v2 was trained on synthetic scenes and evaluated zero-shot on NYU.")
        st.caption("The 100-sample NYU comparison is validation evaluation, not an independent test-set evaluation. Live-upload metrics are separate.")
_show_benchmark()

st.divider()
footer_cols = st.columns([2, 1, 1, 2])
footer_cols[0].markdown("**EdgeDepth AI** · Edge-preserving depth refinement")
footer_cols[1].markdown("Problem **P4**")
footer_cols[2].markdown("Team **Meta · T49**")
footer_cols[3].markdown(f"[GitHub repository]({GITHUB_URL})")
