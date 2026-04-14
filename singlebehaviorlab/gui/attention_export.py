"""Export video with spatial attention heatmap overlay."""

import logging
import os
import math
import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QFileDialog, QMessageBox, QProgressDialog,
)
from PyQt6.QtCore import Qt

logger = logging.getLogger(__name__)


def _crop_frame_to_roi(frame_bgr, bbox_norm, out_w, out_h):
    """Crop a frame to normalized xyxy ROI and resize to output size."""
    if bbox_norm is None:
        return cv2.resize(frame_bgr, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

    h, w = frame_bgr.shape[:2]
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox_norm]
    except Exception:
        return cv2.resize(frame_bgr, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

    fx1 = int(round(x1 * w))
    fy1 = int(round(y1 * h))
    fx2 = int(round(x2 * w))
    fy2 = int(round(y2 * h))
    fx1 = max(0, min(fx1, w - 1))
    fy1 = max(0, min(fy1, h - 1))
    fx2 = max(fx1 + 1, min(fx2, w))
    fy2 = max(fy1 + 1, min(fy2, h))
    crop = frame_bgr[fy1:fy2, fx1:fx2]
    if crop.size == 0:
        crop = frame_bgr
    return cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_LINEAR)


def _get_final_label_for_frame(widget, frame_idx, clip_idx, clip_starts, clip_length, frame_interval, classes, ignore_label):
    """Match overlay export: use precise postprocessed label when available."""
    use_precise = bool(
        hasattr(widget, "frame_aggregation_check")
        and widget.frame_aggregation_check.isChecked()
        and getattr(widget, "aggregated_segments", None)
    )

    if use_precise and getattr(widget, "_use_ovr", False) and isinstance(
        getattr(widget, "_aggregated_frame_scores_norm", None), np.ndarray
    ):
        active_infos = widget._get_precise_active_for_frame(frame_idx)
        if active_infos:
            pred_idx, _ = active_infos[0]
            if pred_idx < 0:
                return ignore_label, (120, 120, 120)
            if 0 <= pred_idx < len(classes):
                return classes[pred_idx], (255, 255, 255)
            return f"class_{pred_idx}", (255, 255, 255)

    if use_precise:
        segments = getattr(widget, "aggregated_segments", None) or []
        for seg in segments:
            s0 = int(seg.get("start", 0))
            s1 = int(seg.get("end", s0))
            if s0 <= frame_idx <= s1:
                pred_idx = int(seg.get("class", -1))
                if pred_idx < 0:
                    return ignore_label, (120, 120, 120)
                if 0 <= pred_idx < len(classes):
                    return classes[pred_idx], (255, 255, 255)
                return f"class_{pred_idx}", (255, 255, 255)

    if clip_idx is not None and clip_starts:
        if hasattr(widget, "_effective_prediction_for_clip"):
            pred_idx = int(widget._effective_prediction_for_clip(clip_idx))
        else:
            pred_idx = None
            if hasattr(widget, "_effective_predictions") and hasattr(widget, "predictions") and widget.predictions:
                effective_preds = widget._effective_predictions()
                if clip_idx < len(effective_preds):
                    pred_idx = int(effective_preds[clip_idx])
        if pred_idx is not None:
            if pred_idx < 0:
                return ignore_label, (120, 120, 120)
            if 0 <= pred_idx < len(classes):
                return classes[pred_idx], (255, 255, 255)
            return f"class_{pred_idx}", (255, 255, 255)

    return None, None


def export_attention_heatmap_video(widget):
    """Generate and save a video with attention heatmaps overlaid on frames.

    Uses effective predictions (with ignore threshold + manual corrections)
    for labels. Applies temporal smoothing and Gaussian blur for smooth
    heatmaps that aren't jumpy or blocky.
    """
    if not hasattr(widget, 'results_cache') or not widget.results_cache:
        QMessageBox.warning(widget, "No results", "Run inference with 'Collect attention maps' enabled first.")
        return

    video_path = None
    if hasattr(widget, 'filter_video_combo') and widget.filter_video_combo.currentData():
        video_path = widget.filter_video_combo.currentData()
    elif widget.video_path:
        video_path = widget.video_path

    if not video_path or video_path not in widget.results_cache:
        QMessageBox.warning(widget, "No video", "Select a video with inference results.")
        return

    res = widget.results_cache[video_path]
    attn_maps = res.get("clip_attention_maps")
    if not attn_maps:
        QMessageBox.warning(
            widget, "No attention data",
            "No attention maps available. Re-run inference with 'Collect attention maps' checked."
        )
        return

    clip_starts = res.get("clip_starts", [])
    classes = getattr(widget, 'classes', [])

    ignore_label = getattr(widget, 'ignore_label_name', 'Ignored')

    default_name = os.path.splitext(os.path.basename(video_path))[0] + "_attention.mp4"
    default_dir = os.path.dirname(video_path)
    output_path, _ = QFileDialog.getSaveFileName(
        widget, "Save attention heatmap video",
        os.path.join(default_dir, default_name),
        "MP4 Video (*.mp4);;AVI Video (*.avi)"
    )
    if not output_path:
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        QMessageBox.critical(widget, "Error", f"Cannot open video: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    if orig_fps <= 0:
        orig_fps = 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    target_fps = getattr(widget, 'target_fps_spin', None)
    target_fps_val = target_fps.value() if target_fps else orig_fps
    frame_interval = max(1, int(round(orig_fps / max(1e-6, float(target_fps_val)))))
    clip_length = getattr(widget, 'clip_length_spin', None)
    clip_length_val = clip_length.value() if clip_length else 8
    use_classification_roi = bool(
        getattr(widget, "localization_bboxes", None)
        and hasattr(widget, "_get_classification_roi_bbox_for_clip_frame")
    )
    crop_resolution = int(getattr(widget, "infer_resolution", 0) or 0)
    if hasattr(widget, "resolution_spin"):
        try:
            crop_resolution = int(widget.resolution_spin.value())
        except Exception as e:
            logger.debug("Could not read resolution from resolution_spin: %s", e)
    crop_resolution = max(64, crop_resolution or min(frame_w, frame_h))
    output_w = crop_resolution if use_classification_roi else frame_w
    output_h = crop_resolution if use_classification_roi else frame_h

    # Build sorted list of (video_frame_idx, heatmap_2d) keyed attention frames
    attn_keyframes = {}
    grid_side = None
    for clip_idx, attn in enumerate(attn_maps):
        if attn is None or clip_idx >= len(clip_starts):
            continue
        attn_arr = np.array(attn, dtype=np.float32)  # [T_clip, num_heads, S]
        attn_avg = attn_arr.mean(axis=1)  # [T_clip, S]
        gs = int(math.isqrt(attn_avg.shape[-1]))
        if gs * gs != attn_avg.shape[-1]:
            continue
        grid_side = gs

        start_frame = clip_starts[clip_idx]
        for t in range(attn_avg.shape[0]):
            vid_frame = start_frame + t * frame_interval
            if vid_frame < total_frames:
                attn_keyframes[vid_frame] = attn_avg[t].reshape(gs, gs)

    if not attn_keyframes or grid_side is None:
        cap.release()
        QMessageBox.warning(widget, "No data", "Could not map attention data to video frames.")
        return

    # Upscale to export resolution so heatmap has maximum detail when overlaid
    interp_size = min(output_w, output_h)

    # Gaussian kernel to smooth grid boundaries (scale with grid cell size at interp)
    blur_ksize = max(3, min(interp_size // 24, grid_side * 4) | 1)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, orig_fps, (output_w, output_h))

    progress = QProgressDialog("Rendering attention heatmap video...", "Cancel", 0, total_frames, widget)
    progress.setWindowModality(Qt.WindowModality.WindowModal)
    progress.setMinimumDuration(0)

    # Temporal EMA smoothing for heatmap continuity
    ema_decay = 0.7
    smooth_heatmap = None

    # Scale label text to video size (readable on small and large frames)
    ref_size = 480
    small = min(output_w, output_h)
    text_scale = max(0.35, min(2.0, small / ref_size))
    font_scale = round(text_scale * 10) / 10.0
    thickness = max(1, int(round(text_scale * 2)))

    for frame_idx in range(total_frames):
        if progress.wasCanceled():
            break

        ret, frame = cap.read()
        if not ret:
            break

        clip_idx = None
        if clip_starts:
            clip_idx = _find_clip_for_frame(frame_idx, clip_starts, clip_length_val, frame_interval)
        if use_classification_roi and clip_idx is not None:
            roi_bbox = widget._get_classification_roi_bbox_for_clip_frame(clip_idx)
            base_frame = _crop_frame_to_roi(frame, roi_bbox, output_w, output_h)
        elif use_classification_roi:
            base_frame = cv2.resize(frame, (output_w, output_h), interpolation=cv2.INTER_LINEAR)
        else:
            base_frame = frame

        if frame_idx in attn_keyframes:
            raw = attn_keyframes[frame_idx]
            if smooth_heatmap is None:
                smooth_heatmap = raw.copy()
            else:
                smooth_heatmap = ema_decay * smooth_heatmap + (1 - ema_decay) * raw
        # else: keep previous smooth_heatmap (carry forward)

        if smooth_heatmap is None:
            writer.write(base_frame)
            if frame_idx % 200 == 0:
                progress.setValue(frame_idx)
            continue

        # Normalize to 0-1
        h_min, h_max = smooth_heatmap.min(), smooth_heatmap.max()
        if h_max > h_min:
            heatmap_01 = (smooth_heatmap - h_min) / (h_max - h_min)
        else:
            heatmap_01 = np.zeros_like(smooth_heatmap)

        # Upscale to intermediate resolution with bicubic
        heatmap_up = cv2.resize(heatmap_01.astype(np.float32), (interp_size, interp_size),
                                interpolation=cv2.INTER_CUBIC)

        # Gaussian blur for smooth appearance
        heatmap_up = cv2.GaussianBlur(heatmap_up, (blur_ksize, blur_ksize), 0)

        # Final resize to video dimensions
        heatmap_final = cv2.resize(heatmap_up, (output_w, output_h),
                                   interpolation=cv2.INTER_LINEAR)

        # Clip to valid range after interpolation
        heatmap_final = np.clip(heatmap_final, 0, 1)
        heatmap_u8 = (heatmap_final * 255).astype(np.uint8)

        heatmap_color = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)

        blended = cv2.addWeighted(base_frame, 0.6, heatmap_color, 0.4, 0)

        label, color = _get_final_label_for_frame(
            widget, frame_idx, clip_idx, clip_starts, clip_length_val, frame_interval, classes, ignore_label
        )
        if label:
            x_label = max(8, int(round(10 * text_scale)))
            y_label = max(20, int(round(30 * text_scale)))
            cv2.putText(blended, label, (x_label, y_label), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, color, thickness, cv2.LINE_AA)

        writer.write(blended)

        if frame_idx % 200 == 0:
            progress.setValue(frame_idx)

    progress.setValue(total_frames)
    cap.release()
    writer.release()

    if not progress.wasCanceled():
        QMessageBox.information(
            widget, "Export complete",
            f"Attention heatmap video saved to:\n{output_path}"
        )

        from .overlay_export import VideoPreviewDialog
        dialog = VideoPreviewDialog(output_path, parent=widget)
        dialog.exec()


def _find_clip_for_frame(frame_idx, clip_starts, clip_length, frame_interval):
    """Find which clip index covers the given video frame."""
    for i in range(len(clip_starts) - 1, -1, -1):
        clip_end = clip_starts[i] + clip_length * frame_interval
        if clip_starts[i] <= frame_idx < clip_end:
            return i
    return None
