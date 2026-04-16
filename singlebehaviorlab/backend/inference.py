"""Headless model inference for long videos.

Mirrors the happy-path of ``InferenceWorker.run`` without Qt signals and
without the optional GUI features (localization, OvR calibration, attention
collection). Output JSON is loadable via the Inference tab's "Load results"
action, where the interactive smoothing, Viterbi decoding, and segment-merge
options can still be applied.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np
import torch

from singlebehaviorlab.backend.data_store import AnnotationManager
from singlebehaviorlab.backend.model import BehaviorClassifier, VideoPrismBackbone

__all__ = ["run_inference_on_video"]


def _center_merge_weights(length: int) -> np.ndarray:
    if length <= 1:
        return np.ones((max(1, length),), dtype=np.float32)
    w = np.hanning(length).astype(np.float32)
    if not np.any(w > 0):
        return np.ones((length,), dtype=np.float32)
    return np.clip(0.1 + 0.9 * w, 1e-3, None).astype(np.float32)


def _read_metadata(model_path: str) -> tuple[dict[str, Any], dict[str, Any]]:
    ckpt: dict[str, Any] = {}
    try:
        payload = torch.load(model_path, map_location="cpu", weights_only=False)
        if isinstance(payload, dict):
            ckpt = payload
    except Exception:
        ckpt = {}
    meta_path = model_path + ".meta.json"
    metadata: dict[str, Any] = {}
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                metadata = loaded
        except Exception:
            metadata = {}
    return ckpt, metadata


def _resolve_classes(metadata: dict[str, Any], annotation_file: Optional[str]) -> list[str]:
    classes = metadata.get("classes") or metadata.get("class_names")
    if isinstance(classes, (list, tuple)) and classes:
        return [str(c) for c in classes]
    if annotation_file and os.path.exists(annotation_file):
        return AnnotationManager(annotation_file).get_classes()
    raise RuntimeError(
        "Model metadata did not include a class list and no fallback annotation file was supplied."
    )


def _iter_clips(
    video_path: str,
    target_fps: float,
    clip_length: int,
    step_frames: int,
    resolution: int,
) -> tuple[list[np.ndarray], list[int], int, float, int]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    if orig_fps <= 0:
        orig_fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(round(orig_fps / max(1e-6, float(target_fps)))))

    clips: list[np.ndarray] = []
    clip_starts: list[int] = []
    buffer: list[np.ndarray] = []
    buffer_starts: list[int] = []

    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval != 0:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (resolution, resolution))
        buffer.append(frame.astype(np.float32) / 255.0)
        buffer_starts.append(frame_idx)
        if len(buffer) >= clip_length:
            clip_arr = np.stack(buffer[:clip_length], axis=0)
            clips.append(clip_arr)
            clip_starts.append(buffer_starts[0])
            drop = step_frames if step_frames > 0 else clip_length
            buffer = buffer[drop:]
            buffer_starts = buffer_starts[drop:]

    cap.release()
    return clips, clip_starts, total_frames, orig_fps, frame_interval


def run_inference_on_video(
    model_path: str | os.PathLike[str],
    video_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    *,
    experiment_dir: Optional[str | os.PathLike[str]] = None,
    target_fps: Optional[float] = None,
    clip_length: Optional[int] = None,
    step_frames: Optional[int] = None,
    batch_size: Optional[int] = None,
    save_arrays: bool = False,
    log_fn: Optional[Callable[[str], None]] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> str:
    """Run a trained classifier on a video and write a GUI-loadable JSON."""
    model_path = str(Path(model_path).expanduser().resolve())
    video_path = str(Path(video_path).expanduser().resolve())
    output_path_obj = Path(output_path).expanduser().resolve()
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    def _log(msg: str) -> None:
        if log_fn:
            log_fn(msg)

    _log(f"Loading model metadata: {model_path}")
    ckpt, metadata = _read_metadata(model_path)

    resolved_target_fps = float(target_fps or metadata.get("target_fps")
                                 or metadata.get("training_config", {}).get("target_fps", 12))
    resolved_clip_length = int(clip_length or metadata.get("clip_length")
                               or metadata.get("training_config", {}).get("clip_length", 16))
    resolved_step_frames = int(step_frames if step_frames is not None else resolved_clip_length)
    resolution = int(metadata.get("resolution")
                     or metadata.get("training_config", {}).get("resolution", 288))

    annotation_file: Optional[str] = None
    if experiment_dir:
        annotation_file = str(Path(experiment_dir) / "data" / "annotations" / "annotations.json")
    classes = _resolve_classes(metadata, annotation_file)
    _log(f"Classes ({len(classes)}): {classes}")
    _log(f"Clip length={resolved_clip_length}, target fps={resolved_target_fps}, step={resolved_step_frames}")
    _log(f"Input resolution: {resolution}")

    head_cfg = metadata.get("head", {}) if isinstance(metadata.get("head"), dict) else {}
    head_kwargs = {"num_heads": int(head_cfg.get("num_heads", 4))}
    dropout = float(head_cfg.get("dropout", 0.1))
    num_stages = int(head_cfg.get("num_stages", 3))
    proj_dim = int(head_cfg.get("proj_dim", 256))
    frame_head_temporal_layers = int(head_cfg.get("frame_head_temporal_layers", 1))
    temporal_pool_frames = int(head_cfg.get("temporal_pool_frames", 1))
    use_temporal_decoder = bool(head_cfg.get("use_temporal_decoder",
                                             metadata.get("training_config", {}).get("use_temporal_decoder", True)))
    multi_scale = bool(ckpt.get("multi_scale", head_cfg.get("multi_scale", False)))
    use_localization = bool(ckpt.get("use_localization", False))

    _log("Loading VideoPrism backbone...")
    backbone_model = metadata.get("backbone_model") or "videoprism_public_v1_base"
    backbone = VideoPrismBackbone(model_name=backbone_model, resolution=resolution, log_fn=log_fn)

    _log("Building classifier...")
    model = BehaviorClassifier(
        backbone,
        num_classes=len(classes),
        class_names=classes,
        dropout=dropout,
        freeze_backbone=True,
        head_kwargs=head_kwargs,
        use_localization=use_localization,
        use_frame_head=True,
        use_temporal_decoder=use_temporal_decoder,
        frame_head_temporal_layers=frame_head_temporal_layers,
        temporal_pool_frames=temporal_pool_frames,
        proj_dim=proj_dim,
        num_stages=num_stages,
        multi_scale=multi_scale,
    )
    model.load_head(model_path)
    if hasattr(model, "frame_head") and model.frame_head is not None:
        model.frame_head.use_ovr = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    _log(f"Running inference on device: {device}")

    _log(f"Reading clips from {video_path}...")
    clips, clip_starts, total_frames, orig_fps, frame_interval = _iter_clips(
        video_path,
        target_fps=resolved_target_fps,
        clip_length=resolved_clip_length,
        step_frames=resolved_step_frames,
        resolution=resolution,
    )
    if not clips:
        raise RuntimeError("No clips could be extracted from the input video.")
    _log(f"Extracted {len(clips)} clips (total frames: {total_frames}, orig fps: {orig_fps:.2f}).")

    has_frame_head = getattr(model, "use_frame_head", True)
    if batch_size is None:
        batch_size = 8 if device.type == "cuda" else 2

    predictions: list[int] = []
    confidences: list[float] = []
    clip_probabilities: list[list[float]] = []
    clip_frame_probabilities: list[list[list[float]]] = []

    total_batches = (len(clips) + batch_size - 1) // batch_size
    for batch_idx in range(total_batches):
        batch_clips = clips[batch_idx * batch_size: (batch_idx + 1) * batch_size]
        tensors = [torch.from_numpy(c).permute(0, 3, 1, 2) for c in batch_clips]
        batch = torch.stack(tensors).to(device)
        with torch.no_grad():
            logits = model(batch, return_frame_logits=has_frame_head)
        if isinstance(logits, tuple):
            logits = logits[0]
        frame_output = getattr(model, "_frame_output", None)
        if frame_output is not None:
            f_logits = frame_output[0]
            batch_frame_probs = torch.sigmoid(f_logits).detach().cpu().numpy()
            for b_i in range(batch_frame_probs.shape[0]):
                clip_frame_probabilities.append(batch_frame_probs[b_i].tolist())
        else:
            clip_frame_probabilities.extend([] for _ in batch_clips)

        probs = torch.sigmoid(logits)
        preds = torch.argmax(probs, dim=1)
        confs = torch.max(probs, dim=1)[0]
        predictions.extend(int(p) for p in preds.cpu().numpy().tolist())
        confidences.extend(float(c) for c in confs.cpu().numpy().tolist())
        clip_probabilities.extend(probs.detach().cpu().numpy().tolist())

        if progress_callback:
            progress_callback(len(predictions), len(clips))

        del batch, logits, probs, preds, confs

    num_classes = len(classes)
    aggregated_frame_probs: Optional[np.ndarray] = None
    if clip_frame_probabilities and any(len(fp) > 0 for fp in clip_frame_probabilities):
        _log("Aggregating per-frame probabilities...")
        agg_probs = np.zeros((total_frames, num_classes), dtype=np.float32)
        agg_counts = np.zeros((total_frames, 1), dtype=np.float32)
        for i, probs_list in enumerate(clip_frame_probabilities):
            if not probs_list or i >= len(clip_starts):
                continue
            probs_arr = np.clip(np.asarray(probs_list, dtype=np.float32), 0.0, None)
            if probs_arr.ndim != 2 or probs_arr.shape[1] != num_classes:
                continue
            t_len = int(probs_arr.shape[0])
            merge_w = _center_merge_weights(t_len)
            frame_conf = np.clip(np.max(probs_arr, axis=1), 0.0, 1.0)
            conf_w = np.clip(0.1 + 0.9 * frame_conf, 0.1, 1.0).astype(np.float32)
            start_f = clip_starts[i]
            for t in range(t_len):
                f_start = start_f + t * frame_interval
                f_end = min(f_start + frame_interval, total_frames)
                if f_start >= total_frames or f_end <= f_start:
                    continue
                w = float(merge_w[t] * conf_w[t])
                agg_probs[f_start:f_end] += probs_arr[t][np.newaxis, :] * w
                agg_counts[f_start:f_end] += w
        agg_probs = agg_probs / np.maximum(agg_counts, 1.0)
        covered = agg_counts.squeeze(-1) > 0
        row_sums = agg_probs[covered].sum(axis=1, keepdims=True)
        safe_sums = np.maximum(row_sums, 1e-8)
        agg_probs[covered] = agg_probs[covered] / safe_sums
        aggregated_frame_probs = agg_probs

    res_entry: dict[str, Any] = {
        "predictions": predictions,
        "confidences": confidences,
        "clip_probabilities": clip_probabilities,
        "clip_starts": clip_starts,
        "total_frames": total_frames,
        "orig_fps": orig_fps,
        "frame_interval": frame_interval,
        "aggregated_frame_probs": aggregated_frame_probs.tolist() if aggregated_frame_probs is not None else None,
    }
    if clip_frame_probabilities:
        res_entry["clip_frame_probabilities"] = clip_frame_probabilities

    payload: dict[str, Any] = {
        "classes": classes,
        "parameters": {
            "target_fps": resolved_target_fps,
            "clip_length": resolved_clip_length,
            "step_frames": resolved_step_frames,
        },
        "results": {video_path: res_entry},
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    arrays_side: dict[str, np.ndarray] = {}
    if save_arrays and aggregated_frame_probs is not None:
        arrays_side["video_00000__aggregated_frame_probs"] = aggregated_frame_probs

    if arrays_side:
        sidecar_path = str(output_path_obj) + ".arrays.npz"
        np.savez_compressed(sidecar_path, **arrays_side)
        payload["external_array_store"] = {
            "format": "npz",
            "file": os.path.basename(sidecar_path),
        }
        _log(f"Wrote arrays sidecar: {sidecar_path}")

    with open(output_path_obj, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=lambda v: v.tolist() if hasattr(v, "tolist") else str(v))
    _log(f"Wrote inference results: {output_path_obj}")
    return str(output_path_obj)
