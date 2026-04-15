"""Headless SAM2 segmentation driven by a prompts JSON file.

The JSON schema mirrors what the GUI Segmentation tab exports via
"Export prompts", so prompts collected interactively on a laptop can be
applied to long videos on a server without touching the GUI.

Schema::

    {
      "video_path": "original_video.mp4",
      "prompts": [
        {"frame_idx": 0, "obj_id": 1, "x": 120, "y": 240, "label": 1},
        ...
      ]
    }

``label`` follows SAM2 convention: ``1`` = foreground click, ``0`` = negative
click. Multiple entries sharing the same ``(frame_idx, obj_id)`` are grouped
into a single SAM2 prompt call.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from singlebehaviorlab._paths import get_sam2_checkpoints_dir
from singlebehaviorlab.backend.video_processor import save_segmentation_data

__all__ = ["load_prompts_json", "save_prompts_json", "run_sam2_segmentation"]


_CHECKPOINT_TO_CONFIG = {
    "sam2.1_hiera_tiny.pt": "configs/sam2.1/sam2.1_hiera_t.yaml",
    "sam2.1_hiera_small.pt": "configs/sam2.1/sam2.1_hiera_s.yaml",
    "sam2.1_hiera_base_plus.pt": "configs/sam2.1/sam2.1_hiera_b+.yaml",
    "sam2.1_hiera_large.pt": "configs/sam2.1/sam2.1_hiera_l.yaml",
}


def load_prompts_json(path: str | os.PathLike[str]) -> dict[tuple[int, int], dict[str, list]]:
    """Parse a prompts JSON into SAM2's ``(frame_idx, obj_id) → {points, labels}`` dict."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    prompts = data.get("prompts", [])
    grouped: dict[tuple[int, int], dict[str, list]] = defaultdict(lambda: {"points": [], "labels": []})
    for entry in prompts:
        frame_idx = int(entry["frame_idx"])
        obj_id = int(entry["obj_id"])
        x = float(entry["x"])
        y = float(entry["y"])
        label = int(entry.get("label", 1))
        key = (frame_idx, obj_id)
        grouped[key]["points"].append([x, y])
        grouped[key]["labels"].append(label)
    return dict(grouped)


def save_prompts_json(
    video_path: str | os.PathLike[str],
    points: list[tuple[float, float, int, int, int]],
    output_path: str | os.PathLike[str],
) -> str:
    """Serialize widget-style ``(x, y, label, frame_idx, obj_id)`` tuples to the prompts JSON.

    Returns the written path.
    """
    entries = [
        {
            "frame_idx": int(frame_idx),
            "obj_id": int(obj_id),
            "x": float(x),
            "y": float(y),
            "label": int(label),
        }
        for (x, y, label, frame_idx, obj_id) in points
    ]
    payload = {"video_path": str(video_path), "prompts": entries}
    output_path_obj = Path(output_path).expanduser().resolve()
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path_obj, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return str(output_path_obj)


def _resolve_checkpoint(model_name: str) -> tuple[str, str]:
    if model_name not in _CHECKPOINT_TO_CONFIG:
        raise ValueError(
            f"Unknown SAM2 checkpoint '{model_name}'. "
            f"Expected one of: {sorted(_CHECKPOINT_TO_CONFIG.keys())}"
        )
    config_name = _CHECKPOINT_TO_CONFIG[model_name]
    checkpoints_root = get_sam2_checkpoints_dir()
    candidates = [
        checkpoints_root / model_name,
        checkpoints_root / "checkpoints" / model_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate), config_name
    raise FileNotFoundError(
        f"SAM2 checkpoint '{model_name}' was not found in {checkpoints_root}. "
        "Launch the GUI once to trigger the automatic download, or place the "
        "checkpoint file manually in that directory."
    )


def run_sam2_segmentation(
    video_path: str | os.PathLike[str],
    prompts_path: str | os.PathLike[str],
    output_mask_path: str | os.PathLike[str],
    *,
    model_name: str = "sam2.1_hiera_large.pt",
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
    log_fn: Optional[Callable[[str], None]] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> str:
    """Propagate SAM2 segmentation across a video using a saved prompts file.

    Returns the written mask HDF5 path.
    """
    import cv2
    import torch
    import sam2  # ensures hydra config registration
    from sam2.build_sam import build_sam2_video_predictor

    video_path = str(Path(video_path).expanduser().resolve())
    output_mask_path_obj = Path(output_mask_path).expanduser().resolve()
    output_mask_path_obj.parent.mkdir(parents=True, exist_ok=True)

    def _log(msg: str) -> None:
        if log_fn:
            log_fn(msg)

    _log(f"Loading prompts from {prompts_path}")
    user_points = load_prompts_json(prompts_path)
    if not user_points:
        raise RuntimeError("Prompts file did not contain any (frame_idx, obj_id) entries.")
    _log(f"Loaded {sum(len(v['points']) for v in user_points.values())} point(s) across "
         f"{len(user_points)} (frame, obj) prompt group(s).")

    checkpoint_path, config_name = _resolve_checkpoint(model_name)
    _log(f"Using SAM2 checkpoint: {checkpoint_path}")
    _log(f"Using SAM2 config: {config_name}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _log(f"Building SAM2 video predictor on {device}")
    predictor = build_sam2_video_predictor(config_name, checkpoint_path, device=device)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    if start_frame is None:
        start_frame = 0
    if end_frame is None or end_frame <= 0 or end_frame > total_frames:
        end_frame = total_frames
    _log(f"Video: {width}x{height}, {total_frames} frames @ {fps:.2f} fps")
    _log(f"Tracking range: frames {start_frame}..{end_frame}")

    _log("Initializing inference state (this loads video frames into SAM2)")
    inference_state = predictor.init_state(video_path=video_path)
    predictor.reset_state(inference_state)

    for (frame_idx, obj_id), data in user_points.items():
        if frame_idx < start_frame or frame_idx >= end_frame:
            continue
        points = np.array(data["points"], dtype=np.float32)
        labels = np.array(data["labels"], dtype=np.int32)
        predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            points=points,
            labels=labels,
            normalize_coords=True,
        )
    _log("Prompt injection complete; starting propagation")

    frame_objects: list[list[dict[str, Any]]] = [[] for _ in range(total_frames)]
    processed = 0
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
        inference_state,
        start_frame_idx=start_frame,
    ):
        if out_frame_idx >= end_frame:
            break
        entries: list[dict[str, Any]] = []
        for idx, obj_id in enumerate(out_obj_ids):
            mask_logit = out_mask_logits[idx]
            if mask_logit.ndim == 3:
                mask_logit = mask_logit[0]
            mask = (mask_logit > 0.0).cpu().numpy().astype(np.uint8)
            if mask.ndim == 3:
                mask = mask.squeeze()
            if not np.any(mask):
                continue
            rows = np.where(np.any(mask, axis=1))[0]
            cols = np.where(np.any(mask, axis=0))[0]
            if len(rows) == 0 or len(cols) == 0:
                continue
            y_min, y_max = int(rows.min()), int(rows.max())
            x_min, x_max = int(cols.min()), int(cols.max())
            cropped = mask[y_min:y_max + 1, x_min:x_max + 1].astype(bool)
            entries.append(
                {
                    "obj_id": int(obj_id),
                    "bbox": (x_min, y_min, x_max, y_max),
                    "mask": cropped,
                }
            )
        frame_objects[out_frame_idx] = entries
        processed += 1
        if progress_callback:
            progress_callback(processed, max(1, end_frame - start_frame))

    mask_data = {
        "video_path": video_path,
        "total_frames": total_frames,
        "original_total_frames": total_frames,
        "height": height,
        "width": width,
        "fps": fps,
        "start_offset": 0,
        "frame_objects": frame_objects,
        "objects_per_frame": [len(objs) for objs in frame_objects],
    }
    save_segmentation_data(str(output_mask_path_obj), mask_data)
    _log(f"Wrote mask file: {output_mask_path_obj}")
    return str(output_mask_path_obj)
