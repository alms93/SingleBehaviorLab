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

Long videos are processed in fixed-size chunks (default 200 frames) so that
the memory-optimised SAM2 fork bundled with this project never has to hold the
entire video in RAM. Between chunks the last-frame masks of every tracked
object are re-injected as mask prompts on frame 0 of the next chunk, which
preserves continuity across chunk boundaries without the widget's growing
buffer.
"""

from __future__ import annotations

import json
import os
from collections import OrderedDict, defaultdict
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

_CHECKPOINT_URLS = {
    "sam2.1_hiera_tiny.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
    "sam2.1_hiera_small.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
    "sam2.1_hiera_base_plus.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
    "sam2.1_hiera_large.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
}

_CHUNK_SIZE = 200


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
    """Serialize widget-style ``(x, y, label, frame_idx, obj_id)`` tuples to the prompts JSON."""
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

    url = _CHECKPOINT_URLS.get(model_name)
    if url:
        dest = checkpoints_root / "checkpoints" / model_name
        dest.parent.mkdir(parents=True, exist_ok=True)
        import urllib.request
        try:
            from tqdm.auto import tqdm as _tqdm
        except Exception:
            _tqdm = None
        print(f"Downloading {model_name} from {url}")
        if _tqdm is None:
            urllib.request.urlretrieve(url, str(dest))
        else:
            with urllib.request.urlopen(url) as resp:
                total = int(resp.headers.get("Content-Length") or 0) or None
                with _tqdm(total=total, unit="B", unit_scale=True, desc=model_name) as bar:
                    with open(dest, "wb") as f:
                        while True:
                            chunk = resp.read(1024 * 256)
                            if not chunk:
                                break
                            f.write(chunk)
                            bar.update(len(chunk))
        if dest.exists() and dest.stat().st_size > 0:
            return str(dest), config_name

    raise FileNotFoundError(
        f"SAM2 checkpoint '{model_name}' could not be downloaded or found in {checkpoints_root}."
    )


def _load_chunk_images(
    video_path: str,
    start: int,
    end: int,
    image_size: int,
    target_dtype: Any,
):
    import decord
    import torch as _torch
    decord.bridge.set_bridge("torch")
    vr = decord.VideoReader(video_path, width=image_size, height=image_size)
    if end > len(vr):
        end = len(vr)
    indices = list(range(start, end))
    frames = vr.get_batch(indices)
    del vr
    images = frames.permute(0, 3, 1, 2).float() / 255.0
    del frames
    img_mean = _torch.tensor([0.485, 0.456, 0.406], dtype=_torch.float32)[:, None, None]
    img_std = _torch.tensor([0.229, 0.224, 0.225], dtype=_torch.float32)[:, None, None]
    images = images.to(dtype=target_dtype)
    img_mean = img_mean.to(dtype=target_dtype)
    img_std = img_std.to(dtype=target_dtype)
    images -= img_mean
    images /= img_std
    return images


def _build_inference_state(
    images_list: list,
    video_height: int,
    video_width: int,
    device,
) -> dict[str, Any]:
    import torch as _torch
    inference_state: dict[str, Any] = {
        "images": images_list,
        "num_frames": len(images_list),
        "offload_video_to_cpu": True,
        "offload_state_to_cpu": True,
        "video_height": video_height,
        "video_width": video_width,
        "device": device,
        "storage_device": _torch.device("cpu"),
        "point_inputs_per_obj": {},
        "mask_inputs_per_obj": {},
        "cached_features": {},
        "constants": {},
        "obj_id_to_idx": OrderedDict(),
        "obj_idx_to_id": OrderedDict(),
        "obj_ids": [],
        "output_dict_per_obj": {},
        "temp_output_dict_per_obj": {},
        "frames_tracked_per_obj": {},
    }
    return inference_state


def _crop_mask_to_entry(mask: np.ndarray, obj_id: int) -> Optional[dict[str, Any]]:
    if not np.any(mask):
        return None
    rows = np.where(np.any(mask, axis=1))[0]
    cols = np.where(np.any(mask, axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return None
    y_min, y_max = int(rows.min()), int(rows.max())
    x_min, x_max = int(cols.min()), int(cols.max())
    cropped = mask[y_min: y_max + 1, x_min: x_max + 1].astype(bool)
    return {
        "obj_id": int(obj_id),
        "bbox": (x_min, y_min, x_max, y_max),
        "mask": cropped,
    }


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
    """Propagate SAM2 segmentation across a video using a saved prompts file."""
    import cv2
    import decord
    import torch
    import sam2  # triggers hydra config registration
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
    _log(
        f"Loaded {sum(len(v['points']) for v in user_points.values())} point(s) across "
        f"{len(user_points)} (frame, obj) prompt group(s)."
    )

    checkpoint_path, config_name = _resolve_checkpoint(model_name)
    _log(f"SAM2 checkpoint: {checkpoint_path}")
    _log(f"SAM2 config: {config_name}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _log(f"Building SAM2 video predictor on {device}")
    predictor = build_sam2_video_predictor(config_name, checkpoint_path, device=device)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    vr_meta = decord.VideoReader(video_path)
    vh, vw, _ = vr_meta[0].shape
    del vr_meta

    if start_frame is None:
        start_frame = 0
    if end_frame is None or end_frame <= 0 or end_frame > total_frames:
        end_frame = total_frames
    _log(f"Video: {vw}x{vh}, {total_frames} frames @ {fps:.2f} fps")
    _log(f"Tracking range: frames {start_frame}..{end_frame} (chunk size {_CHUNK_SIZE})")

    target_dtype = getattr(predictor, "dtype", torch.float32)
    image_size = predictor.image_size
    frame_objects: list[list[dict[str, Any]]] = [[] for _ in range(total_frames)]

    carry_masks: dict[int, np.ndarray] = {}
    total_range = max(1, end_frame - start_frame)
    processed = 0

    chunk_idx = 0
    for chunk_start in range(start_frame, end_frame, _CHUNK_SIZE):
        chunk_end = min(chunk_start + _CHUNK_SIZE, end_frame)
        chunk_idx += 1
        _log(f"Chunk {chunk_idx}: frames {chunk_start}..{chunk_end}")

        images = _load_chunk_images(video_path, chunk_start, chunk_end, image_size, target_dtype)
        images_list = [images[i] for i in range(len(images))]
        inference_state = _build_inference_state(images_list, vh, vw, predictor.device)

        try:
            predictor._get_image_feature(inference_state, frame_idx=0, batch_size=1)
        except Exception:
            pass
        predictor.reset_state(inference_state)

        for (frame_idx, obj_id), data in user_points.items():
            if chunk_start <= frame_idx < chunk_end:
                local_idx = frame_idx - chunk_start
                pts = np.array(data["points"], dtype=np.float32)
                lbls = np.array(data["labels"], dtype=np.int32)
                predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=local_idx,
                    obj_id=obj_id,
                    points=pts,
                    labels=lbls,
                    normalize_coords=True,
                )

        for obj_id, mask in carry_masks.items():
            if mask.shape[0] != vh or mask.shape[1] != vw:
                mask_resized = cv2.resize(
                    mask.astype(np.float32),
                    (vw, vh),
                    interpolation=cv2.INTER_NEAREST,
                )
                mask = (mask_resized > 0.5).astype(np.uint8)
            try:
                predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=0,
                    obj_id=obj_id,
                    mask=mask.astype(bool),
                )
            except Exception as exc:
                _log(f"  [warn] Could not inject carry-over mask for obj {obj_id}: {exc}")

        if not user_points and not carry_masks:
            _log("  [warn] No prompts and no carry-over masks; skipping chunk.")
            del inference_state, images, images_list
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        last_full_masks: dict[int, np.ndarray] = {}
        last_local_idx = (chunk_end - chunk_start) - 1
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
            inference_state,
            start_frame_idx=0,
        ):
            global_idx = out_frame_idx + chunk_start
            if global_idx >= end_frame:
                break
            entries: list[dict[str, Any]] = []
            for i, o_id in enumerate(out_obj_ids):
                mask_logit = out_mask_logits[i]
                if mask_logit.ndim == 3:
                    mask_logit = mask_logit[0]
                mask = (mask_logit > 0.0).cpu().numpy().astype(np.uint8)
                if mask.ndim == 3:
                    mask = mask.squeeze()
                if out_frame_idx == last_local_idx:
                    last_full_masks[int(o_id)] = mask.copy()
                entry = _crop_mask_to_entry(mask, int(o_id))
                if entry is not None:
                    entries.append(entry)
            frame_objects[global_idx] = entries
            processed += 1
            if progress_callback:
                progress_callback(processed, total_range)

        carry_masks = last_full_masks
        del inference_state, images, images_list
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    mask_data = {
        "video_path": video_path,
        "total_frames": total_frames,
        "original_total_frames": total_frames,
        "height": vh,
        "width": vw,
        "fps": fps,
        "start_offset": 0,
        "frame_objects": frame_objects,
        "objects_per_frame": [len(objs) for objs in frame_objects],
    }
    save_segmentation_data(str(output_mask_path_obj), mask_data)
    _log(f"Wrote mask file: {output_mask_path_obj}")
    return str(output_mask_path_obj)
