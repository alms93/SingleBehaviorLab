"""Headless VideoPrism embedding extraction used by the CLI.

Mirrors the happy-path of the GUI registration workers but operates from
plain function arguments — no Qt signals, no widget state. Produces NPZ
matrix/metadata files that the Clustering tab can load directly via
"Load matrix/metadata".
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np
import pandas as pd
import torch

from singlebehaviorlab.backend.model import VideoPrismBackbone
from singlebehaviorlab.backend.video_processor import process_video_to_clips

__all__ = ["RegistrationParams", "run_registration"]


@dataclass
class RegistrationParams:
    """Knobs for ``run_registration``. Defaults match the GUI defaults."""

    box_size: int = 250
    target_size: int = 288
    background_mode: str = "white"
    normalization_method: str = "CLAHE"
    mask_feather_px: int = 0
    anchor_mode: str = "first"
    target_fps: int = 12
    clip_length_frames: int = 16
    step_frames: Optional[int] = None
    backbone_model: str = "videoprism_public_v1_base"
    flip_invariant: bool = False
    experiment_name: Optional[str] = None

    @property
    def effective_step_frames(self) -> int:
        if self.step_frames is None:
            return max(1, self.clip_length_frames // 2)
        return max(1, int(self.step_frames))

    def with_overrides(self, **kwargs: Any) -> "RegistrationParams":
        replacements = {k: v for k, v in kwargs.items() if v is not None}
        return RegistrationParams(**{**self.__dict__, **replacements})


def _load_clip_frames(clip_path: str) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        return None
    frames: list[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.array(frames) if frames else None


def _extract_embedding(
    backbone: VideoPrismBackbone,
    frames: np.ndarray,
    target_size: int,
    flip_invariant: bool = False,
) -> Optional[np.ndarray]:
    try:
        resized = np.array([cv2.resize(f, (target_size, target_size)) for f in frames])
        tensor = torch.from_numpy(np.transpose(resized, (0, 3, 1, 2))).float() / 255.0
        tensor = tensor.unsqueeze(0)
        with torch.no_grad():
            tokens = backbone(tensor)
            embedding = tokens.mean(dim=1).squeeze(0).cpu().numpy()
            if flip_invariant:
                embs = [embedding]
                for dims in [[-1], [-2], [-1, -2]]:
                    t_flip = torch.flip(tensor, dims=dims)
                    embs.append(backbone(t_flip).mean(dim=1).squeeze(0).cpu().numpy())
                embedding = np.mean(embs, axis=0)
        return embedding.astype(np.float32)
    except Exception:
        return None


def run_registration(
    video_path: str | os.PathLike[str],
    mask_path: str | os.PathLike[str],
    output_matrix_path: str | os.PathLike[str],
    *,
    params: Optional[RegistrationParams] = None,
    clips_cache_dir: Optional[str | os.PathLike[str]] = None,
    log_fn: Optional[Callable[[str], None]] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> dict[str, str]:
    """Extract VideoPrism embeddings for a video + mask and save the matrix.

    Args:
        video_path: Source video.
        mask_path: HDF5 mask file produced by the segmentation step.
        output_matrix_path: Destination NPZ path for the feature matrix.
            The sibling metadata file is derived by replacing ``_matrix.npz``
            with ``_metadata.npz`` (or by appending ``_metadata.npz``).
        params: Optional processing parameters; defaults match the GUI.
        clips_cache_dir: Directory used for extracted clip files. Defaults
            to a subdirectory next to the matrix output.
        log_fn: Optional sink for human-readable status lines.
        progress_callback: ``(current, total)`` called after each processed clip.

    Returns:
        Dict with the written ``matrix`` and ``metadata`` paths.
    """
    params = params or RegistrationParams()

    video_path = str(Path(video_path).expanduser().resolve())
    mask_path = str(Path(mask_path).expanduser().resolve())
    output_matrix_path = Path(output_matrix_path).expanduser().resolve()
    output_dir = output_matrix_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    if clips_cache_dir is None:
        clips_cache_dir = output_dir / f"{output_matrix_path.stem}_clips"
    clips_cache_dir = Path(clips_cache_dir).expanduser().resolve()
    clips_cache_dir.mkdir(parents=True, exist_ok=True)

    def _log(msg: str) -> None:
        if log_fn:
            log_fn(msg)

    _log(f"Video: {video_path}")
    _log(f"Mask:  {mask_path}")
    _log(f"Clips cache: {clips_cache_dir}")
    _log(
        f"Clip length {params.clip_length_frames}, step {params.effective_step_frames}, "
        f"target fps {params.target_fps}"
    )

    clip_results = process_video_to_clips(
        video_path=video_path,
        mask_path=mask_path,
        output_dir=str(clips_cache_dir),
        box_size=params.box_size,
        target_size=params.target_size,
        background_mode=params.background_mode,
        normalization_method=params.normalization_method,
        mask_feather_px=params.mask_feather_px,
        anchor_mode=params.anchor_mode,
        target_fps=params.target_fps,
        clip_length_frames=params.clip_length_frames,
        step_frames=params.effective_step_frames,
    )
    if not clip_results:
        raise RuntimeError("No clips produced from the input video and mask.")

    clip_entries: list[tuple[str, Optional[int], Optional[int]]] = []
    for entry in clip_results:
        if isinstance(entry, tuple) and len(entry) >= 3:
            clip_entries.append((entry[0], entry[1], entry[2]))
        else:
            clip_entries.append((entry, None, None))
    _log(f"Produced {len(clip_entries)} clips.")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _log(f"Loading VideoPrism backbone: {params.backbone_model}")
    backbone = VideoPrismBackbone(model_name=params.backbone_model, log_fn=log_fn)
    backbone.eval()

    feature_list: list[list[float]] = []
    metadata_rows: list[dict[str, Any]] = []
    total = len(clip_entries)

    for i, (clip_path, start_frame, end_frame) in enumerate(clip_entries, start=1):
        if progress_callback:
            progress_callback(i, total)
        frames = _load_clip_frames(clip_path)
        if frames is None or len(frames) == 0:
            _log(f"Skipping {os.path.basename(clip_path)}: no frames")
            continue
        embedding = _extract_embedding(backbone, frames, params.target_size, params.flip_invariant)
        del frames
        if embedding is None:
            _log(f"Skipping {os.path.basename(clip_path)}: embedding failed")
            continue
        feature_list.append(embedding.tolist())
        del embedding

        if i % 50 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

        clip_name = os.path.basename(clip_path)
        base = os.path.splitext(clip_name)[0]
        obj_match = re.search(r"_obj(\d+)", base)
        clip_match = re.search(r"clip_(\d+)", base)
        metadata_rows.append(
            {
                "snippet": f"snippet{len(feature_list)}",
                "group": os.path.basename(os.path.dirname(clip_path)),
                "video_id": clip_name,
                "object_id": obj_match.group(1) if obj_match else "",
                "clip_index": int(clip_match.group(1)) if clip_match else i - 1,
                "start_frame": start_frame if start_frame is not None else "",
                "end_frame": end_frame if end_frame is not None else "",
                "clip_path": os.path.abspath(clip_path),
            }
        )

    if not feature_list:
        raise RuntimeError("No embeddings were extracted from the input clips.")

    feature_matrix = np.array(feature_list, dtype=np.float32)
    num_snippets, num_features = feature_matrix.shape
    _log(f"Extracted {num_snippets} embeddings (dim={num_features}).")

    snippet_ids = np.array([row["snippet"] for row in metadata_rows])
    feature_names = np.array([f"behaviorome_embedding_{i}" for i in range(num_features)])
    metadata_df = pd.DataFrame(metadata_rows)

    matrix_path_str = str(output_matrix_path)
    if matrix_path_str.endswith("_matrix.npz"):
        metadata_path_str = matrix_path_str.replace("_matrix.npz", "_metadata.npz")
    elif matrix_path_str.endswith(".npz"):
        metadata_path_str = matrix_path_str[:-4] + "_metadata.npz"
    else:
        metadata_path_str = matrix_path_str + "_metadata.npz"

    np.savez_compressed(
        matrix_path_str,
        matrix=feature_matrix.T,
        feature_names=feature_names,
        snippet_ids=snippet_ids,
    )
    _log(f"Wrote matrix: {matrix_path_str}")

    np.savez_compressed(
        metadata_path_str,
        metadata=metadata_df.values,
        columns=np.array(metadata_df.columns),
    )
    _log(f"Wrote metadata: {metadata_path_str}")

    return {"matrix": matrix_path_str, "metadata": metadata_path_str}
