import logging
import cv2
import os
from typing import Optional

logger = logging.getLogger(__name__)


def extract_clips(
    video_path: str,
    output_dir: str,
    target_fps: int = 16,
    clip_length_frames: int = 16,
    step_frames: int = 16,
    progress_callback: Optional[callable] = None,
    stop_callback: Optional[callable] = None,
) -> tuple[int, str]:
    """Subsample video to target_fps and cut into non-overlapping clips (use step_frames < clip_length_frames for overlap)."""
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    try:
        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        if orig_fps <= 0:
            orig_fps = 30.0

        frame_interval = max(1, int(round(orig_fps / target_fps)))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        total_frames_after_subsampling = total_frames // frame_interval
        if step_frames >= clip_length_frames:
            total_clips = total_frames_after_subsampling // clip_length_frames
        else:
            total_clips = max(0, (total_frames_after_subsampling - clip_length_frames) // step_frames + 1) if total_frames_after_subsampling >= clip_length_frames else 0

        frame_idx = 0
        clip_idx = 0
        frames_buffer = []
        skip_remaining = 0

        while True:
            if stop_callback and stop_callback():
                break
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                if skip_remaining > 0:
                    skip_remaining -= 1
                else:
                    frames_buffer.append(frame)

                if len(frames_buffer) == clip_length_frames:
                    clip_path = os.path.join(output_dir, f"clip_{clip_idx:06d}.mp4")
                    save_clip(frames_buffer, clip_path, target_fps)
                    clip_idx += 1

                    if progress_callback:
                        progress_callback(clip_idx, total_clips)

                    if step_frames < clip_length_frames:
                        frames_buffer = frames_buffer[clip_length_frames - step_frames:]
                    else:
                        frames_buffer = []
                        skip_remaining = max(0, step_frames - clip_length_frames)

            frame_idx += 1

        return clip_idx, output_dir
    finally:
        cap.release()


def save_clip(frames: list, output_path: str, fps: float):
    """Save a list of frames as a standard MP4 clip (mp4v codec)."""
    if not frames:
        return
    
    h, w, c = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    for frame in frames:
        out.write(frame)
    
    out.release()


def load_clip_frames(clip_path: str, target_size: Optional[tuple[int, int]] = None) -> list:
    """Load frames from a video clip.
    
    Args:
        clip_path: Path to video clip
        target_size: Optional (width, height) to resize frames
    
    Returns:
        List of frames as numpy arrays (BGR format)
    """
    cap = cv2.VideoCapture(clip_path)
    frames = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if target_size:
                frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)

            frames.append(frame)

        return frames
    finally:
        cap.release()


def get_video_info(video_path: str) -> dict:
    """Get video metadata."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {}

    try:
        info = {
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }

        return info
    finally:
        cap.release()

