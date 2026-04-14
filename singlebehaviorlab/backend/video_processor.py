"""Video processing with mask-based background removal and centering."""
import logging
import os
import cv2
import h5py
import numpy as np
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def load_segmentation_data(file_path: str) -> dict:
    if not file_path.lower().endswith((".h5", ".hdf5")):
        raise ValueError(f"Unsupported segmentation format: {file_path}")

    with h5py.File(file_path, "r") as f:
        frame_objects_group = f["frame_objects"]
        frame_count = int(f.attrs.get("total_frames", len(frame_objects_group)))
        frame_objects = []
        for frame_idx in range(frame_count):
            frame_key = f"frame_{frame_idx:06d}"
            frame_objs = []
            if frame_key in frame_objects_group:
                frame_group = frame_objects_group[frame_key]
                for obj_key in sorted(frame_group.keys()):
                    obj_group = frame_group[obj_key]
                    bbox = tuple(int(v) for v in obj_group.attrs["bbox"])
                    obj_id = int(obj_group.attrs["obj_id"])
                    mask = obj_group["mask"][()].astype(bool)
                    frame_objs.append({
                        "bbox": bbox,
                        "mask": mask,
                        "obj_id": obj_id,
                    })
            frame_objects.append(frame_objs)

        return {
            "video_path": f.attrs.get("video_path", file_path),
            "total_frames": frame_count,
            "height": int(f.attrs["height"]),
            "width": int(f.attrs["width"]),
            "fps": float(f.attrs.get("fps", 30.0)),
            "frame_objects": frame_objects,
            "objects_per_frame": [int(v) for v in f["objects_per_frame"][()]],
            "tracker": {},
            "format": "hdf5",
            "start_offset": int(f.attrs.get("start_offset", 0)),
            "original_total_frames": int(f.attrs.get("original_total_frames", frame_count)),
        }


def save_segmentation_data(file_path: str, mask_data: dict) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    frame_objects = mask_data.get("frame_objects", [])
    with h5py.File(file_path, "w") as f:
        f.attrs["video_path"] = str(mask_data.get("video_path", ""))
        f.attrs["total_frames"] = int(mask_data.get("total_frames", len(frame_objects)))
        f.attrs["height"] = int(mask_data.get("height", 0))
        f.attrs["width"] = int(mask_data.get("width", 0))
        f.attrs["fps"] = float(mask_data.get("fps", 30.0))
        f.attrs["start_offset"] = int(mask_data.get("start_offset", 0))
        f.attrs["original_total_frames"] = int(
            mask_data.get("original_total_frames", len(frame_objects))
        )

        objects_per_frame = np.asarray(
            mask_data.get("objects_per_frame", [len(objs) for objs in frame_objects]),
            dtype=np.int32,
        )
        f.create_dataset("objects_per_frame", data=objects_per_frame, compression="gzip")

        frames_group = f.create_group("frame_objects")
        for frame_idx, objs in enumerate(frame_objects):
            frame_group = frames_group.create_group(f"frame_{frame_idx:06d}")
            for obj_idx, obj in enumerate(objs):
                obj_group = frame_group.create_group(f"obj_{obj_idx:03d}")
                obj_group.attrs["obj_id"] = int(obj.get("obj_id", 0))
                obj_group.attrs["bbox"] = np.asarray(obj.get("bbox", (0, 0, 0, 0)), dtype=np.int32)
                mask = np.asarray(obj.get("mask", np.zeros((1, 1), dtype=bool)), dtype=np.uint8)
                obj_group.create_dataset(
                    "mask",
                    data=mask,
                    compression="gzip",
                    compression_opts=4,
                    shuffle=True,
                )


def convert_old_format_to_objects(data: np.ndarray) -> list:
    """Convert old format mask array to frame_objects list."""
    frames, channels, height, width = data.shape
    frame_objects = []
    for frame_idx in range(frames):
        frame_mask = data[frame_idx, 0, :, :]
        if np.any(frame_mask):
            rows, cols = np.where(frame_mask)
            if len(rows) > 0 and len(cols) > 0:
                y_min, y_max = np.min(rows), np.max(rows)
                x_min, x_max = np.min(cols), np.max(cols)
                obj_mask = frame_mask[y_min:y_max+1, x_min:x_max+1]
                obj = {
                    'bbox': (x_min, y_min, x_max, y_max),
                    'mask': obj_mask.astype(bool),
                    'obj_id': 0
                }
                frame_objects.append([obj])
            else:
                frame_objects.append([])
        else:
            frame_objects.append([])
    return frame_objects


def read_video_frames(video_path: str, expected_frames: Optional[int] = None) -> list:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    
    while True:
        if expected_frames is not None and len(frames) >= expected_frames:
            break
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    return frames, fps


def process_frame_with_mask(
    frame_bgr: np.ndarray,
    frame_objects: list,
    frame_idx: int,
    box_size: int = 250,
    target_size: int = 288,
    background_mode: str = 'white',
    normalization_method: str = 'CLAHE',
    mask_feather_px: int = 0,
    anchor_cx: Optional[float] = None,
    anchor_cy: Optional[float] = None,
    anchor_mode: str = 'frame',
    obj_id: Optional[int] = None
) -> np.ndarray:
    """
    Process a single frame: crop around centroid and remove background.
    
    Args:
        obj_id: If provided, only process this object ID. Otherwise, use first object.
        normalization_method: 'CLAHE', 'Histogram Equalization', 'Mean-Variance', or 'None'
    
    Returns processed frame (target_size, target_size, 3) in [0,1] float32.
    """
    h, w = frame_bgr.shape[:2]

    if frame_idx >= len(frame_objects) or not frame_objects[frame_idx]:
        return np.zeros((target_size, target_size, 3), dtype=np.float32)

    obj = None
    if obj_id is not None:
        for o in frame_objects[frame_idx]:
            o_id = o.get('obj_id')
            if str(o_id) == str(obj_id):
                obj = o
                break
        if obj is None:
            return np.zeros((target_size, target_size, 3), dtype=np.float32)
    else:
        obj = frame_objects[frame_idx][0]
    mask = obj['mask']
    x_min, y_min, x_max, y_max = obj['bbox']

    full_mask = np.zeros((h, w), dtype=np.uint8)
    bbox_h = max(0, y_max - y_min + 1)
    bbox_w = max(0, x_max - x_min + 1)
    
    if bbox_h > 0 and bbox_w > 0:
        mh, mw = mask.shape
        h_use = min(bbox_h, mh, h - max(0, y_min))
        w_use = min(bbox_w, mw, w - max(0, x_min))
        if h_use > 0 and w_use > 0:
            try:
                full_mask[y_min:y_min + h_use, x_min:x_min + w_use] = (
                    mask[:h_use, :w_use] > 0
                ).astype(np.uint8)
            except Exception:
                pass
    
    m = cv2.moments(full_mask, binaryImage=True)
    if m['m00'] > 0:
        cx = m['m10'] / m['m00']
        cy = m['m01'] / m['m00']
    else:
        cx = (x_min + x_max) / 2.0
        cy = (y_min + y_max) / 2.0

    if anchor_mode == 'first' and anchor_cx is not None and anchor_cy is not None:
        cx_use, cy_use = anchor_cx, anchor_cy
    else:
        cx_use, cy_use = cx, cy
    
    half = box_size // 2
    desired_x1 = int(round(cx_use)) - half
    desired_y1 = int(round(cy_use)) - half
    desired_x2 = desired_x1 + box_size
    desired_y2 = desired_y1 + box_size
    
    src_x1 = max(0, desired_x1)
    src_y1 = max(0, desired_y1)
    src_x2 = min(w, desired_x2)
    src_y2 = min(h, desired_y2)
    
    if src_x2 <= src_x1 or src_y2 <= src_y1:
        crop_rgb = np.zeros((box_size, box_size, 3), dtype=np.uint8)
        crop_mask = np.zeros((box_size, box_size), dtype=np.uint8)
    else:
        crop_bgr = frame_bgr[src_y1:src_y2, src_x1:src_x2]
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        crop_mask = full_mask[src_y1:src_y2, src_x1:src_x2]
        
        pad_left = src_x1 - desired_x1
        pad_top = src_y1 - desired_y1
        pad_right = desired_x2 - src_x2
        pad_bottom = desired_y2 - src_y2
        
        if pad_left or pad_top or pad_right or pad_bottom:
            padded_rgb = np.zeros((box_size, box_size, 3), dtype=np.uint8)
            padded_mask = np.zeros((box_size, box_size), dtype=np.uint8)
            h_c, w_c = crop_rgb.shape[:2]
            padded_rgb[pad_top:pad_top + h_c, pad_left:pad_left + w_c] = crop_rgb
            padded_mask[pad_top:pad_top + h_c, pad_left:pad_left + w_c] = crop_mask
            crop_rgb, crop_mask = padded_rgb, padded_mask
    
    if normalization_method == 'CLAHE':
        gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_eq = clahe.apply(gray)
        crop_rgb = cv2.cvtColor(gray_eq, cv2.COLOR_GRAY2RGB)
        
    elif normalization_method == 'Histogram Equalization':
        gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
        gray_eq = cv2.equalizeHist(gray)
        crop_rgb = cv2.cvtColor(gray_eq, cv2.COLOR_GRAY2RGB)
        
    elif normalization_method == 'Mean-Variance':
        gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        mean, std = cv2.meanStdDev(gray)
        if std[0][0] > 0:
            gray = (gray - mean[0][0]) / std[0][0] * 50 + 127
        else:
            gray = gray - mean[0][0] + 127
        gray = np.clip(gray, 0, 255).astype(np.uint8)
        crop_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    if crop_rgb.shape[0] != box_size or crop_rgb.shape[1] != box_size:
        pad_img = np.zeros((box_size, box_size, 3), dtype=np.uint8)
        h_c, w_c = crop_rgb.shape[:2]
        pad_img[:h_c, :w_c] = crop_rgb
        crop_rgb = pad_img
        
        pad_mask = np.zeros((box_size, box_size), dtype=np.uint8)
        if 'crop_mask' in locals():
            pad_mask[:h_c, :w_c] = crop_mask
        crop_mask = pad_mask
    
    if background_mode in ('black', 'gray', 'blur', 'white'):
        m_float = crop_mask.astype(np.float32)
        if mask_feather_px and mask_feather_px > 0:
            k = max(1, int(mask_feather_px) | 1)
            m_float = cv2.GaussianBlur(m_float, (k, k), 0)
        m_float = np.clip(m_float, 0.0, 1.0)
        m3 = np.repeat(m_float[:, :, None], 3, axis=2)
        
        if background_mode == 'black':
            bg = np.zeros_like(crop_rgb)
        elif background_mode == 'gray':
            bg = np.full_like(crop_rgb, 128)
        elif background_mode == 'white':
            bg = np.full_like(crop_rgb, 255)
        else:  # blur
            bg = cv2.GaussianBlur(crop_rgb, (11, 11), 0)
        
        crop_rgb = (m3 * crop_rgb + (1.0 - m3) * bg).astype(np.uint8)
    
    # Resize to target size
    frame_rgb = cv2.resize(crop_rgb, (target_size, target_size), interpolation=cv2.INTER_AREA)
    
    # Normalize to [0, 1]
    return (frame_rgb.astype(np.float32) / 255.0)


def process_video(
    video_path: str,
    mask_path: str,
    output_path: str,
    box_size: int = 250,
    target_size: int = 288,
    background_mode: str = 'white',
    mask_feather_px: int = 0,
    anchor_mode: str = 'first',
    progress_callback: Optional[callable] = None,
    obj_id: Optional[int] = None
) -> bool:
    """
    Process entire video: load mask, process each frame, save output.
    
    Args:
        video_path: Path to input video
        mask_path: Path to HDF5 mask file
        output_path: Path to save processed video (or base path if obj_id is None and multiple objects exist)
        box_size: Size of crop box in pixels
        target_size: Final output size
        background_mode: 'white', 'black', 'gray', 'blur', 'none'
        mask_feather_px: Feathering radius for mask edges
        anchor_mode: 'frame' (per-frame centroid) or 'first' (fixed at first frame)
        progress_callback: Optional function(frame_num, total_frames, obj_id) for progress updates
        obj_id: If provided, only process this object ID. If None and multiple objects exist, creates separate videos.
        
    Returns:
        True if successful, False otherwise. If multiple objects, returns list of output paths.
    """
    try:
        mask_data = load_segmentation_data(mask_path)
        frame_objects = mask_data['frame_objects']
        num_frames = len(frame_objects)
        start_offset = mask_data.get('start_offset', 0)

        all_obj_ids = set()
        for frame_objs in frame_objects:
            for obj in frame_objs:
                obj_id = obj.get('obj_id', 0)
                all_obj_ids.add(obj_id)
        all_obj_ids = sorted(list(all_obj_ids))

        if obj_id is None and len(all_obj_ids) > 1:
            output_paths = []
            base_path = os.path.splitext(output_path)[0]
            ext = os.path.splitext(output_path)[1]
            
            for oid in all_obj_ids:
                obj_output_path = f"{base_path}_obj{oid}{ext}"
                success = process_video(
                    video_path, mask_path, obj_output_path,
                    box_size, target_size, background_mode,
                    mask_feather_px, anchor_mode, progress_callback, obj_id=oid
                )
                if success:
                    output_paths.append(obj_output_path)
            
            return output_paths if output_paths else False

        if obj_id is None:
            obj_id = all_obj_ids[0] if all_obj_ids else None

        raw_frames, fps = read_video_frames(video_path, expected_frames=num_frames)
        if len(raw_frames) != num_frames:
            logger.warning("Video has %d frames, mask has %d", len(raw_frames), num_frames)
            num_frames = min(len(raw_frames), num_frames)
        
        anchor_cx = None
        anchor_cy = None
        if anchor_mode == 'first':
            for frame_idx in range(min(10, num_frames)):  # Check first 10 frames
                if frame_idx < len(frame_objects) and frame_objects[frame_idx]:
                    obj0 = None
                    for o in frame_objects[frame_idx]:
                        if o.get('obj_id') == obj_id:
                            obj0 = o
                            break
                    if obj0 is None:
                        continue
                    
                    mask0 = obj0['mask']
                    x0_min, y0_min, x0_max, y0_max = obj0['bbox']
                    frame_bgr0 = raw_frames[frame_idx]
                    h0, w0 = frame_bgr0.shape[:2]
                    full_mask0 = np.zeros((h0, w0), dtype=np.uint8)
                    bbox_h0 = max(0, y0_max - y0_min + 1)
                    bbox_w0 = max(0, x0_max - x0_min + 1)
                    if bbox_h0 > 0 and bbox_w0 > 0:
                        mh0, mw0 = mask0.shape
                        h0_use = min(bbox_h0, mh0, h0 - max(0, y0_min))
                        w0_use = min(bbox_w0, mw0, w0 - max(0, x0_min))
                        if h0_use > 0 and w0_use > 0:
                            try:
                                full_mask0[y0_min:y0_min + h0_use, x0_min:x0_min + w0_use] = (
                                    mask0[:h0_use, :w0_use] > 0
                                ).astype(np.uint8)
                            except Exception:
                                pass
                    m0 = cv2.moments(full_mask0, binaryImage=True)
                    if m0['m00'] > 0:
                        anchor_cx = m0['m10'] / m0['m00']
                        anchor_cy = m0['m01'] / m0['m00']
                        break
                    else:
                        anchor_cx = (x0_min + x0_max) / 2.0
                        anchor_cy = (y0_min + y0_max) / 2.0
                        break
        
        # Use mp4v codec for efficient encoding
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (target_size, target_size), isColor=True)

        if not out.isOpened():
            raise ValueError(f"Failed to open video writer for {output_path}")

        try:
            for frame_idx in range(num_frames):
                if progress_callback:
                    if callable(progress_callback):
                        try:
                            progress_callback(frame_idx + 1, num_frames, obj_id)
                        except TypeError:
                            progress_callback(frame_idx + 1, num_frames)

                frame_bgr = raw_frames[frame_idx]

                frame_processed = process_frame_with_mask(
                    frame_bgr,
                    frame_objects,
                    frame_idx,
                    box_size=box_size,
                    target_size=target_size,
                    background_mode=background_mode,
                    mask_feather_px=mask_feather_px,
                    anchor_cx=anchor_cx,
                    anchor_cy=anchor_cy,
                    anchor_mode=anchor_mode,
                    obj_id=obj_id
                )

                frame_bgr_out = (frame_processed * 255.0).astype(np.uint8)
                frame_bgr_out = cv2.cvtColor(frame_bgr_out, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr_out)

            return True
        finally:
            out.release()

    except Exception as e:
        logger.error("Error processing video: %s", e, exc_info=True)
        return False


def process_video_to_clips(
    video_path: str,
    mask_path: str,
    output_dir: str,
    box_size: int = 250,
    target_size: int = 288,
    background_mode: str = 'white',
    normalization_method: str = 'CLAHE',
    mask_feather_px: int = 0,
    anchor_mode: str = 'first',
    target_fps: int = 16,
    clip_length_frames: int = 16,
    step_frames: int = 16,
    progress_callback: Optional[callable] = None,
    obj_id: Optional[int] = None
) -> list:
    """
    Process video into clips with registration: load mask, process frames, save as clips.
    
    Args:
        video_path: Path to input video
        mask_path: Path to HDF5 mask file
        output_dir: Directory to save clips
        box_size: Size of crop box in pixels
        target_size: Final output size
        background_mode: 'white', 'black', 'gray', 'blur', 'none'
        normalization_method: 'CLAHE', 'Histogram Equalization', 'Mean-Variance', or 'None'
        mask_feather_px: Feathering radius for mask edges
        anchor_mode: 'frame' (per-frame centroid) or 'first' (fixed at first frame)
        target_fps: Target FPS for clips (frames will be subsampled)
        clip_length_frames: Number of frames per clip
        step_frames: Step size between clips
        progress_callback: Optional function(clip_num, total_clips, obj_id) for progress updates
        obj_id: If provided, only process this object ID. If None and multiple objects exist, creates separate clips.
        
    Returns:
        List of tuples: (clip_path, start_frame, end_frame). If multiple objects, returns list of lists.
    """
    try:
        # Load mask data
        mask_data = load_segmentation_data(mask_path)
        frame_objects = mask_data['frame_objects']
        num_frames = len(frame_objects)
        # Get start_offset if masks were trimmed to a range
        start_offset = mask_data.get('start_offset', 0)
        
        all_obj_ids = set()
        for frame_objs in frame_objects:
            for obj in frame_objs:
                oid = obj.get('obj_id', 0)
                try:
                    all_obj_ids.add(int(oid))
                except (ValueError, TypeError):
                    all_obj_ids.add(oid)
        try:
            all_obj_ids = sorted(list(all_obj_ids), key=lambda x: int(x))
        except:
            all_obj_ids = sorted(list(all_obj_ids), key=str)

        if obj_id is None and len(all_obj_ids) > 1:
            logger.info("Processing %d objects: %s", len(all_obj_ids), all_obj_ids)
            all_clip_paths = []
            for oid in all_obj_ids:
                logger.info("Starting processing for object %s...", oid)
                try:
                    clip_paths = process_video_to_clips(
                        video_path, mask_path, output_dir,
                        box_size, target_size, background_mode,
                        normalization_method,
                        mask_feather_px, anchor_mode,
                        target_fps, clip_length_frames, step_frames,
                        progress_callback, obj_id=oid
                    )
                    all_clip_paths.extend(clip_paths)
                    logger.info("Completed object %s: %d clips generated.", oid, len(clip_paths))
                except Exception as e:
                    logger.error("Error processing object %s: %s", oid, e, exc_info=True)
            return all_clip_paths
        
        if obj_id is None:
            obj_id = all_obj_ids[0] if all_obj_ids else None

        os.makedirs(output_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        try:
            if start_offset > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_offset)

            orig_fps = cap.get(cv2.CAP_PROP_FPS)
            if orig_fps <= 0:
                orig_fps = 30.0

            frame_interval = max(1, int(round(orig_fps / target_fps)))
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_offset)

            clip_paths = []  # (clip_path, start_frame, end_frame)
            frame_idx = start_offset  # Original video frame index
            clip_idx = 0
            frames_buffer = []
            skip_remaining = 0
            frame_indices_in_buffer = []  # Track original frame indices for each frame in buffer
            clip_start_mask_frame_idx = None  # Track first frame (original index)

            # Persistent anchor for current clip (used when anchor_mode='first')
            clip_anchor_cx = None
            clip_anchor_cy = None

            end_frame_limit = start_offset + num_frames  # process only the trimmed range

            while True:
                if frame_idx >= end_frame_limit:
                    break

                ret, frame_bgr = cap.read()
                if not ret:
                    break

                if frame_idx % frame_interval == 0:
                    if skip_remaining > 0:
                        skip_remaining -= 1
                        frame_idx += 1
                        continue
                    mask_frame_idx = frame_idx - start_offset
                    if mask_frame_idx < 0 or mask_frame_idx >= len(frame_objects):
                        frame_idx += 1
                        continue

                    if len(frames_buffer) == 0:
                        clip_start_mask_frame_idx = mask_frame_idx
                        clip_anchor_cx = None
                        clip_anchor_cy = None

                        if anchor_mode == 'first' and mask_frame_idx < len(frame_objects) and frame_objects[mask_frame_idx]:
                            obj0 = None
                            for o in frame_objects[mask_frame_idx]:
                                o_id = o.get('obj_id')
                                if str(o_id) == str(obj_id):
                                    obj0 = o
                                    break
                            if obj0:
                                mask0 = obj0['mask']
                                x0_min, y0_min, x0_max, y0_max = obj0['bbox']
                                h0, w0 = frame_bgr.shape[:2]
                                full_mask0 = np.zeros((h0, w0), dtype=np.uint8)
                                bbox_h0 = max(0, y0_max - y0_min + 1)
                                bbox_w0 = max(0, x0_max - x0_min + 1)
                                if bbox_h0 > 0 and bbox_w0 > 0:
                                    mh0, mw0 = mask0.shape
                                    h0_use = min(bbox_h0, mh0, h0 - max(0, y0_min))
                                    w0_use = min(bbox_w0, mw0, w0 - max(0, x0_min))
                                    if h0_use > 0 and w0_use > 0:
                                        try:
                                            full_mask0[y0_min:y0_min + h0_use, x0_min:x0_min + w0_use] = (
                                                mask0[:h0_use, :w0_use] > 0
                                            ).astype(np.uint8)
                                        except Exception:
                                            pass
                                m0 = cv2.moments(full_mask0, binaryImage=True)
                                if m0['m00'] > 0:
                                    clip_anchor_cx = m0['m10'] / m0['m00']
                                    clip_anchor_cy = m0['m01'] / m0['m00']
                                else:
                                    clip_anchor_cx = (x0_min + x0_max) / 2.0
                                    clip_anchor_cy = (y0_min + y0_max) / 2.0

                    frame_processed = process_frame_with_mask(
                        frame_bgr,
                        frame_objects,
                        mask_frame_idx,
                        box_size=box_size,
                        target_size=target_size,
                        background_mode=background_mode,
                        normalization_method=normalization_method,
                        mask_feather_px=mask_feather_px,
                        anchor_cx=clip_anchor_cx,
                        anchor_cy=clip_anchor_cy,
                        anchor_mode=anchor_mode,
                        obj_id=obj_id
                    )

                    frame_bgr_out = (frame_processed * 255.0).astype(np.uint8)
                    frame_bgr_out = cv2.cvtColor(frame_bgr_out, cv2.COLOR_RGB2BGR)

                    frames_buffer.append(frame_bgr_out)
                    frame_indices_in_buffer.append(frame_idx)

                    if len(frames_buffer) == clip_length_frames:
                        video_basename = os.path.splitext(os.path.basename(video_path))[0]
                        clip_name = f"{video_basename}_clip_{clip_idx:06d}"
                        if obj_id is not None:
                            clip_name += f"_obj{obj_id}"
                        clip_path = os.path.join(output_dir, f"{clip_name}.mp4")

                        clip_start_mask_frame_idx = frame_indices_in_buffer[0]
                        clip_end_mask_frame_idx = frame_indices_in_buffer[-1]

                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        out = cv2.VideoWriter(clip_path, fourcc, target_fps, (target_size, target_size), isColor=True)

                        if out.isOpened():
                            for f in frames_buffer:
                                out.write(f)
                            out.release()
                            clip_paths.append((clip_path, clip_start_mask_frame_idx, clip_end_mask_frame_idx))

                            if progress_callback:
                                try:
                                    progress_callback(clip_idx + 1, None, obj_id)
                                except TypeError:
                                    try:
                                        progress_callback(clip_idx + 1, None)
                                    except TypeError:
                                        pass

                        clip_idx += 1

                        if step_frames < clip_length_frames:
                            frames_buffer = frames_buffer[clip_length_frames - step_frames:]
                            frame_indices_in_buffer = frame_indices_in_buffer[clip_length_frames - step_frames:]
                            clip_start_mask_frame_idx = frame_indices_in_buffer[0] if frame_indices_in_buffer else None
                        else:
                            frames_buffer = []
                            frame_indices_in_buffer = []
                            clip_start_mask_frame_idx = None
                            skip_remaining = max(0, step_frames - clip_length_frames)

                frame_idx += 1

            return clip_paths
        finally:
            cap.release()

    except Exception as e:
        logger.error("Error processing video to clips: %s", e, exc_info=True)
        return []

