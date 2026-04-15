import sys
import os
import gc
import json
import logging
import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog,
    QRadioButton, QSlider, QButtonGroup, QMessageBox, QProgressBar,
    QComboBox, QDoubleSpinBox, QSpinBox, QFormLayout, QCheckBox, QGroupBox,
    QSizePolicy, QListWidget, QScrollArea, QProgressDialog, QApplication
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QPointF, QEvent
from PyQt6.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QBrush
import shutil
from pathlib import Path
from contextlib import nullcontext
from importlib import metadata as importlib_metadata

# Motion tracking (Kalman filter, OC-SORT) in separate module
from .motion_tracking import MultiObjectMotionTracker


# Colors for different objects (R, G, B)
OBJ_COLORS = [
    (0, 255, 0),    # 1: Green
    (255, 0, 0),    # 2: Red
    (0, 0, 255),    # 3: Blue
    (255, 255, 0),  # 4: Yellow
    (0, 255, 255),  # 5: Cyan
    (255, 0, 255),  # 6: Magenta
    (255, 128, 0),  # 7: Orange
    (128, 0, 255),  # 8: Purple
    (128, 128, 0),  # 9: Olive
    (0, 128, 128),  # 10: Teal
]

def get_obj_color(obj_id):
    idx = (obj_id - 1) % len(OBJ_COLORS)
    return OBJ_COLORS[idx]


class CheckpointDownloadWorker(QThread):
    """Worker thread for downloading SAM2 checkpoints."""
    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, str)
    
    # Model URLs (SAM 2.1)
    MODEL_URLS = {
        "sam2.1_hiera_tiny.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
        "sam2.1_hiera_small.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
        "sam2.1_hiera_base_plus.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
        "sam2.1_hiera_large.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
        # SAM 2.0 (older versions)
        "sam2_hiera_tiny.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt",
        "sam2_hiera_small.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt",
        "sam2_hiera_base_plus.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt",
        "sam2_hiera_large.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt",
    }
    
    def __init__(self, checkpoint_name, checkpoint_path, checkpoint_url):
        super().__init__()
        self.checkpoint_name = checkpoint_name
        self.checkpoint_path = checkpoint_path
        self.checkpoint_url = checkpoint_url
    
    def run(self):
        try:
            # Check if already downloaded
            if os.path.exists(self.checkpoint_path):
                file_size = os.path.getsize(self.checkpoint_path) / (1024**2)  # MB
                if file_size > 10:  # Reasonable size check (should be >100MB)
                    self.finished.emit(True, f"Checkpoint already exists ({file_size:.1f} MB)")
                    return
            
            self.progress.emit(f"Downloading {self.checkpoint_name}...")
            self.progress.emit(f"URL: {self.checkpoint_url}")
            
            # Try wget first, then curl
            import urllib.request
            
            def show_progress(block_num, block_size, total_size):
                if total_size > 0:
                    percent = min(100, (block_num * block_size * 100) / total_size)
                    self.progress.emit(f"Downloading {self.checkpoint_name}: {percent:.1f}%")
            
            # Download with progress
            urllib.request.urlretrieve(
                self.checkpoint_url,
                self.checkpoint_path,
                reporthook=show_progress
            )
            
            # Verify download
            if os.path.exists(self.checkpoint_path):
                file_size = os.path.getsize(self.checkpoint_path) / (1024**2)
                if file_size < 10:  # Suspiciously small
                    os.remove(self.checkpoint_path)
                    raise Exception(f"Downloaded file seems too small ({file_size:.1f} MB). Download may have failed.")
                self.finished.emit(True, f"Downloaded successfully ({file_size:.1f} MB)")
            else:
                raise Exception("Download completed but file not found")
                
        except Exception as e:
            if os.path.exists(self.checkpoint_path):
                try:
                    os.remove(self.checkpoint_path)
                except:
                    pass
            self.finished.emit(False, f"Download failed: {str(e)}")


class TrackingWorker(QThread):
    """Worker thread for running tracking."""
    progress_signal = pyqtSignal(int)
    frame_result_signal = pyqtSignal(int, dict)  # frame_idx, {obj_id: mask}
    finished_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)
    log_message = pyqtSignal(str)
    
    def __init__(self, predictor, video_path, user_points, start_frame, end_frame,
                 mask_threshold=0.0, offload_video=True, offload_state=True,
                 enable_memory_management=True, reseed_between_chunks=False,
                 initial_masks=None, enable_motion_tracking=False,
                 motion_score_threshold=0.3, motion_consecutive_low=3,
                 motion_area_threshold=0.5, enable_ocsort=False,
                 ocsort_inertia=0.2, use_cuda_bf16_autocast=True):
        super().__init__()
        self.predictor = predictor
        self.video_path = video_path
        self.user_points = user_points  # (frame_idx, obj_id) -> {'points': [], 'labels': []}
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.mask_threshold = mask_threshold
        self.offload_video = offload_video
        self.offload_state = offload_state
        self.enable_memory_management = enable_memory_management
        self.reseed_between_chunks = reseed_between_chunks
        self.initial_masks = initial_masks or {}  # {(frame_idx, obj_id): mask_array} for resume conditioning
        self.chunk_size = 200
        self.should_stop = False
        self.use_cuda_bf16_autocast = bool(use_cuda_bf16_autocast)
        
        # Motion-aware tracking
        self.enable_motion_tracking = enable_motion_tracking
        self.motion_score_threshold = motion_score_threshold
        self.motion_tracker = None
        if enable_motion_tracking:
            self.motion_tracker = MultiObjectMotionTracker(
                motion_score_threshold=motion_score_threshold,
                use_kalman=True,
                consecutive_low_threshold=motion_consecutive_low,
                area_change_threshold=motion_area_threshold,
                use_ocsort=enable_ocsort,
                ocsort_inertia=ocsort_inertia
            )

    def _use_cuda_bf16(self):
        """Use bf16 autocast only when SAM2 runs on CUDA."""
        dev = getattr(self.predictor, "device", None)
        dev_type = getattr(dev, "type", str(dev))
        return bool(
            self.use_cuda_bf16_autocast
            and torch.cuda.is_available()
            and dev_type == "cuda"
        )

    def _sam2_autocast(self):
        if self._use_cuda_bf16():
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    def _sam2_call(self, fn, *args, **kwargs):
        with self._sam2_autocast():
            return fn(*args, **kwargs)
    
    def stop(self):
        """Request tracking stop."""
        self.should_stop = True
    
    def run(self):
        """Run tracking with incremental processing."""
        try:
            all_video_segments = {}  # global_frame_idx -> {obj_id: mask}
            MAX_MASKS_IN_MEMORY = 500  # Keep only recent masks, older ones are already emitted via signal
            last_masks_for_reseed = None  # Store last frame masks of previous chunk for optional reseed
            
            try:
                import decord
            except ImportError:
                raise ImportError("decord not found. Please install it: pip install eva-decord")
            
            from collections import OrderedDict
            
            def load_frames(start, end):
                decord.bridge.set_bridge("torch")
                image_size = self.predictor.image_size
                vr = decord.VideoReader(self.video_path, width=image_size, height=image_size)
                target_dtype = getattr(self.predictor, "dtype", torch.float32)
                if self._use_cuda_bf16() and not self.offload_video:
                    target_dtype = torch.bfloat16
                
                if end > len(vr):
                    end = len(vr)
                indices = list(range(start, end))
                frames = vr.get_batch(indices)
                del vr  # Free VideoReader memory after loading frames
                images = frames.permute(0, 3, 1, 2).float() / 255.0
                del frames  # Free original frame tensor after processing
                
                img_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None]
                img_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None]
                
                if not self.offload_video:
                    images = images.to(self.predictor.device, dtype=target_dtype)
                    img_mean = img_mean.to(self.predictor.device, dtype=target_dtype)
                    img_std = img_std.to(self.predictor.device, dtype=target_dtype)
                else:
                    # Keep on CPU but ensure dtype matches model expectations
                    images = images.to(dtype=target_dtype)
                    img_mean = img_mean.to(dtype=target_dtype)
                    img_std = img_std.to(dtype=target_dtype)
                
                images -= img_mean
                images /= img_std
                return images

            def is_cuda_alloc_error(exc):
                msg = str(exc).lower()
                return (
                    "cuda out of memory" in msg
                    or "cublas_status_alloc_failed" in msg
                    or ("cuda error" in msg and "alloc" in msg)
                )

            def run_with_cuda_retry(op_name, fn):
                try:
                    return fn()
                except RuntimeError as e:
                    if not is_cuda_alloc_error(e):
                        raise
                    self.log_message.emit(
                        f"[GPU] Memory error during {op_name}. Clearing cache and retrying once..."
                    )
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    try:
                        return fn()
                    except RuntimeError as e2:
                        if is_cuda_alloc_error(e2):
                            raise RuntimeError(
                                "GPU memory allocation failed while running SAM2. "
                                "Try one or more: enable Offload Video to CPU, enable Offload State to CPU, "
                                "use a smaller SAM2 model, or track a shorter range."
                            ) from e2
                        raise
            
            # Initialize with first chunk
            current_end = min(self.start_frame + self.chunk_size, self.end_frame)
            self.log_message.emit(f"Initializing with chunk: {self.start_frame} to {current_end}")
            
            images = load_frames(self.start_frame, current_end)
            
            # Get original dimensions
            vr_meta = decord.VideoReader(self.video_path)
            vh, vw, _ = vr_meta[0].shape
            del vr_meta  # Free memory immediately after getting dimensions
            
            images_list = [images[i] for i in range(len(images))]
            
            inference_state = {}
            inference_state["images"] = images_list
            inference_state["num_frames"] = len(images_list)
            inference_state["offload_video_to_cpu"] = self.offload_video
            inference_state["offload_state_to_cpu"] = self.offload_state
            inference_state["video_height"] = vh
            inference_state["video_width"] = vw
            inference_state["device"] = self.predictor.device
            inference_state["storage_device"] = torch.device("cpu") if self.offload_state else self.predictor.device
            inference_state["point_inputs_per_obj"] = {}
            inference_state["mask_inputs_per_obj"] = {}
            inference_state["cached_features"] = {}
            inference_state["constants"] = {}
            inference_state["obj_id_to_idx"] = OrderedDict()
            inference_state["obj_idx_to_id"] = OrderedDict()
            inference_state["obj_ids"] = []
            inference_state["output_dict_per_obj"] = {}
            inference_state["temp_output_dict_per_obj"] = {}
            inference_state["frames_tracked_per_obj"] = {}
            
            # Warm up
            try:
                self._sam2_call(self.predictor._get_image_feature, inference_state, frame_idx=0, batch_size=1)
            except:
                pass
            
            self.predictor.reset_state(inference_state)
            
            # Loop through chunks with sliding window memory management
            # Track the global offset (how many frames we've trimmed from the start)
            # Only used when memory management is enabled
            global_offset = 0  # Tracks how many frames we've dropped from the front
            MAX_FRAMES_IN_MEMORY = 800  # Keep ~800 frames in memory
            
            processed_up_to = self.start_frame
            
            while processed_up_to < self.end_frame:
                if self.should_stop:
                    break
                
                # inference_state["images"] grows with each processed frame;
                # frames correspond to self.start_frame + index. When memory
                # management is disabled, global_offset stays at 0.
                buffer_start = self.start_frame + (global_offset if self.enable_memory_management else 0)
                
                chunk_start = processed_up_to
                chunk_end = buffer_start + inference_state["num_frames"]  # End of current buffer
                
                self.log_message.emit(f"Processing range: {chunk_start} to {chunk_end} (buffer: {buffer_start} to {chunk_end})")
                
                for (frame_idx, obj_id), data in self.user_points.items():
                    if chunk_start <= frame_idx < chunk_end:
                        # Local index relative to current buffer (after trimming)
                        local_idx = frame_idx - buffer_start
                        if local_idx < 0 or local_idx >= inference_state["num_frames"]:
                            # Frame was trimmed, skip (shouldn't happen if logic is correct)
                            continue
                        pts = np.array(data['points'], dtype=np.float32)
                        lbls = np.array(data['labels'], dtype=np.int32)
                        
                        run_with_cuda_retry(
                            "add_new_points_or_box",
                            lambda: self._sam2_call(self.predictor.add_new_points_or_box,
                                inference_state=inference_state,
                                frame_idx=local_idx,
                                obj_id=obj_id,
                                points=pts,
                                labels=lbls,
                                normalize_coords=True,
                            ),
                        )
                
                # Inject initial masks (e.g., from pause/resume refinement)
                for (frame_idx, obj_id), mask in self.initial_masks.items():
                    if chunk_start <= frame_idx < chunk_end:
                        local_idx = frame_idx - buffer_start
                        if local_idx < 0 or local_idx >= inference_state["num_frames"]:
                            continue
                        try:
                            # Resize mask to video dimensions if needed
                            vh = inference_state["video_height"]
                            vw = inference_state["video_width"]
                            if mask.shape[0] != vh or mask.shape[1] != vw:
                                import cv2
                                mask_resized = cv2.resize(mask.astype(np.float32), (vw, vh), interpolation=cv2.INTER_NEAREST)
                                mask = (mask_resized > 0.5).astype(np.uint8)
                            
                            run_with_cuda_retry(
                                "add_new_mask",
                                lambda: self._sam2_call(self.predictor.add_new_mask,
                                    inference_state=inference_state,
                                    frame_idx=local_idx,
                                    obj_id=obj_id,
                                    mask=mask.astype(bool),
                                ),
                            )
                            self.log_message.emit(f"Injected refined mask for obj {obj_id} at frame {frame_idx}")
                        except Exception as e:
                            self.log_message.emit(f"Warning: Could not inject mask: {e}")
                
                # Propagate from chunk_start
                # We need local index for propagation start (relative to current buffer)
                prop_start_local = chunk_start - buffer_start
                if prop_start_local < 0:
                    prop_start_local = 0  # Can't propagate from before buffer start
                
                # Memory trimming may drop the initial conditioning frame (the
                # user's first click). The bundled SAM2 fork modifies
                # propagate_in_video_preflight to allow propagation when only
                # tracking history (non_cond_frame_outputs) is present, so
                # explicit mask re-injection is not required here.
                
                with self._sam2_autocast():
                    for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(
                        inference_state,
                        start_frame_idx=prop_start_local
                    ):
                        if self.should_stop:
                            break

                        # Convert local buffer index to global frame index
                        global_idx = out_frame_idx + buffer_start

                        if global_idx not in all_video_segments:
                            all_video_segments[global_idx] = {}

                        frame_scores = {}  # Track scores for this frame per object
                        low_quality_objects = []  # Objects with sustained low scores this frame

                        for i, o_id in enumerate(out_obj_ids):
                            mask_logit = out_mask_logits[i]
                            if mask_logit.ndim == 3:
                                mask_logit = mask_logit[0]
                            mask = (mask_logit > self.mask_threshold).cpu().numpy().astype(np.uint8).squeeze()
                            all_video_segments[global_idx][o_id] = mask

                            # Motion-aware scoring
                            if self.motion_tracker is not None:
                                score, should_use = self.motion_tracker.update(
                                    o_id, mask, mask_logit, global_idx
                                )
                                frame_scores[o_id] = score

                                # Only filter SAM2 memory after sustained low quality.
                                # This avoids dropping useful memory on one-frame glitches.
                                if (not should_use) and self.motion_tracker.check_needs_correction(o_id):
                                    low_quality_objects.append(o_id)

                        # Log motion tracking info periodically
                        if self.motion_tracker is not None and global_idx % 50 == 0:
                            score_str = ", ".join([f"obj{k}:{v:.2f}" for k, v in frame_scores.items()])
                            self.log_message.emit(f"Frame {global_idx} scores: {score_str}")

                        # Memory filtering: remove low-quality frames from memory
                        if self.motion_tracker is not None and low_quality_objects:
                            for obj_id in low_quality_objects:
                                obj_idx = inference_state.get("obj_id_to_idx", {}).get(obj_id)
                                if obj_idx is not None:
                                    # Recency-weighted memory filtering:
                                    # keep recent memory frames and remove older low-score frames first.
                                    obj_output = inference_state.get("output_dict_per_obj", {}).get(obj_idx, {})
                                    non_cond = obj_output.get("non_cond_frame_outputs", {})
                                    if not non_cond:
                                        continue
                                    threshold = self.motion_tracker.get_effective_threshold(obj_id)
                                    recent_keep = 6
                                    max_remove = 2
                                    removed = 0
                                    old_keys = sorted(
                                        k for k in non_cond.keys() if k < (out_frame_idx - recent_keep)
                                    )
                                    for mem_local_idx in old_keys:
                                        if removed >= max_remove:
                                            break
                                        mem_global_idx = mem_local_idx + buffer_start
                                        mem_score = self.motion_tracker.get_frame_score(obj_id, mem_global_idx)
                                        if mem_score is not None and mem_score < threshold:
                                            del non_cond[mem_local_idx]
                                            removed += 1

                                    # Fallback if no older candidate was removable.
                                    if removed == 0 and out_frame_idx in non_cond:
                                        del non_cond[out_frame_idx]

                        # Appearance memory re-seed: when object recovers from long occlusion,
                        # inject golden mask so SAM2 remembers what the object looked like.
                        if self.motion_tracker is not None and self.motion_tracker.appearance_memory is not None:
                            for o_id in out_obj_ids:
                                amem = self.motion_tracker.appearance_memory
                                if amem.is_recovery_pending(o_id):
                                    golden_mask = amem.pop_reseed_mask(o_id)
                                    if golden_mask is not None:
                                        try:
                                            # Inject at current frame so SAM2 uses the golden
                                            # mask immediately (not delayed by one frame).
                                            reseed_local = out_frame_idx
                                            i_vh = inference_state["video_height"]
                                            i_vw = inference_state["video_width"]
                                            gm = golden_mask
                                            if gm.shape[0] != i_vh or gm.shape[1] != i_vw:
                                                import cv2 as _cv2
                                                gm = _cv2.resize(
                                                    gm.astype(np.float32), (i_vw, i_vh),
                                                    interpolation=_cv2.INTER_NEAREST
                                                )
                                                gm = (gm > 0.5).astype(np.uint8)
                                            run_with_cuda_retry(
                                                "appearance_reseed_add_new_mask",
                                                lambda: self._sam2_call(
                                                    self.predictor.add_new_mask,
                                                    inference_state=inference_state,
                                                    frame_idx=reseed_local,
                                                    obj_id=o_id,
                                                    mask=gm.astype(bool),
                                                ),
                                            )
                                            self.log_message.emit(
                                                f"[AppearanceMemory] Re-seeded obj {o_id} at frame {global_idx} with golden mask"
                                            )
                                        except Exception as e:
                                            self.log_message.emit(f"[AppearanceMemory] Re-seed failed for obj {o_id}: {e}")

                        # Automatic prompt injection: when drift detected, inject predicted bbox
                        if self.motion_tracker is not None:
                            for o_id in out_obj_ids:
                                if self.motion_tracker.check_needs_correction(o_id):
                                    # Get Kalman-predicted bbox
                                    pred_bbox = self.motion_tracker.get_predicted_bbox_for_correction(o_id)
                                    if pred_bbox is not None:
                                        if not self.motion_tracker.is_correction_bbox_sane(o_id, pred_bbox):
                                            self.log_message.emit(
                                                f"[Motion] Skipped correction for obj {o_id}: jump/scale too large"
                                            )
                                            self.motion_tracker.reset_correction_flag(o_id)
                                            continue
                                        try:
                                            # Inject predicted bbox as new prompt for next frame
                                            next_local_idx = out_frame_idx + 1
                                            if next_local_idx < inference_state["num_frames"]:
                                                run_with_cuda_retry(
                                                    "motion_correction_add_new_points_or_box",
                                                    lambda: self._sam2_call(
                                                        self.predictor.add_new_points_or_box,
                                                        inference_state,
                                                        frame_idx=next_local_idx,
                                                        obj_id=o_id,
                                                        box=pred_bbox,
                                                        clear_old_points=True,
                                                        normalize_coords=False,
                                                    ),
                                                )
                                                self.log_message.emit(
                                                    f"[Motion] Injected correction for obj {o_id} at frame {global_idx+1}"
                                                )
                                            self.motion_tracker.reset_correction_flag(o_id)
                                        except Exception as e:
                                            self.log_message.emit(f"Correction failed: {e}")

                        self.progress_signal.emit(global_idx)
                        # Emit real-time result for this frame
                        self.frame_result_signal.emit(global_idx, all_video_segments[global_idx])

                        # Clear old masks from memory (they're already emitted to main thread)
                        # Keep only recent MAX_MASKS_IN_MEMORY masks for the final emit
                        if len(all_video_segments) > MAX_MASKS_IN_MEMORY:
                            oldest_frame = min(all_video_segments.keys())
                            del all_video_segments[oldest_frame]

                        # Periodically clear CUDA cache (every 100 frames) to prevent accumulation
                        if global_idx % 100 == 0 and torch.cuda.is_available():
                            torch.cuda.empty_cache()
                
                processed_up_to = chunk_end
                
                # Clear CUDA cache and run garbage collection after each chunk
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                
                # MEMORY MANAGEMENT: Trim old frames if we exceed MAX_FRAMES_IN_MEMORY
                # Only apply if memory management is enabled
                if self.enable_memory_management and inference_state["num_frames"] > MAX_FRAMES_IN_MEMORY:
                    frames_to_trim = inference_state["num_frames"] - MAX_FRAMES_IN_MEMORY
                    self.log_message.emit(f"Trimming {frames_to_trim} old frames from memory (keeping last {MAX_FRAMES_IN_MEMORY} frames)...")
                    
                    # 1. Trim images list (keep last MAX_FRAMES_IN_MEMORY frames)
                    # Since images is a list, this is O(1) pointer manipulation, not O(N) memory copy
                    inference_state["images"] = inference_state["images"][-MAX_FRAMES_IN_MEMORY:]
                    inference_state["num_frames"] = len(inference_state["images"])
                    
                    # 2. Update global offset
                    global_offset += frames_to_trim
                    
                    # 3. Shift all indices in inference_state dictionaries
                    def shift_dict_keys(d, shift):
                        """Shift dictionary keys by subtracting shift, removing negative keys"""
                        new_d = {}
                        for k, v in d.items():
                            new_k = k - shift
                            if new_k >= 0:  # Only keep non-negative keys (frames still in buffer)
                                new_d[new_k] = v
                        return new_d
                    
                    # Shift cached features
                    inference_state["cached_features"] = shift_dict_keys(inference_state["cached_features"], frames_to_trim)
                    
                    # Shift per-object dictionaries
                    for obj_idx in list(inference_state["point_inputs_per_obj"].keys()):
                        inference_state["point_inputs_per_obj"][obj_idx] = shift_dict_keys(
                            inference_state["point_inputs_per_obj"][obj_idx], frames_to_trim
                        )
                        inference_state["mask_inputs_per_obj"][obj_idx] = shift_dict_keys(
                            inference_state["mask_inputs_per_obj"][obj_idx], frames_to_trim
                        )
                        
                        # Shift output dicts (keep conditioning frames if they're still in range)
                        obj_output = inference_state["output_dict_per_obj"][obj_idx]
                        obj_output["cond_frame_outputs"] = shift_dict_keys(
                            obj_output["cond_frame_outputs"], frames_to_trim
                        )
                        # SAM2's memory bank only requires the last num_maskmem
                        # non_cond frames, but all non_cond frames still inside
                        # the (already trimmed) buffer are retained.
                        obj_output["non_cond_frame_outputs"] = shift_dict_keys(
                            obj_output["non_cond_frame_outputs"], frames_to_trim
                        )
                        
                        obj_temp = inference_state["temp_output_dict_per_obj"][obj_idx]
                        obj_temp["cond_frame_outputs"] = shift_dict_keys(
                            obj_temp["cond_frame_outputs"], frames_to_trim
                        )
                        obj_temp["non_cond_frame_outputs"] = shift_dict_keys(
                            obj_temp["non_cond_frame_outputs"], frames_to_trim
                        )
                        
                        # Shift frames_tracked metadata
                        inference_state["frames_tracked_per_obj"][obj_idx] = shift_dict_keys(
                            inference_state["frames_tracked_per_obj"][obj_idx], frames_to_trim
                        )
                    
                    self.log_message.emit(f"Memory trimmed. Global offset now: {global_offset}")
                    
                    # Clear CUDA cache after memory trimming
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                # Load NEXT chunk if needed
                if processed_up_to < self.end_frame:
                    next_end = min(processed_up_to + self.chunk_size, self.end_frame)
                    self.log_message.emit(f"Loading next chunk: {processed_up_to} to {next_end}")
                    
                    # Capture last frame masks of current chunk for optional reseed
                    if self.reseed_between_chunks:
                        last_frame_idx = processed_up_to - 1
                        last_masks_for_reseed = all_video_segments.get(last_frame_idx, None)
                    
                    new_images = load_frames(processed_up_to, next_end)
                    
                    # Append to inference_state
                    # OPTIMIZATION: Since images is a list, we can extend it directly (O(1) per frame)
                    # instead of torch.cat which would copy all existing frames (O(N))
                    new_images_list = [new_images[i] for i in range(len(new_images))]
                    inference_state["images"].extend(new_images_list)
                    inference_state["num_frames"] = len(inference_state["images"])
                    
                    # Optional reseed: add mask from last frame of previous chunk
                    if self.reseed_between_chunks and last_masks_for_reseed:
                        try:
                            seed_frame_local = processed_up_to - buffer_start  # first frame of new chunk in buffer coords
                            vh = inference_state["video_height"]
                            vw = inference_state["video_width"]
                            
                            for obj_id, mask in last_masks_for_reseed.items():
                                if mask is None or mask.max() == 0:
                                    continue
                                
                                # Resize mask to video dimensions if needed
                                if mask.shape[0] != vh or mask.shape[1] != vw:
                                    import cv2
                                    mask_resized = cv2.resize(mask.astype(np.float32), (vw, vh), interpolation=cv2.INTER_NEAREST)
                                    mask = (mask_resized > 0.5).astype(np.uint8)

                                run_with_cuda_retry(
                                    "chunk_reseed_add_new_mask",
                                    lambda: self._sam2_call(self.predictor.add_new_mask,
                                        inference_state=inference_state,
                                        frame_idx=seed_frame_local,
                                        obj_id=obj_id,
                                        mask=mask.astype(bool),
                                    ),
                                )
                                self.log_message.emit(f"Reseeded obj {obj_id} with mask at frame {processed_up_to}")
                        except Exception as e:
                            self.log_message.emit(f"Warning: reseed between chunks failed: {e}")
            
            self.finished_signal.emit(all_video_segments)
            
        except Exception as e:
            import traceback
            self.log_message.emit(f"ERROR: {str(e)}\n{traceback.format_exc()}")
            self.error_signal.emit(str(e))


class VideoLabel(QLabel):
    """Custom label for video display with click handling."""
    click_signal = pyqtSignal(int, int)
    
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.click_signal.emit(event.pos().x(), event.pos().y())


class SegmentationTrackingWidget(QWidget):
    """Widget for segmentation and multi-object tracking using SAM2."""
    
    # Signal emitted when tracking completes with video and mask paths
    tracking_completed = pyqtSignal(str, str)  # video_path, mask_path
    
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.video_path = None
        self.cap = None
        self.total_frames = 0
        self.current_frame_idx = 0
        self.frame = None
        self._frame_rgb = None  # Keep RGB frame reference for QImage memory safety
        self.points = []  # List of (x, y, label, frame_idx, obj_id)
        self.masks = {}   # frame_idx -> {obj_id: mask_array}
        self.last_processed_frame = None
        self.tracking_paused = False
        self.resume_from_frame = None
        self.resume_initial_masks = {}
        self._base_display_pixmap = None
        self.zoom_factor = 1.0
        self.zoom_min = 0.5
        self.zoom_max = 4.0
        self.zoom_step = 0.2
        
        self.obj_ids = [1]
        self.current_obj_id = 1
        
        self.predictor = None
        self.inference_state = None
        self.state_start_frame = 0
        
        # Multi-video support
        self.videos = []  # list of per-video state dicts
        self.current_video_idx = None
        self.batch_queue = []
        self.batch_mode = False
        
        # Settings
        self.mask_threshold = 0.0
        self.fill_hole_area = 0
        self.offload_video = True
        self.offload_state = True
        self.use_cuda_bf16_autocast = True
        self.enable_memory_management = True
        self.max_frames_per_load = 200  # Limit frames loaded at once to prevent RAM issues
        self.save_overlay_video = True
        
        # Motion-aware tracking settings
        self.enable_motion_tracking = False  # Off by default
        self.motion_score_threshold = 0.3
        self.motion_consecutive_low = 3  # Frames before auto-correction
        self.motion_area_threshold = 0.5  # Max allowed area change ratio
        
        # OC-SORT drift correction settings
        self.enable_ocsort = False  # Off by default
        self.ocsort_inertia = 0.2  # Paper default: 0.2
        
        # SAM2 paths - resolved via _paths so this works both from source and pip install
        from singlebehaviorlab._paths import get_sam2_backend_dir, get_sam2_checkpoints_dir
        self.sam2_dir = str(get_sam2_checkpoints_dir())
        self.sam2_backend_dir = str(get_sam2_backend_dir())
        self.checkpoint_path = os.path.join(self.sam2_dir, "checkpoints", "sam2.1_hiera_large.pt")
        self.model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        
        self.tracking_worker = None
        self.download_worker = None
        
        self._setup_ui()
        self._check_sam2_availability()

    def _use_cuda_bf16(self):
        """Use bf16 autocast only for CUDA SAM2 inference."""
        dev = getattr(self.predictor, "device", None)
        dev_type = getattr(dev, "type", str(dev))
        return bool(
            self.use_cuda_bf16_autocast
            and torch.cuda.is_available()
            and dev_type == "cuda"
        )

    def _sam2_autocast(self):
        if self._use_cuda_bf16():
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    def _sam2_call(self, fn, *args, **kwargs):
        with self._sam2_autocast():
            return fn(*args, **kwargs)
        self._check_model_availability()
    
    def _ensure_hydra_initialized(self):
        """Ensure Hydra is initialized before using SAM2."""
        try:
            from hydra.core.global_hydra import GlobalHydra
            from hydra import initialize_config_dir
            
            # Check if Hydra is already initialized
            if GlobalHydra.instance().is_initialized():
                return True
            
            # Find sam2 configs directory
            sam2_configs_dir = None
            
            # Try to find from installed package first (most reliable)
            try:
                import sam2
                if hasattr(sam2, '__file__') and sam2.__file__:
                    sam2_path = os.path.dirname(sam2.__file__)
                    configs_path = os.path.join(sam2_path, "configs")
                    if os.path.exists(configs_path):
                        sam2_configs_dir = configs_path
                elif hasattr(sam2, '__path__'):
                    # Handle namespace packages
                    for path in sam2.__path__:
                        configs_path = os.path.join(path, "configs")
                        if os.path.exists(configs_path):
                            sam2_configs_dir = configs_path
                            break
            except ImportError:
                pass
            
            # Fall back to the bundled sam2_backend configs directory.
            if not sam2_configs_dir:
                sam2_backend_configs = os.path.join(self.sam2_backend_dir, "sam2", "configs")
                if os.path.exists(sam2_backend_configs):
                    sam2_configs_dir = sam2_backend_configs
            
            if sam2_configs_dir:
                # Initialize Hydra with the config directory
                initialize_config_dir(config_dir=sam2_configs_dir, version_base="1.2")
                return True
            
            # Fallback: try initialize_config_module (may work if SAM2 is properly installed)
            try:
                from hydra import initialize_config_module
                initialize_config_module("sam2", version_base="1.2")
                return True
            except Exception:
                pass
            
            return False
            
        except ImportError as e:
            # Hydra not installed
            return False
        except Exception as e:
            # Hydra initialization failed
            return False

    def _has_installed_sam2_distribution(self):
        """Return True when SAM2 is importable as a Python package."""
        for dist_name in ("SAM-2", "sam2"):
            try:
                importlib_metadata.distribution(dist_name)
                return True
            except importlib_metadata.PackageNotFoundError:
                continue
            except Exception:
                continue
        try:
            import importlib.util
            return importlib.util.find_spec("sam2") is not None
        except Exception:
            return False
    
    
    def _check_sam2_availability(self):
        """Check if SAM2 is available."""
        has_installed_pkg = self._has_installed_sam2_distribution()

        # Only report SAM2 as installed when it exists as an actual Python package
        try:
            if has_installed_pkg:
                # Initialize Hydra before importing SAM2
                self._ensure_hydra_initialized()
                from sam2.build_sam import build_sam2_video_predictor
                self.sam2_available = True
                self.setup_status_label.setText("SAM2 is installed")
                self.setup_status_label.setStyleSheet("color: green;")
                self._populate_checkpoints()
                # Set default model selection
                for i in range(self.combo_model.count()):
                    if self.combo_model.itemData(i) == "sam2.1_hiera_large.pt":
                        self.combo_model.setCurrentIndex(i)
                        break
                self._check_model_availability()
                return
        except (ImportError, RuntimeError) as e:
            # If RuntimeError about parent directory, SAM2 needs to be properly installed
            if isinstance(e, RuntimeError) and ("parent directory" in str(e) or "shadowed" in str(e)):
                # Check if sam2_backend directory exists - if so, it needs to be reinstalled
                if os.path.exists(self.sam2_backend_dir):
                    sam2_package = os.path.join(self.sam2_backend_dir, "sam2")
                    if os.path.exists(sam2_package):
                        # SAM2 exists but not properly installed - needs pip install -e
                        self.sam2_available = False
                        self.setup_status_label.setText("SAM2 needs reinstallation")
                        self.setup_status_label.setStyleSheet("color: orange;")
                        return
        
        # If the source tree exists but the package is not installed, report that clearly.
        sam2_folder = self.sam2_backend_dir
        if os.path.exists(sam2_folder):
            sam2_package = os.path.join(sam2_folder, "sam2")
            if os.path.exists(sam2_package) and os.path.exists(os.path.join(sam2_package, "__init__.py")):
                self.sam2_available = False
                self.setup_status_label.setText("SAM2 source found, but not installed")
                self.setup_status_label.setStyleSheet("color: orange;")
                return
        
        self.sam2_available = False
        self.setup_status_label.setText("SAM2 not installed")
        self.setup_status_label.setStyleSheet("color: red;")
    
    def _setup_ui(self):
        """Setup UI components."""
        layout = QVBoxLayout(self)
        
        # Top row: SAM2 Setup and Model Selection side by side
        top_row_layout = QHBoxLayout()
        
        # SAM2 Setup Section (left side)
        setup_group = QGroupBox("SAM2 Setup")
        setup_layout = QVBoxLayout()
        
        setup_info_layout = QHBoxLayout()
        setup_info_layout.addWidget(QLabel("Status:"))
        self.setup_status_label = QLabel("Checking...")
        setup_info_layout.addWidget(self.setup_status_label)
        setup_info_layout.addStretch()
        setup_layout.addLayout(setup_info_layout)
        
        setup_path_layout = QHBoxLayout()
        setup_path_layout.addWidget(QLabel("Package:"))
        self.setup_path_label = QLabel(self.sam2_backend_dir)
        self.setup_path_label.setWordWrap(True)
        self.setup_path_label.setStyleSheet("color: gray;")
        setup_path_layout.addWidget(self.setup_path_label, stretch=1)
        setup_layout.addLayout(setup_path_layout)
        
        ckpt_path_layout = QHBoxLayout()
        ckpt_path_layout.addWidget(QLabel("Checkpoints:"))
        self.ckpt_path_label = QLabel(self.sam2_dir)
        self.ckpt_path_label.setWordWrap(True)
        self.ckpt_path_label.setStyleSheet("color: gray;")
        ckpt_path_layout.addWidget(self.ckpt_path_label, stretch=1)
        setup_layout.addLayout(ckpt_path_layout)
        
        setup_group.setLayout(setup_layout)
        top_row_layout.addWidget(setup_group)
        
        # Model Selection (right side) - matching Video Settings width
        model_group = QGroupBox("Model selection")
        model_group.setFixedWidth(380)  # Match Video Settings & Controls width
        model_layout = QVBoxLayout()
        
        model_select_layout = QHBoxLayout()
        model_select_layout.addWidget(QLabel("Model:"))
        self.combo_model = QComboBox()
        # Add all available models with user-friendly names
        self.model_names = {
            "sam2.1_hiera_tiny.pt": "SAM2.1 Tiny (38.9M, Fastest)",
            "sam2.1_hiera_small.pt": "SAM2.1 Small (46M, Fast)",
            "sam2.1_hiera_base_plus.pt": "SAM2.1 Base+ (80.8M, Balanced)",
            "sam2.1_hiera_large.pt": "SAM2.1 Large (224.4M, Best Quality)",
            "sam2_hiera_tiny.pt": "SAM2.0 Tiny (38.9M, Legacy)",
            "sam2_hiera_small.pt": "SAM2.0 Small (46M, Legacy)",
            "sam2_hiera_base_plus.pt": "SAM2.0 Base+ (80.8M, Legacy)",
            "sam2_hiera_large.pt": "SAM2.0 Large (224.4M, Legacy)",
        }
        for model_file, display_name in self.model_names.items():
            self.combo_model.addItem(display_name, model_file)
        self.combo_model.currentIndexChanged.connect(self._on_model_selected)
        model_select_layout.addWidget(self.combo_model)
        model_layout.addLayout(model_select_layout)
        
        self.model_status_label = QLabel("Select a model to check/download")
        self.model_status_label.setWordWrap(True)
        self.model_status_label.setStyleSheet("color: gray;")
        model_layout.addWidget(self.model_status_label)
        
        self.download_progress = QProgressBar()
        self.download_progress.setVisible(False)
        model_layout.addWidget(self.download_progress)
        
        model_group.setLayout(model_layout)
        top_row_layout.addWidget(model_group)
        
        layout.addLayout(top_row_layout)
        
        # Legacy checkpoint combo (hidden, kept for compatibility)
        self.combo_ckpt = QComboBox()
        self.combo_ckpt.currentTextChanged.connect(self._on_checkpoint_changed)
        
        # Video Range Controls
        range_group = QGroupBox("Processing range")
        range_layout = QHBoxLayout()
        
        self.chk_limit_range = QCheckBox("Limit Range")
        self.chk_limit_range.toggled.connect(self._toggle_range_inputs)
        range_layout.addWidget(self.chk_limit_range)
        
        range_layout.addWidget(QLabel("Start:"))
        self.spin_start = QSpinBox()
        self.spin_start.setRange(0, 999999)
        self.spin_start.setEnabled(False)
        range_layout.addWidget(self.spin_start)
        
        self.btn_set_start = QPushButton("Set")
        self.btn_set_start.clicked.connect(self._set_range_start)
        self.btn_set_start.setEnabled(False)
        range_layout.addWidget(self.btn_set_start)
        
        range_layout.addWidget(QLabel("End:"))
        self.spin_end = QSpinBox()
        self.spin_end.setRange(0, 999999)
        self.spin_end.setEnabled(False)
        range_layout.addWidget(self.spin_end)
        
        self.btn_set_end = QPushButton("Set")
        self.btn_set_end.clicked.connect(self._set_range_end)
        self.btn_set_end.setEnabled(False)
        range_layout.addWidget(self.btn_set_end)
        
        range_group.setLayout(range_layout)
        layout.addWidget(range_group)
        
        # Video Display and Controls side by side
        video_row_layout = QHBoxLayout()
        
        # Video Display (left side)
        self.video_scroll = QScrollArea()
        self.video_scroll.setStyleSheet("background-color: black; border: none;")
        self.video_scroll.setWidgetResizable(False)
        self.video_scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.video_label = VideoLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(1, 1)
        self.video_label.setStyleSheet("background-color: black;")
        self.video_label.click_signal.connect(self._handle_click)
        self.video_scroll.setWidget(self.video_label)
        self.video_scroll.viewport().installEventFilter(self)

        self.btn_zoom_in = QPushButton("+", self.video_scroll.viewport())
        self.btn_zoom_out = QPushButton("-", self.video_scroll.viewport())
        self._style_zoom_button(self.btn_zoom_in)
        self._style_zoom_button(self.btn_zoom_out)
        self.btn_zoom_in.clicked.connect(self._zoom_in)
        self.btn_zoom_out.clicked.connect(self._zoom_out)
        self._position_zoom_buttons()

        video_row_layout.addWidget(self.video_scroll, stretch=2)
        
        # Controls Container (right side) - matching Model Selection width
        controls_group = QGroupBox("Video settings & controls")
        # Set width to match Model Selection container (approximately 350-400px)
        controls_group.setFixedWidth(380)
        controls_layout = QVBoxLayout()
        
        # Load Video button
        self.btn_load = QPushButton("Load videos")
        self.btn_load.clicked.connect(self._load_video)
        controls_layout.addWidget(self.btn_load)
        
        # Video list
        self.video_list_widget = QListWidget()
        self.video_list_widget.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.video_list_widget.currentRowChanged.connect(self._on_video_selected)
        controls_layout.addWidget(self.video_list_widget)
        
        # Object Controls
        obj_layout = QHBoxLayout()
        obj_layout.addWidget(QLabel("Object:"))
        
        self.combo_obj = QComboBox()
        self.combo_obj.addItem("Object 1", 1)
        self.combo_obj.currentIndexChanged.connect(self._on_object_changed)
        obj_layout.addWidget(self.combo_obj)
        
        self.btn_add_obj = QPushButton("+")
        self.btn_add_obj.setFixedWidth(30)
        self.btn_add_obj.clicked.connect(self._add_object)
        obj_layout.addWidget(self.btn_add_obj)
        
        controls_layout.addLayout(obj_layout)
        
        # Point type
        point_type_layout = QHBoxLayout()
        self.radio_pos = QRadioButton("Positive (+)")
        self.radio_neg = QRadioButton("Negative (-)")
        self.radio_pos.setChecked(True)
        self.btn_group = QButtonGroup()
        self.btn_group.addButton(self.radio_pos)
        self.btn_group.addButton(self.radio_neg)
        point_type_layout.addWidget(self.radio_pos)
        point_type_layout.addWidget(self.radio_neg)
        controls_layout.addLayout(point_type_layout)
        
        self.btn_clear_points = QPushButton("Clear points")
        self.btn_clear_points.clicked.connect(self._clear_points)
        controls_layout.addWidget(self.btn_clear_points)

        self.btn_export_prompts = QPushButton("Export prompts")
        self.btn_export_prompts.setToolTip(
            "Save the current point prompts to a JSON file so the headless CLI "
            "(`singlebehaviorlab segment`) can reproduce this segmentation on another machine."
        )
        self.btn_export_prompts.clicked.connect(self._export_prompts)
        controls_layout.addWidget(self.btn_export_prompts)

        self.btn_import_prompts = QPushButton("Import prompts")
        self.btn_import_prompts.setToolTip(
            "Load point prompts from a JSON file exported with 'Export prompts'."
        )
        self.btn_import_prompts.clicked.connect(self._import_prompts)
        controls_layout.addWidget(self.btn_import_prompts)

        controls_layout.addSpacing(10)
        
        self.chk_auto_follow = QCheckBox("Auto-follow")
        self.chk_auto_follow.setChecked(True)
        self.chk_auto_follow.setToolTip("Automatically move slider to the frame being processed.")
        controls_layout.addWidget(self.chk_auto_follow)
        
        self.btn_track = QPushButton("Run tracking (Current)")
        self.btn_track.clicked.connect(self._run_tracking)
        self.btn_track.setEnabled(False)
        controls_layout.addWidget(self.btn_track)
        
        # Pause / Resume tracking controls
        pause_resume_layout = QHBoxLayout()
        self.btn_pause_tracking = QPushButton("Pause tracking")
        self.btn_pause_tracking.setEnabled(False)
        self.btn_pause_tracking.clicked.connect(self._pause_tracking)
        pause_resume_layout.addWidget(self.btn_pause_tracking)
        
        self.btn_resume_tracking = QPushButton("Resume tracking from here")
        self.btn_resume_tracking.setEnabled(False)
        self.btn_resume_tracking.clicked.connect(self._resume_tracking)
        pause_resume_layout.addWidget(self.btn_resume_tracking)
        
        controls_layout.addLayout(pause_resume_layout)
        
        self.btn_track_all = QPushButton("Run tracking (All videos)")
        self.btn_track_all.clicked.connect(self._run_tracking_all)
        self.btn_track_all.setEnabled(False)
        controls_layout.addWidget(self.btn_track_all)

        self.chk_save_overlay = QCheckBox("Save overlay video after tracking")
        self.chk_save_overlay.setChecked(self.save_overlay_video)
        self.chk_save_overlay.setToolTip(
            "Save an MP4 with colored mask overlays for later inspection.\n"
            "Also applies when tracking is paused."
        )
        self.chk_save_overlay.toggled.connect(lambda v: setattr(self, "save_overlay_video", bool(v)))
        controls_layout.addWidget(self.chk_save_overlay)

        # SAM2 tracking resolution
        res_layout = QHBoxLayout()
        res_layout.addWidget(QLabel("Tracking resolution:"))
        self.tracking_res_combo = QComboBox()
        self.tracking_res_combo.addItem("256 (fastest, low quality)", 256)
        self.tracking_res_combo.addItem("384 (fast)", 384)
        self.tracking_res_combo.addItem("512 (balanced)", 512)
        self.tracking_res_combo.addItem("1024 (best quality, default)", 1024)
        self.tracking_res_combo.setCurrentIndex(3)
        self.tracking_res_combo.setToolTip(
            "Resolution at which SAM2 processes frames.\n"
            "Lower = faster tracking but less precise masks.\n"
            "512 is good for centroid/bbox extraction."
        )
        res_layout.addWidget(self.tracking_res_combo)
        controls_layout.addLayout(res_layout)

        self.btn_preview = QPushButton("Preview frame")
        self.btn_preview.clicked.connect(self._preview_frame)
        self.btn_preview.setEnabled(False)
        controls_layout.addWidget(self.btn_preview)
        
        self.btn_settings = QPushButton("Settings")
        self.btn_settings.clicked.connect(self._open_settings)
        controls_layout.addWidget(self.btn_settings)
        
        controls_layout.addStretch()
        
        controls_group.setLayout(controls_layout)
        video_row_layout.addWidget(controls_group)
        
        layout.addLayout(video_row_layout)
        
        # Slider
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.sliderMoved.connect(self._set_frame)
        self.slider.setEnabled(False)
        layout.addWidget(self.slider)
        
        # Frame navigation row
        nav_layout = QHBoxLayout()
        
        self.btn_prev_frame = QPushButton("<")
        self.btn_prev_frame.setFixedWidth(40)
        self.btn_prev_frame.clicked.connect(self._prev_frame)
        self.btn_prev_frame.setEnabled(False)
        nav_layout.addWidget(self.btn_prev_frame)
        
        self.lbl_frame = QLabel("Frame: 0 / 0")
        nav_layout.addWidget(self.lbl_frame, stretch=1)
        
        self.btn_next_frame = QPushButton(">")
        self.btn_next_frame.setFixedWidth(40)
        self.btn_next_frame.clicked.connect(self._next_frame)
        self.btn_next_frame.setEnabled(False)
        nav_layout.addWidget(self.btn_next_frame)
        
        layout.addLayout(nav_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Log area
        self.log_text = QLabel("")
        self.log_text.setWordWrap(True)
        self.log_text.setMaximumHeight(80)
        layout.addWidget(self.log_text)
    
    def _download_checkpoints(self):
        """Prompt user about checkpoint download (now handled automatically)."""
        QMessageBox.information(
            self,
            "Checkpoint Download",
            "Checkpoints are now downloaded automatically when you select a model.\n\n"
            "Simply select a model from the dropdown above, and it will be downloaded if not already present."
        )
    
    def _check_model_availability(self):
        """Check if selected model checkpoint exists."""
        if not self.sam2_available:
            self.model_status_label.setText("SAM2 not installed. Run install.sh and reopen the app.")
            self.model_status_label.setStyleSheet("color: red;")
            return
        
        idx = self.combo_model.currentIndex()
        if idx < 0:
            return
        
        model_name = self.combo_model.itemData(idx)
        if not model_name:
            return
        
        ckpt_path = os.path.join(self.sam2_dir, "checkpoints", model_name)
        
        if os.path.exists(ckpt_path):
            file_size = os.path.getsize(ckpt_path) / (1024**2)
            if file_size > 10:  # Reasonable size check
                self.model_status_label.setText(f"{model_name} available ({file_size:.1f} MB)")
                self.model_status_label.setStyleSheet("color: green;")
            else:
                self.model_status_label.setText(f"{model_name} file seems corrupted ({file_size:.1f} MB). Will re-download.")
                self.model_status_label.setStyleSheet("color: orange;")
        else:
            self.model_status_label.setText(f"{model_name} not found. Will download automatically when selected.")
            self.model_status_label.setStyleSheet("color: orange;")
    
    def _on_model_selected(self):
        """Handle model selection change."""
        idx = self.combo_model.currentIndex()
        if idx < 0:
            return
        
        model_name = self.combo_model.itemData(idx)
        if not model_name:
            return
        
        ckpt_path = os.path.join(self.sam2_dir, "checkpoints", model_name)
        
        # Check if checkpoint exists and is valid
        if os.path.exists(ckpt_path):
            file_size = os.path.getsize(ckpt_path) / (1024**2)
            if file_size > 10:  # Reasonable size check (should be >100MB typically)
                self.model_status_label.setText(f"{model_name} ready ({file_size:.1f} MB)")
                self.model_status_label.setStyleSheet("color: green;")
                self.checkpoint_path = ckpt_path
                self._on_checkpoint_changed(model_name)
                return
        
        # Checkpoint doesn't exist or is invalid, download it
        if model_name not in CheckpointDownloadWorker.MODEL_URLS:
            self.model_status_label.setText(f"Unknown model: {model_name}")
            self.model_status_label.setStyleSheet("color: red;")
            return
        
        # Start download
        self._download_checkpoint(model_name, ckpt_path, CheckpointDownloadWorker.MODEL_URLS[model_name])
    
    def _download_checkpoint(self, model_name, ckpt_path, model_url):
        """Download a checkpoint file."""
        if self.download_worker and self.download_worker.isRunning():
            QMessageBox.warning(self, "Download in progress", "A checkpoint download is already in progress.")
            return
        
        # Ensure checkpoints directory exists
        ckpt_dir = os.path.dirname(ckpt_path)
        os.makedirs(ckpt_dir, exist_ok=True)
        
        self.model_status_label.setText(f"Downloading {model_name}...")
        self.model_status_label.setStyleSheet("color: blue;")
        self.download_progress.setVisible(True)
        self.download_progress.setRange(0, 0)  # Indeterminate
        
        self.download_worker = CheckpointDownloadWorker(model_name, ckpt_path, model_url)
        self.download_worker.progress.connect(self._on_download_progress)
        self.download_worker.finished.connect(self._on_download_finished)
        self.download_worker.start()
    
    def _on_download_progress(self, message):
        """Handle download progress updates."""
        self.model_status_label.setText(message)
    
    def _on_download_finished(self, success, message):
        """Handle download completion."""
        self.download_progress.setVisible(False)
        
        if success:
            self.model_status_label.setText(f"{message}")
            self.model_status_label.setStyleSheet("color: green;")
            model_name = self.combo_model.currentData()
            self.checkpoint_path = os.path.join(self.sam2_dir, "checkpoints", model_name)
            self._on_checkpoint_changed(model_name)
        else:
            self.model_status_label.setText(f"{message}")
            self.model_status_label.setStyleSheet("color: red;")
            QMessageBox.critical(self, "Download failed", message)
    
    def _populate_checkpoints(self):
        """Populate checkpoint combo box (legacy, for compatibility)."""
        self.combo_ckpt.clear()
        ckpt_dir = os.path.join(self.sam2_dir, "checkpoints")
        if os.path.exists(ckpt_dir):
            checkpoints = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
            checkpoints.sort()
            self.combo_ckpt.addItems(checkpoints)
            
            default = "sam2.1_hiera_large.pt"
            if default in checkpoints:
                self.combo_ckpt.setCurrentText(default)
            elif checkpoints:
                self.combo_ckpt.setCurrentIndex(0)
    
    def _on_checkpoint_changed(self, ckpt_name):
        """Handle checkpoint selection change."""
        if not ckpt_name:
            return
        
        # Update checkpoint path
        self.checkpoint_path = os.path.join(self.sam2_dir, "checkpoints", ckpt_name)
        self.model_cfg = self._get_model_config(ckpt_name)
        
        self.predictor = None
        self.inference_state = None
        self.masks = {}
        self.points = []
        # Clean up any incremental mask temp file
        if hasattr(self, '_incremental_mask_file') and self._incremental_mask_file is not None:
            try:
                self._incremental_mask_file.close()
                os.unlink(self._incremental_mask_file.name)
            except:
                pass
            self._incremental_mask_file = None
        self._update_frame()
    
    def _get_model_config(self, ckpt_name):
        """Map checkpoint name to config."""
        if "sam2.1" in ckpt_name:
            prefix = "configs/sam2.1/sam2.1_hiera_"
        else:
            prefix = "configs/sam2/sam2_hiera_"
        
        if "large" in ckpt_name.lower():
            return prefix + "l.yaml"
        elif "base_plus" in ckpt_name.lower() or "b+" in ckpt_name.lower():
            return prefix + "b+.yaml"
        elif "small" in ckpt_name.lower():
            return prefix + "s.yaml"
        elif "tiny" in ckpt_name.lower():
            return prefix + "t.yaml"
        
        return prefix + "l.yaml"
    
    def _toggle_range_inputs(self, checked):
        """Toggle range input widgets."""
        self.spin_start.setEnabled(checked)
        self.spin_end.setEnabled(checked)
        self.btn_set_start.setEnabled(checked)
        self.btn_set_end.setEnabled(checked)
    
    def _set_range_start(self):
        """Apply the start frame chosen in the spin box (clamped to video length)."""
        if not self.cap:
            return
        
        start_val = max(0, min(self.spin_start.value(), self.total_frames - 1))
        
        # Clamp and update start value
        self.spin_start.blockSignals(True)
        self.spin_start.setValue(start_val)
        self.spin_start.blockSignals(False)
        
        # Ensure start is not beyond end
        if start_val > self.spin_end.value():
            self.spin_end.setValue(start_val)
        
        # Jump preview to start frame so the user sees what was set
        self.slider.setValue(start_val)
        self._set_frame(start_val)
    
    def _set_range_end(self):
        """Apply the end frame chosen in the spin box (clamped to video length)."""
        if not self.cap:
            return
        
        end_val = max(0, min(self.spin_end.value(), self.total_frames - 1))
        
        # Clamp and update end value
        self.spin_end.blockSignals(True)
        self.spin_end.setValue(end_val)
        self.spin_end.blockSignals(False)
        
        # Ensure end is not before start
        if end_val < self.spin_start.value():
            self.spin_start.setValue(end_val)
        
        # Jump preview to end frame so the user sees what was set
        self.slider.setValue(end_val)
        self._set_frame(end_val)
    
    def _add_object(self):
        """Add a new object ID."""
        new_id = max(self.obj_ids) + 1
        self.obj_ids.append(new_id)
        self.combo_obj.addItem(f"Object {new_id}", new_id)
        self.combo_obj.setCurrentIndex(self.combo_obj.count() - 1)
    
    def _on_object_changed(self):
        """Handle object selection change."""
        self.current_obj_id = self.combo_obj.currentData()
    
    def _create_video_state(self, path):
        """Create a per-video state dict."""
        cap = cv2.VideoCapture(path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return {
            "path": path,
            "total_frames": total_frames,
            "points": [],
            "masks": {},
            "obj_ids": [1],
            "current_obj_id": 1,
            "current_frame_idx": 0,
            "spin_start": 0,
            "spin_end": max(total_frames - 1, 0),
            "inference_state": None,
            "state_start_frame": 0,
        }
    
    def _save_current_video_state(self):
        """Persist current UI state into the active video entry."""
        if self.current_video_idx is None or self.current_video_idx >= len(self.videos):
            return
        try:
            v = self.videos[self.current_video_idx]
            v["points"] = list(self.points)
            v["masks"] = dict(self.masks)
            v["obj_ids"] = list(self.obj_ids)
            v["current_obj_id"] = self.current_obj_id
            v["current_frame_idx"] = self.current_frame_idx
            v["spin_start"] = self.spin_start.value()
            v["spin_end"] = self.spin_end.value()
            v["inference_state"] = self.inference_state
            v["state_start_frame"] = self.state_start_frame
        except Exception:
            pass
    
    def _apply_video_state(self, idx: int):
        """Load a video's state into the UI and current attributes."""
        if idx < 0 or idx >= len(self.videos):
            return
        self.current_video_idx = idx
        v = self.videos[idx]
        self.video_path = v["path"]
        self.cap = cv2.VideoCapture(self.video_path)
        self.total_frames = v["total_frames"]
        
        # Ranges
        self.slider.setRange(0, max(self.total_frames - 1, 0))
        self.spin_start.setRange(0, max(self.total_frames - 1, 0))
        self.spin_end.setRange(0, max(self.total_frames - 1, 0))
        self.spin_start.setValue(min(v["spin_start"], max(self.total_frames - 1, 0)))
        self.spin_end.setValue(min(v["spin_end"], max(self.total_frames - 1, 0)))
        
        # Restore points/masks/objects
        self.points = list(v["points"])
        self.masks = dict(v["masks"])
        self.obj_ids = list(v["obj_ids"])
        self.current_obj_id = v["current_obj_id"]
        self.current_frame_idx = min(v["current_frame_idx"], max(self.total_frames - 1, 0))
        self.inference_state = v["inference_state"]
        self.state_start_frame = v.get("state_start_frame", 0)
        
        # Rebuild object combo
        self.combo_obj.blockSignals(True)
        self.combo_obj.clear()
        for oid in self.obj_ids:
            self.combo_obj.addItem(f"Object {oid}", oid)
        idx_obj = self.combo_obj.findData(self.current_obj_id)
        if idx_obj >= 0:
            self.combo_obj.setCurrentIndex(idx_obj)
        self.combo_obj.blockSignals(False)
        
        # Enable controls
        self.slider.setEnabled(True)
        self.btn_prev_frame.setEnabled(True)
        self.btn_next_frame.setEnabled(True)
        self.btn_track.setEnabled(self.sam2_available)
        self.btn_track_all.setEnabled(len(self.videos) > 0 and self.sam2_available)
        self.btn_preview.setEnabled(self.sam2_available)
        
        # Move slider to current frame and refresh
        self.slider.blockSignals(True)
        self.slider.setValue(self.current_frame_idx)
        self.slider.blockSignals(False)
        self.zoom_factor = 1.0
        self._update_frame()
    
    def _on_video_selected(self, row: int):
        """Handle selection change from the video list."""
        self._save_current_video_state()
        if row >= 0:
            self._apply_video_state(row)
    
    def _load_video(self):
        """Load one or more video files."""
        video_dir = self.config.get("raw_videos_dir", self.config.get("data_dir", "data/raw_videos"))
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Open Videos", video_dir, "Video Files (*.mp4 *.avi *.mov *.mkv)"
        )
        if not paths:
            return
        
        from .video_utils import ensure_video_in_experiment
        
        added_any = False
        for path in paths:
            path = ensure_video_in_experiment(path, self.config, self)
            # Avoid duplicates
            if any(v["path"] == path for v in self.videos):
                continue
            state = self._create_video_state(path)
            self.videos.append(state)
            self.video_list_widget.addItem(os.path.basename(path))
            added_any = True
        
        if not added_any:
            return
        
        # Auto-select first added video if none active
        if self.current_video_idx is None and self.videos:
            self.video_list_widget.setCurrentRow(0)
            self._apply_video_state(0)
        else:
            # Keep current selection, just enable batch controls
            self.btn_track_all.setEnabled(self.sam2_available and len(self.videos) > 0)
    
    def _ensure_predictor(self):
        """Ensure SAM2 model is loaded (rebuilds if resolution changed)."""
        tracking_res = self.tracking_res_combo.currentData() or 1024
        if self.predictor is not None:
            if getattr(self.predictor, "image_size", 1024) == tracking_res:
                return True
            # Resolution changed — need to rebuild
            del self.predictor
            self.predictor = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        if not self.sam2_available:
            QMessageBox.warning(
                self,
                "SAM2 not available",
                "SAM2 is not installed in this environment.\n\nRun bash install.sh and reopen the app.",
            )
            return False
        
        idx = self.combo_model.currentIndex()
        if idx < 0:
            QMessageBox.warning(self, "No model selected", "Please select a model.")
            return False
        
        model_name = self.combo_model.itemData(idx)
        if not model_name:
            QMessageBox.warning(self, "No model selected", "Please select a model.")
            return False
        
        self.checkpoint_path = os.path.join(self.sam2_dir, "checkpoints", model_name)
        if not os.path.exists(self.checkpoint_path):
            QMessageBox.warning(
                self,
                "Checkpoint Missing",
                f"Checkpoint not found:\n{self.checkpoint_path}\n\n"
                "The checkpoint should download automatically when you select a model.\n"
                "Please wait for the download to complete or select the model again."
            )
            return False
        
        self.model_cfg = self._get_model_config(model_name)
        
        try:
            # Ensure Hydra is initialized before importing SAM2
            if not self._ensure_hydra_initialized():
                QMessageBox.critical(
                    self,
                    "Hydra Initialization Failed",
                    "Failed to initialize Hydra configuration system.\n\n"
                    "Please ensure hydra-core is installed:\n"
                    "pip install hydra-core>=1.3.2"
                )
                return False
            
            # Import sam2 first to trigger its Hydra initialization
            import sam2
            # Then import the build function
            from sam2.build_sam import build_sam2_video_predictor
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cpu":
                QMessageBox.warning(self, "CPU mode", "Running on CPU. This will be very slow.")
            
            hydra_extra = [f"++model.image_size={tracking_res}"]
            self.predictor = build_sam2_video_predictor(
                self.model_cfg, self.checkpoint_path, device=device,
                hydra_overrides_extra=hydra_extra,
            )
            self.predictor.fill_hole_area = self.fill_hole_area
            return True
        except RuntimeError as e:
            if "parent directory" in str(e) or "shadowed" in str(e):
                QMessageBox.critical(
                    self,
                    "SAM2 Import Error",
                    f"SAM2 import conflict:\n{e}\n\n"
                    "Solution: Please install SAM2 to a location outside the behavior_labeling_app directory.\n"
                    "Use the 'Change...' button to select a different installation location."
                )
            else:
                QMessageBox.critical(self, "Error", f"Failed to init SAM2 model:\n{e}")
            return False
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to init SAM2 model:\n{e}")
            return False
    
    def _set_frame(self, frame_idx):
        """Set current frame index."""
        self.current_frame_idx = frame_idx
        self._update_frame()

    def eventFilter(self, source, event):
        if source is self.video_scroll.viewport() and event.type() == QEvent.Type.Resize:
            self._position_zoom_buttons()
        return super().eventFilter(source, event)

    def _style_zoom_button(self, btn):
        btn.setFixedSize(34, 34)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setStyleSheet(
            "QPushButton {"
            "background-color: rgba(20, 20, 20, 190);"
            "color: white;"
            "border: 1px solid rgba(255, 255, 255, 120);"
            "border-radius: 17px;"
            "font-size: 18px;"
            "font-weight: bold;"
            "}"
            "QPushButton:hover {"
            "background-color: rgba(45, 45, 45, 220);"
            "}"
        )

    def _position_zoom_buttons(self):
        if not hasattr(self, "video_scroll") or not hasattr(self, "btn_zoom_in"):
            return
        viewport = self.video_scroll.viewport()
        margin = 10
        spacing = 8
        x = viewport.width() - self.btn_zoom_in.width() - margin
        y = margin
        self.btn_zoom_in.move(x, y)
        self.btn_zoom_out.move(x, y + self.btn_zoom_in.height() + spacing)
        self.btn_zoom_in.raise_()
        self.btn_zoom_out.raise_()

    def _zoom_in(self):
        self.zoom_factor = min(self.zoom_max, self.zoom_factor + self.zoom_step)
        self._apply_zoom()

    def _zoom_out(self):
        self.zoom_factor = max(self.zoom_min, self.zoom_factor - self.zoom_step)
        self._apply_zoom()

    def _apply_zoom(self):
        if self._base_display_pixmap is None:
            return
        w = max(1, int(self._base_display_pixmap.width() * self.zoom_factor))
        h = max(1, int(self._base_display_pixmap.height() * self.zoom_factor))
        scaled = self._base_display_pixmap.scaled(
            w,
            h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.video_label.setPixmap(scaled)
        self.video_label.resize(scaled.size())
    
    def _prev_frame(self):
        """Go to previous frame."""
        if self.current_frame_idx > 0:
            self.current_frame_idx -= 1
            self.slider.blockSignals(True)
            self.slider.setValue(self.current_frame_idx)
            self.slider.blockSignals(False)
            self._update_frame()
    
    def _next_frame(self):
        """Go to next frame."""
        if self.current_frame_idx < self.total_frames - 1:
            self.current_frame_idx += 1
            self.slider.blockSignals(True)
            self.slider.setValue(self.current_frame_idx)
            self.slider.blockSignals(False)
            self._update_frame()
    
    def _update_frame(self):
        """Update video display with current frame and overlays."""
        if not self.cap:
            return
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
        ret, frame = self.cap.read()
        if ret:
            self.frame = frame
            # Keep RGB frame as instance var to prevent QImage memory issues
            self._frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = self._frame_rgb.shape
            
            bytes_per_line = ch * w
            q_img = QImage(self._frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            
            pixmap = QPixmap.fromImage(q_img)
            painter = QPainter(pixmap)
            
            # Draw masks
            if self.current_frame_idx in self.masks:
                frame_masks = self.masks[self.current_frame_idx]
                for obj_id, mask in frame_masks.items():
                    if mask is not None and mask.max() > 0:
                        mask_h, mask_w = mask.shape[:2]
                        if mask_h != h or mask_w != w:
                            mask = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
                        
                        overlay = np.zeros((h, w, 4), dtype=np.uint8)
                        color = get_obj_color(obj_id)
                        overlay[mask > 0] = [color[0], color[1], color[2], 100]
                        
                        overlay_img = QImage(overlay.data, w, h, w * 4, QImage.Format.Format_RGBA8888)
                        painter.drawImage(0, 0, overlay_img)
            
            # Draw points
            for p in self.points:
                if p[3] == self.current_frame_idx:
                    x, y, label, obj_id = p[0], p[1], p[2], p[4]
                    obj_color_rgb = get_obj_color(obj_id)
                    
                    if label == 1:
                        painter.setPen(QPen(QColor(*obj_color_rgb), 5))
                        painter.drawPoint(x, y)
                    else:
                        painter.setPen(QPen(QColor(255, 0, 0), 5))
                        painter.drawPoint(x, y)
                    
                    painter.setPen(QPen(QColor(255, 255, 255), 1))
                    painter.drawText(x + 5, y - 5, str(obj_id))
            
            painter.end()
            self._base_display_pixmap = pixmap
            self._apply_zoom()
            
            self.lbl_frame.setText(f"Frame: {self.current_frame_idx} / {self.total_frames}")
    
    def _handle_click(self, x, y):
        """Handle click on video label."""
        if self.frame is None:
            return
        
        pixmap = self.video_label.pixmap()
        if not pixmap:
            return
        
        label_w = self.video_label.width()
        label_h = self.video_label.height()
        pix_w = pixmap.width()
        pix_h = pixmap.height()
        
        x_offset = (label_w - pix_w) / 2
        y_offset = (label_h - pix_h) / 2
        
        img_x = x - x_offset
        img_y = y - y_offset
        
        if 0 <= img_x < pix_w and 0 <= img_y < pix_h:
            orig_h, orig_w = self.frame.shape[:2]
            scale_x = orig_w / pix_w
            scale_y = orig_h / pix_h
            
            final_x = int(img_x * scale_x)
            final_y = int(img_y * scale_y)
            
            label = 1 if self.radio_pos.isChecked() else 0
            self.points.append((final_x, final_y, label, self.current_frame_idx, self.current_obj_id))
            self._preview_frame()
    
    def _clear_points(self):
        """Clear all points."""
        self.points = []
        self.masks = {}
        # Clean up any incremental mask temp file
        if hasattr(self, '_incremental_mask_file') and self._incremental_mask_file is not None:
            try:
                self._incremental_mask_file.close()
                os.unlink(self._incremental_mask_file.name)
            except:
                pass
            self._incremental_mask_file = None
        # Reset inference state to clear any cached predictions
        if self.inference_state and self.predictor:
            try:
                self.predictor.reset_state(self.inference_state)
            except:
                pass
        self._update_frame()

    def _export_prompts(self):
        """Export current point prompts to a JSON file usable by `singlebehaviorlab segment`."""
        if not self.points:
            QMessageBox.information(self, "No prompts", "Add at least one point prompt before exporting.")
            return
        default_dir = os.path.dirname(self.video_path) if self.video_path else ""
        default_name = "sam2_prompts.json"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export SAM2 prompts", os.path.join(default_dir, default_name),
            "JSON Files (*.json)",
        )
        if not path:
            return
        try:
            from singlebehaviorlab.backend.segmentation import save_prompts_json
            save_prompts_json(self.video_path or "", self.points, path)
            QMessageBox.information(self, "Exported", f"Saved {len(self.points)} prompt point(s) to:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", f"Failed to write prompts:\n{exc}")

    def _import_prompts(self):
        """Load point prompts from a JSON file produced by `Export prompts`."""
        default_dir = os.path.dirname(self.video_path) if self.video_path else ""
        path, _ = QFileDialog.getOpenFileName(
            self, "Import SAM2 prompts", default_dir, "JSON Files (*.json)"
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            entries = data.get("prompts", [])
            if not isinstance(entries, list) or not entries:
                QMessageBox.warning(self, "Empty file", "No prompts found in the selected file.")
                return
            imported: list[tuple[float, float, int, int, int]] = []
            for entry in entries:
                imported.append(
                    (
                        float(entry["x"]),
                        float(entry["y"]),
                        int(entry.get("label", 1)),
                        int(entry["frame_idx"]),
                        int(entry["obj_id"]),
                    )
                )
            self.points = imported
            self._update_frame()
            QMessageBox.information(self, "Imported", f"Loaded {len(imported)} prompt point(s) from:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "Import failed", f"Failed to read prompts:\n{exc}")
    
    def _preview_frame(self):
        """Preview segmentation on current frame."""
        if not self.video_path:
            return
        
        if not self._ensure_predictor():
            return
        
        # Gather points for current frame and object
        current_points = []
        current_labels = []
        for x, y, label, frame_idx, obj_id in self.points:
            if frame_idx == self.current_frame_idx and obj_id == self.current_obj_id:
                current_points.append([x, y])
                current_labels.append(label)
        
        if not current_points:
            return
        
        # Force fresh state load for preview to avoid any stale cached data
        # This is slower but ensures accurate preview
        if not self._load_state(self.current_frame_idx, self.current_frame_idx + 1):
            return
        
        # Reset any previous tracking state to get clean prediction
        try:
            self.predictor.reset_state(self.inference_state)
        except:
            pass
        
        try:
            pts = np.array(current_points, dtype=np.float32)
            lbls = np.array(current_labels, dtype=np.int32)
            
            local_frame_idx = 0  # We just loaded a single frame, so local index is 0
            
            _, out_obj_ids, out_mask_logits = self._sam2_call(
                self.predictor.add_new_points_or_box,
                inference_state=self.inference_state,
                frame_idx=local_frame_idx,
                obj_id=self.current_obj_id,
                points=pts,
                labels=lbls,
                clear_old_points=True,
                normalize_coords=True,
            )
            
            if self.current_obj_id in out_obj_ids:
                idx = out_obj_ids.index(self.current_obj_id)
                mask_logit = out_mask_logits[idx]
                if mask_logit.ndim == 3:
                    mask_logit = mask_logit[0]
                mask = (mask_logit > self.mask_threshold).cpu().numpy().astype(np.uint8).squeeze()
                
                if self.current_frame_idx not in self.masks:
                    self.masks[self.current_frame_idx] = {}
                self.masks[self.current_frame_idx][self.current_obj_id] = mask
                self._update_frame()
            else:
                # If preview did not return this object, clear stale overlay for it.
                if self.current_frame_idx in self.masks and self.current_obj_id in self.masks[self.current_frame_idx]:
                    del self.masks[self.current_frame_idx][self.current_obj_id]
                    self._update_frame()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Preview failed:\n{e}")
    
    def _ensure_state_for_frame(self, frame_idx):
        """Ensure state is loaded for specific frame."""
        if self.inference_state:
            local_idx = frame_idx - self.state_start_frame
            if 0 <= local_idx < self.inference_state["num_frames"]:
                return True
        
        return self._load_state(frame_idx, frame_idx + 1)
    
    def _load_state(self, start_frame, end_frame):
        """Load video frames into SAM2 state."""
        if not self._ensure_predictor():
            return False
        
        if not self.video_path:
            return False
        
        try:
            try:
                import decord
            except ImportError:
                QMessageBox.warning(
                    self,
                    "Missing Dependency",
                    "decord not found. Please install it:\n\n"
                    "pip install eva-decord\n\n"
                    "Or: conda install -c conda-forge decord"
                )
                return False
            
            from collections import OrderedDict
            
            decord.bridge.set_bridge("torch")
            compute_device = self.predictor.device
            image_size = self.predictor.image_size
            
            # Get original video dimensions (needed for coordinate normalization)
            vr_meta = decord.VideoReader(self.video_path)
            video_height, video_width, _ = vr_meta[0].shape
            total_frames = len(vr_meta)
            del vr_meta  # Free memory immediately after getting dimensions
            
            # Load frames at SAM2's internal image_size (square)
            # SAM2 uses original video_height/video_width for coordinate normalization
            vr = decord.VideoReader(self.video_path, width=image_size, height=image_size)
            target_dtype = getattr(self.predictor, "dtype", torch.float32)
            if self._use_cuda_bf16() and not self.offload_video:
                target_dtype = torch.bfloat16
            
            if end_frame is None or end_frame > total_frames:
                end_frame = total_frames
            start_frame = max(0, start_frame)
            
            if start_frame >= end_frame:
                return False
            
            # Limit number of frames loaded at once to prevent RAM issues
            num_frames = end_frame - start_frame
            if num_frames > self.max_frames_per_load:
                # Only load the last max_frames_per_load frames to keep memory usage reasonable
                start_frame = max(start_frame, end_frame - self.max_frames_per_load)
                if start_frame < 0:
                    start_frame = 0
                QMessageBox.warning(
                    self,
                    "Frame Range Limited",
                    f"Requested {num_frames} frames, but limiting to {self.max_frames_per_load} frames "
                    f"to prevent RAM issues.\n\nLoading frames {start_frame} to {end_frame}."
                )
            
            indices = list(range(start_frame, end_frame))
            frames = vr.get_batch(indices)
            del vr  # Free VideoReader memory after loading frames
            images = frames.permute(0, 3, 1, 2).float() / 255.0
            del frames  # Free original frame tensor after processing
            
            img_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None]
            img_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None]
            
            if not self.offload_video:
                images = images.to(compute_device, dtype=target_dtype)
                img_mean = img_mean.to(compute_device, dtype=target_dtype)
                img_std = img_std.to(compute_device, dtype=target_dtype)
            else:
                images = images.to(dtype=target_dtype)
                img_mean = img_mean.to(dtype=target_dtype)
                img_std = img_std.to(dtype=target_dtype)
            
            images -= img_mean
            images /= img_std
            
            # Convert to list format expected by SAM2
            images_list = [images[i] for i in range(len(images))]
            
            inference_state = {}
            inference_state["images"] = images_list
            inference_state["num_frames"] = len(images_list)
            inference_state["offload_video_to_cpu"] = self.offload_video
            inference_state["offload_state_to_cpu"] = self.offload_state
            inference_state["video_height"] = video_height
            inference_state["video_width"] = video_width
            inference_state["device"] = compute_device
            if self.offload_state:
                inference_state["storage_device"] = torch.device("cpu")
            else:
                inference_state["storage_device"] = compute_device
            
            inference_state["point_inputs_per_obj"] = {}
            inference_state["mask_inputs_per_obj"] = {}
            inference_state["cached_features"] = {}
            inference_state["constants"] = {}
            inference_state["obj_id_to_idx"] = OrderedDict()
            inference_state["obj_idx_to_id"] = OrderedDict()
            inference_state["obj_ids"] = []
            inference_state["output_dict_per_obj"] = {}
            inference_state["temp_output_dict_per_obj"] = {}
            inference_state["frames_tracked_per_obj"] = {}
            
            try:
                self._sam2_call(self.predictor._get_image_feature, inference_state, frame_idx=0, batch_size=1)
            except Exception as e:
                pass
            
            self.inference_state = inference_state
            self.state_start_frame = start_frame
            return True
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load video state:\n{e}")
            return False
    
    def _run_tracking(self, from_batch: bool = False):
        """Run tracking on current video."""
        if not self.video_path:
            QMessageBox.warning(self, "No Video", "Please load a video first.")
            return
        if not self.points:
            if from_batch and self.batch_mode and self.batch_queue:
                next_idx = self.batch_queue.pop(0)
                self._apply_video_state(next_idx)
                self._run_tracking(from_batch=True)
                return
            QMessageBox.warning(self, "No Points", "Please add some points first.")
            return
        
        if not self._ensure_predictor():
            return
        
        # Group points by frame and object
        points_grouped = {}
        for x, y, label, frame_idx, obj_id in self.points:
            key = (frame_idx, obj_id)
            if key not in points_grouped:
                points_grouped[key] = {'points': [], 'labels': []}
            points_grouped[key]['points'].append([x, y])
            points_grouped[key]['labels'].append(label)
        
        if not points_grouped:
            if from_batch and self.batch_mode and self.batch_queue:
                next_idx = self.batch_queue.pop(0)
                self._apply_video_state(next_idx)
                self._run_tracking(from_batch=True)
                return
            QMessageBox.warning(self, "Warning", "No points in the selected range.")
            return
        
        # Determine processing range to include user points
        min_frame = min(k[0] for k in points_grouped.keys())
        max_frame = max(k[0] for k in points_grouped.keys())
        
        if hasattr(self, 'chk_limit_range') and self.chk_limit_range.isChecked():
            start_f = max(self.spin_start.value(), 0)
            end_f = min(self.spin_end.value() + 1, self.total_frames)
            # Ensure the range covers the annotated points
            start_f = min(start_f, min_frame)
            end_f = max(end_f, max_frame + 1)
        else:
            start_f = 0
            end_f = self.total_frames
        
        # If resuming, allow forcing the start a bit earlier than the drift point
        if self.resume_from_frame is not None:
            # Resume from the chosen frame (or later), do not jump back to frame 0
            start_f = max(self.resume_from_frame, start_f, 0)
            self.resume_from_frame = None
        
        self.btn_track.setEnabled(False)
        self.btn_track_all.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, end_f - start_f)
        self.progress_bar.setValue(0)
        
        # Get initial masks for resume (if any)
        initial_masks = getattr(self, 'resume_initial_masks', {}) or {}
        
        # Clear resume flags
        self.resume_from_frame = None
        self.resume_initial_masks = {}
        self.tracking_paused = False
        
        self.tracking_worker = TrackingWorker(
            self.predictor,
            self.video_path,
            points_grouped,
            start_f,
            end_f,
            self.mask_threshold,
            self.offload_video,
            self.offload_state,
            enable_memory_management=self.enable_memory_management,
            reseed_between_chunks=getattr(self, "reseed_between_chunks", False),
            initial_masks=initial_masks,
            enable_motion_tracking=getattr(self, "enable_motion_tracking", False),
            motion_score_threshold=getattr(self, "motion_score_threshold", 0.3),
            motion_consecutive_low=getattr(self, "motion_consecutive_low", 3),
            motion_area_threshold=getattr(self, "motion_area_threshold", 0.5),
            enable_ocsort=getattr(self, "enable_ocsort", False),
            ocsort_inertia=getattr(self, "ocsort_inertia", 0.2),
            use_cuda_bf16_autocast=getattr(self, "use_cuda_bf16_autocast", True),
        )
        self.tracking_worker.progress_signal.connect(lambda x: self.progress_bar.setValue(x - start_f) if x >= start_f else None)
        self.tracking_worker.frame_result_signal.connect(self._on_frame_result)
        self.tracking_worker.finished_signal.connect(self._on_tracking_finished)
        self.tracking_worker.error_signal.connect(self._on_tracking_error)
        self.tracking_worker.log_message.connect(lambda msg: self.log_text.setText(msg))
        self.tracking_worker.start()
        
        # Enable pause, disable resume while running
        self.btn_pause_tracking.setEnabled(True)
        self.btn_resume_tracking.setEnabled(False)

    def _pause_tracking(self):
        """Pause the current tracking run to allow adding new prompts."""
        if hasattr(self, "tracking_worker") and self.tracking_worker and self.tracking_worker.isRunning():
            self.tracking_paused = True
            self.tracking_worker.stop()
            self.log_text.setText("Stopping tracking... you can add points and resume.")
            self.btn_pause_tracking.setEnabled(False)
            self.btn_resume_tracking.setEnabled(False)
    
    def _resume_tracking(self):
        """Resume tracking from the current frame (or last processed) after adding prompts."""
        if self.tracking_worker and self.tracking_worker.isRunning():
            return  # already running
        
        # Choose a resume frame: prefer slider position, fallback to last processed
        resume_frame = self.current_frame_idx if hasattr(self, "current_frame_idx") else None
        if resume_frame is None and self.last_processed_frame is not None:
            resume_frame = self.last_processed_frame
        if resume_frame is None:
            resume_frame = 0
        
        # Collect refined masks at the resume frame to use as conditioning
        # This allows the user to refine the mask before resuming
        initial_masks = {}
        if resume_frame in self.masks:
            for obj_id, mask in self.masks[resume_frame].items():
                if mask is not None and mask.max() > 0:
                    initial_masks[(resume_frame, obj_id)] = mask
                    self.log_text.setText(f"Will use refined mask for object {obj_id} at frame {resume_frame}")
        
        # Store initial masks for the worker
        self.resume_initial_masks = initial_masks
        
        # Start a bit before the drift point for stability (but not before the mask frame)
        self.resume_from_frame = resume_frame  # Start exactly from where user refined
        self.tracking_paused = False
        
        # Re-run tracking; it will honor resume_from_frame and initial_masks
        self._run_tracking()

    def _run_tracking_all(self):
        """Run tracking sequentially for all loaded videos."""
        if not self.videos:
            QMessageBox.warning(self, "No Videos", "Please load videos first.")
            return
        if not self.sam2_available:
            QMessageBox.warning(
                self,
                "SAM2 not available",
                "SAM2 is not installed in this environment.\n\nRun bash install.sh and reopen the app.",
            )
            return
        
        # Save current state
        self._save_current_video_state()
        
        # Build queue of indices
        self.batch_queue = list(range(len(self.videos)))
        self.batch_mode = True
        
        # Start with the first video in the queue
        next_idx = self.batch_queue.pop(0)
        self._apply_video_state(next_idx)
        self._run_tracking(from_batch=True)
    
    def _on_frame_result(self, frame_idx, frame_masks):
        """Handle real-time mask updates."""
        if frame_idx not in self.masks:
            self.masks[frame_idx] = {}
        
        for obj_id, mask in frame_masks.items():
            self.masks[frame_idx][obj_id] = mask
        
        # Track last processed frame for potential resume
        self.last_processed_frame = frame_idx
        
        # Incremental save: periodically flush masks to disk to free RAM
        # Trigger when we exceed threshold (not just at exact multiples)
        if len(self.masks) >= 500:
            self._incremental_save_masks()
        
        if frame_idx == self.current_frame_idx:
            self._update_frame()
        
        if self.chk_auto_follow.isChecked():
            self.slider.blockSignals(True)
            self.slider.setValue(frame_idx)
            self.slider.blockSignals(False)
            self.current_frame_idx = frame_idx
            self._update_frame()
    
    def _on_tracking_finished(self, masks):
        """Handle tracking completion."""
        for frame_idx, frame_masks in masks.items():
            if frame_idx not in self.masks:
                self.masks[frame_idx] = {}
            for obj_id, mask in frame_masks.items():
                self.masks[frame_idx][obj_id] = mask
        
        self.btn_track.setEnabled(True)
        self.btn_track_all.setEnabled(self.sam2_available and len(self.videos) > 0)
        self.btn_pause_tracking.setEnabled(False)
        self.btn_resume_tracking.setEnabled(self.tracking_paused)
        self.progress_bar.setVisible(False)
        self._update_frame()
        
        # Persist state for current video
        self._save_current_video_state()
        overlay_path = None
        if self.save_overlay_video:
            overlay_path = self._save_overlay_video(paused=self.tracking_paused)
        
        # If user paused manually, do not save or show completion popups yet
        if self.tracking_paused:
            if overlay_path:
                self.log_text.setText(
                    "Tracking paused. Partial overlay video saved to:\n"
                    f"{overlay_path}\n"
                    "Add points on current frame, click 'Preview frame', then 'Resume tracking from here'."
                )
            else:
                self.log_text.setText(
                    "Tracking paused. Add points on current frame, click 'Preview frame', then 'Resume tracking from here'."
                )
            self.btn_resume_tracking.setEnabled(True)
            return
        
        # Save masks automatically
        mask_path = self._save_masks()
        
        # In batch mode, skip popups to continue processing
        if self.batch_mode:
            pass
        else:
            if mask_path and self.video_path:
                # Show completion message with option to go to registration
                overlay_text = f"Overlay video saved to: {overlay_path}\n\n" if overlay_path else ""
                reply = QMessageBox.question(
                    self,
                    "Tracking Completed",
                    f"Tracking completed successfully!\n\n"
                    f"Masks saved to: {mask_path}\n\n"
                    f"{overlay_text}"
                    "Would you like to proceed to the Registration tab to process this video?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes
                )
                
                if reply == QMessageBox.StandardButton.Yes:
                    # Emit signal to switch to registration tab
                    self.tracking_completed.emit(self.video_path, mask_path)
            else:
                QMessageBox.information(self, "Success", "Tracking completed!")
        
        self.inference_state = None
        self.tracking_paused = False
        self.resume_from_frame = None
        
        # Continue batch if pending
        if self.batch_mode and self.batch_queue:
            next_idx = self.batch_queue.pop(0)
            self._apply_video_state(next_idx)
            self._run_tracking(from_batch=True)
            return
        # End batch
        self.batch_mode = False
        self.batch_queue = []
    
    def _incremental_save_masks(self):
        """Save old masks to temp file and clear from memory to prevent RAM exhaustion."""
        if not self.video_path or len(self.masks) < 500:
            return
        
        import pickle
        import tempfile
        
        # Initialize temp file on first call
        if not hasattr(self, '_incremental_mask_file') or self._incremental_mask_file is None:
            video_basename = os.path.splitext(os.path.basename(self.video_path))[0]
            self._incremental_mask_file = tempfile.NamedTemporaryFile(
                mode='wb', suffix=f'_{video_basename}_masks.pkl', delete=False
            )
            self._incremental_frame_indices = []
        
        # Get frames to save (oldest 400 frames, keep 100 for display)
        sorted_frames = sorted(self.masks.keys())
        frames_to_save = sorted_frames[:400]
        
        # Save to temp file
        chunk_data = {idx: self.masks[idx] for idx in frames_to_save}
        pickle.dump(chunk_data, self._incremental_mask_file)
        self._incremental_frame_indices.extend(frames_to_save)
        
        # Clear from memory
        for idx in frames_to_save:
            del self.masks[idx]
        
        gc.collect()

    def _get_masks_snapshot_for_export(self):
        """Get a merged mask snapshot including incremental chunks without consuming them."""
        snapshot = {}
        for frame_idx, frame_data in self.masks.items():
            snapshot[frame_idx] = dict(frame_data)

        if hasattr(self, "_incremental_mask_file") and self._incremental_mask_file is not None:
            import pickle
            try:
                self._incremental_mask_file.flush()
            except Exception:
                pass
            try:
                with open(self._incremental_mask_file.name, "rb") as f:
                    while True:
                        try:
                            chunk = pickle.load(f)
                            for frame_idx, frame_data in chunk.items():
                                if frame_idx not in snapshot:
                                    snapshot[frame_idx] = dict(frame_data)
                                else:
                                    snapshot[frame_idx].update(frame_data)
                        except EOFError:
                            break
            except Exception as e:
                logger.warning("Could not read incremental masks for overlay export: %s", e)

        return snapshot

    def _save_overlay_video(self, paused=False):
        """Save overlay video (original frame + colored masks) for inspection."""
        if not self.video_path:
            return None

        all_masks = self._get_masks_snapshot_for_export()
        if not all_masks:
            return None

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return None

        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0

        save_start = 0
        save_end = max(all_masks.keys()) if all_masks else 0
        if hasattr(self, "chk_limit_range") and self.chk_limit_range.isChecked():
            save_start = self.spin_start.value()
            save_end = min(self.spin_end.value(), save_end)
        save_start = max(save_start, 0)
        save_end = max(save_end, save_start)

        experiment_path = self.config.get("experiment_path")
        if experiment_path and os.path.exists(experiment_path):
            out_dir = os.path.join(experiment_path, "overlays")
        else:
            from singlebehaviorlab._paths import USER_DATA_DIR
            out_dir = str(USER_DATA_DIR / "data" / "overlays")
        os.makedirs(out_dir, exist_ok=True)

        video_basename = os.path.splitext(os.path.basename(self.video_path))[0]
        if video_basename.endswith("_masks"):
            video_basename = video_basename[:-6]
        suffix = "_tracking_overlay_paused.mp4" if paused else "_tracking_overlay.mp4"
        output_path = os.path.join(out_dir, f"{video_basename}{suffix}")

        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (video_width, video_height),
        )
        if not writer.isOpened():
            cap.release()
            return None

        cap.set(cv2.CAP_PROP_POS_FRAMES, save_start)
        alpha = 0.35
        total_to_write = max(0, save_end - save_start + 1)
        progress = QProgressDialog(
            "Saving overlay video...",
            "",
            0,
            total_to_write,
            self
        )
        progress.setWindowTitle("Exporting Overlay Video")
        progress.setWindowModality(Qt.WindowModality.ApplicationModal)
        progress.setAutoClose(True)
        progress.setAutoReset(True)
        progress.setMinimumDuration(0)
        progress.setCancelButton(None)
        progress.setValue(0)
        QApplication.processEvents()

        try:
            written = 0
            for frame_idx in range(save_start, save_end + 1):
                ret, frame = cap.read()
                if not ret:
                    break

                frame_masks = all_masks.get(frame_idx, {})
                for obj_id, mask in frame_masks.items():
                    if mask is None or mask.max() == 0:
                        continue

                    if mask.shape[0] != video_height or mask.shape[1] != video_width:
                        mask = cv2.resize(
                            mask.astype(np.float32),
                            (video_width, video_height),
                            interpolation=cv2.INTER_NEAREST
                        ).astype(np.uint8)

                    idx = mask > 0
                    if not np.any(idx):
                        continue

                    color_rgb = get_obj_color(obj_id)
                    color_bgr = np.array([color_rgb[2], color_rgb[1], color_rgb[0]], dtype=np.float32)
                    frame_float = frame.astype(np.float32)
                    frame_float[idx] = frame_float[idx] * (1.0 - alpha) + color_bgr * alpha
                    frame = frame_float.astype(np.uint8)

                    ys, xs = np.where(idx)
                    if len(xs) > 0 and len(ys) > 0:
                        cx = int(np.mean(xs))
                        cy = int(np.mean(ys))
                        cv2.putText(
                            frame,
                            f"id:{obj_id}",
                            (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            (255, 255, 255),
                            1,
                            cv2.LINE_AA,
                        )

                writer.write(frame)
                written += 1
                if written % 5 == 0 or written == total_to_write:
                    progress.setValue(written)
                    QApplication.processEvents()
        finally:
            progress.setValue(total_to_write)
            writer.release()
            cap.release()

        return output_path
    
    def _save_masks(self):
        """Save masks in format compatible with animal_registration app."""
        if not self.video_path:
            return None
        
        import cv2
        import pickle
        
        # Merge incremental saves back into self.masks
        if hasattr(self, '_incremental_mask_file') and self._incremental_mask_file is not None:
            self._incremental_mask_file.close()
            try:
                with open(self._incremental_mask_file.name, 'rb') as f:
                    while True:
                        try:
                            chunk = pickle.load(f)
                            for frame_idx, frame_data in chunk.items():
                                if frame_idx not in self.masks:
                                    self.masks[frame_idx] = frame_data
                        except EOFError:
                            break
                # Clean up temp file
                os.unlink(self._incremental_mask_file.name)
            except Exception as e:
                logger.warning("Could not load incremental masks: %s", e)
            finally:
                self._incremental_mask_file = None
                self._incremental_frame_indices = []
        
        if not self.masks:
            return None
        
        # Get video dimensions
        cap = cv2.VideoCapture(self.video_path)
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        # Determine range to save (respect limit range if set)
        save_start = 0
        save_end = max(self.masks.keys()) if self.masks else 0
        if hasattr(self, "chk_limit_range") and self.chk_limit_range.isChecked():
            save_start = self.spin_start.value()
            save_end = self.spin_end.value()
        save_start = max(save_start, 0)
        save_end = max(save_end, save_start)
        
        frame_objects = []
        num_frames = (save_end - save_start + 1) if self.masks else 0
        
        for frame_idx_global in range(save_start, save_end + 1):
            frame_objs = []
            if frame_idx_global in self.masks:
                for obj_id, mask in self.masks[frame_idx_global].items():
                    if mask is not None and mask.max() > 0:
                        # Resize mask to video dimensions if needed
                        if mask.shape[0] != video_height or mask.shape[1] != video_width:
                            mask_resized = cv2.resize(
                                mask.astype(np.float32),
                                (video_width, video_height),
                                interpolation=cv2.INTER_NEAREST
                            ).astype(np.uint8)
                        else:
                            mask_resized = mask
                        
                        # Find bounding box
                        rows, cols = np.where(mask_resized > 0)
                        if len(rows) > 0 and len(cols) > 0:
                            y_min, y_max = np.min(rows), np.max(rows)
                            x_min, x_max = np.min(cols), np.max(cols)
                            
                            # Extract mask within bbox
                            bbox_mask = mask_resized[y_min:y_max+1, x_min:x_max+1]
                            
                            obj = {
                                'bbox': (int(x_min), int(y_min), int(x_max), int(y_max)),
                                'mask': bbox_mask.astype(bool),
                                'obj_id': int(obj_id)
                            }
                            frame_objs.append(obj)
            frame_objects.append(frame_objs)
        
        # Create mask data dictionary
        mask_data = {
            'video_path': self.video_path,
            'total_frames': num_frames,
            'height': video_height,
            'width': video_width,
            'fps': fps,
            'frame_objects': frame_objects,
            'objects_per_frame': [len(objs) for objs in frame_objects],
            'tracker': {},
            'format': 'new',
            'start_offset': save_start,
            'original_total_frames': self.total_frames
        }
        
        # Save to HDF5 file - use experiment folder if available
        experiment_path = self.config.get("experiment_path")
        if experiment_path and os.path.exists(experiment_path):
            masks_dir = os.path.join(experiment_path, "masks")
        else:
            from singlebehaviorlab._paths import USER_DATA_DIR
            masks_dir = str(USER_DATA_DIR / "data" / "masks")
        os.makedirs(masks_dir, exist_ok=True)
        
        video_basename = os.path.splitext(os.path.basename(self.video_path))[0]
        # Remove "_masks" suffix if present to avoid duplication
        if video_basename.endswith("_masks"):
            video_basename = video_basename[:-6]
        mask_path = os.path.join(masks_dir, f"{video_basename}.h5")

        from singlebehaviorlab.backend.video_processor import save_segmentation_data
        save_segmentation_data(mask_path, mask_data)
        return mask_path
    
    def _on_tracking_error(self, err):
        """Handle tracking error."""
        self.btn_track.setEnabled(True)
        self.btn_track_all.setEnabled(self.sam2_available and len(self.videos) > 0)
        self.btn_pause_tracking.setEnabled(False)
        self.btn_resume_tracking.setEnabled(False)
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Error", f"Tracking failed:\n{err}")
        self.inference_state = None
        # Stop batch mode on error
        self.batch_mode = False
        self.batch_queue = []
    
    def _open_settings(self):
        """Open settings dialog."""
        from PyQt6.QtWidgets import QDialog, QFormLayout
        
        dialog = QDialog(self)
        dialog.setWindowTitle("SAM2 Settings")
        dialog.resize(450, 550)
        layout = QFormLayout(dialog)
        
        spin_threshold = QDoubleSpinBox()
        spin_threshold.setRange(-10.0, 10.0)
        spin_threshold.setSingleStep(0.1)
        spin_threshold.setValue(self.mask_threshold)
        layout.addRow("Mask Threshold:", spin_threshold)
        
        spin_fill_hole = QSpinBox()
        spin_fill_hole.setRange(0, 10000)
        spin_fill_hole.setValue(self.fill_hole_area)
        layout.addRow("Fill Hole Area:", spin_fill_hole)
        
        chk_offload_video = QCheckBox()
        chk_offload_video.setChecked(self.offload_video)
        layout.addRow("Offload Video to CPU:", chk_offload_video)
        
        chk_offload_state = QCheckBox()
        chk_offload_state.setChecked(self.offload_state)
        layout.addRow("Offload State to CPU:", chk_offload_state)

        chk_bf16_autocast = QCheckBox()
        chk_bf16_autocast.setChecked(self.use_cuda_bf16_autocast)
        chk_bf16_autocast.setToolTip(
            "Use CUDA bfloat16 autocast for SAM2 inference.\n"
            "Usually speeds up segmentation on newer NVIDIA GPUs.\n"
            "Ignored on CPU."
        )
        layout.addRow("Use CUDA bf16 autocast:", chk_bf16_autocast)
        
        chk_memory_management = QCheckBox()
        chk_memory_management.setChecked(self.enable_memory_management)
        layout.addRow("Enable Memory Management:", chk_memory_management)

        chk_reseed_chunks = QCheckBox()
        chk_reseed_chunks.setToolTip("When processing in chunks, re-seed the next chunk with the last frame mask as a mask prompt.")
        chk_reseed_chunks.setChecked(getattr(self, "reseed_between_chunks", False))
        layout.addRow("Re-seed each chunk with last mask:", chk_reseed_chunks)
        
        layout.addRow(QLabel("<b>Motion-Aware Tracking</b>"))
        
        chk_motion_tracking = QCheckBox()
        chk_motion_tracking.setToolTip(
            "Enable motion-aware tracking:\n"
            "- Uses Kalman filter to predict object motion\n"
            "- Scores each frame by mask quality and motion consistency\n"
            "- Filters low-quality frames from memory to prevent drift\n"
            "Requires: pip install filterpy"
        )
        chk_motion_tracking.setChecked(getattr(self, "enable_motion_tracking", False))
        layout.addRow("Enable motion-aware tracking:", chk_motion_tracking)
        
        spin_motion_threshold = QDoubleSpinBox()
        spin_motion_threshold.setRange(0.0, 1.0)
        spin_motion_threshold.setSingleStep(0.05)
        spin_motion_threshold.setValue(getattr(self, "motion_score_threshold", 0.3))
        spin_motion_threshold.setToolTip(
            "Minimum score for a frame to be used in memory.\n"
            "Lower = more permissive, Higher = stricter filtering.\n"
            "Score combines mask confidence and motion IoU."
        )
        layout.addRow("Motion score threshold:", spin_motion_threshold)
        
        spin_consecutive_low = QSpinBox()
        spin_consecutive_low.setRange(1, 20)
        spin_consecutive_low.setValue(getattr(self, "motion_consecutive_low", 3))
        spin_consecutive_low.setToolTip(
            "Number of consecutive low-score frames before auto-correction.\n"
            "Lower = faster correction but more sensitive.\n"
            "Higher = more tolerant but slower to correct drift."
        )
        layout.addRow("Frames before auto-correct:", spin_consecutive_low)
        
        spin_area_threshold = QDoubleSpinBox()
        spin_area_threshold.setRange(0.1, 2.0)
        spin_area_threshold.setSingleStep(0.1)
        spin_area_threshold.setValue(getattr(self, "motion_area_threshold", 0.5))
        spin_area_threshold.setToolTip(
            "Max allowed mask area change ratio.\n"
            "0.5 = mask can shrink/grow by 50% max.\n"
            "Lower = stricter, Higher = more permissive."
        )
        layout.addRow("Area change tolerance:", spin_area_threshold)
        
        layout.addRow(QLabel("<b>OC-SORT Drift Correction</b>"))
        
        chk_ocsort = QCheckBox()
        chk_ocsort.setToolTip(
            "Enable OC-SORT enhancements for drift correction:\n\n"
            "-Virtual Trajectory: During occlusions, maintains tracking\n"
            "  using predicted motion (prevents state collapse)\n\n"
            "-ORU (Observation-Centric Re-Update): When object reappears,\n"
            "  corrects accumulated drift by re-estimating past states\n\n"
            "Based on: 'Observation-Centric SORT' (arXiv:2203.14360)"
        )
        chk_ocsort.setChecked(getattr(self, "enable_ocsort", False))
        layout.addRow("Enable OC-SORT drift correction:", chk_ocsort)
        
        spin_ocsort_inertia = QDoubleSpinBox()
        spin_ocsort_inertia.setRange(0.0, 1.0)
        spin_ocsort_inertia.setSingleStep(0.05)
        spin_ocsort_inertia.setValue(getattr(self, "ocsort_inertia", 0.2))
        spin_ocsort_inertia.setToolTip(
            "Velocity smoothing factor for ORU (paper default: 0.2).\n\n"
            "When object reappears after occlusion, this blends\n"
            "old velocity with newly computed velocity:\n"
            "  smoothed = inertia * old_vel + (1-inertia) * new_vel\n\n"
            "Higher = more momentum, smoother but slower correction.\n"
            "Lower = faster correction but may be jerky."
        )
        layout.addRow("ORU inertia (velocity smoothing):", spin_ocsort_inertia)
        
        btn_ok = QPushButton("OK")
        btn_ok.clicked.connect(dialog.accept)
        layout.addRow(btn_ok)
        
        if dialog.exec():
            self.mask_threshold = spin_threshold.value()
            self.fill_hole_area = spin_fill_hole.value()
            self.offload_video = chk_offload_video.isChecked()
            self.offload_state = chk_offload_state.isChecked()
            self.use_cuda_bf16_autocast = chk_bf16_autocast.isChecked()
            self.enable_memory_management = chk_memory_management.isChecked()
            self.reseed_between_chunks = chk_reseed_chunks.isChecked()
            self.enable_motion_tracking = chk_motion_tracking.isChecked()
            self.motion_score_threshold = spin_motion_threshold.value()
            self.motion_consecutive_low = spin_consecutive_low.value()
            self.motion_area_threshold = spin_area_threshold.value()
            self.enable_ocsort = chk_ocsort.isChecked()
            self.ocsort_inertia = spin_ocsort_inertia.value()
            
            if self.predictor:
                self.predictor.fill_hole_area = self.fill_hole_area
    
    def update_config(self, config: dict):
        """Update configuration."""
        self.config = config

