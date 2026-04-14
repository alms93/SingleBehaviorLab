"""Inference worker thread: runs model on video(s) and emits results."""
from PyQt6.QtCore import QThread, pyqtSignal
import cv2
import os
import torch
import numpy as np


def _sanitize_bbox_coords(x1, y1, x2, y2, crop_padding=0.35, crop_min_size=0.04):
    """Pad bbox proportionally to its own size (matches training _sanitize_bboxes)."""
    x1, y1 = max(0.0, min(1.0, x1)), max(0.0, min(1.0, y1))
    x2, y2 = max(0.0, min(1.0, x2)), max(0.0, min(1.0, y2))
    cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    w = min(max(abs(x2 - x1), crop_min_size) * (1.0 + 2.0 * crop_padding), 1.0)
    h = min(max(abs(y2 - y1), crop_min_size) * (1.0 + 2.0 * crop_padding), 1.0)
    x1, y1 = max(0.0, cx - 0.5 * w), max(0.0, cy - 0.5 * h)
    x2, y2 = min(1.0, cx + 0.5 * w), min(1.0, cy + 0.5 * h)
    x2, y2 = max(x2, x1 + crop_min_size), max(y2, y1 + crop_min_size)
    return (max(0.0, min(1.0, x1)), max(0.0, min(1.0, y1)),
            max(0.0, min(1.0, x2)), max(0.0, min(1.0, y2)))


class InferenceWorker(QThread):
    """Worker thread for running inference."""
    progress = pyqtSignal(int, int)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    log_message = pyqtSignal(str)
    video_done = pyqtSignal(str, dict)
    
    def __init__(
        self,
        model,
        video_paths,
        target_fps,
        clip_length,
        step_frames,
        resolution=288,
        classes=None,
        use_localization_pipeline=False,
        crop_padding=0.35,
        crop_min_size=0.04,
        use_ovr=False,
        ovr_temperatures=None,
        collect_attention=False,
        sample_ranges_by_video=None,
    ):
        super().__init__()
        self.model = model
        self.video_paths = video_paths if isinstance(video_paths, list) else [video_paths]
        self.target_fps = target_fps
        self.clip_length = clip_length
        self.step_frames = step_frames
        self.resolution = int(resolution)
        self.classes = classes or []
        self.use_localization_pipeline = bool(use_localization_pipeline)
        self.crop_padding = float(crop_padding)
        self.crop_min_size = float(crop_min_size)
        self.use_ovr = bool(use_ovr)
        self._ovr_temperatures = dict(ovr_temperatures or {})
        self.collect_attention = bool(collect_attention)
        self.sample_ranges_by_video = dict(sample_ranges_by_video or {})
        self.should_stop = False
    
    def _apply_ovr_temperature(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale logits by per-class calibrated temperature before sigmoid.

        logits: [..., C] tensor.  Divides each class dimension by its
        temperature (default 1.0 = no change).
        """
        if not self._ovr_temperatures or not self.classes:
            return logits
        C = logits.shape[-1]
        temps = torch.ones(C, device=logits.device, dtype=logits.dtype)
        for ci, cls_name in enumerate(self.classes):
            if ci < C and cls_name in self._ovr_temperatures:
                t = float(self._ovr_temperatures[cls_name])
                temps[ci] = max(t, 0.01)
        return logits / temps

    def stop(self):
        self.should_stop = True

    def _build_center_merge_weights(self, length: int) -> np.ndarray:
        """Center-heavy temporal weights for overlap aggregation.

        Clip-edge predictions tend to be less reliable because they have less
        temporal context. Use a Hann-shaped profile with a small floor so edge
        frames still contribute when only one clip covers them.
        """
        if length <= 1:
            return np.ones((max(1, length),), dtype=np.float32)
        w = np.hanning(length).astype(np.float32)
        if not np.any(w > 0):
            return np.ones((length,), dtype=np.float32)
        return np.clip(0.1 + 0.9 * w, 1e-3, None).astype(np.float32)
    
    def _iter_clips_prefetched(self, video_path, target_fps, clip_length, step_frames, chunk_size=64):
        from threading import Thread
        from queue import Queue
        q = Queue(maxsize=2)

        def _producer():
            try:
                for chunk in self._iter_clips_in_chunks(video_path, target_fps, clip_length, step_frames, chunk_size):
                    q.put(chunk)
                    if self.should_stop:
                        break
            finally:
                q.put(None)

        t = Thread(target=_producer, daemon=True)
        t.start()
        while True:
            item = q.get()
            if item is None:
                break
            yield item
        t.join(timeout=2.0)

    def _normalize_sample_ranges(self, sample_ranges, total_frames):
        """Clamp and sort sample ranges as [(start, end), ...] with end-exclusive bounds."""
        if not sample_ranges:
            return []
        normalized = []
        for rng in sample_ranges:
            if not isinstance(rng, (list, tuple)) or len(rng) != 2:
                continue
            start, end = int(rng[0]), int(rng[1])
            start = max(0, min(total_frames, start))
            end = max(start, min(total_frames, end))
            if end > start:
                normalized.append((start, end))
        normalized.sort()
        merged = []
        for start, end in normalized:
            if not merged or start > merged[-1][1]:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)
        return [(start, end) for start, end in merged]

    def _estimate_subsampled_frames_in_ranges(self, sample_ranges, frame_interval):
        total = 0
        for start, end in sample_ranges:
            first = start if start % frame_interval == 0 else start + (frame_interval - (start % frame_interval))
            if first >= end:
                continue
            total += 1 + ((end - 1 - first) // frame_interval)
        return total

    def _iter_clips_in_chunks(self, video_path, target_fps, clip_length, step_frames, chunk_size=64):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        if orig_fps <= 0:
            orig_fps = 30.0
        
        frame_interval = max(1, int(round(orig_fps / target_fps)))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_ranges = self._normalize_sample_ranges(
            self.sample_ranges_by_video.get(video_path),
            total_frames,
        )
        if sample_ranges:
            total_subsampled = self._estimate_subsampled_frames_in_ranges(sample_ranges, frame_interval)
        else:
            total_subsampled = total_frames // frame_interval
        if total_subsampled < clip_length:
            total_clips_estimate = 0
        elif step_frames <= 0:
            total_clips_estimate = max(1, total_subsampled // max(1, clip_length))
        else:
            total_clips_estimate = 1 + max(0, (total_subsampled - clip_length) // step_frames)
        total_clips_estimate = max(1, total_clips_estimate)

        clip_starts = []
        frames_buffer = []
        frames_buffer_fullres = []
        keep_fullres_cache = self.use_localization_pipeline and getattr(self.model, "use_localization", False)
        chunk_clips = []
        chunk_starts = []
        chunk_fullres = []
        clip_idx = 0

        def _iter_ranges():
            if sample_ranges:
                for start, end in sample_ranges:
                    yield start, end
            else:
                yield 0, total_frames

        for range_start, range_end in _iter_ranges():
            if self.should_stop:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, range_start)
            frames_buffer = []
            frames_buffer_fullres = []
            skip_remaining = 0
            frame_idx = range_start

            while frame_idx < range_end:
                if self.should_stop:
                    break
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % frame_interval == 0:
                    if skip_remaining > 0:
                        skip_remaining -= 1
                        frame_idx += 1
                        continue
                    if keep_fullres_cache:
                        frames_buffer_fullres.append(frame.copy())
                    h_src, w_src = frame.shape[:2]
                    is_upscale = (w_src < self.resolution) or (h_src < self.resolution)
                    interp = cv2.INTER_LANCZOS4 if is_upscale else cv2.INTER_AREA
                    frame_resized = cv2.resize(frame, (self.resolution, self.resolution), interpolation=interp)
                    if is_upscale:
                        blurred = cv2.GaussianBlur(frame_resized, (0, 0), sigmaX=1.0)
                        frame_resized = cv2.addWeighted(frame_resized, 1.5, blurred, -0.5, 0)
                    frame_resized = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                    frames_buffer.append(frame_resized)
                    if len(frames_buffer) == clip_length:
                        clip_array = np.stack(frames_buffer).astype(np.float32) / 255.0
                        clip_start = frame_idx - (clip_length - 1) * frame_interval
                        clip_starts.append(clip_start)
                        chunk_clips.append(clip_array)
                        chunk_starts.append(clip_start)
                        if keep_fullres_cache:
                            chunk_fullres.append(np.stack(frames_buffer_fullres))
                        clip_idx += 1
                        if self.progress:
                            self.progress.emit(clip_idx, total_clips_estimate)
                        if step_frames < clip_length:
                            frames_buffer = frames_buffer[step_frames:]
                            if keep_fullres_cache:
                                frames_buffer_fullres = frames_buffer_fullres[step_frames:]
                        else:
                            frames_buffer = []
                            frames_buffer_fullres = []
                            skip_remaining = max(0, step_frames - clip_length)
                        if len(chunk_clips) >= max(1, int(chunk_size)):
                            yield chunk_clips, chunk_starts, chunk_fullres
                            chunk_clips = []
                            chunk_starts = []
                            chunk_fullres = []
                frame_idx += 1
        cap.release()
        if chunk_clips:
            yield chunk_clips, chunk_starts, chunk_fullres

    def _predict_localization(self, clips, device, batch_size=4):
        loc_bboxes = []
        for i in range(0, len(clips), batch_size):
            if self.should_stop:
                break
            batch_clips = clips[i:i+batch_size]
            batch_tensors = []
            for clip in batch_clips:
                clip_tensor = torch.from_numpy(clip).permute(0, 3, 1, 2)
                batch_tensors.append(clip_tensor)
            batch = torch.stack(batch_tensors).to(device)
            with torch.no_grad():
                out = self.model(batch, return_localization=True)
            del batch
            loc_part = None
            if isinstance(out, tuple) and len(out) >= 2 and torch.is_tensor(out[-1]) and out[-1].shape[-1] == 4:
                loc_part = out[-1]
            if loc_part is None:
                loc_part = torch.tensor([[0.0, 0.0, 1.0, 1.0]] * len(batch_clips), device=device)
            loc_np = loc_part.detach().cpu().numpy()
            if loc_part.dim() == 3:
                loc_bboxes.extend(loc_np.tolist())
            else:
                loc_bboxes.extend(loc_np.tolist())
        return loc_bboxes

    def _build_refined_clips(self, fullres_clip_frames, loc_bboxes):
        refined = []
        for idx, frames_bgr in enumerate(fullres_clip_frames):
            bbox = loc_bboxes[idx] if idx < len(loc_bboxes) else [0.0, 0.0, 1.0, 1.0]
            has_temporal_boxes = (
                isinstance(bbox, (list, tuple))
                and len(bbox) > 0
                and isinstance(bbox[0], (list, tuple))
                and len(bbox[0]) == 4
            )
            if has_temporal_boxes:
                b0 = [float(v) for v in bbox[0]]
                x1_fixed, y1_fixed, x2_fixed, y2_fixed = _sanitize_bbox_coords(*b0, self.crop_padding, self.crop_min_size)
            clip_frames = []
            for fi, frame_bgr in enumerate(frames_bgr):
                if has_temporal_boxes:
                    x1, y1, x2, y2 = x1_fixed, y1_fixed, x2_fixed, y2_fixed
                else:
                    x1, y1, x2, y2 = _sanitize_bbox_coords(*[float(v) for v in bbox], self.crop_padding, self.crop_min_size)
                h, w = frame_bgr.shape[:2]
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
                # Match training: BGR→RGB, [0,1] float, then PyTorch bilinear resize
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                crop_f = torch.from_numpy(crop_rgb.astype(np.float32) / 255.0)
                crop_f = crop_f.permute(2, 0, 1).unsqueeze(0)  # [1,C,H,W]
                crop_f = torch.nn.functional.interpolate(
                    crop_f, size=(self.resolution, self.resolution),
                    mode='bilinear', align_corners=False,
                )
                clip_frames.append(crop_f.squeeze(0))  # [C,H,W]
            if not clip_frames:
                clip_frames = [torch.zeros(3, self.resolution, self.resolution)] * self.clip_length
            while len(clip_frames) < self.clip_length:
                clip_frames.append(clip_frames[-1].clone())
            clip_frames = clip_frames[:self.clip_length]
            # Stack to [T,C,H,W] then back to [T,H,W,C] numpy for the existing pipeline
            clip_tensor = torch.stack(clip_frames)  # [T,C,H,W]
            clip_np = clip_tensor.permute(0, 2, 3, 1).numpy()  # [T,H,W,C]
            refined.append(clip_np)
        return refined

    def _is_cuda_oom(self, err: Exception) -> bool:
        if not isinstance(err, RuntimeError):
            return False
        msg = str(err).lower()
        return ("out of memory" in msg) and ("cuda" in msg or "cublas" in msg or "cudnn" in msg)

    def _cleanup_memory(self, device=None):
        import gc
        gc.collect()
        if device is not None and getattr(device, "type", None) == "cuda" and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        try:
            jax.clear_caches()
        except Exception:
            pass
    
    def run(self):
        import traceback
        try:
            def log_fn(msg):
                self.log_message.emit(msg)
            results = {}
            total_videos = len(self.video_paths)
            for v_idx, video_path in enumerate(self.video_paths):
                if self.should_stop:
                    break
                log_fn(f"Processing video {v_idx+1}/{total_videos}: {os.path.basename(video_path)}")
                log_fn("Extracting clips and running inference in memory-safe chunks...")
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                clip_frame_embeddings = None
                aggregated_frame_embs = None
                try:
                    self.model.to(device)
                    self.model.eval()
                    predictions = []
                    confidences = []
                    clip_probabilities = []
                    clip_frame_probabilities = []
                    clip_frame_logits = []
                    clip_frame_embeddings = []
                    clip_attention_maps = []
                    clip_starts = []
                    localization_bboxes = []
                    has_frame_head = getattr(self.model, "use_frame_head", True)
                    if self.use_localization_pipeline and getattr(self.model, "use_localization", False):
                        log_fn("Localization pipeline enabled")
                    resolution = getattr(self.model.backbone, 'resolution', 288)
                    if device.type == "cuda":
                        try:
                            free_mb = torch.cuda.mem_get_info(device)[0] / (1024 * 1024)
                        except Exception:
                            free_mb = 4000
                        pixels = resolution * resolution
                        # Scale relative to 288x288 baseline (~250 MB/clip including activations)
                        scale = pixels / (288 * 288)
                        est_per_clip_mb = 250 * scale
                        batch_size = max(1, int(free_mb * 0.5 / est_per_clip_mb))
                        batch_size = min(batch_size, 32)
                        if resolution > 300:
                            batch_size = min(batch_size, 16)
                        if resolution > 400:
                            batch_size = min(batch_size, 4)
                    else:
                        batch_size = 2
                    log_fn(f"Inference batch size: {batch_size} (resolution={resolution}, device={device})")
                    cap = cv2.VideoCapture(video_path)
                    orig_fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()
                    if orig_fps <= 0:
                        orig_fps = 30.0
                    frame_interval = max(1, int(round(orig_fps / self.target_fps)))
                    sample_ranges = self._normalize_sample_ranges(
                        self.sample_ranges_by_video.get(video_path),
                        frame_count,
                    )
                    if sample_ranges:
                        sample_desc = ", ".join(
                            f"{start/orig_fps:.1f}s-{end/orig_fps:.1f}s" for start, end in sample_ranges[:6]
                        )
                        if len(sample_ranges) > 6:
                            sample_desc += ", ..."
                        log_fn(
                            f"Quick-check inference on {len(sample_ranges)} sample range(s): {sample_desc}"
                        )
                    chunk_idx = 0
                    for chunk_clips, chunk_starts, chunk_fullres in self._iter_clips_prefetched(
                        video_path, self.target_fps, self.clip_length, self.step_frames, chunk_size=64,
                    ):
                        if self.should_stop:
                            break
                        chunk_idx += 1
                        clip_starts.extend(chunk_starts)
                        work_clips = chunk_clips
                        used_refined_roi = False
                        if self.use_localization_pipeline and getattr(self.model, "use_localization", False):
                            loc_input = chunk_clips
                            loc_bboxes = self._predict_localization(loc_input, device, batch_size=batch_size)
                            if len(loc_bboxes) == len(chunk_clips):
                                localization_bboxes.extend(loc_bboxes)
                            else:
                                localization_bboxes.extend([[0.0, 0.0, 1.0, 1.0]] * len(chunk_clips))
                            if len(loc_bboxes) == len(chunk_clips) and chunk_fullres:
                                work_clips = self._build_refined_clips(chunk_fullres, loc_bboxes)
                                used_refined_roi = True
                        for i in range(0, len(work_clips), batch_size):
                            if self.should_stop:
                                break
                            batch_clips = work_clips[i:i+batch_size]
                            batch_tensors = []
                            for clip in batch_clips:
                                clip_tensor = torch.from_numpy(clip)
                                clip_tensor = clip_tensor.permute(0, 3, 1, 2)
                                batch_tensors.append(clip_tensor)
                            batch = torch.stack(batch_tensors).to(device)
                            with torch.no_grad():
                                logits = self.model(batch, return_frame_logits=has_frame_head,
                                                    return_attn_weights=self.collect_attention)
                            batch_frame_probs = None
                            batch_frame_logits = None
                            batch_frame_embs = None
                            batch_attn_maps = None
                            _fo = getattr(self.model, '_frame_output', None)
                            if _fo is not None:
                                f_logits = _fo[0]
                                batch_frame_logits = f_logits.detach().cpu().numpy()
                                if self.use_ovr:
                                    batch_frame_probs = torch.sigmoid(self._apply_ovr_temperature(f_logits)).detach().cpu().numpy()
                                else:
                                    batch_frame_probs = torch.softmax(f_logits, dim=-1).detach().cpu().numpy()
                                _emb_src = (_fo[7] if len(_fo) > 7 and _fo[7] is not None else
                                            (_fo[6] if len(_fo) > 6 and _fo[6] is not None else None))
                                if _emb_src is not None:
                                    batch_frame_embs = _emb_src.detach().cpu().numpy()
                                if self.collect_attention and len(_fo) > 8 and _fo[8] is not None:
                                    batch_attn_maps = _fo[8].detach().cpu().numpy()
                            if batch_frame_probs is not None:
                                for b_i in range(batch_frame_probs.shape[0]):
                                    clip_frame_probabilities.append(batch_frame_probs[b_i].tolist())
                                    if batch_frame_logits is not None:
                                        clip_frame_logits.append(batch_frame_logits[b_i])
                                    else:
                                        clip_frame_logits.append(None)
                                    if batch_frame_embs is not None:
                                        clip_frame_embeddings.append(batch_frame_embs[b_i])
                                    else:
                                        clip_frame_embeddings.append(None)
                                    if batch_attn_maps is not None:
                                        clip_attention_maps.append(batch_attn_maps[b_i])
                                    elif self.collect_attention:
                                        clip_attention_maps.append(None)
                            else:
                                for b_i in range(len(batch_clips)):
                                    clip_frame_probabilities.append(None)
                                    clip_frame_logits.append(None)
                                    clip_frame_embeddings.append(None)
                                    if self.collect_attention:
                                        clip_attention_maps.append(None)
                            del batch
                            if isinstance(logits, tuple):
                                logits = logits[0]
                            if self.use_ovr:
                                probs = torch.sigmoid(self._apply_ovr_temperature(logits))
                            else:
                                probs = torch.softmax(logits, dim=1)
                            preds = torch.argmax(probs, dim=1)
                            confs = torch.max(probs, dim=1)[0]
                            predictions.extend(preds.cpu().numpy().tolist())
                            confidences.extend(confs.cpu().numpy().tolist())
                            clip_probabilities.extend(probs.detach().cpu().numpy().tolist())
                            del logits, probs, preds, confs
                        self._cleanup_memory(device)
                        if chunk_idx % 5 == 0:
                            log_fn(f"Processed {len(predictions)} clips so far...")
                    if not predictions:
                        log_fn(f"Warning: No clips extracted from {os.path.basename(video_path)}")
                        continue
                    if clip_frame_probabilities:
                        sample = clip_frame_probabilities[:min(3, len(clip_frame_probabilities))]
                        for ci, fp in enumerate(sample):
                            arr = np.asarray(fp, dtype=np.float32)
                            if arr.ndim == 2:
                                means = arr.mean(axis=0)
                                maxs = arr.max(axis=0)
                                log_fn(f"  Clip {ci} frame probs — mean: {np.array2string(means, precision=3)}, max: {np.array2string(maxs, precision=3)}")
                    cap_info = cv2.VideoCapture(video_path)
                    video_total_frames = int(cap_info.get(cv2.CAP_PROP_FRAME_COUNT))
                    video_orig_fps = cap_info.get(cv2.CAP_PROP_FPS)
                    if video_orig_fps <= 0:
                        video_orig_fps = 30.0
                    cap_info.release()
                    video_frame_interval = max(1, int(round(video_orig_fps / max(1e-6, float(self.target_fps)))))
                    aggregated_frame_probs = None
                    aggregated_frame_logits = None
                    aggregated_frame_embs = None
                    has_embeddings = any(e is not None for e in clip_frame_embeddings)
                    has_frame_logits = any(fl is not None for fl in clip_frame_logits)
                    embed_dim = clip_frame_embeddings[0].shape[-1] if has_embeddings and clip_frame_embeddings[0] is not None else 0
                    if clip_frame_probabilities and any(p is not None for p in clip_frame_probabilities):
                        log_fn("Aggregating per-frame outputs (center/confidence-weighted overlap merge)...")
                        num_classes = len(self.classes)
                        agg_probs = np.zeros((frame_count, num_classes), dtype=np.float32)
                        agg_logits = np.zeros((frame_count, num_classes), dtype=np.float32) if has_frame_logits else None
                        agg_counts = np.zeros((frame_count, 1), dtype=np.float32)
                        if has_embeddings and embed_dim > 0:
                            agg_embs = np.zeros((frame_count, embed_dim), dtype=np.float32)
                        else:
                            agg_embs = None
                        for i, probs in enumerate(clip_frame_probabilities):
                            if probs is None:
                                continue
                            if i >= len(clip_starts):
                                break
                            start_f = clip_starts[i]
                            probs_arr = np.clip(np.array(probs, dtype=np.float32), 0.0, None)
                            if probs_arr.ndim != 2 or probs_arr.shape[1] != num_classes:
                                continue
                            T = int(probs_arr.shape[0])
                            merge_w = self._build_center_merge_weights(T)
                            # Confidence-weight each frame's contribution, but keep a
                            # floor so uncertain clips still add a little evidence.
                            frame_conf = np.clip(np.max(probs_arr, axis=1), 0.0, 1.0)
                            conf_w = np.clip(0.1 + 0.9 * frame_conf, 0.1, 1.0).astype(np.float32)
                            logits_arr = None
                            emb_arr = None
                            if agg_logits is not None and i < len(clip_frame_logits) and clip_frame_logits[i] is not None:
                                logits_arr = np.array(clip_frame_logits[i], dtype=np.float32)
                            if agg_embs is not None and clip_frame_embeddings[i] is not None:
                                emb_arr = np.array(clip_frame_embeddings[i], dtype=np.float32)
                            for t in range(T):
                                f_start = start_f + t * frame_interval
                                f_end = min(f_start + frame_interval, frame_count)
                                if f_start >= frame_count:
                                    break
                                if f_end <= f_start:
                                    continue
                                w = float(merge_w[t] * conf_w[t])
                                agg_probs[f_start:f_end] += probs_arr[t][np.newaxis, :] * w
                                agg_counts[f_start:f_end] += w
                                if logits_arr is not None and t < logits_arr.shape[0]:
                                    agg_logits[f_start:f_end] += logits_arr[t][np.newaxis, :] * w
                                if emb_arr is not None and t < emb_arr.shape[0]:
                                    agg_embs[f_start:f_end] += emb_arr[t][np.newaxis, :] * w
                        agg_probs = agg_probs / np.maximum(agg_counts, 1.0)
                        if agg_logits is not None:
                            agg_logits = agg_logits / np.maximum(agg_counts, 1.0)
                            aggregated_frame_logits = agg_logits
                        if agg_embs is not None:
                            agg_embs = agg_embs / np.maximum(agg_counts, 1.0)
                            aggregated_frame_embs = agg_embs
                        if not self.use_ovr:
                            covered_mask = agg_counts.squeeze(-1) > 0
                            row_sums = agg_probs[covered_mask].sum(axis=1, keepdims=True)
                            safe_sums = np.maximum(row_sums, 1e-8)
                            agg_probs[covered_mask] = agg_probs[covered_mask] / safe_sums
                        aggregated_frame_probs = agg_probs
                    clip_frame_embeddings = None
                    res_entry = {
                        "predictions": predictions,
                        "confidences": confidences,
                        "clip_probabilities": clip_probabilities,
                        "clip_starts": clip_starts,
                        "total_frames": video_total_frames,
                        "orig_fps": video_orig_fps,
                        "frame_interval": video_frame_interval,
                        "aggregated_frame_probs": aggregated_frame_probs,
                    }
                    if sample_ranges:
                        res_entry["sample_ranges"] = sample_ranges
                    if clip_frame_probabilities:
                        res_entry["clip_frame_probabilities"] = clip_frame_probabilities
                    if localization_bboxes:
                        res_entry["localization_bboxes"] = localization_bboxes
                    if self.collect_attention and clip_attention_maps:
                        res_entry["clip_attention_maps"] = clip_attention_maps
                    results[video_path] = res_entry
                    self.video_done.emit(video_path, res_entry)
                except RuntimeError as video_err:
                    if self._is_cuda_oom(video_err):
                        log_fn(
                            f"CUDA OOM while processing {os.path.basename(video_path)}. "
                            f"Skipping this video and continuing."
                        )
                    else:
                        log_fn(
                            f"Error processing {os.path.basename(video_path)}: {video_err}\n"
                            f"{traceback.format_exc()}"
                        )
                    continue
                except Exception as video_err:
                    log_fn(
                        f"Error processing {os.path.basename(video_path)}: {video_err}\n"
                        f"{traceback.format_exc()}"
                    )
                    continue
                finally:
                    clip_frame_embeddings = None
                    aggregated_frame_embs = None
                    self._cleanup_memory(device)
            if self.should_stop:
                if results:
                    log_fn("Inference stopped by user. Keeping results for completed videos.")
                else:
                    log_fn("Inference stopped by user. No videos were completed.")
            if not results:
                if not self.should_stop:
                    self.error.emit("No results generated.")
                else:
                    self.finished.emit({})
                return
            if not self.should_stop:
                log_fn("Inference complete!")
            self.finished.emit(results)
        except Exception as e:
            error_msg = f"Inference failed: {str(e)}\n{traceback.format_exc()}"
            self.log_message.emit(f"ERROR: {error_msg}")
            self.error.emit(error_msg)
