import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from typing import Callable, Optional, Dict, Any
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from collections import Counter
import random
import math
import re

logger = logging.getLogger(__name__)


def _slugify_class_name(name: str) -> str:
    """Create a filesystem/column-safe slug for class names."""
    slug = re.sub(r'[^0-9a-zA-Z]+', '_', name).strip('_').lower()
    return slug or "class"


class FocalLoss(nn.Module):
    """
    Focal Loss for dense object detection and classification.
    focuses training on hard examples and down-weights easy ones.
    Loss = -alpha * (1 - pt)^gamma * log(pt)
    """
    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: [B, C] logits (not probabilities)
            targets: [B] labels (use -100 to ignore)
        """
        valid = targets >= 0
        ce_loss = F.cross_entropy(inputs, targets.clamp(min=0), reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        focal_loss = focal_loss * valid.float()

        if self.reduction == 'mean':
            n = valid.sum().clamp(min=1)
            return focal_loss.sum() / n
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class BinaryFocalLoss(nn.Module):
    """Per-element binary focal loss for OvR heads."""
    def __init__(self, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor,
                weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            inputs: [B, C] logits
            targets: [B, C] binary targets (0 or 1)
            weight: optional [B, C] per-element weight
        """
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce)
        focal = ((1 - pt) ** self.gamma) * bce
        if weight is not None:
            focal = focal * weight
        if self.reduction == 'mean':
            return focal.mean()
        elif self.reduction == 'sum':
            return focal.sum()
        return focal


class AsymmetricLoss(nn.Module):
    """Asymmetric Loss for multi-label / OvR classification.

    Applies focal-style down-weighting only to *negatives* (gamma_neg),
    while keeping full gradient signal from positives (gamma_pos, default 0).
    Optionally applies probability shifting (clip) to hard-threshold easy
    negatives, further reducing their contribution.

    Reference: Ridnik et al., "Asymmetric Loss For Multi-Label Classification"
    (https://arxiv.org/abs/2009.14119)
    """

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 0.0,
        clip: float = 0.05,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.reduction = reduction

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            inputs: [*, C] raw logits
            targets: [*, C] binary targets in [0, 1]
            weight: optional [*, C] per-element weight
        """
        p = torch.sigmoid(inputs)
        
        # Derive hard targets to prevent ASL from over-penalizing easy negatives under label smoothing.
        hard_targets = (targets >= 0.5).float()

        pos_part = hard_targets * torch.log(p.clamp(min=1e-8))
        neg_p = 1.0 - p

        # Probability shifting: suppress easy negatives to reduce their gradient contribution
        if self.clip > 0:
            neg_p = (neg_p + self.clip).clamp(max=1.0)

        neg_part = (1.0 - hard_targets) * torch.log(neg_p.clamp(min=1e-8))

        # Asymmetric focusing
        if self.gamma_pos > 0:
            pos_part = pos_part * ((1.0 - p) ** self.gamma_pos)
        if self.gamma_neg > 0:
            neg_part = neg_part * (p ** self.gamma_neg)

        loss = -(pos_part + neg_part)

        if weight is not None:
            loss = loss * weight
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss



class BehaviorDataset(Dataset):
    """PyTorch Dataset for behavior clips."""
    
    def __init__(
        self,
        clips: list,
        annotation_manager,
        classes: list,
        clip_base_dir: str,
        transform: Optional[Callable] = None,
        target_size: tuple[int, int] = (288, 288),
        clip_length: int = 16,
        virtual_size_multiplier: int = 1,
        grid_size: int = 0,
        stitch_prob: float = 0.0,
        crop_jitter: bool = False,
        crop_jitter_strength: float = 0.15,
        ovr_background_classes: Optional[list[str]] = None,
    ):
        self.clips = clips
        self.annotation_manager = annotation_manager
        self.raw_classes = classes # Original labels
        self.clip_base_dir = clip_base_dir
        self.transform = transform
        self.target_size = target_size
        self.clip_length = clip_length
        self.virtual_size_multiplier = virtual_size_multiplier
        self.grid_size = grid_size  # Spatial grid size (e.g. 16 for 288px). 0 = disabled.
        # Masks stored at 16×16 reference grid; rescaled to training grid_size if different.
        STORED_GRID = 16
        self.spatial_masks = []
        self.spatial_bboxes = []
        self.spatial_bbox_valid = []
        num_patches = grid_size * grid_size if grid_size > 0 else 0
        for clip in clips:
            clip_id = clip["id"]
            mask_indices = clip.get("spatial_mask")
            if mask_indices and grid_size > 0:
                # Build mask at stored 16×16 resolution
                ref_mask = torch.zeros(STORED_GRID, STORED_GRID, dtype=torch.float32)
                for idx in mask_indices:
                    if 0 <= idx < STORED_GRID * STORED_GRID:
                        row, col = divmod(idx, STORED_GRID)
                        ref_mask[row, col] = 1.0
                
                if grid_size != STORED_GRID:
                    # Rescale to training grid size via bilinear interpolation
                    ref_4d = ref_mask.unsqueeze(0).unsqueeze(0)  # [1,1,16,16]
                    scaled = torch.nn.functional.interpolate(
                        ref_4d, size=(grid_size, grid_size), mode='bilinear', align_corners=False
                    )
                    mask = (scaled.squeeze() > 0.25).float()  # threshold to keep binary
                else:
                    mask = ref_mask
                
                self.spatial_masks.append(mask.reshape(-1))  # flatten to [G*G]
            else:
                self.spatial_masks.append(torch.zeros(max(1, num_patches), dtype=torch.float32))

            # Per-frame bboxes: [T, 4] and [T] validity
            T = self.clip_length
            bbox_frames_data = clip.get("spatial_bbox_frames")
            if bbox_frames_data and isinstance(bbox_frames_data, (list, tuple)):
                frame_boxes = torch.zeros(T, 4, dtype=torch.float32)
                frame_valid = torch.zeros(T, dtype=torch.float32)
                for fi in range(min(T, len(bbox_frames_data))):
                    b = bbox_frames_data[fi]
                    if b and isinstance(b, (list, tuple)) and len(b) == 4:
                        x1, y1, x2, y2 = [max(0.0, min(1.0, float(v))) for v in b]
                        if x2 > x1 and y2 > y1:
                            frame_boxes[fi] = torch.tensor([x1, y1, x2, y2])
                            frame_valid[fi] = 1.0
                self.spatial_bboxes.append(frame_boxes)
                self.spatial_bbox_valid.append(frame_valid)
            else:
                # Legacy single bbox: replicate to all frames
                bbox = clip.get("spatial_bbox")
                if bbox and isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                    x1, y1, x2, y2 = [max(0.0, min(1.0, float(v))) for v in bbox]
                    if x2 > x1 and y2 > y1:
                        single = torch.tensor([x1, y1, x2, y2], dtype=torch.float32)
                        self.spatial_bboxes.append(single.unsqueeze(0).expand(T, -1).clone())
                        # Mark only frame 0 valid: legacy single-frame annotations shouldn't penalize per-frame tracking.
                        legacy_valid = torch.zeros(T, dtype=torch.float32)
                        legacy_valid[0] = 1.0
                        self.spatial_bbox_valid.append(legacy_valid)
                    else:
                        self.spatial_bboxes.append(torch.zeros(T, 4, dtype=torch.float32))
                        self.spatial_bbox_valid.append(torch.zeros(T, dtype=torch.float32))
                else:
                    self.spatial_bboxes.append(torch.zeros(T, 4, dtype=torch.float32))
                    self.spatial_bbox_valid.append(torch.zeros(T, dtype=torch.float32))
        
        self.classes = classes
        self.attributes = []
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.attr_to_idx = {}
        self.ovr_background_classes = set(ovr_background_classes or [])
        # Primary label index per clip (-1 if not in class list, e.g. near_negative_*).
        self.labels = []
        # Multi-label: list of all label indices per clip (for OvR multi-hot targets).
        self.multi_labels = []
        for clip in clips:
            clip_labels = clip.get("labels")
            if not isinstance(clip_labels, list) or not clip_labels:
                clip_labels = [clip.get("label", "")]
            indices = [self.class_to_idx[l] for l in clip_labels if l in self.class_to_idx]
            if indices:
                self.labels.append(indices[0])
                self.multi_labels.append(indices)
            else:
                self.labels.append(-1)
                self.multi_labels.append([])
        self.class_labels = self.labels
        self.attr_labels = []

        # OvR: for each clip, store the matched suppression class index (or -1).
        # near_negative_X → suppress class X; also check hard_negative_for_class metadata.
        self.ovr_suppress_idx = []
        for clip in clips:
            suppress = -1
            label = clip.get("label", "")
            meta = clip.get("meta") or {}
            hn_for = meta.get("hard_negative_for_class")
            if hn_for and hn_for in self.class_to_idx:
                suppress = self.class_to_idx[hn_for]
            elif label.startswith("near_negative_"):
                suffix = label[len("near_negative_"):]
                for cls_name, cls_idx in self.class_to_idx.items():
                    if cls_name == suffix or cls_name.replace(" ", "_") == suffix:
                        suppress = cls_idx
                        break
            self.ovr_suppress_idx.append(suppress)

        # Per-frame labels: [T] tensor per clip.
        # - If explicit frame labels exist, use them.
        # - Else if a valid clip class exists, supervise all frames with that class.
        # - Else use -1 (ignored by frame loss).
        # has_real_frame_labels is used by training to route samples to frame
        # supervision (and exclude them from clip-loss metrics/path).
        self.frame_labels = []
        self.ovr_background_frame_mask = []
        self.ovr_background_clip = []
        self.has_real_frame_labels = []
        for i, clip in enumerate(clips):
            clip_fl = clip.get("frame_labels")
            T = self.clip_length
            primary = self.labels[i]
            raw_label = clip.get("label", "")
            clip_is_background = raw_label in self.ovr_background_classes
            if clip_fl and isinstance(clip_fl, (list, tuple)) and len(clip_fl) > 0:
                fl = []
                bg = []
                for lbl_name in clip_fl:
                    if lbl_name is None:
                        fl.append(-1)
                        bg.append(False)
                    elif isinstance(lbl_name, str):
                        if lbl_name in self.ovr_background_classes:
                            fl.append(-1)
                            bg.append(True)
                        else:
                            fl.append(self.class_to_idx.get(lbl_name, -1))
                            bg.append(False)
                    else:
                        v = int(lbl_name)
                        if 0 <= v < len(self.class_to_idx):
                            fl.append(v)
                            bg.append(False)
                        else:
                            fl.append(-1)
                            bg.append(False)
                fl_tensor = torch.tensor(fl, dtype=torch.long)
                bg_tensor = torch.tensor(bg, dtype=torch.bool)
                if len(fl_tensor) < T:
                    fl_tensor = torch.cat([fl_tensor, fl_tensor[-1:].expand(T - len(fl_tensor))])
                    bg_tensor = torch.cat([bg_tensor, bg_tensor[-1:].expand(T - len(bg_tensor))])
                elif len(fl_tensor) > T:
                    start = (len(fl_tensor) - T) // 2
                    fl_tensor = fl_tensor[start:start + T]
                    bg_tensor = bg_tensor[start:start + T]
                # Safety: convert any out-of-range numeric label to ignore (-1).
                fl_tensor[(fl_tensor < 0) | (fl_tensor >= len(self.class_to_idx))] = -1
                self.frame_labels.append(fl_tensor)
                self.ovr_background_frame_mask.append(bg_tensor)
                self.ovr_background_clip.append(bool(bg_tensor.any().item()) or clip_is_background)
                self.has_real_frame_labels.append(True)
            else:
                if primary >= 0:
                    self.frame_labels.append(torch.full((T,), int(primary), dtype=torch.long))
                    self.ovr_background_frame_mask.append(torch.zeros(T, dtype=torch.bool))
                    self.ovr_background_clip.append(False)
                    self.has_real_frame_labels.append(True)
                else:
                    bg_tensor = torch.full((T,), bool(clip_is_background), dtype=torch.bool)
                    self.frame_labels.append(torch.full((T,), -1, dtype=torch.long))
                    self.ovr_background_frame_mask.append(bg_tensor)
                    self.ovr_background_clip.append(bool(clip_is_background))
                    # Background clips have real negative supervision in hybrid OvR mode.
                    self.has_real_frame_labels.append(bool(clip_is_background))

        # Clip-stitching augmentation params.
        self.stitch_prob = float(stitch_prob)
        self.stitch_exclude_classes: set[int] = set()
        # Map class index → list of clip indices for fast same/different-class sampling.
        self._label_to_clip_indices: dict[int, list[int]] = {}
        for i, lbl in enumerate(self.labels):
            if lbl >= 0:
                self._label_to_clip_indices.setdefault(lbl, []).append(i)

        # Crop jitter augmentation (only used with ROI cache).
        self.crop_jitter = bool(crop_jitter)
        self.crop_jitter_strength = float(crop_jitter_strength)

        # Runtime toggles used by training curriculum.
        self._roi_cache_mode = False
        self._roi_cache_dir = None
        # Embedding-space stitch cache (backbone tokens pre-computed).
        self._emb_cache_mode = False
        self._emb_cache_dir = None
        self._emb_clip_length = self.clip_length

    def __len__(self):
        return len(self.clips) * self.virtual_size_multiplier

    def _resolve_clip_path(self, clip_id: str) -> str:
        clip_path = os.path.join(self.clip_base_dir, clip_id)
        found = False
        if os.path.exists(clip_path):
            found = True
        else:
            base_name, ext = os.path.splitext(clip_id)
            clip_basename = os.path.basename(clip_id)
            clip_dir_part = os.path.dirname(clip_id) if os.path.dirname(clip_id) else None
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV', '.MKV']

            if not ext:
                for video_ext in video_extensions:
                    test_path = os.path.join(self.clip_base_dir, clip_id + video_ext)
                    if os.path.exists(test_path):
                        clip_path = test_path
                        found = True
                        break
            else:
                base_name_only = os.path.basename(base_name)
                ext_lower = ext.lower()

                for video_ext in video_extensions:
                    if video_ext.lower() == ext_lower:
                        continue
                    test_path = os.path.join(self.clip_base_dir, base_name + video_ext)
                    if os.path.exists(test_path):
                        clip_path = test_path
                        found = True
                        break

                if not found:
                    for video_ext in video_extensions:
                        test_path = os.path.join(self.clip_base_dir, base_name_only + video_ext)
                        if os.path.exists(test_path):
                            clip_path = test_path
                            found = True
                            break

                if not found:
                    for root, dirs, files in os.walk(self.clip_base_dir):
                        for file in files:
                            file_base, file_ext = os.path.splitext(file)
                            if file_base == base_name_only or file_base == base_name:
                                if file_ext.lower() in [e.lower() for e in video_extensions]:
                                    clip_path = os.path.join(root, file)
                                    found = True
                                    break
                        if found:
                            break

                if not found and clip_dir_part:
                    subdir_path = os.path.join(self.clip_base_dir, clip_dir_part)
                    if os.path.exists(subdir_path):
                        for video_ext in video_extensions:
                            test_path = os.path.join(subdir_path, clip_basename)
                            if os.path.exists(test_path):
                                clip_path = test_path
                                found = True
                                break
                            test_path = os.path.join(subdir_path, base_name_only + video_ext)
                            if os.path.exists(test_path):
                                clip_path = test_path
                                found = True
                                break

        if not found:
            error_msg = f"Clip not found: {clip_path}\n"
            error_msg += f"Clip ID from annotation: {clip_id}\n"
            error_msg += f"Base directory: {self.clip_base_dir}\n"
            error_msg += "Please check if the file exists or update the annotation."
            raise FileNotFoundError(error_msg)
        return clip_path
    
    _NATIVE_RES = object()  # Sentinel: load at native resolution (no resize)

    def _load_clip(self, clip_path: str, target_size=None, apply_transform: bool = True,
                   apply_aoi: bool = True) -> torch.Tensor:
        """Load and preprocess a video clip.
        target_size: (w,h) to resize to, _NATIVE_RES for no resize, None for default self.target_size.
        apply_aoi: unused (kept for API compatibility).
        """
        import cv2

        cap = cv2.VideoCapture(clip_path)
        frames = []
        if target_size is self._NATIVE_RES:
            resize_to = None
        elif target_size is not None:
            resize_to = target_size
        else:
            resize_to = self.target_size

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if resize_to is not None:
                h_src, w_src = frame.shape[:2]
                is_upscale = (w_src < resize_to[0]) or (h_src < resize_to[1])
                interp = cv2.INTER_LANCZOS4 if is_upscale else cv2.INTER_AREA
                frame = cv2.resize(frame, resize_to, interpolation=interp)
                if is_upscale:
                    blurred = cv2.GaussianBlur(frame, (0, 0), sigmaX=1.0)
                    frame = cv2.addWeighted(frame, 1.5, blurred, -0.5, 0)
            frames.append(frame)
        
        cap.release()
        
        while len(frames) < self.clip_length:
            if frames:
                frames.append(frames[-1])
            else:
                fallback_size = resize_to if resize_to is not None else self.target_size
                frames.append(np.zeros((*fallback_size[::-1], 3), dtype=np.uint8))
        
        if len(frames) > self.clip_length:
            # Center-crop temporally: pick the middle clip_length frames
            start = (len(frames) - self.clip_length) // 2
            frames = frames[start:start + self.clip_length]
        
        clip_array = np.stack(frames).astype(np.float32) / 255.0
        clip_tensor = torch.from_numpy(clip_array)
        clip_tensor = clip_tensor.permute(0, 3, 1, 2)
        
        if apply_transform and self.transform:
            clip_tensor = self.transform(clip_tensor)
        
        return clip_tensor

    def load_fullres_clip_by_index(self, actual_idx: int, apply_aoi: bool = False) -> torch.Tensor:
        """Load clip at native resolution without augmentation.
        
        By default skip AOI crop (localization needs the original frame).
        """
        clip_info = self.clips[actual_idx]
        clip_id = clip_info["id"]
        clip_path = self._resolve_clip_path(clip_id)
        return self._load_clip(clip_path, target_size=self._NATIVE_RES, apply_transform=False, apply_aoi=apply_aoi)

    def load_modelres_clip_by_index(self, actual_idx: int) -> torch.Tensor:
        """Load clip at model input resolution without augmentation."""
        clip_info = self.clips[actual_idx]
        clip_id = clip_info["id"]
        clip_path = self._resolve_clip_path(clip_id)
        return self._load_clip(clip_path, target_size=self.target_size, apply_transform=False)

    def _apply_crop_jitter(self, x: torch.Tensor) -> torch.Tensor:
        """Randomly shift the crop to vary background context.

        x: [T, C, H, W] in [0,1]. Returns same shape with a random
        translation applied (pixels that shift out are filled with edge values).
        """
        _, _, H, W = x.shape
        max_dx = int(self.crop_jitter_strength * W)
        max_dy = int(self.crop_jitter_strength * H)
        if max_dx < 1 and max_dy < 1:
            return x
        dx = random.randint(-max_dx, max_dx)
        dy = random.randint(-max_dy, max_dy)
        if dx == 0 and dy == 0:
            return x
        # Use affine_grid + grid_sample for smooth sub-pixel shifting
        theta = torch.tensor([
            [1.0, 0.0, -2.0 * dx / W],
            [0.0, 1.0, -2.0 * dy / H],
        ], dtype=x.dtype).unsqueeze(0).expand(x.size(0), -1, -1)
        grid = F.affine_grid(theta, x.shape, align_corners=False)
        return F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=False)

    def _apply_spatial_label_augment(
        self,
        spatial_mask: torch.Tensor,
        spatial_bbox: torch.Tensor,
        spatial_bbox_valid: torch.Tensor,
        aug_params: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply spatial label transforms using the same params as clip augmentation."""
        hflip = bool(aug_params.get("hflip", aug_params.get("flip", False))) if aug_params else False
        vflip = bool(aug_params.get("vflip", False)) if aug_params else False
        if not hflip and not vflip:
            return spatial_mask, spatial_bbox

        out_mask = spatial_mask
        out_bbox = spatial_bbox.clone()

        # Flip mask on width/height axes when mask grid is enabled.
        if self.grid_size > 0 and out_mask.numel() == self.grid_size * self.grid_size:
            out_mask_grid = out_mask.reshape(self.grid_size, self.grid_size)
            if hflip:
                out_mask_grid = out_mask_grid.flip(1)
            if vflip:
                out_mask_grid = out_mask_grid.flip(0)
            out_mask = out_mask_grid.reshape(-1)

        # Flip per-frame bboxes in normalized xyxy coordinates.
        # spatial_bbox is [T, 4], spatial_bbox_valid is [T]
        if out_bbox.dim() == 2:
            valid = spatial_bbox_valid > 0.5  # [T]
            if valid.any():
                if hflip:
                    new_x1 = (1.0 - out_bbox[:, 2]).clamp(0.0, 1.0)
                    new_x2 = (1.0 - out_bbox[:, 0]).clamp(0.0, 1.0)
                    out_bbox[:, 0] = new_x1
                    out_bbox[:, 2] = new_x2
                if vflip:
                    new_y1 = (1.0 - out_bbox[:, 3]).clamp(0.0, 1.0)
                    new_y2 = (1.0 - out_bbox[:, 1]).clamp(0.0, 1.0)
                    out_bbox[:, 1] = new_y1
                    out_bbox[:, 3] = new_y2
                # Zero out invalid frames to avoid corrupted coords
                out_bbox[~valid] = 0.0
        elif out_bbox.dim() == 1 and out_bbox.numel() == 4:
            # Legacy fallback (single bbox)
            if float(spatial_bbox_valid.sum().item()) > 0.5:
                x1, y1, x2, y2 = [float(v) for v in out_bbox]
                if hflip:
                    x1, x2 = max(0.0, min(1.0, 1.0 - x2)), max(0.0, min(1.0, 1.0 - x1))
                if vflip:
                    y1, y2 = max(0.0, min(1.0, 1.0 - y2)), max(0.0, min(1.0, 1.0 - y1))
                out_bbox = torch.tensor([x1, y1, x2, y2], dtype=out_bbox.dtype)

        return out_mask, out_bbox
    
    def _do_stitch(self, actual_idx: int, x_a: torch.Tensor):
        """Splice clip A with a clip from a different class at a fixed 50/50 boundary.

        Returns (x_stitched, fl_stitched, y_stitched, spatial_mask, bbox, bbox_valid, bg_mask).
        The clip-level label is set to -1 because the mixed clip has no single ground
        truth; the per-frame labels carry all supervision.
        """
        T = self.clip_length
        label_a = self.labels[actual_idx]

        # Don't stitch if the source clip is an excluded class (e.g. "Other")
        if label_a in self.stitch_exclude_classes:
            return (x_a, self.frame_labels[actual_idx],
                    self.labels[actual_idx],
                    self.spatial_masks[actual_idx].clone(),
                    self.spatial_bboxes[actual_idx].clone(),
                    self.spatial_bbox_valid[actual_idx].clone(),
                    self.ovr_background_frame_mask[actual_idx].clone())

        # Gather candidate indices from any class other than clip A's class,
        # excluding F1-excluded classes (e.g. "Other") to avoid contaminating
        # real behavior clips with catch-all content.
        other_indices: list[int] = []
        for cls_idx, idxs in self._label_to_clip_indices.items():
            if cls_idx != label_a and cls_idx not in self.stitch_exclude_classes:
                other_indices.extend(idxs)
        if not other_indices:
            # Fall back to a different clip from the same class.
            same = [i for i in self._label_to_clip_indices.get(label_a, []) if i != actual_idx]
            if not same:
                # Only one clip in the whole dataset — skip stitching.
                return (x_a, self.frame_labels[actual_idx],
                        self.labels[actual_idx],
                        self.spatial_masks[actual_idx].clone(),
                        self.spatial_bboxes[actual_idx].clone(),
                        self.spatial_bbox_valid[actual_idx].clone(),
                        self.ovr_background_frame_mask[actual_idx].clone())
            other_indices = same

        b_idx = random.choice(other_indices)
        clip_b_info = self.clips[b_idx]
        clip_b_path = self._resolve_clip_path(clip_b_info["id"])
        x_b = self._load_clip(clip_b_path, apply_transform=False)

        # 50/50 split: each half gets T//2 frames for maximum temporal context.
        stitch_t = T // 2

        x_stitched = torch.cat([x_a[:stitch_t], x_b[stitch_t:]], dim=0)

        fl_a = self.frame_labels[actual_idx].clone()
        fl_b = self.frame_labels[b_idx].clone()
        fl_stitched = torch.cat([fl_a[:stitch_t], fl_b[stitch_t:]], dim=0)
        bg_a = self.ovr_background_frame_mask[actual_idx].clone()
        bg_b = self.ovr_background_frame_mask[b_idx].clone()
        bg_stitched = torch.cat([bg_a[:stitch_t], bg_b[stitch_t:]], dim=0)

        # Splice per-frame bboxes; use clip A's spatial grid mask (no better option
        # for a synthetic composite clip).
        bbox_a = self.spatial_bboxes[actual_idx].clone()
        bbox_b = self.spatial_bboxes[b_idx].clone()
        bbox_stitched = torch.cat([bbox_a[:stitch_t], bbox_b[stitch_t:]], dim=0)

        bv_a = self.spatial_bbox_valid[actual_idx].clone()
        bv_b = self.spatial_bbox_valid[b_idx].clone()
        bv_stitched = torch.cat([bv_a[:stitch_t], bv_b[stitch_t:]], dim=0)

        spatial_mask = self.spatial_masks[actual_idx].clone()
        return x_stitched, fl_stitched, -1, spatial_mask, bbox_stitched, bv_stitched, bg_stitched

    def __getitem__(self, idx):
        # Map virtual index to actual clip index
        actual_idx = idx % len(self.clips)
        clip_info = self.clips[actual_idx]

        # Embedding-space stitch: backbone tokens are pre-cached.
        # Stitch happens on token tensors so VideoPrism always sees clean clips.
        if getattr(self, '_emb_cache_mode', False):
            cache_dir = getattr(self, '_emb_cache_dir', None)
            num_versions = getattr(self, '_emb_num_versions', 1)
            use_multi_scale = getattr(self, '_emb_multi_scale', False)

            def _pick_version() -> int:
                return random.randint(0, num_versions - 1) if num_versions > 1 else 0

            def _load_emb(clip_idx: int, v: int, short: bool = False) -> torch.Tensor:
                suffix = f"_{v}_s.pt" if short else f"_{v}.pt"
                path = os.path.join(cache_dir, f"{clip_idx}{suffix}")
                return torch.load(path, map_location='cpu', weights_only=True).float()

            v_a = _pick_version()
            emb_a = _load_emb(actual_idx, v_a) if cache_dir else torch.zeros(self.clip_length * 256, 768)
            emb_a_s = _load_emb(actual_idx, v_a, short=True) if (cache_dir and use_multi_scale) else None

            label_a = self.labels[actual_idx]
            do_stitch = (
                self.stitch_prob > 0.0
                and label_a not in self.stitch_exclude_classes
                and torch.rand(1).item() < self.stitch_prob
            )
            other_indices = [
                i for cls, idxs in self._label_to_clip_indices.items()
                if cls != label_a and cls not in self.stitch_exclude_classes
                for i in idxs
            ]

            if do_stitch and other_indices:
                b_idx = random.choice(other_indices)
                v_b = _pick_version()
                emb_b = _load_emb(b_idx, v_b)
                emb_b_s = _load_emb(b_idx, v_b, short=True) if use_multi_scale else None

                T = self._emb_clip_length
                S = emb_a.shape[0] // T
                stitch_t = T // 2

                x_out = torch.cat([emb_a[:stitch_t * S], emb_b[stitch_t * S:]], dim=0)

                if use_multi_scale and emb_a_s is not None and emb_b_s is not None:
                    T_s = T // 2
                    S_s = emb_a_s.shape[0] // T_s
                    stitch_t_s = T_s // 2
                    x_short = torch.cat([emb_a_s[:stitch_t_s * S_s], emb_b_s[stitch_t_s * S_s:]], dim=0)
                else:
                    x_short = torch.empty(0)

                fl_a = self.frame_labels[actual_idx].clone()
                fl_b = self.frame_labels[b_idx].clone()
                fl = torch.cat([fl_a[:stitch_t], fl_b[stitch_t:]], dim=0)
                bg_a = self.ovr_background_frame_mask[actual_idx].clone()
                bg_b = self.ovr_background_frame_mask[b_idx].clone()
                bg_mask = torch.cat([bg_a[:stitch_t], bg_b[stitch_t:]], dim=0)
                bbox_a = self.spatial_bboxes[actual_idx].clone()
                bbox_b = self.spatial_bboxes[b_idx].clone()
                spatial_bbox = torch.cat([bbox_a[:stitch_t], bbox_b[stitch_t:]], dim=0)
                bv_a = self.spatial_bbox_valid[actual_idx].clone()
                bv_b = self.spatial_bbox_valid[b_idx].clone()
                spatial_bbox_valid = torch.cat([bv_a[:stitch_t], bv_b[stitch_t:]], dim=0)
                spatial_mask = self.spatial_masks[actual_idx].clone()
                y = -1
            else:
                x_out = emb_a
                x_short = emb_a_s if (use_multi_scale and emb_a_s is not None) else torch.empty(0)
                y = self.labels[actual_idx]
                fl = self.frame_labels[actual_idx].clone()
                bg_mask = self.ovr_background_frame_mask[actual_idx].clone()
                spatial_mask = self.spatial_masks[actual_idx].clone()
                spatial_bbox = self.spatial_bboxes[actual_idx].clone()
                spatial_bbox_valid = self.spatial_bbox_valid[actual_idx]

            return x_out, y, spatial_mask, spatial_bbox, spatial_bbox_valid, actual_idx, fl, x_short, bg_mask

        # When ROI cache is active, load the precomputed crop from disk instead
        # of decoding the original video. This lets DataLoader workers operate
        # normally (parallel prefetch) so classification training runs at the
        # same speed as standard (no-localization) training.
        if getattr(self, '_roi_cache_mode', False):
            cache_dir = getattr(self, '_roi_cache_dir', None)
            x_full = None
            if cache_dir:
                pt_path = os.path.join(cache_dir, f"{actual_idx}.pt")
                cached = torch.load(pt_path, map_location='cpu', weights_only=True)
                if isinstance(cached, dict) and torch.is_tensor(cached.get("roi")):
                    x = cached["roi"].float()
                    if torch.is_tensor(cached.get("full")):
                        x_full = cached["full"].float()
                elif torch.is_tensor(cached):
                    x = cached.float()
                else:
                    x = torch.zeros(self.clip_length, 3, 1, 1, dtype=torch.float32)
            else:
                x = torch.zeros(self.clip_length, 3, 1, 1, dtype=torch.float32)
            if x_full is None:
                x_full = x.clone()

            if self.crop_jitter and self.crop_jitter_strength > 0:
                x = self._apply_crop_jitter(x)

            spatial_mask = self.spatial_masks[actual_idx].clone()
            spatial_bbox = self.spatial_bboxes[actual_idx].clone()
            spatial_bbox_valid = self.spatial_bbox_valid[actual_idx]

            if self.transform:
                if hasattr(self.transform, "augment_with_params"):
                    x, aug_params = self.transform.augment_with_params(x)
                    spatial_mask, spatial_bbox = self._apply_spatial_label_augment(
                        spatial_mask, spatial_bbox, spatial_bbox_valid, aug_params
                    )
                else:
                    x = self.transform(x)

            y = self.labels[actual_idx]
            fl = self.frame_labels[actual_idx]
            bg_mask = self.ovr_background_frame_mask[actual_idx].clone()
            return x, y, spatial_mask, spatial_bbox, spatial_bbox_valid, actual_idx, fl, torch.empty(0), bg_mask

        clip_id = clip_info["id"]
        clip_path = self._resolve_clip_path(clip_id)

        x = self._load_clip(clip_path, apply_transform=False)

        # Clip-stitching augmentation: splice two clips from different classes so
        # the model learns per-frame features independent of clip-level context.
        if self.stitch_prob > 0.0 and torch.rand(1).item() < self.stitch_prob:
            x, fl, y, spatial_mask, spatial_bbox, spatial_bbox_valid, bg_mask = self._do_stitch(actual_idx, x)
        else:
            spatial_mask = self.spatial_masks[actual_idx].clone()
            spatial_bbox = self.spatial_bboxes[actual_idx].clone()
            spatial_bbox_valid = self.spatial_bbox_valid[actual_idx]
            y = self.labels[actual_idx]
            fl = self.frame_labels[actual_idx]
            bg_mask = self.ovr_background_frame_mask[actual_idx].clone()

        if self.transform:
            if hasattr(self.transform, "augment_with_params"):
                x, aug_params = self.transform.augment_with_params(x)
                spatial_mask, spatial_bbox = self._apply_spatial_label_augment(
                    spatial_mask, spatial_bbox, spatial_bbox_valid, aug_params
                )
            else:
                x = self.transform(x)

        return x, y, spatial_mask, spatial_bbox, spatial_bbox_valid, actual_idx, fl, torch.empty(0), bg_mask


def compute_class_weights(labels: list, num_classes: int) -> torch.Tensor:
    """Compute class weights based on inverse frequency."""
    if not labels:
        return torch.ones(num_classes)
    
    counter = Counter(labels)
    total = len(labels)
    
    weights = torch.ones(num_classes)
    for class_idx, count in counter.items():
        if count > 0:
            weights[class_idx] = total / (num_classes * count)
    
    weights = weights / weights.sum() * num_classes
    return weights


class BalancedBatchSampler:
    """
    Batch sampler that ensures at least `min_samples_per_class` samples per selected class inside each batch.
    Strategy:
      - Pick K = min(num_eligible_classes, batch_size // min_samples_per_class) classes per batch
      - Draw `min_samples_per_class` samples for each selected class (with cycling/shuffle per class)
      - Fill any remaining slots from the selected classes
    Notes:
      - Classes with fewer than `min_samples_per_class` examples are ignored for balancing
      - When no eligible classes exist, the sampler yields empty and should be ignored by caller
    """
    def __init__(
        self,
        labels: list[int],
        batch_size: int,
        min_samples_per_class: int = 2,
        drop_last: bool = False,
        seed: Optional[int] = None,
        excluded_classes: Optional[list[int]] = None,
        virtual_size_multiplier: int = 1,
        background_indices: Optional[list[int]] = None,
        background_per_batch: int = 0,
    ):
        self.labels = list(labels)
        self.batch_size = int(batch_size)
        self.min_samples_per_class = int(min_samples_per_class)
        self.drop_last = bool(drop_last)
        self.excluded_classes = set(excluded_classes or [])
        self.virtual_size_multiplier = max(1, int(virtual_size_multiplier))
        self.background_indices = list(background_indices or [])
        self.background_per_batch = max(0, int(background_per_batch))
        self._effective_length = len(self.labels) * self.virtual_size_multiplier
        
        # Use global random state if seed is None (respects random.seed())
        # Otherwise use private generator
        if seed is None:
            self.rng = random
        else:
            self.rng = random.Random(seed)
        
        # Build index lists per class
        self.class_to_indices: Dict[int, list[int]] = {}
        for idx, y in enumerate(self.labels):
            self.class_to_indices.setdefault(int(y), []).append(idx)
        
        # Eligible classes must have at least `min_samples_per_class` items and NOT be excluded
        self.eligible_classes = [
            c for c, idxs in self.class_to_indices.items()
            if len(idxs) >= self.min_samples_per_class and c not in self.excluded_classes
        ]
        self.enabled = len(self.eligible_classes) > 0 and self.batch_size > 0
        
        # Identify dropped classes
        self.dropped_classes = [c for c, idxs in self.class_to_indices.items() if len(idxs) < self.min_samples_per_class]
        
        # Prepare per-class shuffled pools with cursors
        self._pools: Dict[int, list[int]] = {}
        self._cursors: Dict[int, int] = {}
        for c in self.eligible_classes:
            pool = self.class_to_indices[c][:]
            self.rng.shuffle(pool)
            self._pools[c] = pool
            self._cursors[c] = 0
        self._bg_pool = self.background_indices[:]
        if self._bg_pool:
            self.rng.shuffle(self._bg_pool)
        self._bg_cursor = 0
    
    def __len__(self) -> int:
        if self._effective_length <= 0:
            return 0
        if self.drop_last:
            return self._effective_length // max(1, self.batch_size)
        return math.ceil(self._effective_length / max(1, self.batch_size))
    
    def _draw_from_class(self, cls: int) -> int:
        pool = self._pools[cls]
        cur = self._cursors[cls]
        if cur >= len(pool):
            # Reshuffle and reset
            pool = self.class_to_indices[cls][:]
            self.rng.shuffle(pool)
            self._pools[cls] = pool
            cur = 0
        idx = pool[cur]
        self._cursors[cls] = cur + 1
        return idx

    def _draw_background(self) -> Optional[int]:
        if not self._bg_pool:
            return None
        cur = self._bg_cursor
        if cur >= len(self._bg_pool):
            self._bg_pool = self.background_indices[:]
            self.rng.shuffle(self._bg_pool)
            cur = 0
        idx = self._bg_pool[cur]
        self._bg_cursor = cur + 1
        return idx
    
    def __iter__(self):
        if not self.enabled:
            # Yield nothing - caller should fallback
            return
        num_batches = len(self)
        for _ in range(num_batches):
            batch: list[int] = []
            if self.batch_size <= 0:
                yield batch
                continue
            
            # Number of classes to include this batch
            k = max(1, min(len(self.eligible_classes), self.batch_size // self.min_samples_per_class))
            if k <= len(self.eligible_classes):
                selected = self.rng.sample(self.eligible_classes, k)
            else:
                # Not enough distinct classes; sample with replacement to fill
                selected = self.eligible_classes[:] + [self.rng.choice(self.eligible_classes) for _ in range(k - len(self.eligible_classes))]
            
            # Ensure min_samples_per_class per selected class
            for cls in selected:
                needed = min(self.min_samples_per_class, max(0, self.batch_size - len(batch)))
                for _ in range(needed):
                    batch.append(self._draw_from_class(cls))
                    if len(batch) >= self.batch_size:
                        break
                if len(batch) >= self.batch_size:
                    break

            # Optional hybrid-OvR negatives: reserve a few slots for background clips
            # that teach all target heads to stay low without becoming their own class.
            bg_slots = min(
                self.background_per_batch,
                max(0, self.batch_size - len(batch)),
            )
            for _ in range(bg_slots):
                bg_idx = self._draw_background()
                if bg_idx is None:
                    break
                batch.append(bg_idx)
                if len(batch) >= self.batch_size:
                    break
            
            # Fill remaining slots from the selected classes (round-robin)
            si = 0
            while len(batch) < self.batch_size:
                cls = selected[si % len(selected)]
                batch.append(self._draw_from_class(cls))
                si += 1
            
            yield batch


class ConfusionAwareSampler(BalancedBatchSampler):
    """BalancedBatchSampler extended with per-sample confusion-based weights for OvR hard mining.

    After each training epoch, call update_weights() with per-clip confusion scores.
    Clips with high confusion scores (model fires wrong heads) are sampled more often
    within their class pool. Weights are EMA-blended to avoid sudden jumps.

    Confusion score for clip with true class c:
        blend of:
          - strongest rival-head activation
          - low activation of the true head
          - rival-over-true margin violation
    This keeps the score pair-aware while still producing a single scalar weight.
    """

    def __init__(self, *args, weight_temperature: float = 2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_temperature = max(0.1, float(weight_temperature))
        n = len(self.labels)
        self._confusion_scores = np.zeros(n, dtype=np.float32)
        self._top_rival = np.full(n, -1, dtype=np.int32)
        # Rebuild per-class weighted pools (initially uniform)
        self._class_weights: Dict[int, tuple] = {}
        self._rebuild_class_weights()

    def update_weights(
        self,
        confusion_scores: np.ndarray,
        top_rivals: Optional[np.ndarray] = None,
        ema_alpha: float = 0.4,
    ) -> None:
        """Blend new confusion scores into running weights and rebuild class pools."""
        if len(confusion_scores) != len(self._confusion_scores):
            return
        self._confusion_scores = (
            ema_alpha * confusion_scores.astype(np.float32)
            + (1.0 - ema_alpha) * self._confusion_scores
        )
        if top_rivals is not None and len(top_rivals) == len(self._top_rival):
            valid = top_rivals.astype(np.int32) >= 0
            self._top_rival[valid] = top_rivals.astype(np.int32)[valid]
        self._rebuild_class_weights()

    def _rebuild_class_weights(self) -> None:
        """Compute normalised sampling probability per class from confusion scores."""
        self._class_weights = {}
        for c, indices in self.class_to_indices.items():
            scores = self._confusion_scores[indices]
            # Base weight 1.0 + confusion boost; apply temperature sharpening
            w = (1.0 + scores) ** self.weight_temperature
            total = w.sum()
            if total <= 0:
                w = np.ones_like(w, dtype=np.float32)
                total = w.sum()
            self._class_weights[c] = (np.asarray(indices), w / total)

    def _draw_from_class(self, cls: int) -> int:
        """Weighted sample from class pool; falls back to parent if weights unavailable."""
        if cls in self._class_weights:
            indices, probs = self._class_weights[cls]
            return int(np.random.choice(indices, p=probs))
        return super()._draw_from_class(cls)

    def log_top_confused(self, class_names: list, dataset_clips: list, n: int = 5) -> list[str]:
        """Return log lines describing the most-confused clip per class for transparency."""
        lines = []
        for ci, cname in enumerate(class_names):
            indices = self.class_to_indices.get(ci, [])
            if not indices:
                continue
            scores = self._confusion_scores[indices]
            top_local = int(np.argmax(scores))
            top_global = indices[top_local]
            top_score = float(scores[top_local])
            if top_score < 0.05:
                continue
            clip_id = dataset_clips[top_global % len(dataset_clips)].get("id", "?") if dataset_clips else str(top_global)
            rival_idx = int(self._top_rival[top_global]) if 0 <= top_global < len(self._top_rival) else -1
            rival_txt = ""
            if 0 <= rival_idx < len(class_names):
                rival_txt = f", rival={class_names[rival_idx]}"
            lines.append(
                f"  [{cname}] hardest: {os.path.basename(clip_id)} "
                f"(confusion={top_score:.3f}{rival_txt})"
            )
        return lines[:n]


def _run_augmentation_ablation_eval(
    model: nn.Module,
    dataset,
    config: Dict[str, Any],
    device: torch.device,
    log_fn: Optional[Callable] = None,
):
    """Post-training evaluation: measure per-augmentation impact on accuracy.

    For each enabled augmentation, runs all clips through the model with ONLY that
    augmentation applied (3 random trials averaged) and compares to the clean baseline.
    Reports which augmentations help/hurt and the worst-affected clips per augmentation.
    """
    from .augmentations import ClipAugment

    aug_opts = config.get("augmentation_options") or {}
    use_ovr = config.get("use_ovr", False)

    # Map of augmentation toggle names → ClipAugment kwargs that isolate that aug
    aug_toggles = {
        "horizontal_flip": "use_horizontal_flip",
        "vertical_flip": "use_vertical_flip",
        "color_jitter": "use_color_jitter",
        "gaussian_blur": "use_gaussian_blur",
        "random_noise": "use_random_noise",
        "small_rotation": "use_small_rotation",
        "speed_perturb": "use_speed_perturb",
        "random_shapes": "use_random_shapes",
        "grayscale": "use_grayscale",
    }

    # Only evaluate augmentations that were actually enabled during training
    enabled = []
    for name, kwarg in aug_toggles.items():
        if aug_opts.get(kwarg, False):
            enabled.append((name, kwarg))

    if not enabled:
        if log_fn:
            log_fn("Augmentation ablation: no augmentations were enabled — skipping.")
        return

    if log_fn:
        log_fn(f"\n{'='*60}")
        log_fn("AUGMENTATION ABLATION EVALUATION")
        log_fn(f"{'='*60}")
        log_fn(f"Evaluating {len(enabled)} enabled augmentation(s)...")

    n_clips = len(dataset.clips)
    class_names = dataset.classes
    n_classes = len(class_names)
    n_trials = 3  # average over multiple random augmentation rolls

    saved_transform = dataset.transform
    saved_stitch = getattr(dataset, "stitch_prob", 0.0)
    saved_mult = getattr(dataset, "virtual_size_multiplier", 1)
    dataset.transform = None
    dataset.stitch_prob = 0.0
    dataset.virtual_size_multiplier = 1

    model.eval()

    def _eval_clips(transform_fn) -> np.ndarray:
        """Run all clips, return per-clip frame accuracy array."""
        dataset.transform = transform_fn
        clip_accs = np.zeros(n_clips, dtype=np.float32)
        loader = DataLoader(dataset, batch_size=config.get("batch_size", 8),
                            shuffle=False, num_workers=0, pin_memory=False)
        clip_cursor = 0
        with torch.no_grad():
            for batch_data in loader:
                if not isinstance(batch_data, (list, tuple)) or len(batch_data) < 7:
                    continue
                clips_t = batch_data[0].to(device)
                frame_labels_t = batch_data[6].to(device) if batch_data[6] is not None else None
                indices_t = batch_data[5]

                _emb_mode = getattr(dataset, '_emb_cache_mode', False)
                _cl = config.get("clip_length", 8)
                if _emb_mode:
                    clips_short_t = None
                    if len(batch_data) >= 8:
                        _cs_abl = batch_data[7]
                        if isinstance(_cs_abl, torch.Tensor) and _cs_abl.numel() > 0:
                            clips_short_t = _cs_abl.to(device)
                    out = model(
                        None, backbone_tokens=clips_t, num_frames=_cl,
                        backbone_tokens_short=clips_short_t,
                        num_frames_short=_cl // 2 if clips_short_t is not None else None,
                        return_frame_logits=True,
                    )
                else:
                    out = model(clips_t, return_frame_logits=True)

                fo = getattr(model, '_frame_output', None)
                if fo is None:
                    continue
                f_logits = fo[0]  # [B, T, C]

                if use_ovr:
                    preds = torch.argmax(torch.sigmoid(f_logits), dim=-1)
                else:
                    preds = torch.argmax(f_logits, dim=-1)

                B = preds.shape[0]
                for bi in range(B):
                    idx = int(indices_t[bi].item()) % n_clips
                    if frame_labels_t is not None:
                        valid = frame_labels_t[bi] >= 0
                        if valid.any():
                            acc = float((preds[bi][valid] == frame_labels_t[bi][valid]).float().mean().item())
                        else:
                            acc = 1.0
                    else:
                        acc = 1.0
                    clip_accs[idx] = acc
        return clip_accs

    # 1. Clean baseline (no augmentation)
    if log_fn:
        log_fn("Running clean baseline (no augmentation)...")
    baseline_accs = _eval_clips(None)
    baseline_mean = float(baseline_accs.mean()) * 100

    # 2. Per-augmentation evaluation
    results = {}
    for aug_name, aug_kwarg in enabled:
        if log_fn:
            log_fn(f"Evaluating: {aug_name}...")
        trial_accs = []
        for trial in range(n_trials):
            # Build a ClipAugment with ONLY this one augmentation on
            kwargs = {k: False for _, k in aug_toggles.items()}
            kwargs[aug_kwarg] = True
            # Pass through augmentation-specific params
            for pkey in ["color_jitter_brightness", "color_jitter_contrast",
                         "color_jitter_saturation", "color_jitter_hue",
                         "noise_std", "rotation_degrees"]:
                if pkey in aug_opts:
                    kwargs[pkey] = aug_opts[pkey]
            aug_fn = ClipAugment(**kwargs)
            trial_accs.append(_eval_clips(aug_fn))
        avg_accs = np.mean(trial_accs, axis=0)
        delta = avg_accs - baseline_accs  # per-clip change
        results[aug_name] = {
            "mean_acc": float(avg_accs.mean()) * 100,
            "delta_mean": float(delta.mean()) * 100,
            "n_hurt": int((delta < -0.05).sum()),
            "n_helped": int((delta > 0.05).sum()),
            "per_clip_delta": delta,
        }

    # Restore
    dataset.transform = saved_transform
    dataset.stitch_prob = saved_stitch
    dataset.virtual_size_multiplier = saved_mult

    # 3. Log results
    if log_fn:
        log_fn(f"\nClean baseline accuracy: {baseline_mean:.1f}%\n")
        log_fn(f"{'Augmentation':<20} {'Acc':>7} {'Δ':>7} {'Hurt':>6} {'Helped':>8}")
        log_fn("-" * 52)
        for aug_name, r in sorted(results.items(), key=lambda x: x[1]["delta_mean"]):
            log_fn(
                f"{aug_name:<20} {r['mean_acc']:>6.1f}% {r['delta_mean']:>+6.1f}% "
                f"{r['n_hurt']:>5} {r['n_helped']:>7}"
            )

        # Worst clips per augmentation
        log_fn(f"\nWorst-affected clips per augmentation:")
        for aug_name, r in results.items():
            deltas = r["per_clip_delta"]
            worst_idx = int(np.argmin(deltas))
            worst_delta = float(deltas[worst_idx]) * 100
            if worst_delta > -1.0:
                continue
            clip_id = dataset.clips[worst_idx].get("id", "?")
            clip_label_idx = dataset.labels[worst_idx]
            clip_cls = class_names[clip_label_idx] if 0 <= clip_label_idx < n_classes else "?"
            log_fn(
                f"  {aug_name}: {os.path.basename(clip_id)} [{clip_cls}] "
                f"(clean={baseline_accs[worst_idx]*100:.0f}% → aug={float(r['per_clip_delta'][worst_idx] + baseline_accs[worst_idx])*100:.0f}%, "
                f"Δ={worst_delta:+.1f}%)"
            )

        log_fn(f"\n{'='*60}")


def train_model(
    model: nn.Module,
    train_dataset: Dataset,
    val_dataset: Optional[Dataset],
    config: Dict[str, Any],
    log_fn: Optional[Callable[[str], None]] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    stop_callback: Optional[Callable[[], bool]] = None,
    metrics_callback: Optional[Callable[[Dict[str, Any]], None]] = None
):
    """Training loop for behavior classifier.
    
    Args:
        model: BehaviorClassifier model
        train_dataset: Training dataset
        val_dataset: Optional validation dataset
        config: Training configuration dict
        log_fn: Optional callback for logging messages
        progress_callback: Optional callback(epoch, total_epochs) for progress
        stop_callback: Optional callback returning True if training should stop
        metrics_callback: Optional callback(metrics_dict) called after each epoch
    """
    import traceback
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if log_fn:
            log_fn(f"Using device: {device}")
            if device.type == "cuda":
                log_fn(f"CUDA device: {torch.cuda.get_device_name(0)}")
                log_fn(f"CUDA available: {torch.cuda.is_available()}")
                log_fn(f"CUDA device count: {torch.cuda.device_count()}")
                log_fn(f"Current CUDA device: {torch.cuda.current_device()}")
                log_fn(f"CUDA memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        
        # Move model to device (head will be on GPU, backbone stays on CPU for JAX)
        model.to(device)

        # Load pretrained weights if specified (fine-tuning)
        pretrained_path = config.get("pretrained_path")
        if pretrained_path and os.path.exists(pretrained_path):
            if log_fn:
                log_fn(f"Loading pretrained weights from {pretrained_path}...")
            try:
                payload = torch.load(pretrained_path, map_location=device)
                if isinstance(payload, dict):
                    frame_head_state_dict = payload.get("frame_head_state_dict", {})
                    localization_state_dict = payload.get("localization_state_dict", {})
                else:
                    frame_head_state_dict = {}
                    localization_state_dict = {}

                if (
                    getattr(model, "frame_head", None) is not None
                    and isinstance(frame_head_state_dict, dict)
                    and frame_head_state_dict
                ):
                    frame_model_state = model.frame_head.state_dict()
                    filtered_frame = {}
                    mismatched_frame = []
                    for k, v in frame_head_state_dict.items():
                        if k in frame_model_state:
                            if v.shape == frame_model_state[k].shape:
                                filtered_frame[k] = v
                            else:
                                mismatched_frame.append(k)
                    model.frame_head.load_state_dict(filtered_frame, strict=False)
                    if log_fn:
                        log_fn(
                            f"Loaded frame head weights: {len(filtered_frame)} tensors"
                            + (f" (skipped mismatched: {mismatched_frame})" if mismatched_frame else "")
                        )

                # Also restore localization head when available and compatible.
                if (
                    getattr(model, "use_localization", False)
                    and getattr(model, "localization_head", None) is not None
                    and isinstance(localization_state_dict, dict)
                    and localization_state_dict
                ):
                    loc_model_state = model.localization_head.state_dict()
                    filtered_loc = {}
                    mismatched_loc = []
                    for k, v in localization_state_dict.items():
                        if k in loc_model_state:
                            if v.shape == loc_model_state[k].shape:
                                filtered_loc[k] = v
                            else:
                                mismatched_loc.append(k)
                    model.localization_head.load_state_dict(filtered_loc, strict=False)
                    if log_fn:
                        log_fn(
                            f"Loaded localization head weights: {len(filtered_loc)} tensors"
                            + (f" (skipped mismatched: {mismatched_loc})" if mismatched_loc else "")
                        )
                elif log_fn and getattr(model, "use_localization", False):
                    log_fn("No localization weights found in pretrained checkpoint; localization head will train from initialization.")

                if log_fn:
                    log_fn("Pretrained weights loaded successfully (partial load if class count changed).")
            except Exception as e:
                if log_fn:
                    log_fn(f"WARNING: Failed to load pretrained weights: {e}")
        
        # Ensure frame head is on GPU
        if device.type == "cuda":
            model.frame_head.to(device)
            fh_device = next(model.frame_head.parameters()).device
            if log_fn:
                log_fn(f"Frame head device: {fh_device}")
                if fh_device.type != "cuda":
                    log_fn("ERROR: Frame head is not on GPU! Training will be very slow.")
                else:
                    log_fn(f"GPU memory after moving head: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        
        if log_fn:
            log_fn(f"Model moved to {device}")
            log_fn("Note: VideoPrism backbone runs on GPU (JAX) if available, classification heads run on GPU (PyTorch)")
        
        if log_fn:
            log_fn(f"Creating data loaders (batch_size={config['batch_size']})...")
        
        use_weighted_sampler = config.get("use_weighted_sampler", False)
        use_ovr = config.get("use_ovr", False)
        _confusion_warmup_pct = float(config.get("confusion_sampler_warmup_pct", 0.2))
        sampler = None
        batch_sampler = None
        shuffle = True
        
        # OvR needs balanced batches so each binary head sees positives every batch.
        if use_ovr and hasattr(train_dataset, 'labels'):
            labels_for_balance = train_dataset.labels
            ovr_min_samples = 1
            _confusion_temperature = float(config.get("confusion_sampler_temperature", 2.0))
            _use_confusion_sampler = bool(config.get("use_confusion_sampler", True)) and use_ovr
            _hybrid_bg = bool(config.get("ovr_background_as_negative", False))
            _bg_indices = []
            _bg_per_batch = 0
            if _hybrid_bg and hasattr(train_dataset, "ovr_background_clip"):
                _bg_indices = [
                    i for i, is_bg in enumerate(train_dataset.ovr_background_clip)
                    if bool(is_bg)
                ]
                if _bg_indices:
                    _bg_per_batch = max(1, config["batch_size"] // 4)
            _SamplerClass = ConfusionAwareSampler if _use_confusion_sampler else BalancedBatchSampler
            _sampler_kwargs = dict(weight_temperature=_confusion_temperature) if _use_confusion_sampler else {}
            bbs = _SamplerClass(
                labels=labels_for_balance,
                batch_size=config["batch_size"],
                min_samples_per_class=ovr_min_samples,
                drop_last=False,
                excluded_classes=[-1],
                virtual_size_multiplier=getattr(train_dataset, "virtual_size_multiplier", 1),
                background_indices=_bg_indices,
                background_per_batch=_bg_per_batch,
                **_sampler_kwargs,
            )
            if bbs and getattr(bbs, "enabled", False):
                batch_sampler = bbs
                if log_fn:
                    reason = "OvR" if use_ovr else "contrastive loss"
                    sampler_type = "ConfusionAwareSampler" if _use_confusion_sampler else "BalancedBatchSampler"
                    log_fn(f"Using {sampler_type} (>={ovr_min_samples} per class) for training ({reason} enabled)")
                    if _bg_indices:
                        log_fn(
                            f"Hybrid OvR background negatives: {len(_bg_indices)} clips "
                            f"(up to {_bg_per_batch} per batch)"
                        )
                    if _use_confusion_sampler:
                        warmup_ep = int(config["epochs"] * _confusion_warmup_pct)
                        log_fn(f"Confusion sampler warmup: {int(_confusion_warmup_pct * 100)}% ({warmup_ep} epochs) — uniform sampling until epoch {warmup_ep}")
                    if hasattr(bbs, "dropped_classes") and bbs.dropped_classes:
                        # Map indices back to names safely.
                        dropped_names = []
                        for idx in bbs.dropped_classes:
                            name = None
                            if hasattr(train_dataset, "classes") and 0 <= idx < len(train_dataset.classes):
                                name = train_dataset.classes[idx]
                            elif hasattr(train_dataset, "raw_classes") and 0 <= idx < len(train_dataset.raw_classes):
                                name = train_dataset.raw_classes[idx]
                            else:
                                name = str(idx)
                            dropped_names.append(name)
                        log_fn(
                            f"WARNING: The following classes have < {ovr_min_samples} samples and will be SKIPPED by BalancedBatchSampler: "
                            f"{dropped_names}"
                        )
            else:
                if log_fn:
                    log_fn("BalancedBatchSampler disabled (insufficient class counts); falling back to standard batching")
        
        # If not using balanced batches, optionally use weighted sampler
        if batch_sampler is None and use_weighted_sampler and hasattr(train_dataset, 'labels'):
            if log_fn:
                log_fn("Creating weighted random sampler...")

            labels_for_sampling = list(train_dataset.labels)

            virtual_mult = int(getattr(train_dataset, "virtual_size_multiplier", 1) or 1)
            if virtual_mult > 1:
                labels_for_sampling = labels_for_sampling * virtual_mult

            class_counts = Counter(labels_for_sampling)
            if log_fn:
                log_fn(f"Class counts: {dict(class_counts)}")
            num_samples = len(labels_for_sampling)
            weights = [1.0 / class_counts[label] for label in labels_for_sampling]
            sampler = WeightedRandomSampler(weights=weights, num_samples=num_samples, replacement=True)
            shuffle = False
            if log_fn:
                log_fn("Using weighted random sampler for training")
        
        num_workers = config.get("num_workers", 4)
        
        if batch_sampler is not None:
            train_loader = DataLoader(
                train_dataset,
                batch_sampler=batch_sampler,
                num_workers=num_workers,
                pin_memory=True if device.type == "cuda" else False,
                persistent_workers=True if num_workers > 0 else False
            )
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=config["batch_size"],
                shuffle=shuffle,
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=True if device.type == "cuda" else False,
                persistent_workers=True if num_workers > 0 else False
            )
        
        if log_fn:
            log_fn(f"Train loader created: {len(train_loader)} batches (workers={num_workers})")
        
        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=config["batch_size"],
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True if device.type == "cuda" else False,
                persistent_workers=True if num_workers > 0 else False
            )
            if log_fn:
                log_fn(f"Val loader created: {len(val_loader)} batches")
        
        if log_fn:
            log_fn("Creating optimizer and loss function...")
        
        base_lr = config["lr"]
        localization_lr = float(config.get("localization_lr", base_lr))
        classification_lr = float(config.get("classification_lr", base_lr))
        wd = config.get("weight_decay", 0.001)
        param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
        raw_model = model.module if hasattr(model, "module") else model
        use_separate_lr = (
            getattr(raw_model, "use_localization", False)
            and getattr(raw_model, "localization_head", None) is not None
        )
        if use_separate_lr:
            loc_params = [p for n, p in param_dict.items() if "localization_head" in n]
            other_params = [p for n, p in param_dict.items() if "localization_head" not in n]
            loc_decay = [p for p in loc_params if p.dim() >= 2]
            loc_nodecay = [p for p in loc_params if p.dim() < 2]
            cls_decay = [p for p in other_params if p.dim() >= 2]
            cls_nodecay = [p for p in other_params if p.dim() < 2]
            optim_groups = []
            if loc_decay:
                optim_groups.append({"params": loc_decay, "weight_decay": wd, "lr": localization_lr})
            if loc_nodecay:
                optim_groups.append({"params": loc_nodecay, "weight_decay": 0.0, "lr": localization_lr})
            if cls_decay:
                optim_groups.append({"params": cls_decay, "weight_decay": wd, "lr": classification_lr})
            if cls_nodecay:
                optim_groups.append({"params": cls_nodecay, "weight_decay": 0.0, "lr": classification_lr})
        else:
            decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
            nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
            optim_groups = [
                {"params": decay_params, "weight_decay": wd, "lr": classification_lr},
                {"params": nodecay_params, "weight_decay": 0.0, "lr": classification_lr},
            ]
        optimizer = AdamW(optim_groups, lr=classification_lr)
        
        # --- Scheduler: Cosine Annealing with Warm Restarts + Linear Warmup ---
        use_scheduler = bool(config.get('use_scheduler', True))
        total_epochs = config['epochs']
        warmup_epochs = 0
        scheduler = None
        warmup_scheduler = None
        restart_period = None
        eta_min = None
        if use_scheduler:
            eta_min = 0.2 * classification_lr
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, total_epochs - warmup_epochs), eta_min=eta_min
            )
            warmup_scheduler = None
            if warmup_epochs > 0:
                warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=0.01, end_factor=1.0,
                    total_iters=warmup_epochs,
                )

        # --- EMA (Exponential Moving Average) for classification head only ---
        use_ema = bool(config.get("use_ema", True))
        ema_decay = 0.99
        ema_state: dict[str, torch.Tensor] = {}
        ema_active = False  # enabled later when ema_start_epoch is reached
        def _init_ema():
            raw = model.module if hasattr(model, "module") else model
            ema_state.clear()
            for name, param in raw.named_parameters():
                if param.requires_grad and not name.startswith("localization_head."):
                    ema_state[name] = param.data.clone()
        def _update_ema():
            if not ema_active:
                return
            raw = model.module if hasattr(model, "module") else model
            for name, param in raw.named_parameters():
                if name in ema_state:
                    ema_state[name].mul_(ema_decay).add_(param.data, alpha=1.0 - ema_decay)
        def _apply_ema():
            """Swap model weights with EMA weights. Call again to restore."""
            if not ema_active:
                return
            raw = model.module if hasattr(model, "module") else model
            for name, param in raw.named_parameters():
                if name in ema_state:
                    param.data, ema_state[name] = ema_state[name].clone(), param.data.clone()

        if log_fn:
            log_fn("Using AdamW with separate weight decay for biases/norm")
            if use_separate_lr:
                log_fn(f"Localization LR: {localization_lr:.2e}, Classification LR: {classification_lr:.2e}")
            else:
                log_fn(f"Learning rate: {classification_lr:.2e}")
            log_fn("Gradient clipping: max_norm=1.0 (NaN-guarded)")
            if use_scheduler:
                if warmup_epochs > 0:
                    log_fn(f"LR schedule: {warmup_epochs}-epoch linear warmup → CosineAnnealingLR (single decay, {total_epochs - warmup_epochs} epochs)")
                else:
                    log_fn(f"LR schedule: CosineAnnealingLR (single cosine decay, {total_epochs} epochs, no warmup)")
            else:
                log_fn("LR scheduler disabled (fixed learning rate)")
            if use_ema:
                if use_separate_lr or getattr(raw_model, "use_localization", False):
                    log_fn(f"EMA: classification head only (decay={ema_decay}); localization head unchanged")
                else:
                    log_fn(f"EMA model averaging: enabled (decay={ema_decay})")
            else:
                log_fn("EMA model averaging: disabled")
        
        use_class_weights = config.get("use_class_weights", False)

        # Focal Loss settings
        use_focal_loss = config.get("use_focal_loss", False)
        focal_gamma = config.get("focal_gamma", 2.0)
        use_supcon_loss = bool(config.get("use_supcon_loss", False))
        supcon_weight = float(config.get("supcon_weight", 0.2)) if use_supcon_loss else 0.0
        supcon_temperature = float(config.get("supcon_temperature", 0.1))

        # Asymmetric Loss settings
        use_asl = config.get("use_asl", True)  # Asymmetric Loss on by default for OvR
        asl_gamma_neg = float(config.get("asl_gamma_neg", 4.0))
        asl_gamma_pos = float(config.get("asl_gamma_pos", 0.0))
        asl_clip = float(config.get("asl_clip", 0.05))

        # ASL's 'clip' parameter already applies negative smoothing, so default
        # to 0.0 when ASL is active to avoid contradictory loss signals.
        default_smoothing = 0.0 if (use_ovr and use_asl) else (0.05 if use_ovr else 0.0)
        ovr_label_smoothing = float(config.get("ovr_label_smoothing", default_smoothing))
        ovr_background_as_negative = bool(config.get("ovr_background_as_negative", False))
        allowed_cooccurrence = []
        cooccur_lookup: dict[int, set[int]] = {}
        hard_pair_mining = bool(config.get("use_hard_pair_mining", False) and use_ovr)
        hard_pair_margin = float(config.get("hard_pair_margin", 0.5))
        hard_pair_loss_weight = float(config.get("hard_pair_loss_weight", 0.2)) if hard_pair_mining else 0.0
        hard_pair_confusion_boost = max(1.0, float(config.get("hard_pair_confusion_boost", 1.5)))
        hard_pair_index_pairs: list[tuple[int, int]] = []
        hard_pair_name_pairs: list[list[str]] = []

        if use_ovr and getattr(model, "frame_head", None) is not None:
            model.frame_head.use_ovr = True
            if ovr_background_as_negative and log_fn:
                bg_names = config.get("ovr_background_class_names", [])
                bg_txt = ", ".join(bg_names) if bg_names else "background"
                log_fn(f"OvR hybrid background negatives: {bg_txt} kept as all-zero targets")
            if hard_pair_mining:
                class_to_idx = {c: i for i, c in enumerate(getattr(train_dataset, "classes", []))}
                seen_pairs = set()
                skipped_pairs = []
                for pair in (config.get("hard_pairs", []) or []):
                    if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                        continue
                    a_name = str(pair[0]).strip()
                    b_name = str(pair[1]).strip()
                    if not a_name or not b_name or a_name == b_name:
                        continue
                    a_idx = class_to_idx.get(a_name, -1)
                    b_idx = class_to_idx.get(b_name, -1)
                    if a_idx < 0 or b_idx < 0:
                        skipped_pairs.append((a_name, b_name))
                        continue
                    key = (min(a_idx, b_idx), max(a_idx, b_idx))
                    if key in seen_pairs:
                        continue
                    seen_pairs.add(key)
                    hard_pair_index_pairs.append(key)
                    hard_pair_name_pairs.append([
                        train_dataset.classes[key[0]],
                        train_dataset.classes[key[1]],
                    ])
                if skipped_pairs and log_fn:
                    skipped_txt = ", ".join(f"{a}<->{b}" for a, b in skipped_pairs[:8])
                    if len(skipped_pairs) > 8:
                        skipped_txt += ", ..."
                    log_fn(f"Hard-pair mining: skipped unknown pair(s): {skipped_txt}")
                if hard_pair_index_pairs:
                    if log_fn:
                        pair_txt = ", ".join(f"{a}<->{b}" for a, b in hard_pair_name_pairs)
                        log_fn(
                            "Hard-pair mining: "
                            f"{pair_txt} | margin={hard_pair_margin:.2f}, "
                            f"loss_weight={hard_pair_loss_weight:.3f}, "
                            f"confusion_boost={hard_pair_confusion_boost:.2f}x"
                        )
                else:
                    hard_pair_mining = False
                    hard_pair_loss_weight = 0.0
                    if log_fn:
                        log_fn("Hard-pair mining enabled, but no valid configured pairs matched the active classes.")

        # Frame-level loss is always the sole objective
        use_frame_loss = True
        if log_fn:
            log_fn("Frame-level classification: enabled (sole classification loss)")

        # Boundary and smoothness loss settings
        use_temporal_decoder = bool(config.get("use_temporal_decoder", True))
        boundary_loss_weight = float(config.get("boundary_loss_weight", 0.3)) if use_temporal_decoder else 0.0
        smoothness_loss_weight = float(config.get("smoothness_loss_weight", 0.05))
        boundary_tolerance = int(config.get("boundary_tolerance", 1))
        if log_fn:
            if use_temporal_decoder:
                log_fn(f"Boundary loss: weight={boundary_loss_weight}, tolerance={boundary_tolerance}")
            else:
                log_fn("Temporal decoder: disabled (direct per-frame classifier after spatial pooling)")
                log_fn("Boundary loss: disabled (no boundary branch)")
            log_fn(f"Smoothness loss: weight={smoothness_loss_weight}")
            if use_supcon_loss and supcon_weight > 0:
                log_fn(
                    f"SupCon on MAP embeddings: enabled "
                    f"(weight={supcon_weight:.3f}, temperature={supcon_temperature:.3f})"
                )
            else:
                log_fn("SupCon on MAP embeddings: disabled")

        use_frame_bout_balance = bool(config.get("use_frame_bout_balance", True))
        frame_bout_balance_power = float(config.get("frame_bout_balance_power", 1.0))
        if use_frame_bout_balance and log_fn:
            log_fn(
                f"Frame bout balancing: enabled (power={frame_bout_balance_power:.2f}) "
                f"[reduces dominance of long behavior bouts]"
            )

        def _generate_boundary_labels(frame_labels: torch.Tensor, tolerance: int = 1) -> torch.Tensor:
            """Generate boundary labels from frame labels.
            
            boundary[t] = 1 if there is a label transition within ±tolerance frames.
            Frames with label=-1 get boundary label=-1 (ignored).
            """
            B, T = frame_labels.shape
            boundaries = torch.zeros(B, T, dtype=torch.float32, device=frame_labels.device)
            for bi in range(B):
                for t in range(1, T):
                    cur = int(frame_labels[bi, t].item())
                    prev = int(frame_labels[bi, t - 1].item())
                    if cur < 0 or prev < 0:
                        continue
                    if cur != prev:
                        lo = max(0, t - tolerance)
                        hi = min(T, t + tolerance + 1)
                        boundaries[bi, lo:hi] = 1.0
            # Mark invalid frames as -1
            boundaries[frame_labels < 0] = -1.0
            return boundaries

        def _pool_frame_labels(frame_labels: Optional[torch.Tensor], pool: int) -> Optional[torch.Tensor]:
            """Downsample [B, T] labels to pooled timeline by majority vote (ignore=-1)."""
            if frame_labels is None or pool <= 1:
                return frame_labels
            if frame_labels.dim() != 2:
                return frame_labels
            B, T = frame_labels.shape
            pad = (pool - (T % pool)) % pool
            lbl = frame_labels
            if pad > 0:
                pad_vals = lbl[:, -1:].repeat(1, pad)
                lbl = torch.cat([lbl, pad_vals], dim=1)
            Tp = lbl.shape[1] // pool
            chunks = lbl.view(B, Tp, pool)
            pooled = torch.full((B, Tp), -1, dtype=lbl.dtype, device=lbl.device)
            for bi in range(B):
                for ti in range(Tp):
                    vals = chunks[bi, ti]
                    valid = vals[vals >= 0]
                    if valid.numel() == 0:
                        continue
                    uniq, counts = torch.unique(valid, return_counts=True)
                    pooled[bi, ti] = uniq[torch.argmax(counts)]
            return pooled

        def _pool_binary_mask(mask: Optional[torch.Tensor], pool: int) -> Optional[torch.Tensor]:
            """Downsample a boolean [B, T] mask by any() over each pooled window."""
            if mask is None or pool <= 1:
                return mask
            if mask.dim() != 2:
                return mask
            B, T = mask.shape
            pad = (pool - (T % pool)) % pool
            m = mask
            if pad > 0:
                pad_vals = m[:, -1:].repeat(1, pad)
                m = torch.cat([m, pad_vals], dim=1)
            Tp = m.shape[1] // pool
            return m.view(B, Tp, pool).any(dim=2)

        def _pool_frame_embeddings(frame_embeddings: Optional[torch.Tensor], pool: int) -> Optional[torch.Tensor]:
            """Average [B, T, D] embeddings over pooled windows to match pooled labels."""
            if frame_embeddings is None or pool <= 1:
                return frame_embeddings
            if frame_embeddings.dim() != 3:
                return frame_embeddings
            B, T, D = frame_embeddings.shape
            pad = (pool - (T % pool)) % pool
            emb = frame_embeddings
            if pad > 0:
                pad_vals = emb[:, -1:, :].repeat(1, pad, 1)
                emb = torch.cat([emb, pad_vals], dim=1)
            Tp = emb.shape[1] // pool
            return emb.view(B, Tp, pool, D).mean(dim=2)

        def _supervised_contrastive_loss(
            frame_embeddings: Optional[torch.Tensor],
            frame_labels: Optional[torch.Tensor],
            temperature: float = 0.1,
            max_samples: int = 512,
        ) -> torch.Tensor:
            """SupCon over pooled frame embeddings using valid frame labels only."""
            if frame_embeddings is None or frame_labels is None:
                if frame_embeddings is not None:
                    return frame_embeddings.sum() * 0.0
                return torch.tensor(0.0, device=device)
            if frame_embeddings.dim() != 3 or frame_labels.dim() != 2:
                return frame_embeddings.sum() * 0.0
            feats = frame_embeddings.reshape(-1, frame_embeddings.shape[-1])
            labels_flat = frame_labels.reshape(-1)
            valid = labels_flat >= 0
            if valid.sum().item() < 2:
                return feats.sum() * 0.0
            feats = feats[valid]
            labels_flat = labels_flat[valid]
            if feats.shape[0] > max_samples:
                perm = torch.randperm(feats.shape[0], device=feats.device)[:max_samples]
                feats = feats[perm]
                labels_flat = labels_flat[perm]
            uniq_labels, counts = torch.unique(labels_flat, return_counts=True)
            keep_labels = uniq_labels[counts >= 2]
            if keep_labels.numel() == 0:
                return feats.sum() * 0.0
            keep_mask = torch.zeros_like(labels_flat, dtype=torch.bool)
            for cls_id in keep_labels:
                keep_mask |= labels_flat == cls_id
            feats = feats[keep_mask]
            labels_flat = labels_flat[keep_mask]
            if feats.shape[0] < 2:
                return feats.sum() * 0.0

            feats = F.normalize(feats, p=2, dim=1)
            logits = torch.matmul(feats, feats.T) / max(float(temperature), 1e-6)
            logits = logits - logits.max(dim=1, keepdim=True).values.detach()

            same_class = labels_flat.unsqueeze(0) == labels_flat.unsqueeze(1)
            self_mask = torch.eye(feats.shape[0], device=feats.device, dtype=torch.bool)
            pos_mask = same_class & (~self_mask)
            valid_anchor = pos_mask.any(dim=1)
            if not valid_anchor.any():
                return feats.sum() * 0.0

            exp_logits = torch.exp(logits) * (~self_mask)
            log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp(min=1e-12))
            mean_log_prob_pos = (pos_mask.float() * log_prob).sum(dim=1) / pos_mask.sum(dim=1).clamp(min=1)
            return -mean_log_prob_pos[valid_anchor].mean()

        def _lookup_ovr_suppress_for_batch(sample_indices, dataset, device) -> Optional[torch.Tensor]:
            """Map batch sample indices back to per-clip OvR suppression targets."""
            suppress_list = getattr(dataset, "ovr_suppress_idx", None)
            if not suppress_list:
                return None
            if isinstance(sample_indices, torch.Tensor):
                indices = sample_indices.detach().cpu().tolist()
            else:
                indices = list(sample_indices)
            mapped = []
            for idx in indices:
                try:
                    clip_idx = int(idx)
                except Exception as e:
                    logger.debug("Could not convert index to int: %s", e)
                    clip_idx = -1
                if 0 <= clip_idx < len(suppress_list):
                    mapped.append(int(suppress_list[clip_idx]))
                else:
                    mapped.append(-1)
            return torch.tensor(mapped, dtype=torch.long, device=device)

        def _build_bout_weights(frame_labels: torch.Tensor, power: float = 1.0) -> torch.Tensor:
            """Per-frame weights inversely proportional to contiguous bout length."""
            B, T = frame_labels.shape
            weights = torch.zeros((B, T), dtype=torch.float32, device=frame_labels.device)
            for bi in range(B):
                t = 0
                while t < T:
                    lbl = int(frame_labels[bi, t].item())
                    if lbl < 0:
                        t += 1
                        continue
                    t_end = t + 1
                    while t_end < T and int(frame_labels[bi, t_end].item()) == lbl:
                        t_end += 1
                    seg_len = max(1, t_end - t)
                    w = float(seg_len) ** (-float(power))
                    weights[bi, t:t_end] = w
                    t = t_end
            valid = frame_labels >= 0
            if valid.any():
                mean_w = weights[valid].mean().clamp(min=1e-8)
                weights = weights / mean_w
            return weights

        def _hard_pair_margin_loss(
            frame_logits: torch.Tensor,
            frame_labels: torch.Tensor,
            pair_indices: list[tuple[int, int]],
            margin: float,
            use_bout_balance: bool = False,
            bout_power: float = 1.0,
        ) -> torch.Tensor:
            """Extra pairwise margin pressure for configured confusing class pairs."""
            if not pair_indices or margin <= 0:
                return frame_logits.sum() * 0.0
            B, T, _ = frame_logits.shape
            frame_w = torch.ones((B, T), dtype=torch.float32, device=frame_logits.device)
            if use_bout_balance:
                frame_w = _build_bout_weights(frame_labels, power=bout_power)
            total_loss = frame_logits.new_tensor(0.0)
            total_weight = frame_logits.new_tensor(0.0)
            for a_idx, b_idx in pair_indices:
                mask_a = frame_labels == a_idx
                if mask_a.any():
                    loss_a = F.relu(margin - (frame_logits[..., a_idx] - frame_logits[..., b_idx]))
                    total_loss = total_loss + (loss_a[mask_a] * frame_w[mask_a]).sum()
                    total_weight = total_weight + frame_w[mask_a].sum()
                mask_b = frame_labels == b_idx
                if mask_b.any():
                    loss_b = F.relu(margin - (frame_logits[..., b_idx] - frame_logits[..., a_idx]))
                    total_loss = total_loss + (loss_b[mask_b] * frame_w[mask_b]).sum()
                    total_weight = total_weight + frame_w[mask_b].sum()
            if float(total_weight.item()) <= 0:
                return frame_logits.sum() * 0.0
            return total_loss / total_weight.clamp(min=1e-6)

        def _frame_loss_balanced(
            frame_logits: torch.Tensor,
            frame_labels: torch.Tensor,
            use_ovr_local: bool = False,
            ovr_targets: Optional[torch.Tensor] = None,
            ovr_weight: Optional[torch.Tensor] = None,
            use_bout_balance: bool = False,
            bout_power: float = 1.0,
            valid_mask_override: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            """Frame classification loss with optional inverse-bout-length weighting."""
            B, T, C = frame_logits.shape
            valid = valid_mask_override if valid_mask_override is not None else (frame_labels >= 0)
            if not valid.any():
                return frame_logits.sum() * 0.0

            frame_w = torch.ones((B, T), dtype=torch.float32, device=frame_logits.device)
            if use_bout_balance:
                base_w = _build_bout_weights(frame_labels, power=bout_power)
                if valid_mask_override is not None:
                    frame_w = torch.where(
                        valid,
                        torch.where(frame_labels >= 0, base_w, torch.ones_like(base_w)),
                        torch.zeros_like(base_w),
                    )
                else:
                    frame_w = base_w

            if use_ovr_local and ovr_targets is not None:
                if use_asl and isinstance(criterion, AsymmetricLoss):
                    per_elem = criterion(frame_logits, ovr_targets)
                else:
                    per_elem = F.binary_cross_entropy_with_logits(
                        frame_logits, ovr_targets, reduction='none'
                    )
                    if use_focal_loss:
                        pt = torch.exp(-per_elem)
                        per_elem = ((1 - pt) ** focal_gamma) * per_elem

                elem_w = valid.unsqueeze(-1).float() * frame_w.unsqueeze(-1)
                if ovr_weight is not None:
                    elem_w = elem_w * ovr_weight
                denom = elem_w.sum().clamp(min=1.0)
                return (per_elem * elem_w).sum() / denom

            logits_flat = frame_logits.reshape(B * T, C)
            labels_flat = frame_labels.reshape(B * T)
            valid_flat = labels_flat >= 0
            if not valid_flat.any():
                return frame_logits.sum() * 0.0
            logits_valid = logits_flat[valid_flat]
            labels_valid = labels_flat[valid_flat]
            raw_ce = F.cross_entropy(logits_valid, labels_valid, reduction='none')
            w_flat = frame_w.reshape(B * T)[valid_flat]
            denom = w_flat.sum().clamp(min=1.0)
            return (raw_ce * w_flat).sum() / denom

        # Localization supervision settings (autonomous staged curriculum)
        use_localization = bool(config.get("use_localization", False) and getattr(model, "use_localization", False))
        has_any_localization = use_localization and any(
            float(v.sum().item()) > 0.5 for v in getattr(train_dataset, "spatial_bbox_valid", [])
        )
        # Default localization cap follows total training epochs from GUI unless explicitly overridden.
        loc_max_stage_epochs = int(config.get("localization_stage_max_epochs", config.get("epochs", 20)))
        use_manual_loc_switch = bool(config.get("use_manual_localization_switch", False))
        manual_loc_switch_epoch = int(config.get("manual_localization_switch_epoch", 20))
        loc_gate_patience = int(config.get("localization_gate_patience", 2))
        loc_gate_iou_threshold = float(config.get("localization_gate_iou", 0.55))
        loc_gate_center_error = float(config.get("localization_gate_center_error", 0.15))
        loc_gate_valid_rate = float(config.get("localization_gate_valid_rate", 0.9))
        crop_mix_start_gt = float(config.get("classification_crop_gt_prob_start", 1.0))
        crop_mix_end_gt = float(config.get("classification_crop_gt_prob_end", 0.0))
        crop_padding = float(config.get("classification_crop_padding", 0.35))
        crop_min_size = float(config.get("classification_crop_min_size_norm", 0.04))
        enable_roi_cache = True  # Always use precomputed crops for classification stage
        center_heatmap_weight = float(config.get("center_heatmap_weight", 1.0))
        center_heatmap_sigma = float(config.get("center_heatmap_sigma", 2.5))
        direct_center_weight = float(config.get("direct_center_weight", 2.0))
        # Use the largest bbox across all classes so every crop has the same
        # extent. This prevents the classifier from using background size as a
        # class cue and keeps training/inference consistent (inference doesn't
        # know the class label).
        global_fixed_wh = (0.2, 0.2)
        if use_localization and has_any_localization:
            all_w = []
            all_h = []
            max_n = min(
                len(getattr(train_dataset, "spatial_bboxes", [])),
                len(getattr(train_dataset, "spatial_bbox_valid", [])),
            )
            for i in range(max_n):
                bboxes_i = train_dataset.spatial_bboxes[i]   # [T, 4]
                valid_i = train_dataset.spatial_bbox_valid[i] # [T]
                for t in range(bboxes_i.size(0)):
                    if float(valid_i[t].item()) <= 0.5:
                        continue
                    x1, y1, x2, y2 = [float(v) for v in bboxes_i[t].tolist()]
                    w = max(1e-4, min(1.0, x2 - x1))
                    h = max(1e-4, min(1.0, y2 - y1))
                    all_w.append(w)
                    all_h.append(h)

            if all_w and all_h:
                gw = float(max(all_w))
                gh = float(max(all_h))
                global_fixed_wh = (max(1e-4, min(1.0, gw)), max(1e-4, min(1.0, gh)))

            raw_model = model.module if hasattr(model, "module") else model
            if getattr(raw_model, "use_localization", False) and getattr(raw_model, "localization_head", None) is not None:
                raw_model.localization_head.set_fixed_box_size(global_fixed_wh[0], global_fixed_wh[1])

        def _fixed_wh_for_labels(label_tensor: torch.Tensor, device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
            """Return the single global fixed box size for each sample in the batch."""
            if label_tensor is None:
                return None
            if not torch.is_tensor(label_tensor):
                return None
            if label_tensor.dim() == 0:
                label_tensor = label_tensor.view(1)
            B = label_tensor.size(0)
            wh = [list(global_fixed_wh)] * B
            return torch.tensor(wh, device=device, dtype=dtype)
        if use_localization and log_fn:
            if has_any_localization:
                n_with_bbox = sum(1 for v in train_dataset.spatial_bbox_valid if float(v.sum().item()) > 0.5)
                log_fn(
                    f"Localization Supervision: enabled ({n_with_bbox} clips with bbox)"
                )
                if enable_roi_cache:
                    log_fn("Classification ROI cache: enabled (precompute crops once, then augment in-memory).")
                if center_heatmap_weight > 0:
                    log_fn(
                        f"Center Heatmap Loss (Gaussian Focal): enabled (weight={center_heatmap_weight}, "
                        f"sigma_patches={center_heatmap_sigma})"
                    )
                if direct_center_weight > 0:
                    log_fn(
                        f"Direct Center Loss: enabled (weight={direct_center_weight})"
                    )
                log_fn(
                    f"Localization fixed box size: max across all classes, "
                    f"w={global_fixed_wh[0]:.4f}, h={global_fixed_wh[1]:.4f}"
                )
                if use_manual_loc_switch:
                    log_fn(
                        f"Manual localization switch enabled: phase will switch at epoch {manual_loc_switch_epoch}"
                    )
            else:
                log_fn("Localization Supervision: enabled but no clips have bbox annotations — will be skipped")

        def _split_localization_output(output):
            if (
                use_localization
                and isinstance(output, tuple)
                and len(output) >= 2
                and torch.is_tensor(output[-1])
                and output[-1].dim() in (2, 3)
                and output[-1].size(-1) == 4
            ):
                head_out = output[:-1]
                if len(head_out) == 1:
                    head_out = head_out[0]
                return head_out, output[-1]
            return output, None

        def _bbox_iou(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            x1 = torch.maximum(pred[:, 0], target[:, 0])
            y1 = torch.maximum(pred[:, 1], target[:, 1])
            x2 = torch.minimum(pred[:, 2], target[:, 2])
            y2 = torch.minimum(pred[:, 3], target[:, 3])
            inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
            area_p = (pred[:, 2] - pred[:, 0]).clamp(min=0) * (pred[:, 3] - pred[:, 1]).clamp(min=0)
            area_t = (target[:, 2] - target[:, 0]).clamp(min=0) * (target[:, 3] - target[:, 1]).clamp(min=0)
            union = area_p + area_t - inter
            return inter / (union + 1e-6)

        def _localization_metrics(pred_bboxes: torch.Tensor, target_bboxes: torch.Tensor, valid_mask: torch.Tensor):
            if pred_bboxes is None:
                return 0.0, 1.0, 0.0
            if pred_bboxes.dim() == 3:
                pred_bboxes = pred_bboxes[:, 0, :]
            valid = valid_mask > 0.5
            if int(valid.sum().item()) == 0:
                return 0.0, 1.0, 0.0
            pred = pred_bboxes[valid]
            tgt = target_bboxes[valid]
            iou = _bbox_iou(pred, tgt).mean().item()
            pred_cx = 0.5 * (pred[:, 0] + pred[:, 2])
            pred_cy = 0.5 * (pred[:, 1] + pred[:, 3])
            tgt_cx = 0.5 * (tgt[:, 0] + tgt[:, 2])
            tgt_cy = 0.5 * (tgt[:, 1] + tgt[:, 3])
            center_err = torch.sqrt((pred_cx - tgt_cx) ** 2 + (pred_cy - tgt_cy) ** 2).mean().item()
            valid_pred_rate = float(
                ((pred[:, 2] > pred[:, 0]) & (pred[:, 3] > pred[:, 1])).float().mean().item()
            )
            return float(iou), float(center_err), valid_pred_rate

        def _sanitize_bboxes(bboxes: torch.Tensor) -> torch.Tensor:
            """Pad bboxes proportionally to their own size, not the full frame.
            crop_padding is now a fraction of the bbox dimension (e.g. 0.20 = 20% of bbox w/h).
            """
            boxes = bboxes.clone()
            orig_shape = boxes.shape
            boxes = boxes.view(-1, 4)
            x1 = boxes[:, 0].clamp(0.0, 1.0)
            y1 = boxes[:, 1].clamp(0.0, 1.0)
            x2 = boxes[:, 2].clamp(0.0, 1.0)
            y2 = boxes[:, 3].clamp(0.0, 1.0)
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            w = (x2 - x1).abs().clamp(min=crop_min_size)
            h = (y2 - y1).abs().clamp(min=crop_min_size)
            # Proportional padding: add crop_padding * bbox_size on each side
            w = torch.clamp(w * (1.0 + 2.0 * crop_padding), max=1.0)
            h = torch.clamp(h * (1.0 + 2.0 * crop_padding), max=1.0)
            x1 = (cx - 0.5 * w).clamp(0.0, 1.0)
            y1 = (cy - 0.5 * h).clamp(0.0, 1.0)
            x2 = (cx + 0.5 * w).clamp(0.0, 1.0)
            y2 = (cy + 0.5 * h).clamp(0.0, 1.0)
            boxes[:, 0] = x1
            boxes[:, 1] = y1
            boxes[:, 2] = torch.maximum(x2, x1 + crop_min_size).clamp(0.0, 1.0)
            boxes[:, 3] = torch.maximum(y2, y1 + crop_min_size).clamp(0.0, 1.0)
            return boxes.view(orig_shape)

        def _clamp_bboxes_no_expand(bboxes: torch.Tensor) -> torch.Tensor:
            """Clamp/reorder bbox coordinates without padding or min-size expansion."""
            boxes = bboxes.clone()
            orig_shape = boxes.shape
            boxes = boxes.view(-1, 4)
            x1 = boxes[:, 0].clamp(0.0, 1.0)
            y1 = boxes[:, 1].clamp(0.0, 1.0)
            x2 = boxes[:, 2].clamp(0.0, 1.0)
            y2 = boxes[:, 3].clamp(0.0, 1.0)
            lo_x = torch.minimum(x1, x2)
            hi_x = torch.maximum(x1, x2)
            lo_y = torch.minimum(y1, y2)
            hi_y = torch.maximum(y1, y2)
            boxes[:, 0] = lo_x
            boxes[:, 1] = lo_y
            boxes[:, 2] = torch.maximum(hi_x, lo_x + 1e-4).clamp(0.0, 1.0)
            boxes[:, 3] = torch.maximum(hi_y, lo_y + 1e-4).clamp(0.0, 1.0)
            return boxes.view(orig_shape)

        def _crop_clips_with_bboxes(clips: torch.Tensor, bboxes: torch.Tensor) -> torch.Tensor:
            # clips: [B, T, C, H, W], bboxes: [B, 4] normalized.
            B, T, C, H, W = clips.shape
            out = torch.empty_like(clips)
            boxes = _sanitize_bboxes(bboxes)
            for i in range(B):
                x1 = int(round(float(boxes[i, 0].item()) * (W - 1)))
                y1 = int(round(float(boxes[i, 1].item()) * (H - 1)))
                x2 = int(round(float(boxes[i, 2].item()) * W))
                y2 = int(round(float(boxes[i, 3].item()) * H))
                x1 = max(0, min(x1, W - 1))
                y1 = max(0, min(y1, H - 1))
                x2 = max(x1 + 1, min(x2, W))
                y2 = max(y1 + 1, min(y2, H))
                sample = clips[i]  # [T, C, H, W]
                cropped = sample[:, :, y1:y2, x1:x2]
                if cropped.size(-1) < 2 or cropped.size(-2) < 2:
                    out[i] = sample
                    continue
                out[i] = F.interpolate(cropped, size=(H, W), mode="bilinear", align_corners=False)
            return out

        def _crop_single_clip_to_target(clip_tchw: torch.Tensor, bbox_xyxy: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
            # clip_tchw: [T,C,H,W], bbox_xyxy: [4] or [T,4] normalized
            T, C, H, W = clip_tchw.shape
            box = _sanitize_bboxes(bbox_xyxy)
            if box.dim() == 1:
                box = box.view(1, 4).repeat(T, 1)
            elif box.dim() == 2 and box.size(0) != T:
                if box.size(0) < T:
                    box = torch.cat([box, box[-1:].repeat(T - box.size(0), 1)], dim=0)
                else:
                    box = box[:T]

            out_frames = []
            for ti in range(T):
                bt = box[ti]
                x1 = int(round(float(bt[0].item()) * (W - 1)))
                y1 = int(round(float(bt[1].item()) * (H - 1)))
                x2 = int(round(float(bt[2].item()) * W))
                y2 = int(round(float(bt[3].item()) * H))
                x1 = max(0, min(x1, W - 1))
                y1 = max(0, min(y1, H - 1))
                x2 = max(x1 + 1, min(x2, W))
                y2 = max(y1 + 1, min(y2, H))
                frame = clip_tchw[ti : ti + 1]  # [1,C,H,W]
                cropped = frame[:, :, y1:y2, x1:x2]
                if cropped.size(-2) < 2 or cropped.size(-1) < 2:
                    resized = F.interpolate(frame, size=(out_h, out_w), mode="bilinear", align_corners=False)
                else:
                    resized = F.interpolate(cropped, size=(out_h, out_w), mode="bilinear", align_corners=False)
                out_frames.append(resized[0])
            return torch.stack(out_frames, dim=0)

        def _lock_temporal_box_size_from_first(bboxes: torch.Tensor) -> torch.Tensor:
            """For [B,T,4], use the frame-0 box for all frames (fully fixed crop)."""
            if bboxes.dim() != 3 or bboxes.size(1) < 1:
                return bboxes
            first = bboxes[:, 0:1, :].clone()  # [B,1,4]
            return first.expand(-1, bboxes.size(1), -1).contiguous()

        def _precompute_roi_cache(dataset_obj, split_name: str = "train") -> str:
            """Precompute localized crops and save to disk as .pt files.
            
            Returns the directory path containing the saved crops.
            The dataset __getitem__ can then load these directly with workers.
            """
            cache_dir = os.path.join(output_dir_base, f"{basename}_roi_crops_{split_name}")
            os.makedirs(cache_dir, exist_ok=True)
            was_training = model.training
            model.eval()
            try:
                total_items = len(dataset_obj.clips)
                for ds_idx in range(total_items):
                    loc_clip_noaoi = dataset_obj.load_modelres_clip_by_index(ds_idx).float().unsqueeze(0).to(device)
                    loc_wh = torch.tensor([[float(global_fixed_wh[0]), float(global_fixed_wh[1])]], device=device, dtype=loc_clip_noaoi.dtype)
                    with torch.no_grad():
                        loc_out = model(loc_clip_noaoi, return_localization=True, localization_box_wh=loc_wh)
                    _, pred = _split_localization_output(loc_out)
                    if pred is None:
                        pred = torch.tensor([[0.0, 0.0, 1.0, 1.0]], device=device, dtype=loc_clip_noaoi.dtype)
                    pred = _sanitize_bboxes(pred)
                    if pred.dim() == 3:
                        pred = _lock_temporal_box_size_from_first(pred)
                    bbox_for_crop = pred[0].detach().cpu() if pred.dim() == 3 else pred[0].detach().cpu()

                    # Safety fallback: use GT when predicted geometry is invalid.
                    if pred.dim() == 3:
                        invalid = bool(((bbox_for_crop[:, 2] <= bbox_for_crop[:, 0]) | (bbox_for_crop[:, 3] <= bbox_for_crop[:, 1])).any().item())
                    else:
                        invalid = bool(((bbox_for_crop[2] <= bbox_for_crop[0]) or (bbox_for_crop[3] <= bbox_for_crop[1])))
                    ds_valid = dataset_obj.spatial_bbox_valid[ds_idx]
                    if invalid and float(ds_valid[0].item() if ds_valid.dim() > 0 else ds_valid.item()) > 0.5:
                        ds_bbox = dataset_obj.spatial_bboxes[ds_idx]
                        ds_bbox_f0 = ds_bbox[0] if ds_bbox.dim() == 2 else ds_bbox
                        gt_box = _sanitize_bboxes(ds_bbox_f0.view(1, 4))[0].detach().cpu()
                        if pred.dim() == 3:
                            bbox_for_crop = gt_box.unsqueeze(0).expand(pred.size(1), -1).contiguous()
                        else:
                            bbox_for_crop = gt_box

                    # Localization crops come from the original frame (no AOI)
                    raw_clip = dataset_obj.load_fullres_clip_by_index(ds_idx, apply_aoi=False).float()
                    out_h = int(loc_clip_noaoi.shape[-2])
                    out_w = int(loc_clip_noaoi.shape[-1])
                    cropped = _crop_single_clip_to_target(raw_clip, bbox_for_crop, out_h, out_w).detach().cpu()
                    torch.save(cropped, os.path.join(cache_dir, f"{ds_idx}.pt"))

                    if log_fn and ((ds_idx + 1) % 50 == 0 or (ds_idx + 1) == total_items):
                        log_fn(f"Saving cropped clips [{split_name}]: {ds_idx+1}/{total_items}")
            finally:
                if was_training:
                    model.train()
            if log_fn:
                log_fn(f"Saved {total_items} cropped clips to {cache_dir}")
            return cache_dir

        def _precompute_embedding_cache(
            dataset_obj,
            split_name: str = "train",
            num_aug_versions: int = 1,
            use_augmentation: bool = False,
            multi_scale: bool = False,
        ) -> str:
            """Run each clip through the frozen backbone and save token tensors.

            When num_aug_versions > 1 and use_augmentation=True, each clip is
            passed through the backbone num_aug_versions times with different
            random augmentations applied, giving the temporal head diverse
            inputs each epoch without re-running the backbone at training time.
            Version 0 is always unaugmented (clean reference).
            Embeddings are stored as float16 to halve disk usage.

            When multi_scale=True, each clip is also passed through the backbone
            at half fps (every-other frame subsampled), and saved as {idx}_{v}_s.pt.
            This gives the temporal head both fine-grained and broader temporal context.

            Returns the cache directory path.
            """
            cache_dir = os.path.join(output_dir_base, f"{basename}_emb_cache_{split_name}")
            os.makedirs(cache_dir, exist_ok=True)
            was_training = model.training
            model.eval()
            raw_model = model.module if hasattr(model, "module") else model
            total_items = len(dataset_obj.clips)
            # Multi-scale doubles the number of backbone passes
            passes_per_item = (2 if multi_scale else 1) * num_aug_versions
            total_ops = total_items * passes_per_item
            ops_done = 0
            try:
                for ds_idx in range(total_items):
                    clip_info = dataset_obj.clips[ds_idx]
                    clip_id = clip_info["id"]
                    clip_path = dataset_obj._resolve_clip_path(clip_id)

                    # If the dataset has no transform, augmentation is a no-op regardless
                    can_augment = use_augmentation and dataset_obj.transform is not None
                    for v in range(num_aug_versions):
                        # Version 0: always clean (no augmentation)
                        apply_aug = can_augment and v > 0
                        clip_t = dataset_obj._load_clip(
                            clip_path,
                            target_size=dataset_obj.target_size,
                            apply_transform=apply_aug,
                        ).float().unsqueeze(0).to(device)  # [1, T, C, H, W]
                        with torch.no_grad():
                            tokens = raw_model.backbone(clip_t)  # [1, T*S, D]
                        # Save as float16 — halves disk usage with negligible precision loss
                        torch.save(tokens[0].cpu().half(), os.path.join(cache_dir, f"{ds_idx}_{v}.pt"))
                        ops_done += 1
                        if log_fn and (ops_done % 50 == 0 or ops_done == total_ops):
                            log_fn(
                                f"Caching backbone embeddings [{split_name}]: "
                                f"{ops_done}/{total_ops} "
                                f"(clip {ds_idx+1}/{total_items}, aug v{v})"
                            )

                        if multi_scale:
                            # Short scale: subsample every-other frame (half fps, same duration)
                            clip_short = clip_t[:, ::2, :, :, :]  # [1, T//2, C, H, W]
                            with torch.no_grad():
                                tokens_s = raw_model.backbone(clip_short)  # [1, T_s*S, D]
                            torch.save(tokens_s[0].cpu().half(), os.path.join(cache_dir, f"{ds_idx}_{v}_s.pt"))
                            ops_done += 1
                            if log_fn and (ops_done % 50 == 0 or ops_done == total_ops):
                                log_fn(
                                    f"Caching backbone embeddings [{split_name}]: "
                                    f"{ops_done}/{total_ops} "
                                    f"(clip {ds_idx+1}/{total_items}, aug v{v}, short-scale)"
                                )
            finally:
                if was_training:
                    model.train()
            scale_note = " + short-scale" if multi_scale else ""
            if log_fn:
                log_fn(
                    f"Saved {total_ops} embedding tensors to {cache_dir} "
                    f"({num_aug_versions} version(s) per clip{scale_note}, float16)"
                )
            return cache_dir

        if log_fn:
            if use_focal_loss:
                log_fn(f"Using Focal Loss (Active Learning): gamma={focal_gamma} (replaces CrossEntropy)")
        
        # Determine class-only loss function
        criterion = None
        if use_class_weights and hasattr(train_dataset, 'labels') and not use_ovr:
            if log_fn:
                log_fn("Computing class weights for loss...")
            class_weights = compute_class_weights(
                [l for l in train_dataset.labels if l >= 0],
                len(train_dataset.classes)
            ).to(device)
            if log_fn:
                log_fn(f"Class weights: {class_weights.tolist()}")
        if use_ovr:
            if use_asl:
                criterion = AsymmetricLoss(
                    gamma_neg=asl_gamma_neg,
                    gamma_pos=asl_gamma_pos,
                    clip=asl_clip,
                    reduction="none",
                )
            elif use_focal_loss:
                criterion = BinaryFocalLoss(gamma=focal_gamma)
            else:
                criterion = None  # handled inline with F.binary_cross_entropy_with_logits
            n_hn = sum(1 for s in train_dataset.ovr_suppress_idx if s >= 0)
            n_real = sum(1 for lbl in train_dataset.labels if lbl >= 0)

            # Per-head pos_weight: upweight positives for minority classes so each
            # binary head sees balanced effective pos/neg counts.
            num_c = len(train_dataset.classes)
            pos_counts = torch.zeros(num_c, dtype=torch.float32)
            
            if hasattr(train_dataset, 'frame_labels') and getattr(train_dataset, 'frame_labels', None) is not None:
                if log_fn:
                    log_fn("Computing OvR pos_weight using frame labels...")
                total_frames = 0
                for fl in train_dataset.frame_labels:
                    if fl is not None:
                        for lbl in fl:
                            if 0 <= lbl < num_c:
                                pos_counts[lbl] += 1.0
                                total_frames += 1
                neg_counts = float(total_frames) - pos_counts
            else:
                if log_fn:
                    log_fn("Computing OvR pos_weight using clip labels...")
                total_real = 0
                for ml in train_dataset.multi_labels:
                    if ml:
                        total_real += 1
                        for lbl in ml:
                            if 0 <= lbl < num_c:
                                pos_counts[lbl] += 1.0
                neg_counts = float(total_real) - pos_counts

            ovr_pos_weight = torch.ones(num_c, device=device)
            for ci in range(num_c):
                if pos_counts[ci] > 0:
                    raw_ratio = neg_counts[ci].item() / pos_counts[ci].item()
                    # ASL already suppresses easy negatives via its focusing term,
                    # so the full neg/pos ratio double-corrects for imbalance.
                    # Use sqrt when ASL is active for a softer complementary weight.
                    if use_asl:
                        ovr_pos_weight[ci] = max(1.0, raw_ratio ** 0.5)
                    else:
                        ovr_pos_weight[ci] = max(1.0, raw_ratio)
            ovr_pos_weight = ovr_pos_weight.clamp(max=50.0)

            # Helper/background classes (e.g. "Other"): set pos_weight to a fixed value
            # (default 1.0) so the head trains gently. Use ovr_pos_weight_f1_excluded=1.5
            # to slightly upweight them if desired.
            _f1_exclude_names_early = set(config.get("f1_exclude_classes", []))
            _ovr_pw_excluded = float(config.get("ovr_pos_weight_f1_excluded", 1.5))
            for ci in range(num_c):
                if train_dataset.classes[ci] in _f1_exclude_names_early:
                    ovr_pos_weight[ci] = _ovr_pw_excluded

            # Detect co-occurrence pairs from multi-label annotations
            cooccur_set = set()
            for ml in train_dataset.multi_labels:
                if len(ml) >= 2:
                    for a in ml:
                        for b in ml:
                            if a < b:
                                cooccur_set.add((a, b))
            allowed_cooccurrence = [[train_dataset.classes[a], train_dataset.classes[b]] for a, b in sorted(cooccur_set)]
            cooccur_lookup = {i: set() for i in range(num_c)}
            for a, b in cooccur_set:
                cooccur_lookup[a].add(b)
                cooccur_lookup[b].add(a)

            if log_fn:
                if use_asl:
                    loss_msg = f" + ASL(γ-={asl_gamma_neg}, γ+={asl_gamma_pos}, clip={asl_clip})"
                elif use_focal_loss:
                    loss_msg = f" + BinaryFocal(gamma={focal_gamma})"
                else:
                    loss_msg = ""
                log_fn(f"OvR mode: {num_c} heads, {n_real} real clips, "
                       f"{n_hn} near-negative clips{loss_msg}")
                pw_str = ", ".join(f"{train_dataset.classes[i]}={ovr_pos_weight[i]:.1f}" for i in range(num_c))
                log_fn(f"OvR per-head pos_weight: {pw_str}")
                if ovr_label_smoothing > 0:
                    log_fn(f"OvR label smoothing: {ovr_label_smoothing} (targets [{ovr_label_smoothing:.2f}, {1-ovr_label_smoothing:.2f}])")
                bs = config["batch_size"]
                n_distinct = num_c + (1 if n_hn > 0 else 0)
                if bs < num_c:
                    log_fn(f"WARNING: batch_size ({bs}) < num_classes ({num_c}). "
                           f"Some OvR heads will see no positives per batch. "
                           f"Recommend batch_size >= {num_c * 2} for stable OvR training.")
                elif bs < num_c * 2:
                    log_fn(f"Note: batch_size ({bs}) is small for {num_c}-class OvR. "
                           f"Consider increasing to {num_c * 2}+ for better per-head gradient signal.")
                if allowed_cooccurrence:
                    pairs_str = ", ".join(f"{a}+{b}" for a, b in allowed_cooccurrence)
                    log_fn(f"OvR allowed co-occurrence pairs: {pairs_str}")
        elif use_class_weights:
            if use_focal_loss:
                criterion = FocalLoss(gamma=focal_gamma, alpha=class_weights)
            else:
                criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            if use_focal_loss:
                criterion = FocalLoss(gamma=focal_gamma)
            else:
                criterion = nn.CrossEntropyLoss()

        if log_fn and not use_ovr:
            if use_focal_loss:
                w_msg = "weighted" if use_class_weights else "unweighted"
                log_fn(f"Using {w_msg} FocalLoss(gamma={focal_gamma})")
            else:
                if use_class_weights:
                    log_fn(f"Using Class-Weighted CrossEntropyLoss")
                else:
                    log_fn(f"Using standard CrossEntropyLoss (no class weights)")
        
        # Use the ACTUAL classes from the dataset for metadata to ensure correct mapping order
        class_names = train_dataset.classes

        # Classes excluded from F1 metrics (e.g., "Other"/"Background" helper classes).
        # They still train normally but don't affect best-model selection.
        _f1_exclude_names = set(config.get("f1_exclude_classes", []))
        _f1_include_indices = [i for i, c in enumerate(class_names) if c not in _f1_exclude_names]
        _f1_exclude_indices = {i for i, c in enumerate(class_names) if c in _f1_exclude_names}
        train_dataset.stitch_exclude_classes = _f1_exclude_indices
        slug_counts = {}
        class_key_map = {}
        class_label_map = {}
        for idx, cls_name in enumerate(class_names):
            base_slug = _slugify_class_name(cls_name)
            slug = base_slug
            counter = 1
            while slug in slug_counts:
                slug = f"{base_slug}_{counter}"
                counter += 1
            slug_counts[slug] = True
            key = f"val_f1_{slug}"
            class_key_map[idx] = key
            class_label_map[idx] = cls_name
        
        class_counts = Counter(train_dataset.labels)
             
        class_counts_named = {
            train_dataset.classes[idx]: count 
            for idx, count in class_counts.items()
            if 0 <= idx < len(train_dataset.classes)
        }
        
        clip_length_value = config.get("clip_length")
        if clip_length_value is None and hasattr(train_dataset, "clip_length"):
            clip_length_value = train_dataset.clip_length

        resolution_value = config.get("resolution")
        if resolution_value is None and hasattr(train_dataset, "target_size"):
            target_size = train_dataset.target_size
            if isinstance(target_size, (tuple, list)) and target_size:
                resolution_value = int(target_size[0])
            elif isinstance(target_size, int):
                resolution_value = int(target_size)

        def _json_safe(value):
            if value is None or isinstance(value, (bool, int, float, str)):
                return value
            if isinstance(value, dict):
                return {str(k): _json_safe(v) for k, v in value.items()}
            if isinstance(value, (list, tuple, set)):
                return [_json_safe(v) for v in value]
            return str(value)

        def _calibrate_ignore_thresholds_from_validation(
            score_chunks_by_class: dict[int, list[np.ndarray]],
            target_chunks_by_class: dict[int, list[np.ndarray]],
            class_names_local: list[str],
        ) -> Optional[dict]:
            """Pick per-class ignore thresholds from validation scores by F1."""
            per_class_thresholds = {}
            per_class_stats = {}
            weighted_thresholds = []
            weighted_supports = []
            for cls_idx, cls_name in enumerate(class_names_local):
                score_chunks = score_chunks_by_class.get(cls_idx, [])
                target_chunks = target_chunks_by_class.get(cls_idx, [])
                if not score_chunks or not target_chunks:
                    continue
                scores = np.concatenate(score_chunks).astype(np.float32, copy=False)
                targets = np.concatenate(target_chunks).astype(np.uint8, copy=False)
                if scores.size == 0 or targets.size != scores.size:
                    continue
                pos_support = int(targets.sum())
                neg_support = int(targets.size - pos_support)
                if pos_support < 3 or neg_support < 3:
                    continue

                base_grid = np.linspace(0.35, 0.90, 56, dtype=np.float32)
                quantiles = np.quantile(scores, np.linspace(0.05, 0.95, 19)).astype(np.float32)
                candidates = np.unique(np.clip(np.concatenate([base_grid, quantiles]), 0.35, 0.90))

                best_tau = 0.60
                best_f1 = -1.0
                best_precision = -1.0
                best_recall = -1.0
                for tau in candidates:
                    pred_pos = scores >= float(tau)
                    tp = float(np.sum(pred_pos & (targets == 1)))
                    fp = float(np.sum(pred_pos & (targets == 0)))
                    fn = float(np.sum((~pred_pos) & (targets == 1)))
                    precision = tp / max(1.0, tp + fp)
                    recall = tp / max(1.0, tp + fn)
                    f1 = (2.0 * precision * recall) / max(1e-8, precision + recall)
                    if (
                        (f1 > best_f1 + 1e-8)
                        or (abs(f1 - best_f1) <= 1e-8 and precision > best_precision + 1e-8)
                        or (
                            abs(f1 - best_f1) <= 1e-8
                            and abs(precision - best_precision) <= 1e-8
                            and float(tau) > best_tau
                        )
                    ):
                        best_tau = float(tau)
                        best_f1 = float(f1)
                        best_precision = float(precision)
                        best_recall = float(recall)

                per_class_thresholds[cls_name] = float(best_tau)
                per_class_stats[cls_name] = {
                    "positive_support": pos_support,
                    "negative_support": neg_support,
                    "best_f1": float(best_f1),
                    "best_precision": float(best_precision),
                    "best_recall": float(best_recall),
                }
                weighted_thresholds.append(float(best_tau))
                weighted_supports.append(max(1, pos_support))

            if not per_class_thresholds:
                return None

            global_threshold = float(
                np.average(np.asarray(weighted_thresholds, dtype=np.float32), weights=np.asarray(weighted_supports, dtype=np.float32))
            )
            global_threshold = max(0.35, min(0.90, global_threshold))
            return {
                "source": "validation_f1_calibration",
                "global_threshold": global_threshold,
                "per_class_thresholds": per_class_thresholds,
                "per_class_stats": per_class_stats,
            }

        ovr_pos_weight_named = None
        if use_ovr:
            ovr_pos_weight_named = {
                train_dataset.classes[i]: float(ovr_pos_weight[i].item())
                for i in range(len(train_dataset.classes))
            }
        stitch_excluded_names = []
        if hasattr(train_dataset, "stitch_exclude_classes"):
            stitch_excluded_names = [
                class_names[i] for i in sorted(train_dataset.stitch_exclude_classes)
                if 0 <= int(i) < len(class_names)
            ]
        sampler_mode = "balanced_batch" if batch_sampler is not None else ("weighted_random" if sampler is not None else "shuffle")

        head_metadata = {
            "classes": class_names,
            "num_classes": len(class_names),
            "clip_length": clip_length_value,
            "resolution": resolution_value,
            "training_samples": class_counts_named,
            "backbone_model": config.get("backbone_model", "videoprism_public_v1_base"),
            "head": {
                "type": "DilatedTemporalHead" if use_temporal_decoder else "SpatialPoolLinearHead",
                "dropout": config.get("dropout", 0.1),
                "use_localization": use_localization,
                "localization_hidden_dim": int(config.get("localization_hidden_dim", 256)),
                "localization_dropout": float(config.get("localization_dropout", 0.0)),
                "use_temporal_decoder": use_temporal_decoder,
                "frame_head_temporal_layers": int(config.get("frame_head_temporal_layers", 1)),
                "temporal_pool_frames": int(config.get("temporal_pool_frames", 1)),
                "num_stages": int(config.get("num_stages", 3)),
                "proj_dim": int(config.get("proj_dim", 256)),
            },
            "training_config": {
                "batch_size": config["batch_size"],
                "epochs": config["epochs"],
                "lr": config["lr"],
                "weight_decay": config.get("weight_decay", 0.001),
                "use_scheduler": use_scheduler,
                "scheduler_name": "CosineAnnealingLR" if use_scheduler else None,
                "warmup_epochs": int(warmup_epochs),
                "t_max_epochs": int(total_epochs - warmup_epochs),
                "eta_min": float(eta_min) if eta_min is not None else None,
                "use_ovr": use_ovr,
                "ovr_background_as_negative": bool(config.get("ovr_background_as_negative", False)) if use_ovr else False,
                "ovr_background_class_names": config.get("ovr_background_class_names", []) if use_ovr else [],
                "allowed_cooccurrence": allowed_cooccurrence if use_ovr else [],
                "cooccurrence_loss_mode": "ignore_negative_pairs" if use_ovr else None,
                "ovr_pos_weight": ovr_pos_weight_named if use_ovr else None,
                "use_hard_pair_mining": hard_pair_mining if use_ovr else False,
                "hard_pair_mode": "pairwise_margin" if (use_ovr and hard_pair_mining) else None,
                "hard_pairs": hard_pair_name_pairs if (use_ovr and hard_pair_mining) else [],
                "hard_pair_margin": hard_pair_margin if (use_ovr and hard_pair_mining) else None,
                "hard_pair_loss_weight": hard_pair_loss_weight if (use_ovr and hard_pair_mining) else None,
                "hard_pair_confusion_boost": hard_pair_confusion_boost if (use_ovr and hard_pair_mining) else None,
                "use_class_weights": use_class_weights,
                "resolution": resolution_value,
                "use_weighted_sampler": config.get("use_weighted_sampler", False),
                "use_balanced_sampler": (batch_sampler is not None),
                "sampler_mode": sampler_mode,
                "use_augmentation": config.get("use_augmentation", False),
                "augmentation_options": config.get("augmentation_options") or {},
                "stitch_augmentation_prob": float(getattr(train_dataset, "stitch_prob", 0.0)),
                "emb_cache": bool(config.get("emb_cache", False)),
                "emb_aug_versions": int(config.get("emb_aug_versions", 1)),
                "stitch_excluded_classes": stitch_excluded_names,
                "use_focal_loss": use_focal_loss,
                "focal_gamma": focal_gamma if use_focal_loss else None,
                "use_supcon_loss": use_supcon_loss,
                "supcon_weight": supcon_weight if use_supcon_loss else None,
                "supcon_temperature": supcon_temperature if use_supcon_loss else None,
                "use_asl": use_asl if use_ovr else False,
                "asl_gamma_neg": asl_gamma_neg if (use_ovr and use_asl) else None,
                "asl_gamma_pos": asl_gamma_pos if (use_ovr and use_asl) else None,
                "asl_clip": asl_clip if (use_ovr and use_asl) else None,
                "use_ema": use_ema,
                "ema_decay": float(ema_decay) if use_ema else None,
                "ema_start_epoch": int(max(warmup_epochs, 3)) if use_ema else None,
                "frame_loss_weight": 1.0,
                "use_temporal_decoder": use_temporal_decoder,
                "use_frame_bout_balance": use_frame_bout_balance,
                "frame_bout_balance_power": frame_bout_balance_power if use_frame_bout_balance else None,
                "temporal_pool_frames": int(config.get("temporal_pool_frames", 1)),
                "frame_head_temporal_layers": int(config.get("frame_head_temporal_layers", 1)),
                "num_stages": int(config.get("num_stages", 3)),
                "boundary_loss_weight": boundary_loss_weight,
                "smoothness_loss_weight": smoothness_loss_weight,
                "boundary_tolerance": boundary_tolerance,
                "proj_dim": int(config.get("proj_dim", 256)),
                "use_localization": use_localization,
                "use_manual_localization_switch": use_manual_loc_switch if use_localization else None,
                "manual_localization_switch_epoch": manual_loc_switch_epoch if (use_localization and use_manual_loc_switch) else None,
                "localization_stage_max_epochs": loc_max_stage_epochs if use_localization else None,
                "localization_gate_patience": loc_gate_patience if use_localization else None,
                "localization_gate_iou": loc_gate_iou_threshold if use_localization else None,
                "localization_gate_center_error": loc_gate_center_error if use_localization else None,
                "localization_gate_valid_rate": loc_gate_valid_rate if use_localization else None,
                "classification_crop_gt_prob_start": crop_mix_start_gt if use_localization else None,
                "classification_crop_gt_prob_end": crop_mix_end_gt if use_localization else None,
                "classification_crop_padding": crop_padding if use_localization else None,
                "crop_jitter": config.get("crop_jitter", False),
                "crop_jitter_strength": config.get("crop_jitter_strength", 0.15),
                "classification_crop_min_size_norm": crop_min_size if use_localization else None,
                "center_heatmap_weight": center_heatmap_weight if use_localization else None,
                "center_heatmap_sigma": center_heatmap_sigma if use_localization else None,
                "direct_center_weight": direct_center_weight if use_localization else None,
                "localization_fixed_size_stat": "max" if use_localization else None,
                "localization_fixed_box_global_w": global_fixed_wh[0] if use_localization else None,
                "localization_fixed_box_global_h": global_fixed_wh[1] if use_localization else None,
                "val_split": config.get("val_split", 0.2),
                "use_all_for_training": config.get("use_all_for_training", False),
                "config_snapshot": _json_safe(config),
            }
        }
        
        if log_fn:
            log_fn(f"Training with classes: {class_names}")
            log_fn(f"Training samples per class (primary label): {class_counts_named}")
            # Multi-class breakdown
            mc_count = sum(1 for ml in train_dataset.multi_labels if len(ml) > 1)
            if mc_count > 0:
                from collections import Counter as _Counter
                mc_combos = _Counter(
                    tuple(sorted(train_dataset.classes[i] for i in ml))
                    for ml in train_dataset.multi_labels if len(ml) > 1
                )
                log_fn(f"Multi-class clips in training set: {mc_count} of {len(train_dataset.labels)}")
                for combo, cnt in mc_combos.most_common():
                    log_fn(f"  {' + '.join(combo)}: {cnt}")

        best_val_frame_acc = 0.0
        best_val_f1 = -1.0
        
        if log_fn:
            log_fn(f"Starting training for {config['epochs']} epochs...")

        history = {
            "epoch": [],
            "train_loss": [],
            "train_loss_class": [],  # primary classification loss only
            "train_acc": [],
            "train_frame_acc": [],
            "val_loss": [],
            "val_acc": [],
            "val_frame_acc": [],
            "val_f1": [],
            "loc_val_iou": [],
            "loc_val_center_error": [],
            "loc_val_valid_rate": [],
        }
        for key in class_key_map.values():
            history[key] = []
            

        # Curriculum state: stage 1 localization -> stage 2 classification on crops.
        in_localization_stage = bool(use_localization and has_any_localization)
        localization_gate_streak = 0
        classification_stage_start_epoch = None
        output_dir_base = os.path.dirname(config["output_path"])
        basename = os.path.splitext(os.path.basename(config["output_path"]))[0]
        crop_progress_num_samples = int(config.get("crop_progress_num_samples", 5))
        crop_progress_dir = None
        train_roi_cache = None
        val_roi_cache = None
        train_emb_cache = None
        val_emb_cache_dir = None
        
        # Save sample clips exactly as VideoPrism sees them (with augmentation)
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            import cv2
            
            samples_dir = os.path.join(output_dir_base, f"{basename}_input_samples")
            os.makedirs(samples_dir, exist_ok=True)
            loc_samples_dir = os.path.join(output_dir_base, f"{basename}_localized_input_samples")
            crop_progress_dir = os.path.join(output_dir_base, f"{basename}_crop_progress")
            if use_localization:
                os.makedirs(loc_samples_dir, exist_ok=True)
                os.makedirs(crop_progress_dir, exist_ok=True)
            
            resolution = config.get("resolution", 288)
            grid_g = resolution // 18
            sample_indices = np.random.choice(len(train_dataset), size=min(5, len(train_dataset)), replace=False)
            
            for si, idx in enumerate(sample_indices):
                batch = train_dataset[idx]
                clip_tensor = batch[0]  # [T, C, H, W]
                T = clip_tensor.shape[0]
                frames = clip_tensor.permute(0, 2, 3, 1).numpy()  # [T, H, W, C]
                H, W = frames.shape[1], frames.shape[2]
                
                label_idx = batch[1] if isinstance(batch[1], int) else batch[1].item()
                if 0 <= label_idx < len(class_names):
                    label_name = class_names[label_idx]
                elif label_idx < 0:
                    label_name = "mixed/stitched"
                else:
                    label_name = f"Class {label_idx}"
                
                actual_idx = idx % len(train_dataset.clips)
                clip_id = train_dataset.clips[actual_idx].get("id", "?")
                orig_label = train_dataset.clips[actual_idx].get("label", "?")
                # Count valid frame labels
                fl_tensor = train_dataset.frame_labels[actual_idx]
                num_labeled_frames = (fl_tensor >= 0).sum().item()
                total_frames = fl_tensor.numel()
                
                if log_fn:
                    log_fn(f"  Sample {si+1}: idx={idx} actual={actual_idx} clip_id={clip_id} "
                           f"dataset_label={label_name} orig_label={orig_label} "
                           f"({num_labeled_frames}/{total_frames} frames labeled)")
                
                num_show = min(T, 10)
                frame_indices_show = np.linspace(0, T - 1, num_show, dtype=int)
                
                # Scale figure to true pixel size: each frame at 1:1 pixels
                dpi = 150
                fig_w = (W * num_show + 40) / dpi  # 40px padding between frames
                fig_h = (H + 80) / dpi  # 80px for two-line title
                fig, axes = plt.subplots(1, num_show, figsize=(fig_w, fig_h))
                if num_show == 1:
                    axes = [axes]
                
                for j, fi in enumerate(frame_indices_show):
                    frame_rgb = (np.clip(frames[fi], 0, 1) * 255).astype(np.uint8)
                    axes[j].imshow(frame_rgb)
                    # Draw patch grid
                    for g in range(1, grid_g):
                        axes[j].axhline(y=g * H / grid_g, color='white', linewidth=0.3, alpha=0.5)
                        axes[j].axvline(x=g * W / grid_g, color='white', linewidth=0.3, alpha=0.5)
                    axes[j].axis('off')
                    
                    # Show per-frame label if available
                    f_lbl_idx = int(fl_tensor[fi].item())
                    f_lbl_text = f'f{fi}'
                    if f_lbl_idx >= 0 and f_lbl_idx < len(class_names):
                        f_lbl_text += f'\n{class_names[f_lbl_idx]}'
                    elif f_lbl_idx >= 0:
                        f_lbl_text += f'\n{f_lbl_idx}'
                        
                    axes[j].set_title(f_lbl_text, fontsize=7)
                
                clip_id_short = os.path.basename(clip_id)
                fig.suptitle(
                    f'"{label_name}" — clip {idx} (actual={actual_idx})  |  '
                    f'res {H}×{W}, grid {grid_g}×{grid_g}, {T} frames, patch 18px\n'
                    f'file: {clip_id_short}  |  orig_label: {orig_label}  |  '
                    f'Frames Labeled: {num_labeled_frames}/{total_frames}',
                    fontsize=8, fontweight='bold'
                )
                plt.tight_layout(rect=[0, 0, 1, 0.90])
                safe_label = label_name.replace(" ", "_").replace("/", "-")
                plt.savefig(os.path.join(samples_dir, f'sample_{si+1}_{safe_label}.png'), dpi=150, bbox_inches='tight')
                plt.close()

                # Optional localization preview: crop from native-res video
                # so the preview matches what the classification head actually receives.
                if use_localization:
                    bbox_target = batch[3] if len(batch) >= 4 else None
                    bbox_valid = batch[4] if len(batch) >= 5 else None
                    if (
                        torch.is_tensor(bbox_target)
                        and torch.is_tensor(bbox_valid)
                        and float(bbox_valid.sum().item()) > 0.5
                    ):
                        # Use first valid frame bbox for the preview
                        if bbox_target.dim() == 2:
                            valid_mask = bbox_valid > 0.5
                            first_valid_t = int(valid_mask.float().argmax().item()) if valid_mask.any() else 0
                            bbox_target = bbox_target[first_valid_t]
                        x1, y1, x2, y2 = [float(v) for v in bbox_target.tolist()]
                        x1 = max(0.0, min(1.0, x1))
                        y1 = max(0.0, min(1.0, y1))
                        x2 = max(0.0, min(1.0, x2))
                        y2 = max(0.0, min(1.0, y2))
                        if x2 > x1 and y2 > y1:
                            # Load full-resolution clip for high-quality crop
                            actual_idx = idx % len(train_dataset.clips)
                            try:
                                raw_clip = train_dataset.load_fullres_clip_by_index(actual_idx)
                                raw_frames = raw_clip.permute(0, 2, 3, 1).numpy()  # [T, Hraw, Wraw, C]
                                Hraw, Wraw = raw_frames.shape[1], raw_frames.shape[2]
                            except Exception as e:
                                logger.debug("Could not load full-res clip by index: %s", e)
                                raw_frames = frames
                                Hraw, Wraw = H, W

                            fig2, axes2 = plt.subplots(2, num_show, figsize=(fig_w, (H * 2 + 80) / dpi))
                            if num_show == 1:
                                axes2 = axes2.reshape(2, 1)

                            # Bbox in pixel coords on the low-res frames (for display row 1)
                            ix1 = int(round(x1 * W))
                            iy1 = int(round(y1 * H))
                            ix2 = int(round(x2 * W))
                            iy2 = int(round(y2 * H))
                            ix1 = max(0, min(ix1, W - 1))
                            iy1 = max(0, min(iy1, H - 1))
                            ix2 = max(ix1 + 1, min(ix2, W))
                            iy2 = max(iy1 + 1, min(iy2, H))

                            # Bbox in pixel coords on the full-res frames (for crop)
                            rx1 = int(round(x1 * Wraw))
                            ry1 = int(round(y1 * Hraw))
                            rx2 = int(round(x2 * Wraw))
                            ry2 = int(round(y2 * Hraw))
                            rx1 = max(0, min(rx1, Wraw - 1))
                            ry1 = max(0, min(ry1, Hraw - 1))
                            rx2 = max(rx1 + 1, min(rx2, Wraw))
                            ry2 = max(ry1 + 1, min(ry2, Hraw))

                            for j, fi in enumerate(frame_indices_show):
                                frame_rgb = (np.clip(frames[fi], 0, 1) * 255).astype(np.uint8)

                                # Row 1: model-res frame with target bbox
                                axes2[0, j].imshow(frame_rgb)
                                bbox_rect = mpatches.Rectangle(
                                    (ix1, iy1), ix2 - ix1, iy2 - iy1,
                                    fill=False, edgecolor='orange', linewidth=1.5
                                )
                                axes2[0, j].add_patch(bbox_rect)
                                axes2[0, j].axis('off')
                                axes2[0, j].set_title(f'f{fi}', fontsize=7)

                                # Row 2: crop from full-res, resize to model resolution
                                raw_fi = min(fi, len(raw_frames) - 1)
                                raw_rgb = (np.clip(raw_frames[raw_fi], 0, 1) * 255).astype(np.uint8)
                                crop = raw_rgb[ry1:ry2, rx1:rx2]
                                if crop.size == 0:
                                    crop = raw_rgb
                                crop = cv2.resize(crop, (W, H), interpolation=cv2.INTER_AREA)
                                axes2[1, j].imshow(crop)
                                axes2[1, j].axis('off')

                            fig2.suptitle(
                                f'"{label_name}" — localization target crop preview (from native {Hraw}×{Wraw})',
                                fontsize=9, fontweight='bold'
                            )
                            axes2[0, 0].set_ylabel('Original+bbox', fontsize=9)
                            axes2[1, 0].set_ylabel('Crop preview', fontsize=9)
                            plt.tight_layout(rect=[0, 0, 1, 0.92])
                            plt.savefig(
                                os.path.join(loc_samples_dir, f'loc_sample_{si+1}_{label_name.replace(" ", "_")}.png'),
                                dpi=150,
                                bbox_inches='tight'
                            )
                            plt.close()
            
            if log_fn:
                log_fn(f"Saved {len(sample_indices)} input sample visualizations to {samples_dir}")
                if use_localization:
                    log_fn(f"Saved localization crop previews to {loc_samples_dir}")
                    log_fn(f"Crop progress visualizations will be saved every 2 epochs to {crop_progress_dir}")

            def _save_epoch_crop_progress(epoch_idx: int, phase_name: str):
                if not (use_localization and crop_progress_dir and len(train_dataset) > 0):
                    return
                if (epoch_idx + 1) % 2 != 0:
                    return
                was_training = model.training
                model.eval()
                try:
                    sample_count = min(max(1, crop_progress_num_samples), len(train_dataset))
                    sample_indices_epoch = np.random.choice(len(train_dataset), size=sample_count, replace=False)

                    if classification_stage_start_epoch is None:
                        gt_crop_prob = crop_mix_end_gt
                    else:
                        cls_stage_steps = max(1, config["epochs"] - classification_stage_start_epoch)
                        cls_epoch_idx = max(0, epoch_idx - classification_stage_start_epoch)
                        alpha = min(1.0, cls_epoch_idx / cls_stage_steps)
                        gt_crop_prob = crop_mix_start_gt + (crop_mix_end_gt - crop_mix_start_gt) * alpha

                    for si, idx in enumerate(sample_indices_epoch):
                        actual_idx = idx % len(train_dataset.clips)
                        # Mirror classification crop construction: non-augmented model-res clip.
                        try:
                            clip_tensor = train_dataset.load_modelres_clip_by_index(actual_idx).unsqueeze(0).to(device)
                        except Exception as e:
                            logger.debug("Could not load model-res clip by index: %s", e)
                            batch = train_dataset[idx]
                            clip_tensor = batch[0].unsqueeze(0).to(device)  # fallback

                        cp_valid = train_dataset.spatial_bbox_valid[actual_idx]
                        valid_gt = bool(float(cp_valid[0].item() if cp_valid.dim() > 0 else cp_valid.item()) > 0.5)
                        gt_bbox_raw = None
                        if valid_gt:
                            cp_bbox = train_dataset.spatial_bboxes[actual_idx]
                            cp_bbox_f0 = cp_bbox[0] if cp_bbox.dim() == 2 else cp_bbox
                            gt_bbox_raw = _clamp_bboxes_no_expand(
                                cp_bbox_f0.view(1, 4).to(device)
                            )[0]

                        with torch.no_grad():
                            loc_wh = torch.tensor([[float(global_fixed_wh[0]), float(global_fixed_wh[1])]], device=device, dtype=clip_tensor.dtype)
                            loc_out = model(clip_tensor, return_localization=True, localization_box_wh=loc_wh)
                            _, pred_bbox = _split_localization_output(loc_out)
                            if pred_bbox is None:
                                pred_bbox = torch.zeros((1, 4), device=device)
                            pred_bbox_raw_all = _clamp_bboxes_no_expand(pred_bbox)
                            if pred_bbox_raw_all.dim() == 3:
                                pred_bbox_raw = pred_bbox_raw_all[0, 0]
                            else:
                                pred_bbox_raw = pred_bbox_raw_all[0]

                        cls_bbox_raw = pred_bbox_raw.clone()
                        if (phase_name == "classification") and (gt_bbox_raw is not None) and (gt_crop_prob >= 0.5):
                            cls_bbox_raw = gt_bbox_raw.clone()

                        # Actual crop boxes used by the pipeline (with padding/min-size sanitization).
                        pred_bbox_used = _sanitize_bboxes(pred_bbox_raw.view(1, 4))[0]
                        cls_bbox_used = _sanitize_bboxes(cls_bbox_raw.view(1, 4))[0]

                        # Crop from native-resolution video
                        full_clip = clip_tensor[0].detach().cpu()
                        T = int(full_clip.shape[0])
                        H = int(full_clip.shape[2])  # model resolution
                        W = int(full_clip.shape[3])

                        try:
                            raw_clip = train_dataset.load_fullres_clip_by_index(actual_idx).float()
                            Hraw, Wraw = int(raw_clip.shape[2]), int(raw_clip.shape[3])
                        except Exception as e:
                            logger.debug("Could not load full-res clip for crop preview: %s", e)
                            raw_clip = full_clip.clone()
                            Hraw, Wraw = H, W

                        pred_crop = _crop_single_clip_to_target(raw_clip, pred_bbox_used, H, W)
                        cls_crop = _crop_single_clip_to_target(raw_clip, cls_bbox_used, H, W)

                        num_show = min(T, 8)
                        frame_indices_show = np.linspace(0, T - 1, num_show, dtype=int)
                        fig_w = max(8.0, 2.4 * num_show)
                        fig, axes = plt.subplots(3, num_show, figsize=(fig_w, 6.8))
                        if num_show == 1:
                            axes = axes.reshape(3, 1)

                        def _bbox_px(b):
                            x1 = int(round(float(b[0].item()) * (W - 1)))
                            y1 = int(round(float(b[1].item()) * (H - 1)))
                            x2 = int(round(float(b[2].item()) * W))
                            y2 = int(round(float(b[3].item()) * H))
                            x1 = max(0, min(x1, W - 1))
                            y1 = max(0, min(y1, H - 1))
                            x2 = max(x1 + 1, min(x2, W))
                            y2 = max(y1 + 1, min(y2, H))
                            return x1, y1, x2, y2

                        px_pred_used = _bbox_px(pred_bbox_used)
                        px_pred_raw = _bbox_px(pred_bbox_raw)
                        px_cls_used = _bbox_px(cls_bbox_used)

                        for j, fi in enumerate(frame_indices_show):
                            frame_full = (full_clip[fi].permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8)
                            frame_pred = (pred_crop[fi].permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8)
                            frame_cls = (cls_crop[fi].permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8)

                            axes[0, j].imshow(frame_full)
                            pred_raw_rect = mpatches.Rectangle(
                                (px_pred_raw[0], px_pred_raw[1]),
                                px_pred_raw[2] - px_pred_raw[0],
                                px_pred_raw[3] - px_pred_raw[1],
                                fill=False,
                                edgecolor='magenta',
                                linewidth=1.2,
                                linestyle='--'
                            )
                            axes[0, j].add_patch(pred_raw_rect)
                            pred_rect = mpatches.Rectangle(
                                (px_pred_used[0], px_pred_used[1]),
                                px_pred_used[2] - px_pred_used[0],
                                px_pred_used[3] - px_pred_used[1],
                                fill=False,
                                edgecolor='cyan',
                                linewidth=1.4
                            )
                            axes[0, j].add_patch(pred_rect)
                            cls_rect = mpatches.Rectangle(
                                (px_cls_used[0], px_cls_used[1]),
                                px_cls_used[2] - px_cls_used[0],
                                px_cls_used[3] - px_cls_used[1],
                                fill=False,
                                edgecolor='lime',
                                linewidth=1.4
                            )
                            axes[0, j].add_patch(cls_rect)
                            axes[0, j].axis('off')
                            axes[0, j].set_title(f"f{fi}", fontsize=7)

                            axes[1, j].imshow(frame_pred)
                            axes[1, j].axis('off')
                            axes[2, j].imshow(frame_cls)
                            axes[2, j].axis('off')

                        axes[0, 0].set_ylabel("Full+boxes (clip-level)", fontsize=9)
                        axes[1, 0].set_ylabel("Pred crop", fontsize=9)
                        axes[2, 0].set_ylabel("Cls input", fontsize=9)

                        cp_clip_id = train_dataset.clips[actual_idx].get("id", "?")
                        cp_label_name = class_names[train_dataset.labels[actual_idx]] if train_dataset.labels[actual_idx] >= 0 else "?"
                        cp_clip_short = os.path.basename(str(cp_clip_id))
                        if phase_name == "localization":
                            title_note = (
                                f"Epoch {epoch_idx+1} | LOCALIZATION STAGE | \"{cp_label_name}\" | {cp_clip_short}\n"
                                f"row0: model-res {H}×{W}, crops from native {Hraw}×{Wraw} | crop_padding={crop_padding} | "
                                "magenta=pred raw, cyan=pred+padding"
                            )
                        else:
                            title_note = (
                                f"Epoch {epoch_idx+1} | CLASSIFICATION STAGE | \"{cp_label_name}\" | {cp_clip_short} | gt_mix={gt_crop_prob:.2f}\n"
                                f"row0: model-res {H}×{W}, crops from native {Hraw}×{Wraw} | crop_padding={crop_padding} | "
                                "magenta=pred raw, cyan=pred+padding, lime=cls input"
                            )
                        fig.suptitle(title_note, fontsize=9, fontweight='bold')
                        plt.tight_layout(rect=[0, 0, 1, 0.93])
                        out_name = f"epoch_{epoch_idx+1:03d}_sample_{si+1}.png"
                        plt.savefig(os.path.join(crop_progress_dir, out_name), dpi=140, bbox_inches='tight')
                        plt.close(fig)

                    if log_fn:
                        log_fn(
                            f"Saved crop progress previews for epoch {epoch_idx+1} to {crop_progress_dir} "
                            f"(visualization only; classification uses precomputed ROI cache)"
                        )
                except Exception as e_crop:
                    if log_fn:
                        log_fn(f"Note: Could not save crop progress previews for epoch {epoch_idx+1}: {e_crop}")
                finally:
                    if was_training:
                        model.train()
        except Exception as e:
            if log_fn:
                log_fn(f"Note: Could not save input samples: {e}")
            def _save_epoch_crop_progress(epoch_idx: int, phase_name: str):
                return
        
        for epoch in range(config["epochs"]):
            if stop_callback and stop_callback():
                if log_fn:
                    log_fn("Training stopped by user.")
                break
                
            if progress_callback:
                progress_callback(epoch + 1, config["epochs"])
            
            current_lr = optimizer.param_groups[0]['lr']
            epoch_phase = "localization" if in_localization_stage else "classification"
            if log_fn:
                log_fn(f"\n=== Epoch {epoch+1}/{config['epochs']} | phase={epoch_phase} (LR: {current_lr:.2e}) ===")

            # On first classification epoch: crop all clips using the trained
            # localization head, save to disk, and switch datasets to load from
            # those .pt files. DataLoader uses normal num_workers so classification
            # training runs at the same speed as standard (no-localization) training.
            if not in_localization_stage and use_localization and has_any_localization and enable_roi_cache:
                if train_roi_cache is None:
                    if log_fn:
                        log_fn("Cropping clips using localization model and saving to disk (train split)...")
                    train_roi_cache = _precompute_roi_cache(train_dataset, split_name="train")
                    train_dataset._roi_cache_mode = True
                    train_dataset._roi_cache_dir = train_roi_cache
                    if log_fn:
                        log_fn("Dataset switched to disk-cached crops. Training now identical to standard classifier.")
                    # Recreate loader with normal workers (loading .pt files is fast)
                    if batch_sampler is not None:
                        train_loader = DataLoader(
                            train_dataset, batch_sampler=batch_sampler,
                            num_workers=num_workers,
                            pin_memory=True if device.type == "cuda" else False,
                            persistent_workers=True if num_workers > 0 else False,
                        )
                    else:
                        train_loader = DataLoader(
                            train_dataset, batch_size=config["batch_size"],
                            shuffle=shuffle, sampler=sampler, num_workers=num_workers,
                            pin_memory=True if device.type == "cuda" else False,
                            persistent_workers=True if num_workers > 0 else False,
                        )
                if val_roi_cache is None and val_dataset is not None:
                    if log_fn:
                        log_fn("Cropping clips using localization model and saving to disk (val split)...")
                    val_roi_cache = _precompute_roi_cache(val_dataset, split_name="val")
                    val_dataset._roi_cache_mode = True
                    val_dataset._roi_cache_dir = val_roi_cache
                    val_loader = DataLoader(
                        val_dataset, batch_size=config["batch_size"],
                        shuffle=False, num_workers=num_workers,
                        pin_memory=True if device.type == "cuda" else False,
                        persistent_workers=True if num_workers > 0 else False,
                    )

            # Backbone embeddings are pre-computed once for classification training.
            # If clip-stitch is enabled, stitching then happens on cached embeddings.
            # Backbone is frozen during classification — caching is always equivalent
            # to running it live and avoids 10-50x redundant computation per epoch.
            emb_aug_versions = max(1, int(config.get("emb_aug_versions", 1)))
            use_multi_scale = bool(config.get("multi_scale", False))
            if not in_localization_stage and not use_localization:
                if not getattr(train_dataset, '_emb_cache_mode', False):
                    aug_note = f" × {emb_aug_versions} augmented versions" if emb_aug_versions > 1 else ""
                    ms_note = " + short-scale (multi-scale)" if use_multi_scale else ""
                    if log_fn:
                        log_fn(
                            f"Pre-computing backbone embeddings "
                            f"(train split{aug_note}{ms_note})..."
                        )
                    emb_cache_dir = _precompute_embedding_cache(
                        train_dataset, split_name="train",
                        num_aug_versions=emb_aug_versions,
                        use_augmentation=emb_aug_versions > 1,
                        multi_scale=use_multi_scale,
                    )
                    train_emb_cache = emb_cache_dir
                    train_dataset._emb_cache_mode = True
                    train_dataset._emb_cache_dir = emb_cache_dir
                    train_dataset._emb_clip_length = config.get("clip_length", 8)
                    train_dataset._emb_num_versions = emb_aug_versions
                    train_dataset._emb_multi_scale = use_multi_scale
                    # batch_sampler is mutually exclusive with batch_size/shuffle/sampler
                    _pin = device.type == "cuda"
                    _pw = num_workers > 0
                    if batch_sampler is not None:
                        train_loader = DataLoader(
                            train_dataset, batch_sampler=batch_sampler,
                            num_workers=num_workers, pin_memory=_pin, persistent_workers=_pw,
                        )
                    else:
                        train_loader = DataLoader(
                            train_dataset, batch_size=config["batch_size"],
                            shuffle=shuffle, sampler=sampler,
                            num_workers=num_workers, pin_memory=_pin, persistent_workers=_pw,
                        )
                if val_dataset is not None and not getattr(val_dataset, '_emb_cache_mode', False):
                    if log_fn:
                        log_fn("Pre-computing backbone embeddings (val split, no aug)...")
                    # Validation always uses a single clean (unaugmented) version
                    val_emb_cache = _precompute_embedding_cache(
                        val_dataset, split_name="val",
                        num_aug_versions=1, use_augmentation=False,
                        multi_scale=use_multi_scale,
                    )
                    val_emb_cache_dir = val_emb_cache
                    val_dataset._emb_cache_mode = True
                    val_dataset._emb_cache_dir = val_emb_cache
                    val_dataset._emb_clip_length = config.get("clip_length", 8)
                    val_dataset._emb_num_versions = 1
                    val_dataset._emb_multi_scale = use_multi_scale
                    # No stitching during validation — we want clean per-clip evaluation
                    val_dataset.stitch_prob = 0.0
                    val_loader = DataLoader(
                        val_dataset, batch_size=config["batch_size"],
                        shuffle=False, num_workers=num_workers,
                        pin_memory=device.type == "cuda",
                        persistent_workers=num_workers > 0,
                    )

            model.train()
            total_loss = 0.0
            total_loss_class = 0.0
            correct = 0
            total = 0
            frame_correct = 0
            frame_total = 0
            train_targets_all = []
            train_preds_all = []
            # Per-clip confusion scores accumulated over this epoch (OvR only).
            # Use real clip count, not the virtual-expanded dataset length.
            _n_clips = len(train_dataset.clips) if hasattr(train_dataset, 'clips') else len(train_dataset)
            _epoch_confusion = np.zeros(_n_clips, dtype=np.float32)
            _epoch_confusion_count = np.zeros(_n_clips, dtype=np.int32)
            _epoch_top_rival = np.full(_n_clips, -1, dtype=np.int32)
            
            try:
                for batch_idx, batch_data in enumerate(train_loader):
                    if stop_callback and stop_callback():
                        break
                        
                    try:
                        frame_labels_batch = None
                        clips_short_batch = None
                        bg_mask_batch = None
                        suppress_batch = None
                        if isinstance(batch_data, (list, tuple)) and len(batch_data) >= 7:
                            clips, labels, spatial_masks_batch, spatial_bboxes_batch, spatial_bbox_valid_batch, sample_indices_batch, frame_labels_batch = batch_data[:7]
                        else:
                            clips, labels, spatial_masks_batch, spatial_bboxes_batch, spatial_bbox_valid_batch, sample_indices_batch = batch_data[:6]
                        if isinstance(batch_data, (list, tuple)) and len(batch_data) >= 8:
                            _cs = batch_data[7]
                            if isinstance(_cs, torch.Tensor) and _cs.numel() > 0:
                                clips_short_batch = _cs
                        if isinstance(batch_data, (list, tuple)) and len(batch_data) >= 9:
                            _bg = batch_data[8]
                            if isinstance(_bg, torch.Tensor) and _bg.numel() > 0:
                                bg_mask_batch = _bg.to(device=device, dtype=torch.bool)
                        clips = clips.to(device)
                        labels = labels.to(device)
                        if frame_labels_batch is not None:
                            frame_labels_batch = frame_labels_batch.to(device)
                        if use_ovr:
                            suppress_batch = _lookup_ovr_suppress_for_batch(sample_indices_batch, train_dataset, device)
                        spatial_bboxes_batch = spatial_bboxes_batch.to(device)
                        spatial_bbox_valid_batch = spatial_bbox_valid_batch.to(device)
                        
                        optimizer.zero_grad()
                        attn_w = None

                        # First-frame bbox slices for classification stage crop decisions
                        spatial_bboxes_f0 = spatial_bboxes_batch[:, 0, :] if spatial_bboxes_batch.dim() == 3 else spatial_bboxes_batch
                        spatial_bbox_valid_f0 = spatial_bbox_valid_batch[:, 0] if spatial_bbox_valid_batch.dim() == 2 else spatial_bbox_valid_batch

                        # Stage 1: train localization on full frames (all frames supervised).
                        # spatial_bboxes_batch: [B, T, 4], spatial_bbox_valid_batch: [B, T]
                        if in_localization_stage:
                            loc_fixed_wh = _fixed_wh_for_labels(labels, device=clips.device, dtype=clips.dtype)
                            loc_out = model(
                                clips,
                                return_localization=True,
                                cache_backbone_tokens=True,
                                localization_box_wh=loc_fixed_wh,
                            )
                            _, loc_pred_bboxes = _split_localization_output(loc_out)
                            if loc_pred_bboxes is None or not has_any_localization:
                                raise RuntimeError("Localization stage enabled but localization predictions/targets are unavailable.")

                            raw_model = model.module if hasattr(model, "module") else model
                            backbone_tokens = raw_model._backbone_tokens

                            # Flatten temporal dimension for per-frame supervision:
                            # loc_pred_bboxes: [B, T, 4] or [B, 4]
                            B_loc = spatial_bboxes_batch.size(0)
                            T_loc = spatial_bboxes_batch.size(1) if spatial_bboxes_batch.dim() == 3 else 1
                            tgt_flat = spatial_bboxes_batch.view(B_loc * T_loc, 4)       # [B*T, 4]
                            valid_flat = spatial_bbox_valid_batch.view(B_loc * T_loc)     # [B*T]
                            if loc_pred_bboxes.dim() == 2:
                                # [B, 4] → repeat for T frames
                                pred_flat = loc_pred_bboxes.unsqueeze(1).expand(-1, T_loc, -1).reshape(B_loc * T_loc, 4)
                            else:
                                pred_flat = loc_pred_bboxes.view(B_loc * T_loc, 4)       # [B*T, 4]

                            # Primary: center heatmap with Gaussian focal loss (all frames)
                            obj_logits = raw_model.localization_head.get_objectness_logits(
                                backbone_tokens, num_frames=train_dataset.clip_length,
                                all_frames=(T_loc > 1),
                            )  # [B*T, S] when all_frames=True, else [B, S]
                            from .model import center_heatmap_loss, direct_center_loss
                            chm_loss = center_heatmap_loss(
                                obj_logits,
                                tgt_flat,
                                valid_flat,
                                sigma_in_patches=center_heatmap_sigma,
                            )
                            loss = center_heatmap_weight * chm_loss
                            chm_loss_val = chm_loss.item()

                            # Secondary: direct center-to-center regression (all frames)
                            dc_loss = direct_center_loss(
                                pred_flat,
                                tgt_flat,
                                valid_flat,
                            )
                            loss = loss + direct_center_weight * dc_loss
                            dc_loss_val = dc_loss.item()

                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            if any(torch.isnan(p.grad).any() for p in model.parameters() if p.grad is not None):
                                optimizer.zero_grad()
                                if log_fn and batch_idx % 10 == 0:
                                    log_fn(f"[loc] Skipped batch {batch_idx} (NaN grad)")
                            else:
                                optimizer.step()
                                _update_ema()

                            batch_size = clips.size(0)
                            total_loss += loss.item() * batch_size
                            total_loss_class += 0.0
                            total += batch_size

                            if log_fn and batch_idx % 10 == 0:
                                iou_val, center_err, valid_rate = _localization_metrics(
                                    pred_flat.detach(),
                                    tgt_flat,
                                    valid_flat,
                                )
                                log_fn(
                                    f"Epoch {epoch+1}/{config['epochs']}, Batch {batch_idx}/{len(train_loader)}, "
                                    f"Loc Loss: {loss.item():.4f}, CHM: {chm_loss_val:.4f}, DC: {dc_loss_val:.4f}, "
                                    f"CErr: {center_err:.3f}, IoU: {iou_val:.3f}, VRate: {valid_rate:.3f}"
                                )
                            continue

                        # Classification forward: if embedding cache is active, clips
                        # are pre-computed tokens [B, T*S, D]; otherwise (localization
                        # pipeline) clips are raw pixels [B, T, C, H, W].
                        _clip_len = config.get("clip_length", 8)
                        _emb_active = getattr(train_dataset, '_emb_cache_mode', False)
                        if _emb_active:
                            _cs = clips_short_batch.to(device) if clips_short_batch is not None else None
                            logits = model(
                                None,
                                backbone_tokens=clips,
                                num_frames=_clip_len,
                                backbone_tokens_short=_cs,
                                num_frames_short=_clip_len // 2 if _cs is not None else None,
                                return_localization=False,
                                return_frame_logits=True,
                            )
                        else:
                            logits = model(
                                clips,
                                return_localization=False,
                                return_frame_logits=True,
                            )
                        
                        # Frame-level multi-task loss
                        _fo = getattr(model, '_frame_output', None)
                        if _fo is not None and frame_labels_batch is not None and not in_localization_stage:
                            f_logits = _fo[0]
                            f_logits_pooled = _fo[3] if len(_fo) > 3 else f_logits
                            pool_n = int(_fo[4]) if len(_fo) > 4 else 1
                            boundary_logits_out = _fo[5] if len(_fo) > 5 else None
                            frame_embeddings_out = _fo[6] if len(_fo) > 6 else None
                            labels_for_loss = _pool_frame_labels(frame_labels_batch, pool_n)
                            logits_for_loss = f_logits_pooled if pool_n > 1 else f_logits
                            embeddings_for_loss = _pool_frame_embeddings(frame_embeddings_out, pool_n)
                            bg_mask_for_loss = None
                            if use_ovr and ovr_background_as_negative and bg_mask_batch is not None:
                                bg_mask_for_loss = _pool_binary_mask(bg_mask_batch, pool_n)

                            # L_state: frame classification loss
                            if use_ovr:
                                B_f, T_f, C_f = logits_for_loss.shape
                                fl_ovr_targets = torch.full((B_f, T_f, C_f), ovr_label_smoothing, device=device)
                                fl_ovr_weight = torch.ones(B_f, T_f, C_f, device=device)
                                for bi in range(B_f):
                                    for ti in range(T_f):
                                        lbl = int(labels_for_loss[bi, ti].item())
                                        if 0 <= lbl < C_f:
                                            fl_ovr_targets[bi, ti, lbl] = 1.0 - ovr_label_smoothing
                                            fl_ovr_weight[bi, ti, lbl] = ovr_pos_weight[lbl]
                                            for cj in cooccur_lookup.get(lbl, ()):
                                                if cj != lbl and 0 <= cj < C_f:
                                                    fl_ovr_weight[bi, ti, cj] = 0.0
                                suppress_mask_for_loss = None
                                if suppress_batch is not None:
                                    suppress_mask_for_loss = (labels_for_loss < 0) & (suppress_batch.view(-1, 1) >= 0)
                                    if suppress_mask_for_loss.any():
                                        fl_ovr_weight = fl_ovr_weight.masked_fill(
                                            suppress_mask_for_loss.unsqueeze(-1), 0.0
                                        )
                                        hn_b, hn_t = torch.nonzero(suppress_mask_for_loss, as_tuple=True)
                                        hn_c = suppress_batch[hn_b]
                                        fl_ovr_targets[hn_b, hn_t, hn_c] = 0.0
                                        fl_ovr_weight[hn_b, hn_t, hn_c] = 1.0
                                if bg_mask_for_loss is not None and bg_mask_for_loss.any():
                                    fl_ovr_targets = fl_ovr_targets.masked_fill(
                                        bg_mask_for_loss.unsqueeze(-1), 0.0
                                    )
                                valid_ovr_mask = (
                                    (labels_for_loss >= 0) |
                                    (suppress_mask_for_loss if suppress_mask_for_loss is not None else torch.zeros_like(labels_for_loss, dtype=torch.bool)) |
                                    (bg_mask_for_loss if bg_mask_for_loss is not None else torch.zeros_like(labels_for_loss, dtype=torch.bool))
                                )
                                loss = _frame_loss_balanced(
                                    logits_for_loss, labels_for_loss,
                                    use_ovr_local=True, ovr_targets=fl_ovr_targets,
                                    ovr_weight=fl_ovr_weight,
                                    use_bout_balance=use_frame_bout_balance,
                                    bout_power=frame_bout_balance_power,
                                    valid_mask_override=valid_ovr_mask,
                                )
                            else:
                                loss = _frame_loss_balanced(
                                    logits_for_loss, labels_for_loss,
                                    use_ovr_local=False,
                                    use_bout_balance=use_frame_bout_balance,
                                    bout_power=frame_bout_balance_power,
                                )
                            if hard_pair_mining and hard_pair_loss_weight > 0:
                                pair_loss = _hard_pair_margin_loss(
                                    logits_for_loss,
                                    labels_for_loss,
                                    hard_pair_index_pairs,
                                    hard_pair_margin,
                                    use_bout_balance=use_frame_bout_balance,
                                    bout_power=frame_bout_balance_power,
                                )
                                loss = loss + hard_pair_loss_weight * pair_loss

                            # L_boundary: boundary detection loss
                            if boundary_logits_out is not None and boundary_loss_weight > 0:
                                boundary_labels_batch = _generate_boundary_labels(
                                    frame_labels_batch, tolerance=boundary_tolerance,
                                )
                                from .model import boundary_detection_loss
                                b_loss = boundary_detection_loss(
                                    boundary_logits_out, boundary_labels_batch,
                                )
                                loss = loss + boundary_loss_weight * b_loss

                            # L_smooth: temporal smoothness regularizer
                            if smoothness_loss_weight > 0:
                                from .model import temporal_smoothness_loss
                                s_loss = temporal_smoothness_loss(logits_for_loss, labels_for_loss)
                                loss = loss + smoothness_loss_weight * s_loss

                            if use_supcon_loss and supcon_weight > 0:
                                sc_loss = _supervised_contrastive_loss(
                                    embeddings_for_loss,
                                    labels_for_loss,
                                    temperature=supcon_temperature,
                                )
                                loss = loss + supcon_weight * sc_loss

                            with torch.no_grad():
                                valid_fl = labels_for_loss >= 0
                                if valid_fl.any():
                                    if use_ovr:
                                        f_pred = torch.argmax(torch.sigmoid(logits_for_loss.detach()), dim=-1)
                                    else:
                                        f_pred = torch.argmax(logits_for_loss.detach(), dim=-1)
                                    frame_total += int(valid_fl.sum().item())
                                    frame_correct += int((f_pred[valid_fl] == labels_for_loss[valid_fl]).sum().item())
                                    train_targets_all.extend(labels_for_loss[valid_fl].detach().cpu().tolist())
                                    train_preds_all.extend(f_pred[valid_fl].detach().cpu().tolist())

                                # Accumulate per-clip confusion scores for ConfusionAwareSampler
                                # Skip during warmup: model is still learning basics, scores are noise
                                _confusion_warmup_epoch = int(config["epochs"] * _confusion_warmup_pct)
                                if (use_ovr and isinstance(batch_sampler, ConfusionAwareSampler)
                                        and not in_localization_stage
                                        and (epoch + 1) > _confusion_warmup_epoch):
                                    probs = torch.sigmoid(logits_for_loss.detach())  # [B, T, C]
                                    B_cs = probs.shape[0]
                                    C_cs = probs.shape[-1]
                                    for bi in range(B_cs):
                                        raw_idx = int(sample_indices_batch[bi].item())
                                        clip_idx = raw_idx % _n_clips
                                        # Skip stitched clips (y=-1): mixed classes corrupt the score
                                        clip_label = int(labels[bi].item())
                                        if clip_label < 0 or clip_label >= C_cs:
                                            continue
                                        valid_t = labels_for_loss[bi] >= 0
                                        if not valid_t.any() or C_cs < 2:
                                            continue
                                        avg_p = probs[bi][valid_t].mean(0)  # [C]
                                        # Pair-aware confusion:
                                        # favor clips where a specific rival head stays high,
                                        # the true head stays weak, or the rival overtakes the true head.
                                        wrong_mask = torch.ones(C_cs, dtype=torch.bool, device=avg_p.device)
                                        wrong_mask[clip_label] = False
                                        if not wrong_mask.any():
                                            continue
                                        wrong_idx = torch.where(wrong_mask)[0]
                                        wrong_probs = avg_p[wrong_mask]
                                        top_local = int(torch.argmax(wrong_probs).item())
                                        top_rival = int(wrong_idx[top_local].item())
                                        top_wrong = float(wrong_probs[top_local].item())
                                        true_p = float(avg_p[clip_label].item())
                                        rival_margin = max(0.0, top_wrong - true_p)
                                        true_deficit = max(0.0, 1.0 - true_p)
                                        confusion = 0.55 * top_wrong + 0.30 * rival_margin + 0.15 * true_deficit
                                        if hard_pair_mining and hard_pair_confusion_boost > 1.0:
                                            pair_key = (min(clip_label, top_rival), max(clip_label, top_rival))
                                            if pair_key in hard_pair_index_pairs:
                                                confusion *= hard_pair_confusion_boost
                                        confusion = min(2.0, confusion)
                                        _epoch_confusion[clip_idx] += confusion
                                        _epoch_confusion_count[clip_idx] += 1
                                        _epoch_top_rival[clip_idx] = top_rival
                        else:
                            loss = logits.sum() * 0.0

                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        if any(torch.isnan(p.grad).any() for p in model.parameters() if p.grad is not None):
                            optimizer.zero_grad()
                            if log_fn and batch_idx % 10 == 0:
                                log_fn(f"[cls] Skipped batch {batch_idx} (NaN grad)")
                            continue
                        optimizer.step()
                        _update_ema()

                        total_loss += loss.item() * clips.size(0)
                        total_loss_class += loss.item() * clips.size(0)

                        with torch.no_grad():
                            if use_ovr:
                                predicted = torch.argmax(torch.sigmoid(logits.data), dim=1)
                            else:
                                _, predicted = torch.max(logits.data, 1)
                            valid_mask = labels >= 0
                            total += int(valid_mask.sum().item())
                            correct += int((predicted[valid_mask] == labels[valid_mask]).sum().item())
                        
                        if log_fn and batch_idx % 10 == 0:
                            log_fn(f"Epoch {epoch+1}/{config['epochs']}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
                    except Exception as e:
                        error_msg = f"Error in training batch {batch_idx}: {str(e)}\n{traceback.format_exc()}"
                        if log_fn:
                            log_fn(f"ERROR: {error_msg}")
                        raise
                
                if len(train_dataset) > 0:
                    avg_loss = total_loss / len(train_dataset)
                    avg_loss_class = total_loss_class / len(train_dataset)
                else:
                    avg_loss = 0.0
                    avg_loss_class = 0.0

                # Update ConfusionAwareSampler weights from this epoch's scores
                _confusion_warmup_epoch = int(config["epochs"] * _confusion_warmup_pct)
                if (use_ovr and isinstance(batch_sampler, ConfusionAwareSampler)
                        and not in_localization_stage
                        and (epoch + 1) > _confusion_warmup_epoch):
                    nonzero = _epoch_confusion_count > 0
                    avg_epoch_confusion = np.where(
                        nonzero,
                        _epoch_confusion / np.maximum(_epoch_confusion_count, 1),
                        batch_sampler._confusion_scores,  # keep last known score for unseen clips
                    )
                    avg_epoch_top_rival = np.where(
                        nonzero,
                        _epoch_top_rival,
                        batch_sampler._top_rival,
                    )
                    batch_sampler.update_weights(avg_epoch_confusion, top_rivals=avg_epoch_top_rival)
                    if log_fn and (epoch + 1) % 5 == 0:
                        log_fn("Confusion sampler — hardest clips per class:")
                        for line in batch_sampler.log_top_confused(
                            class_names, getattr(train_dataset, "clips", [])
                        ):
                            log_fn(line)
                
                train_acc = 100.0 * correct / total if total > 0 else 0.0
                train_frame_acc = 100.0 * frame_correct / frame_total if frame_total > 0 else 0.0
                
                if log_fn:
                    if use_frame_loss and not in_localization_stage:
                        log_fn(
                            f"Epoch {epoch+1}/{config['epochs']} - Train Loss: {avg_loss:.4f}, "
                            f"Train Acc (clip): {train_acc:.2f}%, Train Acc (frame): {train_frame_acc:.2f}%"
                        )
                    else:
                        log_fn(f"Epoch {epoch+1}/{config['epochs']} - Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%")
                
                # Record train metrics
                history["epoch"].append(epoch + 1)
                history["train_loss"].append(avg_loss)
                history["train_loss_class"].append(avg_loss_class)
                history["train_acc"].append(train_acc)
                history["train_frame_acc"].append(train_frame_acc)
                history["val_loss"].append(0.0)
                history["val_acc"].append(0.0)
                history["val_frame_acc"].append(0.0)
                history["val_f1"].append(0.0)
                history["loc_val_iou"].append(0.0)
                history["loc_val_center_error"].append(1.0 if in_localization_stage else 0.0)
                history["loc_val_valid_rate"].append(0.0)
                for key in class_key_map.values():
                    history[key].append(0.0)
                
                val_acc = 0.0
                avg_val_loss = 0.0
                per_attr_f1 = {}

                if val_loader:
                    _apply_ema()  # swap to EMA weights for validation
                    model.eval()
                    val_correct = 0
                    val_total = 0
                    val_frame_correct = 0
                    val_frame_total = 0
                    val_loss = 0.0
                    val_targets_all = []
                    val_preds_all = []
                    val_score_chunks_by_class = {i: [] for i in range(len(class_names))}
                    val_target_chunks_by_class = {i: [] for i in range(len(class_names))}
                    val_loc_iou_sum = 0.0
                    val_loc_center_sum = 0.0
                    val_loc_valid_sum = 0.0
                    val_loc_batches = 0
                    
                    try:
                        with torch.no_grad():
                            for batch_data in val_loader:
                                frame_labels_batch = None
                                clips_short_val = None
                                bg_mask_batch = None
                                suppress_batch = None
                                if isinstance(batch_data, (list, tuple)) and len(batch_data) >= 7:
                                    clips, labels, _, spatial_bboxes_batch, spatial_bbox_valid_batch, sample_indices_batch, frame_labels_batch = batch_data[:7]
                                else:
                                    clips, labels, _, spatial_bboxes_batch, spatial_bbox_valid_batch, sample_indices_batch = batch_data[:6]
                                if isinstance(batch_data, (list, tuple)) and len(batch_data) >= 8:
                                    _cs_v = batch_data[7]
                                    if isinstance(_cs_v, torch.Tensor) and _cs_v.numel() > 0:
                                        clips_short_val = _cs_v.to(device)
                                if isinstance(batch_data, (list, tuple)) and len(batch_data) >= 9:
                                    _bg_v = batch_data[8]
                                    if isinstance(_bg_v, torch.Tensor) and _bg_v.numel() > 0:
                                        bg_mask_batch = _bg_v.to(device=device, dtype=torch.bool)
                                clips = clips.to(device)
                                labels = labels.to(device)
                                if frame_labels_batch is not None:
                                    frame_labels_batch = frame_labels_batch.to(device)
                                if use_ovr:
                                    suppress_batch = _lookup_ovr_suppress_for_batch(sample_indices_batch, val_dataset, device)
                                spatial_bboxes_batch = spatial_bboxes_batch.to(device)
                                spatial_bbox_valid_batch = spatial_bbox_valid_batch.to(device)

                                spatial_bboxes_f0 = spatial_bboxes_batch[:, 0, :] if spatial_bboxes_batch.dim() == 3 else spatial_bboxes_batch
                                spatial_bbox_valid_f0 = spatial_bbox_valid_batch[:, 0] if spatial_bbox_valid_batch.dim() == 2 else spatial_bbox_valid_batch

                                if in_localization_stage:
                                    val_wh = _fixed_wh_for_labels(labels, device=clips.device, dtype=clips.dtype)
                                    val_out = model(clips, return_localization=True, localization_box_wh=val_wh)
                                    _, loc_pred_bboxes = _split_localization_output(val_out)
                                    if loc_pred_bboxes is None:
                                        continue

                                    # Flatten temporal dim for per-frame eval
                                    B_v = spatial_bboxes_batch.size(0)
                                    T_v = spatial_bboxes_batch.size(1) if spatial_bboxes_batch.dim() == 3 else 1
                                    tgt_flat_v = spatial_bboxes_batch.view(B_v * T_v, 4)
                                    valid_flat_v = spatial_bbox_valid_batch.view(B_v * T_v)
                                    if loc_pred_bboxes.dim() == 2:
                                        pred_flat_v = loc_pred_bboxes.unsqueeze(1).expand(-1, T_v, -1).reshape(B_v * T_v, 4)
                                    else:
                                        pred_flat_v = loc_pred_bboxes.view(B_v * T_v, 4)

                                    from .model import direct_center_loss
                                    loss = direct_center_loss(
                                        pred_flat_v,
                                        tgt_flat_v,
                                        valid_flat_v,
                                    )
                                    val_loss += loss.item() * clips.size(0)
                                    iou_val, center_err, valid_rate = _localization_metrics(
                                        pred_flat_v,
                                        tgt_flat_v,
                                        valid_flat_v,
                                    )
                                    val_loc_iou_sum += iou_val
                                    val_loc_center_sum += center_err
                                    val_loc_valid_sum += valid_rate
                                    val_loc_batches += 1
                                    continue

                                _val_clip_len = config.get("clip_length", 8)
                                _val_emb_active = getattr(val_dataset, '_emb_cache_mode', False) if val_dataset is not None else False
                                if _val_emb_active:
                                    logits = model(
                                        None,
                                        backbone_tokens=clips,
                                        num_frames=_val_clip_len,
                                        backbone_tokens_short=clips_short_val,
                                        num_frames_short=_val_clip_len // 2 if clips_short_val is not None else None,
                                        return_localization=False,
                                        return_frame_logits=True,
                                    )
                                else:
                                    logits = model(
                                        clips,
                                        return_localization=False,
                                        return_frame_logits=True,
                                    )

                                # Frame-level validation metrics
                                if not in_localization_stage and frame_labels_batch is not None:
                                    _fo_val = getattr(model, "_frame_output", None)
                                    if _fo_val is not None:
                                        f_logits_val = _fo_val[0]
                                        f_logits_val_pooled = _fo_val[3] if len(_fo_val) > 3 else f_logits_val
                                        pool_n_val = int(_fo_val[4]) if len(_fo_val) > 4 else 1
                                        frame_embeddings_val = _fo_val[6] if len(_fo_val) > 6 else None
                                        labels_for_val = _pool_frame_labels(frame_labels_batch, pool_n_val)
                                        logits_for_val = f_logits_val_pooled if pool_n_val > 1 else f_logits_val
                                        embeddings_for_val = _pool_frame_embeddings(frame_embeddings_val, pool_n_val)
                                        bg_mask_for_val = None
                                        if use_ovr and ovr_background_as_negative and bg_mask_batch is not None:
                                            bg_mask_for_val = _pool_binary_mask(bg_mask_batch, pool_n_val)
                                        valid_fl_val = labels_for_val >= 0
                                        if valid_fl_val.any():
                                            if use_ovr:
                                                score_tensor_val = torch.sigmoid(logits_for_val)
                                            else:
                                                score_tensor_val = torch.softmax(logits_for_val, dim=-1)
                                            valid_scores_np = score_tensor_val[valid_fl_val].detach().cpu().numpy()
                                            valid_targets_np = labels_for_val[valid_fl_val].detach().cpu().numpy()
                                            for cls_idx in range(len(class_names)):
                                                val_score_chunks_by_class[cls_idx].append(valid_scores_np[:, cls_idx].astype(np.float32, copy=False))
                                                val_target_chunks_by_class[cls_idx].append((valid_targets_np == cls_idx).astype(np.uint8, copy=False))
                                            if use_ovr:
                                                f_pred_val = torch.argmax(torch.sigmoid(logits_for_val), dim=-1)
                                            else:
                                                f_pred_val = torch.argmax(logits_for_val, dim=-1)
                                            val_frame_total += int(valid_fl_val.sum().item())
                                            val_frame_correct += int((f_pred_val[valid_fl_val] == labels_for_val[valid_fl_val]).sum().item())
                                        
                                        if use_ovr:
                                            B_fv, T_fv, C_fv = logits_for_val.shape
                                            fl_ovr_targets_v = torch.full((B_fv, T_fv, C_fv), ovr_label_smoothing, device=device)
                                            fl_ovr_weight_v = torch.ones(B_fv, T_fv, C_fv, device=device)
                                            for bi in range(B_fv):
                                                for ti in range(T_fv):
                                                    lbl = int(labels_for_val[bi, ti].item())
                                                    if 0 <= lbl < C_fv:
                                                        fl_ovr_targets_v[bi, ti, lbl] = 1.0 - ovr_label_smoothing
                                                        fl_ovr_weight_v[bi, ti, lbl] = ovr_pos_weight[lbl]
                                                        for cj in cooccur_lookup.get(lbl, ()):
                                                            if cj != lbl and 0 <= cj < C_fv:
                                                                fl_ovr_weight_v[bi, ti, cj] = 0.0
                                            suppress_mask_for_val = None
                                            if suppress_batch is not None:
                                                suppress_mask_for_val = (labels_for_val < 0) & (suppress_batch.view(-1, 1) >= 0)
                                                if suppress_mask_for_val.any():
                                                    fl_ovr_weight_v = fl_ovr_weight_v.masked_fill(
                                                        suppress_mask_for_val.unsqueeze(-1), 0.0
                                                    )
                                                    hn_bv, hn_tv = torch.nonzero(suppress_mask_for_val, as_tuple=True)
                                                    hn_cv = suppress_batch[hn_bv]
                                                    fl_ovr_targets_v[hn_bv, hn_tv, hn_cv] = 0.0
                                                    fl_ovr_weight_v[hn_bv, hn_tv, hn_cv] = 1.0
                                            if bg_mask_for_val is not None and bg_mask_for_val.any():
                                                fl_ovr_targets_v = fl_ovr_targets_v.masked_fill(
                                                    bg_mask_for_val.unsqueeze(-1), 0.0
                                                )
                                            valid_ovr_mask_v = (
                                                (labels_for_val >= 0) |
                                                (suppress_mask_for_val if suppress_mask_for_val is not None else torch.zeros_like(labels_for_val, dtype=torch.bool)) |
                                                (bg_mask_for_val if bg_mask_for_val is not None else torch.zeros_like(labels_for_val, dtype=torch.bool))
                                            )
                                            loss = _frame_loss_balanced(
                                                logits_for_val, labels_for_val,
                                                use_ovr_local=True, ovr_targets=fl_ovr_targets_v,
                                                ovr_weight=fl_ovr_weight_v,
                                                use_bout_balance=use_frame_bout_balance,
                                                bout_power=frame_bout_balance_power,
                                                valid_mask_override=valid_ovr_mask_v,
                                            )
                                        else:
                                            loss = _frame_loss_balanced(
                                                logits_for_val, labels_for_val,
                                                use_ovr_local=False,
                                                use_bout_balance=use_frame_bout_balance,
                                                bout_power=frame_bout_balance_power,
                                            )
                                        if hard_pair_mining and hard_pair_loss_weight > 0:
                                            pair_loss_v = _hard_pair_margin_loss(
                                                logits_for_val,
                                                labels_for_val,
                                                hard_pair_index_pairs,
                                                hard_pair_margin,
                                                use_bout_balance=use_frame_bout_balance,
                                                bout_power=frame_bout_balance_power,
                                            )
                                            loss = loss + hard_pair_loss_weight * pair_loss_v
                                        if use_supcon_loss and supcon_weight > 0:
                                            sc_loss_v = _supervised_contrastive_loss(
                                                embeddings_for_val,
                                                labels_for_val,
                                                temperature=supcon_temperature,
                                            )
                                            loss = loss + supcon_weight * sc_loss_v
                                        val_loss += loss.item() * clips.size(0)

                                        if valid_fl_val.any():
                                            val_targets_all.extend(labels_for_val[valid_fl_val].detach().cpu().tolist())
                                            val_preds_all.extend(f_pred_val[valid_fl_val].detach().cpu().tolist())
                        
                        avg_val_loss = val_loss / len(val_dataset)
                        val_acc = 100.0 * val_correct / val_total if val_total > 0 else 0.0
                        val_frame_acc = 100.0 * val_frame_correct / val_frame_total if val_frame_total > 0 else 0.0
                        val_macro_f1 = 0.0
                        per_class_f1 = np.zeros(len(class_names), dtype=float)
                        per_class_support = np.zeros(len(class_names), dtype=int)
                        conf_matrix = np.zeros((len(class_names), len(class_names)), dtype=int)
                        
                        per_attr_f1 = {}
                        val_ignore_thresholds = _calibrate_ignore_thresholds_from_validation(
                            val_score_chunks_by_class,
                            val_target_chunks_by_class,
                            class_names,
                        )
                        
                        # val_targets_all / val_preds_all are from frame-level validation when frame_labels_batch was used.
                        if val_targets_all:
                            per_class_f1 = f1_score(
                                val_targets_all,
                                val_preds_all,
                                labels=list(range(len(class_names))),
                                average=None,
                                zero_division=0
                            ) * 100.0
                            if _f1_include_indices and len(_f1_include_indices) < len(class_names):
                                val_macro_f1 = float(per_class_f1[_f1_include_indices].mean())
                            else:
                                val_macro_f1 = f1_score(
                                    val_targets_all,
                                    val_preds_all,
                                    average='macro',
                                    zero_division=0
                                ) * 100.0
                            per_class_support = np.bincount(
                                np.asarray(val_targets_all, dtype=np.int64),
                                minlength=len(class_names),
                            ).astype(int)
                            for t, p in zip(val_targets_all, val_preds_all):
                                if 0 <= int(t) < len(class_names) and 0 <= int(p) < len(class_names):
                                    conf_matrix[int(t), int(p)] += 1
                        
                        loc_val_iou = (val_loc_iou_sum / val_loc_batches) if val_loc_batches > 0 else 0.0
                        loc_val_center = (val_loc_center_sum / val_loc_batches) if val_loc_batches > 0 else 1.0
                        loc_val_valid = (val_loc_valid_sum / val_loc_batches) if val_loc_batches > 0 else 0.0

                        # Update history: when validation is frame-level, val_acc and val_f1 are per-frame.
                        history["val_loss"][-1] = avg_val_loss
                        if val_frame_total > 0:
                            history["val_acc"][-1] = val_frame_acc
                        else:
                            history["val_acc"][-1] = val_acc
                        history["val_frame_acc"][-1] = val_frame_acc
                        history["val_f1"][-1] = val_macro_f1
                        history["loc_val_iou"][-1] = loc_val_iou
                        history["loc_val_center_error"][-1] = loc_val_center
                        history["loc_val_valid_rate"][-1] = loc_val_valid
                        for idx, key in class_key_map.items():
                            if idx < len(per_class_f1):
                                history[key][-1] = per_class_f1[idx]
                        
                        if log_fn:
                            if in_localization_stage:
                                log_fn(
                                    f"Epoch {epoch+1}/{config['epochs']} - Val Loc Loss: {avg_val_loss:.4f}, "
                                    f"IoU: {loc_val_iou:.3f}, CErr: {loc_val_center:.3f}, VRate: {loc_val_valid:.3f}"
                                )
                            else:
                                log_fn(
                                    f"Epoch {epoch+1}/{config['epochs']} - Val Loss: {avg_val_loss:.4f}, "
                                    f"Val Acc (frame): {val_frame_acc:.2f}%, "
                                    f"Val Macro F1: {val_macro_f1:.2f}%"
                                )
                                if val_targets_all:
                                    metric_scope = "frame-labeled" if (use_frame_loss and not in_localization_stage) else "clip"
                                    log_fn(f"Val class diagnostics ({metric_scope}):")
                                    for ci, cname in enumerate(class_names):
                                        excl_tag = " [excluded from F1]" if cname in _f1_exclude_names else ""
                                        log_fn(
                                            f"  - {cname}: support={int(per_class_support[ci])}, "
                                            f"F1={float(per_class_f1[ci]):.2f}%{excl_tag}"
                                        )
                                    if len(class_names) <= 12:
                                        log_fn("Val confusion matrix rows=true, cols=pred:")
                                        for ci, cname in enumerate(class_names):
                                            row_vals = " ".join(str(int(v)) for v in conf_matrix[ci].tolist())
                                            log_fn(f"  {ci}:{cname} | {row_vals}")
                                if val_ignore_thresholds:
                                    log_fn(
                                        "Validation-calibrated ignore thresholds: "
                                        + ", ".join(
                                            f"{cls}={float(tau):.2f}"
                                            for cls, tau in sorted(
                                                val_ignore_thresholds.get("per_class_thresholds", {}).items(),
                                                key=lambda item: item[0],
                                            )
                                        )
                                    )

                        if head_metadata:
                            head_metadata["training_config"]["validation_calibrated_ignore_thresholds"] = _json_safe(val_ignore_thresholds)

                        metric_improved = False if in_localization_stage else (
                            val_macro_f1 > best_val_f1 + 1e-6
                        )

                        if in_localization_stage:
                            gate_pass = (
                                (loc_val_iou >= loc_gate_iou_threshold)
                                and (loc_val_center <= loc_gate_center_error)
                                and (loc_val_valid >= loc_gate_valid_rate)
                            )
                            localization_gate_streak = localization_gate_streak + 1 if gate_pass else 0
                            reached_epoch_cap = (epoch + 1) >= max(1, loc_max_stage_epochs)
                            reached_manual_switch = use_manual_loc_switch and ((epoch + 1) >= max(1, manual_loc_switch_epoch))
                            if gate_pass and log_fn:
                                log_fn(
                                    f"Localization gate check passed ({localization_gate_streak}/{loc_gate_patience})"
                                )
                            if reached_manual_switch or (localization_gate_streak >= max(1, loc_gate_patience)) or reached_epoch_cap:
                                in_localization_stage = False
                                classification_stage_start_epoch = epoch + 1
                                if log_fn:
                                    if reached_manual_switch:
                                        reason = f"manual switch epoch reached ({manual_loc_switch_epoch})"
                                    elif localization_gate_streak >= max(1, loc_gate_patience):
                                        reason = "metrics gate reached"
                                    else:
                                        reason = "max localization epochs reached"
                                    log_fn(
                                        f"Switching to classification stage at epoch {epoch+1} ({reason}). "
                                        "Classifier will train on localized crops."
                                    )
                                # Reset LR schedule, EMA, and optimizer state for classification
                                cls_remaining = total_epochs - (epoch + 1)
                                cls_warmup = 0
                                if use_scheduler:
                                    eta_min = 0.2 * classification_lr
                                    for pg in optimizer.param_groups:
                                        pg['lr'] = classification_lr
                                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                                        optimizer, T_max=max(1, cls_remaining - cls_warmup), eta_min=eta_min
                                    )
                                    warmup_scheduler = None
                                    warmup_epochs = cls_warmup
                                    if log_fn:
                                        log_fn(f"Reset LR schedule: CosineAnnealingLR (single decay, {cls_remaining} epochs)")
                                optimizer.state.clear()
                                if log_fn:
                                    log_fn("Reset optimizer momentum/adaptive state for classification")
                                ema_active = False
                                ema_state.clear()
                        
                        if metric_improved:
                            best_val_f1 = val_macro_f1
                            best_val_frame_acc = val_frame_acc
                            if head_metadata:
                                head_metadata["training_config"]["best_val_f1"] = best_val_f1
                            if config.get("save_best", True):
                                # Create folder for this best epoch
                                output_dir = os.path.dirname(config["output_path"])
                                basename = os.path.splitext(os.path.basename(config["output_path"]))[0]
                                from datetime import datetime
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                
                                best_folder = os.path.join(
                                    output_dir,
                                    f"{basename}_checkpoints",
                                    f"epoch_{epoch+1}_f1_{val_macro_f1:.1f}_frameacc_{val_frame_acc:.1f}_{timestamp}"
                                )
                                os.makedirs(best_folder, exist_ok=True)
                                
                                # Save Model
                                best_path = os.path.join(best_folder, "model.pt")
                                if head_metadata:
                                    model.save_head(best_path, metadata=head_metadata)
                                else:
                                    model.save_head(best_path)
                                    
                                # Also update main best file
                                best_main_path = config["output_path"].replace(".pt", "_best.pt")
                                if head_metadata:
                                    model.save_head(best_main_path, metadata=head_metadata)
                                else:
                                    model.save_head(best_main_path)
                                
                                # Save Logs & Plots
                                import pandas as pd
                                import matplotlib
                                matplotlib.use('Agg')
                                import matplotlib.pyplot as plt
                                
                                # Construct temp history including current epoch
                                curr_hist = {k: v.copy() for k, v in history.items()}
                                # Train metrics and epoch are ALREADY in history (updated before validation)
                                # And val metrics were just updated in-place at index -1
                                
                                pd.DataFrame(curr_hist).to_csv(os.path.join(best_folder, "history.csv"), index=False)
                                
                                plt.style.use('ggplot')
                                fig, axes = plt.subplots(4, 1, figsize=(10, 18))
                                ax1, ax2, ax3, ax4 = axes
                                epochs_hist = curr_hist['epoch']
                                
                                ax1.plot(epochs_hist, curr_hist['train_acc'], label='Train Acc', marker='o')
                                ax1.plot(epochs_hist, curr_hist['val_acc'], label='Val Acc (frame)', marker='s')
                                ax1.set_title(f'Accuracy - Epoch {epoch+1}')
                                ax1.set_ylabel('Accuracy (%)')
                                ax1.legend()
                                ax1.grid(True)
                                
                                ax2.plot(epochs_hist, curr_hist['train_loss'], label='Train Loss', marker='o')
                                ax2.plot(epochs_hist, curr_hist['val_loss'], label='Val Loss', marker='s')
                                ax2.set_ylabel('Loss')
                                ax2.legend()
                                ax2.grid(True)
                                
                                ax3.plot(epochs_hist, curr_hist['val_f1'], label='Val Macro F1 (frame)', linewidth=2, color='tab:purple')
                                for idx in range(len(class_names)):
                                    class_key = class_key_map.get(idx)
                                    if class_key in curr_hist:
                                        ax3.plot(
                                            epochs_hist,
                                            curr_hist[class_key],
                                            label=f"{class_names[idx]}",
                                            linestyle='--',
                                            alpha=0.6
                                        )
                                ax3.set_ylabel('F1 (%)')
                                ax3.legend(ncol=2, fontsize=8)
                                ax3.grid(True)
                                
                                per_class_keys_ordered = [class_key_map[idx] for idx in range(len(class_names))]
                                if per_class_keys_ordered:
                                    per_class_matrix = np.array([
                                        curr_hist[key] for key in per_class_keys_ordered
                                    ])
                                else:
                                    per_class_matrix = np.zeros((0, len(epochs_hist)))
                                
                                im = ax4.imshow(
                                    per_class_matrix,
                                    aspect='auto',
                                    cmap='magma',
                                    vmin=0,
                                    vmax=100
                                )
                                ax4.set_yticks(range(len(class_names)))
                                ax4.set_yticklabels(class_names)
                                ax4.set_xlabel('Epoch')
                                ax4.set_ylabel('Class')
                                ax4.set_title('Validation F1 Heatmap (%)')
                                if epochs_hist:
                                    max_ticks = min(len(epochs_hist), 12)
                                    tick_positions = np.linspace(0, len(epochs_hist) - 1, max_ticks, dtype=int)
                                    ax4.set_xticks(tick_positions)
                                    ax4.set_xticklabels([str(epochs_hist[i]) for i in tick_positions])
                                cbar = fig.colorbar(im, ax=ax4, orientation='vertical', pad=0.01)
                                cbar.set_label('F1 (%)')
                                
                                plt.tight_layout()
                                plt.savefig(os.path.join(best_folder, "training_plot.pdf"))
                                plt.close()
                                
                                if log_fn:
                                    log_fn(f"Saved best model checkpoint to {best_folder}")

                    except Exception as e:
                        error_msg = f"Error in validation: {str(e)}\n{traceback.format_exc()}"
                        if log_fn:
                            log_fn(f"ERROR: {error_msg}")
                        raise
                
                # Record val metrics (use train metrics if no validation)
                if not val_loader:
                     history["val_loss"][-1] = avg_loss
                     history["val_acc"][-1] = train_acc
                     history["val_frame_acc"][-1] = train_frame_acc
                     history["val_f1"][-1] = 0.0

                     # Print train confusion matrix every 5 epochs when there is no val set
                     if (log_fn and train_targets_all and not in_localization_stage
                             and ((epoch + 1) % 5 == 0 or epoch == 0)):
                         train_macro_f1 = f1_score(
                             train_targets_all, train_preds_all,
                             average='macro', zero_division=0,
                         ) * 100.0
                         per_class_f1_tr = f1_score(
                             train_targets_all, train_preds_all,
                             labels=list(range(len(class_names))),
                             average=None, zero_division=0,
                         ) * 100.0
                         per_class_support_tr = np.bincount(
                             np.asarray(train_targets_all, dtype=np.int64),
                             minlength=len(class_names),
                         ).astype(int)
                         conf_matrix_tr = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
                         for t, p in zip(train_targets_all, train_preds_all):
                             if 0 <= int(t) < len(class_names) and 0 <= int(p) < len(class_names):
                                 conf_matrix_tr[int(t), int(p)] += 1
                         history["val_f1"][-1] = train_macro_f1
                         log_fn(f"Train Macro F1: {train_macro_f1:.2f}%")
                         log_fn("Train class diagnostics (frame-labeled):")
                         for ci, cname in enumerate(class_names):
                             log_fn(
                                 f"  - {cname}: support={int(per_class_support_tr[ci])}, "
                                 f"F1={float(per_class_f1_tr[ci]):.2f}%"
                             )
                         if len(class_names) <= 12:
                             log_fn("Train confusion matrix rows=true, cols=pred:")
                             for ci, cname in enumerate(class_names):
                                 row_vals = " ".join(str(int(v)) for v in conf_matrix_tr[ci].tolist())
                                 log_fn(f"  {ci}:{cname} | {row_vals}")
                     reached_manual_switch = use_manual_loc_switch and ((epoch + 1) >= max(1, manual_loc_switch_epoch))
                     reached_epoch_cap = (epoch + 1) >= max(1, loc_max_stage_epochs)
                     if in_localization_stage and (reached_manual_switch or reached_epoch_cap):
                         in_localization_stage = False
                         classification_stage_start_epoch = epoch + 1
                         if log_fn:
                             if reached_manual_switch:
                                 reason = f"manual switch epoch reached ({manual_loc_switch_epoch})"
                             else:
                                 reason = "max localization epochs reached"
                             log_fn(
                                 f"Switching to classification stage at epoch {epoch+1} ({reason}, no validation set)."
                             )
                         # Reset LR schedule, EMA, and optimizer state for classification
                         cls_remaining = total_epochs - (epoch + 1)
                         cls_warmup = 0
                         if use_scheduler:
                             eta_min = 0.2 * classification_lr
                             for pg in optimizer.param_groups:
                                 pg['lr'] = classification_lr
                             scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                                 optimizer, T_max=max(1, cls_remaining - cls_warmup), eta_min=eta_min
                             )
                             warmup_scheduler = None
                             warmup_epochs = cls_warmup
                             if log_fn:
                                 log_fn(f"Reset LR schedule: CosineAnnealingLR (single decay, {cls_remaining} epochs)")
                         optimizer.state.clear()
                         if log_fn:
                             log_fn("Reset optimizer momentum/adaptive state for classification")
                         ema_active = False
                         ema_state.clear()
                     
                     # Save checkpoints only after classification stage starts.
                     # During localization stage we skip periodic model saves to
                     # avoid generating unusable intermediate checkpoints.
                     _apply_ema()  # swap to EMA weights for saving
                     if config.get("save_best", True) and (not in_localization_stage):
                         output_dir = os.path.dirname(config["output_path"])
                         basename = os.path.splitext(os.path.basename(config["output_path"]))[0]
                         from datetime import datetime
                         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                         
                         epoch_folder = os.path.join(
                             output_dir,
                             f"{basename}_checkpoints",
                             f"epoch_{epoch+1}_trainloss_{avg_loss:.4f}_acc_{train_acc:.1f}_{timestamp}"
                         )
                         os.makedirs(epoch_folder, exist_ok=True)
                         
                         epoch_path = os.path.join(epoch_folder, "model.pt")
                         if head_metadata:
                             model.save_head(epoch_path, metadata=head_metadata)
                         else:
                             model.save_head(epoch_path)
                         
                         if log_fn:
                             log_fn(f"Saved epoch checkpoint to {epoch_folder}")
                     elif config.get("save_best", True) and in_localization_stage and log_fn:
                         log_fn("Checkpoint save skipped (still in localization phase; classification not started yet).")

                # Restore training weights after EMA-based validation/saving
                _apply_ema()

                # Save crop-progress visualization every 2nd epoch.
                _save_epoch_crop_progress(epoch, epoch_phase)

                # Incremental history CSV every 2 epochs for offline plotting
                if (epoch + 1) % 2 == 0:
                    try:
                        import pandas as pd
                        inc_csv_dir = os.path.join(output_dir_base, f"{basename}_training_history")
                        os.makedirs(inc_csv_dir, exist_ok=True)
                        inc_csv_path = os.path.join(inc_csv_dir, "history.csv")
                        pd.DataFrame(history).to_csv(inc_csv_path, index=False)
                    except Exception as e:
                        logger.debug("Could not save incremental history CSV: %s", e)

                # Step LR scheduler once per epoch.
                sched_epoch_base = classification_stage_start_epoch if classification_stage_start_epoch is not None else 0
                sched_epoch = epoch - sched_epoch_base
                if use_scheduler and sched_epoch >= 0:
                    if warmup_scheduler is not None and warmup_epochs > 0 and sched_epoch < warmup_epochs:
                        warmup_scheduler.step()
                        if use_ema and sched_epoch == warmup_epochs - 1:
                            ema_active = True
                            _init_ema()
                            if log_fn:
                                log_fn(f"EMA activated after {warmup_epochs}-epoch warmup")
                    elif scheduler is not None:
                        scheduler.step()
                ema_start_epoch = max(warmup_epochs, 3)
                if use_ema and (not ema_active) and sched_epoch >= ema_start_epoch - 1:
                    ema_active = True
                    _init_ema()
                    if log_fn:
                        log_fn(f"EMA activated at epoch {epoch + 1}")
                current_lr = max(pg['lr'] for pg in optimizer.param_groups)
                if log_fn and (epoch + 1) % 5 == 0:
                    log_fn(f"Current Learning Rate: {current_lr:.8f}")
                
                if metrics_callback:
                    current_metrics = {
                        "epoch": epoch + 1,
                        "train_loss": history["train_loss"][-1],
                        "train_loss_class": history["train_loss_class"][-1],
                        "train_acc": history["train_acc"][-1],
                        "train_frame_acc": history["train_frame_acc"][-1],
                        "val_loss": history["val_loss"][-1],
                        "val_acc": history["val_acc"][-1],
                        "val_frame_acc": history["val_frame_acc"][-1],
                        "val_f1": history["val_f1"][-1],
                        "training_phase": epoch_phase,
                        "loc_val_iou": history["loc_val_iou"][-1],
                        "loc_val_center_error": history["loc_val_center_error"][-1],
                        "loc_val_valid_rate": history["loc_val_valid_rate"][-1],
                        "per_class_f1": {
                            class_names[idx]: history[class_key_map[idx]][-1]
                            for idx in range(len(class_names))
                            if class_key_map.get(idx) in history
                            and class_names[idx] not in _f1_exclude_names
                        },
                        "per_attr_f1": per_attr_f1,
                        "crop_progress_dir": crop_progress_dir,
                    }
                    metrics_callback(current_metrics)
                
            except Exception as e:
                error_msg = f"Error in epoch {epoch+1}: {str(e)}\n{traceback.format_exc()}"
                if log_fn:
                    log_fn(f"ERROR: {error_msg}")
                raise
        
        if log_fn:
            log_fn("Training complete!")

        # --- Augmentation ablation evaluation ---
        if (config.get("use_augmentation", False)
                and not (stop_callback and stop_callback())):
            try:
                # Load best model for evaluation
                best_main_path = config["output_path"].replace(".pt", "_best.pt")
                if os.path.exists(best_main_path):
                    model.load_head(best_main_path)
                _run_augmentation_ablation_eval(
                    model, train_dataset, config, device, log_fn=log_fn
                )
            except Exception as abl_err:
                if log_fn:
                    log_fn(f"Augmentation ablation eval failed (non-fatal): {abl_err}")

        # --- Per-head temperature calibration (OvR only) ---
        if use_ovr and val_loader is not None and not (stop_callback and stop_callback()):
            try:
                # Load best checkpoint for calibration
                best_main_path = config["output_path"].replace(".pt", "_best.pt")
                if os.path.exists(best_main_path):
                    model.load_head(best_main_path)
                model.eval()
                all_logits = []
                all_labels = []
                _val_emb_mode = getattr(val_dataset, '_emb_cache_mode', False)
                _cal_clip_len = config.get("clip_length", 8)
                with torch.no_grad():
                    for batch in val_loader:
                        # Batch is a tuple: (clips, labels, spatial_mask, bboxes, bbox_valid, indices, frame_labels)
                        if not isinstance(batch, (list, tuple)) or len(batch) < 7:
                            continue
                        clips = batch[0].to(device)
                        labels = batch[1].to(device)
                        frame_labels_cal = batch[6].to(device)
                        _cs_cal = batch[7] if len(batch) > 7 and isinstance(batch[7], torch.Tensor) and batch[7].numel() > 0 else None
                        if _cs_cal is not None:
                            _cs_cal = _cs_cal.to(device)

                        if _val_emb_mode:
                            out = model(None, backbone_tokens=clips,
                                        num_frames=_cal_clip_len,
                                        backbone_tokens_short=_cs_cal,
                                        num_frames_short=_cal_clip_len // 2 if _cs_cal is not None else None,
                                        return_frame_logits=True)
                        else:
                            out = model(clips, return_frame_logits=True)

                        fo = getattr(model, '_frame_output', None)
                        if fo is not None:
                            f_logits = fo[0]  # [B, T, C]
                            all_logits.append(f_logits.cpu())
                            all_labels.append(frame_labels_cal.cpu())

                if all_logits:
                    cat_logits = torch.cat(all_logits, dim=0)  # [N, T, C]
                    cat_labels = torch.cat(all_labels, dim=0)  # [N, T]
                    B_cal, T_cal, C_cal = cat_logits.shape
                    flat_logits = cat_logits.reshape(-1, C_cal)  # [N*T, C]
                    flat_labels = cat_labels.reshape(-1)          # [N*T]
                    valid_mask = flat_labels >= 0

                    if valid_mask.sum() > 50:
                        temperatures = torch.ones(C_cal)
                        for ci in range(C_cal):
                            ci_logits = flat_logits[valid_mask, ci]
                            ci_targets = (flat_labels[valid_mask] == ci).float()
                            if ci_targets.sum() < 5 or (1 - ci_targets).sum() < 5:
                                continue
                            # Grid search for temperature that maximizes F1
                            best_f1 = -1.0
                            best_t = 1.0
                            for t_cand in [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]:
                                probs = torch.sigmoid(ci_logits / t_cand)
                                preds = (probs >= 0.5).float()
                                tp = (preds * ci_targets).sum()
                                fp = (preds * (1 - ci_targets)).sum()
                                fn = ((1 - preds) * ci_targets).sum()
                                prec = tp / (tp + fp + 1e-8)
                                rec = tp / (tp + fn + 1e-8)
                                f1 = 2 * prec * rec / (prec + rec + 1e-8)
                                if f1.item() > best_f1:
                                    best_f1 = f1.item()
                                    best_t = t_cand
                            temperatures[ci] = best_t

                        temp_dict = {
                            class_names[ci]: round(float(temperatures[ci]), 3)
                            for ci in range(C_cal)
                        }
                        if head_metadata:
                            head_metadata["ovr_temperatures"] = temp_dict
                        if log_fn:
                            t_str = ", ".join(f"{k}={v}" for k, v in temp_dict.items())
                            log_fn(f"OvR per-head calibrated temperatures: {t_str}")
                    elif log_fn:
                        log_fn("Skipping OvR temperature calibration: too few valid validation frames.")
                elif log_fn:
                    log_fn("Skipping OvR temperature calibration: no frame logits from validation.")
            except Exception as cal_err:
                if log_fn:
                    log_fn(f"OvR temperature calibration failed (non-fatal): {cal_err}")

        if head_metadata:
            head_metadata["training_config"]["best_val_f1"] = best_val_f1 if best_val_f1 >= 0 else None
            head_metadata["training_config"]["best_val_frame_acc"] = best_val_frame_acc
            # Legacy field retained for compatibility with existing readers.
            head_metadata["training_config"]["best_val_acc"] = best_val_frame_acc
            per_class_final = {}
            for idx in range(len(class_names)):
                class_key = class_key_map.get(idx)
                if class_key in history and history[class_key]:
                    per_class_final[class_label_map[idx]] = history[class_key][-1]
            head_metadata["training_config"]["final_epoch_val_f1_per_class"] = per_class_final
        
        # Apply EMA weights for final model save (if EMA was activated)
        if ema_active:
            _apply_ema()

        try:
            if head_metadata:
                model.save_head(config["output_path"], metadata=head_metadata)
            else:
                model.save_head(config["output_path"])
            if log_fn:
                tag = " (EMA-averaged)" if ema_active else ""
                log_fn(f"Saved final model{tag} to {config['output_path']}")
                
            # --- Save Training Logs and Plots ---
            import pandas as pd
            from datetime import datetime
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import json
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.dirname(config["output_path"])
            output_basename = os.path.splitext(os.path.basename(config["output_path"]))[0]

            # Save training config snapshot for inference resolution fallback
            training_config_path = os.path.join(output_dir, f"{output_basename}_training_config.json")
            training_snapshot = {
                "classes": class_names,
                "attributes": getattr(train_dataset, "attributes", []),
                "clip_length": clip_length_value,
                "resolution": resolution_value,
                "backbone_model": config.get("backbone_model", "videoprism_public_v1_base"),
                "training_config": head_metadata.get("training_config", {}) if head_metadata else {},
            }
            with open(training_config_path, "w", encoding="utf-8") as cfg_file:
                json.dump(training_snapshot, cfg_file, indent=2)
            if log_fn:
                log_fn(f"Saved training config to {training_config_path}")
            
            # Save CSV log
            csv_path = os.path.join(output_dir, f"{output_basename}_training_log_{timestamp}.csv")
            df = pd.DataFrame(history)
            df.to_csv(csv_path, index=False)
            if log_fn:
                log_fn(f"Saved training log to {csv_path}")
            
            # Generate Plots
            plt.style.use('ggplot')
            fig, axes = plt.subplots(4, 1, figsize=(10, 18))
            ax1, ax2, ax3, ax4 = axes
            epochs_hist = history['epoch']
            
            ax1.plot(epochs_hist, history['train_acc'], label='Train Accuracy', marker='o')
            ax1.plot(epochs_hist, history['val_acc'], label='Val Acc (frame)', marker='s')
            ax1.set_title(f'Training Accuracy - {output_basename}')
            ax1.set_ylabel('Accuracy (%)')
            ax1.legend()
            ax1.grid(True)
            
            ax2.plot(epochs_hist, history['train_loss'], label='Train Loss', marker='o')
            ax2.plot(epochs_hist, history['val_loss'], label='Val Loss', marker='s')
            ax2.set_title(f'Training Loss - {output_basename}')
            ax2.set_ylabel('Loss')
            ax2.legend()
            ax2.grid(True)
            
            ax3.plot(epochs_hist, history['val_f1'], label='Val Macro F1 (frame)', linewidth=2, color='tab:purple')
            for idx in range(len(class_names)):
                class_key = class_key_map.get(idx)
                if class_key in history:
                    ax3.plot(
                        epochs_hist,
                        history[class_key],
                        label=f"{class_names[idx]}",
                        linestyle='--',
                        alpha=0.6
                    )
            ax3.set_ylabel('F1 (%)')
            ax3.legend(ncol=2, fontsize=8)
            ax3.grid(True)
            
            per_class_keys_ordered = [class_key_map[idx] for idx in range(len(class_names))]
            if per_class_keys_ordered:
                per_class_matrix = np.array([
                    history[key] for key in per_class_keys_ordered
                ])
            else:
                per_class_matrix = np.zeros((0, len(epochs_hist)))
            
            im = ax4.imshow(
                per_class_matrix,
                aspect='auto',
                cmap='magma',
                vmin=0,
                vmax=100
            )
            ax4.set_yticks(range(len(class_names)))
            ax4.set_yticklabels(class_names)
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Class')
            ax4.set_title('Validation F1 Heatmap (%)')
            if epochs_hist:
                max_ticks = min(len(epochs_hist), 12)
                tick_positions = np.linspace(0, len(epochs_hist) - 1, max_ticks, dtype=int)
                ax4.set_xticks(tick_positions)
                ax4.set_xticklabels([str(epochs_hist[i]) for i in tick_positions])
            cbar = fig.colorbar(im, ax=ax4, orientation='vertical', pad=0.01)
            cbar.set_label('F1 (%)')
            
            # Save Plot as PDF
            plot_path = os.path.join(output_dir, f"{output_basename}_training_plot_{timestamp}.pdf")
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
            
            if log_fn:
                log_fn(f"Saved training plot to {plot_path}")
                
        except Exception as e:
            error_msg = f"Error saving model/logs: {str(e)}\n{traceback.format_exc()}"
            if log_fn:
                log_fn(f"ERROR: {error_msg}")
        
        final_train_acc = history["train_acc"][-1] if history["train_acc"] else 0.0
        best_val_f1_out = best_val_f1 if best_val_f1 >= 0 else 0.0

        final_per_class_f1 = {}
        for idx in range(len(class_names)):
            if class_names[idx] in _f1_exclude_names:
                continue
            class_key = class_key_map.get(idx)
            if class_key in history and history[class_key]:
                final_per_class_f1[class_label_map[idx]] = history[class_key][-1]
                
        return {
            "best_val_acc": best_val_frame_acc,
            "best_val_frame_acc": best_val_frame_acc,
            "best_val_f1": best_val_f1_out, 
            "final_train_acc": final_train_acc,
            "per_class_f1": final_per_class_f1
        }
    
    except Exception as e:
        error_msg = f"Training failed: {str(e)}\n{traceback.format_exc()}"
        if log_fn:
            log_fn(f"FATAL ERROR: {error_msg}")
        raise RuntimeError(error_msg) from e

    finally:
        # Clean up embedding caches — they are only valid for this training run
        # (specific backbone weights, augmentation settings, dataset split).
        import shutil
        for _cache_path in [train_emb_cache, val_emb_cache_dir]:
            if _cache_path and os.path.isdir(_cache_path):
                try:
                    shutil.rmtree(_cache_path)
                    if log_fn:
                        log_fn(f"Cleaned up embedding cache: {_cache_path}")
                except Exception as e:
                    logger.debug("Could not remove embedding cache %s: %s", _cache_path, e)
