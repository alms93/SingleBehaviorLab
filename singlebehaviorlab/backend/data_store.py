import json
import logging
import os
from typing import Optional, List, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class AnnotationManager:
    
    def __init__(self, annotation_file: str):
        self.annotation_file = annotation_file
        self.data = self._load_or_create()
    
    def _load_or_create(self) -> dict:
        if os.path.exists(self.annotation_file):
            try:
                with open(self.annotation_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning("Error loading annotations: %s, creating new file", e)
        
        return {
            "clips": [],
            "classes": []
        }

    def _normalize_clip_id(self, clip_id: str) -> str:
        """Normalize clip IDs to match labeling list paths."""
        normalized = (clip_id or "").replace('\\', '/')
        if normalized.startswith("./"):
            normalized = normalized[2:]
        for prefix in ("../clips/", "clips/", "data/clips/"):
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]
                break
        return normalized
    
    def save(self):
        """Save annotations to file (with safe write)."""
        os.makedirs(os.path.dirname(self.annotation_file), exist_ok=True)
        
        temp_file = self.annotation_file + '.tmp'
        try:
            with open(temp_file, 'w') as f:
                json.dump(self.data, f, indent=2)
            os.replace(temp_file, self.annotation_file)
        except Exception as e:
            if os.path.exists(temp_file):
                os.remove(temp_file)
            raise e

    def reload(self):
        self.data = self._load_or_create()
    
    def add_class(self, class_name: str):
        if class_name not in self.data["classes"]:
            self.data["classes"].append(class_name)
            self.save()
    
    def remove_class(self, class_name: str):
        if class_name in self.data["classes"]:
            self.data["classes"].remove(class_name)
            self.save()
    
    def rename_class(self, old_name: str, new_name: str):
        """Rename a behavior class and update all associated clips."""
        if old_name not in self.data["classes"]:
            return False
            
        if new_name not in self.data["classes"]:
            idx = self.data["classes"].index(old_name)
            self.data["classes"][idx] = new_name
        else:
            # New name already exists, merge
            self.data["classes"].remove(old_name)
            
        # Update clips (both label and labels fields)
        for clip in self.data["clips"]:
            if clip.get("label") == old_name:
                clip["label"] = new_name
            labels = clip.get("labels")
            if isinstance(labels, list):
                clip["labels"] = [new_name if l == old_name else l for l in labels]
                if clip["labels"]:
                    clip["label"] = clip["labels"][0]
                
        self.save()
        return True
    
    def get_classes(self) -> List[str]:
        return self.data["classes"].copy()
    
    def add_clip(self, clip_id: str, label, meta: Optional[Dict[str, Any]] = None,
                 _defer_save: bool = False) -> str:
        """Add or update a clip annotation. label can be str or list of str.
        Returns the clip id that was updated or created (for callers that need to set_frame_labels).
        Set _defer_save=True when adding many clips in a loop, then call save() once at the end.
        Ensures every label in labels_list is in the classes list so training sees no stray labels.
        """
        clip_id_normalized = self._normalize_clip_id(clip_id)
        labels_list = label if isinstance(label, list) else [label] if label else []
        primary = labels_list[0] if labels_list else ""

        for lbl in labels_list:
            if lbl and lbl not in self.data["classes"]:
                self.data["classes"].append(lbl)

        for clip in self.data["clips"]:
            if self._normalize_clip_id(clip["id"]) == clip_id_normalized:
                clip["id"] = clip_id_normalized
                clip["label"] = primary
                clip["labels"] = labels_list
                if meta:
                    clip.setdefault("meta", {})
                    clip["meta"].update(meta)
                if not _defer_save:
                    self.save()
                return clip_id_normalized

        # No id match: if adding from inference (single-clip, not segment), try to update an
        # existing unlabeled bulk-extracted clip for the same source video + start frame.
        if meta and primary and isinstance(meta, dict) and not meta.get("added_from_inference_segment"):
            src_video = meta.get("source_video")
            src_frame = meta.get("source_frame")
            if src_video is not None and src_frame is not None:
                for clip in self.data["clips"]:
                    if clip.get("label"):
                        continue
                    cmeta = clip.get("meta") or {}
                    if cmeta.get("source_video") != src_video:
                        continue
                    if cmeta.get("sub_start_frame") is not None and cmeta.get("sub_start_frame") == src_frame:
                        clip["label"] = primary
                        clip["labels"] = labels_list
                        clip.setdefault("meta", {})
                        clip["meta"].update(meta)
                        if not _defer_save:
                            self.save()
                        return self._normalize_clip_id(clip["id"])

        new_clip = {
            "id": clip_id_normalized,
            "label": primary,
            "labels": labels_list,
            "meta": meta or {}
        }
        self.data["clips"].append(new_clip)
        if not _defer_save:
            self.save()
        return clip_id_normalized

    def get_clip_labels(self, clip_id: str) -> List[str]:
        """Get all labels for a clip (multi-label aware). Falls back to single label."""
        clip_id_normalized = self._normalize_clip_id(clip_id)
        for clip in self.data["clips"]:
            stored_id = self._normalize_clip_id(clip["id"])
            if stored_id == clip_id_normalized:
                labels = clip.get("labels")
                if labels and isinstance(labels, list):
                    return list(labels)
                lbl = clip.get("label")
                return [lbl] if lbl else []
            stored_base = os.path.splitext(stored_id)[0]
            clip_base = os.path.splitext(clip_id_normalized)[0]
            if stored_base == clip_base or stored_id == clip_base or clip_id_normalized == stored_base:
                labels = clip.get("labels")
                if labels and isinstance(labels, list):
                    return list(labels)
                lbl = clip.get("label")
                return [lbl] if lbl else []
        return []
    
    def set_spatial_mask(self, clip_id: str, patch_indices: List[int]):
        clip_id_normalized = self._normalize_clip_id(clip_id)
        for clip in self.data["clips"]:
            if self._normalize_clip_id(clip["id"]) == clip_id_normalized:
                clip["spatial_mask"] = sorted(patch_indices)
                self.save()
                return
    
    def clear_spatial_mask(self, clip_id: str):
        clip_id_normalized = self._normalize_clip_id(clip_id)
        for clip in self.data["clips"]:
            if self._normalize_clip_id(clip["id"]) == clip_id_normalized:
                clip.pop("spatial_mask", None)
                self.save()
                return
    
    def get_spatial_mask(self, clip_id: str) -> Optional[List[int]]:
        clip_id_normalized = self._normalize_clip_id(clip_id)
        for clip in self.data["clips"]:
            stored_id = self._normalize_clip_id(clip["id"])
            if stored_id == clip_id_normalized:
                return clip.get("spatial_mask")
            stored_base = os.path.splitext(stored_id)[0]
            clip_base = os.path.splitext(clip_id_normalized)[0]
            if stored_base == clip_base or stored_id == clip_base or clip_id_normalized == stored_base:
                return clip.get("spatial_mask")
        return None

    def set_spatial_bbox(self, clip_id: str, bbox_norm: List[float]):
        """Set spatial bbox [x1,y1,x2,y2] normalized to [0,1] for a clip."""
        if not bbox_norm or len(bbox_norm) != 4:
            return
        x1, y1, x2, y2 = [float(v) for v in bbox_norm]
        x1 = max(0.0, min(1.0, x1))
        y1 = max(0.0, min(1.0, y1))
        x2 = max(0.0, min(1.0, x2))
        y2 = max(0.0, min(1.0, y2))
        if x2 <= x1 or y2 <= y1:
            return

        clip_id_normalized = self._normalize_clip_id(clip_id)
        for clip in self.data["clips"]:
            if self._normalize_clip_id(clip["id"]) == clip_id_normalized:
                clip["spatial_bbox"] = [x1, y1, x2, y2]
                self.save()
                return

    def clear_spatial_bbox(self, clip_id: str):
        clip_id_normalized = self._normalize_clip_id(clip_id)
        for clip in self.data["clips"]:
            if self._normalize_clip_id(clip["id"]) == clip_id_normalized:
                clip.pop("spatial_bbox", None)
                self.save()
                return

    def get_spatial_bbox(self, clip_id: str) -> Optional[List[float]]:
        clip_id_normalized = self._normalize_clip_id(clip_id)
        for clip in self.data["clips"]:
            stored_id = self._normalize_clip_id(clip["id"])
            if stored_id == clip_id_normalized:
                return clip.get("spatial_bbox")
            stored_base = os.path.splitext(stored_id)[0]
            clip_base = os.path.splitext(clip_id_normalized)[0]
            if stored_base == clip_base or stored_id == clip_base or clip_id_normalized == stored_base:
                return clip.get("spatial_bbox")
        return None

    def set_spatial_bbox_frames(self, clip_id: str, frame_bboxes: List):
        """Set per-frame bboxes for a clip.

        Args:
            frame_bboxes: list of length T, each element is [x1,y1,x2,y2] or None.
        """
        if not frame_bboxes:
            return

        def _clamp(b):
            if b is None or len(b) != 4:
                return None
            x1, y1, x2, y2 = [float(v) for v in b]
            x1, y1 = max(0.0, min(1.0, x1)), max(0.0, min(1.0, y1))
            x2, y2 = max(0.0, min(1.0, x2)), max(0.0, min(1.0, y2))
            if x2 <= x1 or y2 <= y1:
                return None
            return [x1, y1, x2, y2]

        cleaned = [_clamp(b) for b in frame_bboxes]
        clip_id_normalized = self._normalize_clip_id(clip_id)
        for clip in self.data["clips"]:
            if self._normalize_clip_id(clip["id"]) == clip_id_normalized:
                clip["spatial_bbox_frames"] = cleaned
                # Also update legacy spatial_bbox to first valid frame for backward compat
                first_valid = next((b for b in cleaned if b is not None), None)
                if first_valid is not None:
                    clip["spatial_bbox"] = first_valid
                self.save()
                return

    def get_spatial_bbox_frames(self, clip_id: str) -> Optional[List]:
        """Get per-frame bboxes for a clip, or None if not set.

        Returns list of [x1,y1,x2,y2] or None per frame.
        """
        clip_id_normalized = self._normalize_clip_id(clip_id)
        for clip in self.data["clips"]:
            stored_id = self._normalize_clip_id(clip["id"])
            if stored_id == clip_id_normalized:
                return clip.get("spatial_bbox_frames")
            stored_base = os.path.splitext(stored_id)[0]
            clip_base = os.path.splitext(clip_id_normalized)[0]
            if stored_base == clip_base or stored_id == clip_base or clip_id_normalized == stored_base:
                return clip.get("spatial_bbox_frames")
        return None

    def clear_spatial_bbox_frames(self, clip_id: str):
        clip_id_normalized = self._normalize_clip_id(clip_id)
        for clip in self.data["clips"]:
            if self._normalize_clip_id(clip["id"]) == clip_id_normalized:
                clip.pop("spatial_bbox_frames", None)
                self.save()
                return

    def set_frame_labels(self, clip_id: str, frame_labels: List[Optional[str]], _defer_save: bool = False):
        """Set per-frame behavior labels for a clip.

        Args:
            frame_labels: list of length T, each element is a class name or None.
        """
        clip_id_normalized = self._normalize_clip_id(clip_id)
        for clip in self.data["clips"]:
            if self._normalize_clip_id(clip["id"]) == clip_id_normalized:
                clip["frame_labels"] = list(frame_labels)
                if not _defer_save:
                    self.save()
                return

    def get_frame_labels(self, clip_id: str) -> Optional[List[Optional[str]]]:
        """Get per-frame behavior labels for a clip, or None if not set."""
        clip_id_normalized = self._normalize_clip_id(clip_id)
        for clip in self.data["clips"]:
            stored_id = self._normalize_clip_id(clip["id"])
            if stored_id == clip_id_normalized:
                return clip.get("frame_labels")
            stored_base = os.path.splitext(stored_id)[0]
            clip_base = os.path.splitext(clip_id_normalized)[0]
            if stored_base == clip_base or stored_id == clip_base or clip_id_normalized == stored_base:
                return clip.get("frame_labels")
        return None

    def clear_frame_labels(self, clip_id: str):
        clip_id_normalized = self._normalize_clip_id(clip_id)
        for clip in self.data["clips"]:
            if self._normalize_clip_id(clip["id"]) == clip_id_normalized:
                clip.pop("frame_labels", None)
                self.save()
                return

    def get_clip_label(self, clip_id: str) -> Optional[str]:
        """Returns None if not labeled. Handles extension mismatches between clip_id and stored annotation."""
        clip_id_normalized = self._normalize_clip_id(clip_id)
        
        for clip in self.data["clips"]:
            stored_id = self._normalize_clip_id(clip["id"])
            if stored_id == clip_id_normalized:
                return clip["label"]
            
            stored_base, stored_ext = os.path.splitext(stored_id)
            clip_base, clip_ext = os.path.splitext(clip_id_normalized)
            
            if stored_base == clip_base or stored_id == clip_base or clip_id_normalized == stored_base:
                return clip["label"]
        
        return None
    
    def get_all_clips(self) -> List[Dict[str, Any]]:
        return self.data["clips"].copy()
    
    def get_labeled_clips(self) -> List[Dict[str, Any]]:
        return [c for c in self.data["clips"] if c.get("label")]
    
    def get_unlabeled_clips(self, all_clip_paths: List[str]) -> List[str]:
        """Get list of clip paths that don't have labels."""
        labeled_ids = {c["id"] for c in self.data["clips"] if c.get("label")}
        labeled_bases = {os.path.splitext(cid)[0] for cid in labeled_ids}
        
        unlabeled = []
        for cp in all_clip_paths:
            cp_normalized = cp.replace('\\', '/')
            if cp_normalized not in labeled_ids:
                cp_base = os.path.splitext(cp_normalized)[0]
                if cp_base not in labeled_bases:
                    unlabeled.append(cp)
        
        return unlabeled
    
    def remove_clip(self, clip_id: str):
        clip_id_normalized = self._normalize_clip_id(clip_id)
        self.data["clips"] = [
            c for c in self.data["clips"]
            if self._normalize_clip_id(c["id"]) != clip_id_normalized
        ]
        self.save()
    
    def get_clip_count_by_label(self) -> Dict[str, int]:
        """Get count of clips per label (counts each label in multi-label clips)."""
        counts = {}
        for clip in self.data["clips"]:
            labels = clip.get("labels")
            if isinstance(labels, list) and labels:
                for lbl in labels:
                    counts[lbl] = counts.get(lbl, 0) + 1
            else:
                label = clip.get("label", "unlabeled")
                counts[label] = counts.get(label, 0) + 1
        return counts

    def get_multilabel_stats(self) -> dict:
        """Return per-label exclusive/multi-class counts and combo frequencies.

        Returns dict with:
            exclusive: {label: count} — clips where this is the only label
            shared:    {label: count} — clips where this label co-occurs with others
            combos:    {(sorted tuple of labels): count}
        """
        exclusive: Dict[str, int] = {}
        shared: Dict[str, int] = {}
        combos: Dict[tuple, int] = {}
        for clip in self.data["clips"]:
            labels = clip.get("labels")
            if isinstance(labels, list) and labels:
                lbl_list = list(labels)
            else:
                lbl_list = [clip.get("label", "unlabeled")]
            if len(lbl_list) == 1:
                exclusive[lbl_list[0]] = exclusive.get(lbl_list[0], 0) + 1
            else:
                combo_key = tuple(sorted(lbl_list))
                combos[combo_key] = combos.get(combo_key, 0) + 1
                for lbl in lbl_list:
                    shared[lbl] = shared.get(lbl, 0) + 1
        return {"exclusive": exclusive, "shared": shared, "combos": combos}
    
    def clear_all_clips(self):
        self.data["clips"] = []
        self.save()

