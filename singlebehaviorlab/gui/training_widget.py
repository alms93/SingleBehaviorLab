import logging
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSpinBox,
    QDoubleSpinBox, QPlainTextEdit, QProgressBar, QGroupBox, QFormLayout,
    QFileDialog, QMessageBox, QLineEdit, QCheckBox, QToolButton,
    QScrollArea, QListWidget, QListWidgetItem, QTableWidget, QTableWidgetItem,
    QHeaderView, QDialog, QGridLayout, QSizePolicy
)

logger = logging.getLogger(__name__)
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QPixmap
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import os
import copy
import shutil
import tempfile
import json
import time
import glob
import torch
import yaml
import numpy as np
import cv2
import random
import re
from singlebehaviorlab.backend.data_store import AnnotationManager
from singlebehaviorlab.backend.model import VideoPrismBackbone, BehaviorClassifier
from singlebehaviorlab.backend.train import BehaviorDataset, train_model
from singlebehaviorlab.backend.video_utils import load_clip_frames
from .training_profiles import TrainingProfileDialog


class TrainingWorker(QThread):
    """Worker thread for training."""
    log_message = pyqtSignal(str)
    progress = pyqtSignal(int, int)
    finished = pyqtSignal()
    error = pyqtSignal(str)
    training_complete = pyqtSignal(float, float, float, dict)  # (best_val_acc, best_val_f1, final_train_acc, per_class_f1)
    epoch_complete = pyqtSignal(dict)  # New signal for real-time metrics

    def __init__(self, config, train_config, annotation_file, clips_dir, output_path):
        super().__init__()
        self.config = config
        self.train_config = train_config
        self.annotation_file = annotation_file
        self.clips_dir = clips_dir
        self.output_path = output_path
        self.should_stop = False

    def stop(self):
        """Request training stop."""
        self.should_stop = True

    def _build_model_for_config(self, train_dataset, cfg, log_fn):
        """Create a fresh classifier for one training run."""
        model_name = cfg.get("backbone_model", self.config.get("backbone_model", "videoprism_public_v1_base"))
        resolution = cfg.get("resolution", 288)
        log_fn(f"Loading VideoPrism backbone ({model_name}) at resolution {resolution}×{resolution}...")
        backbone = VideoPrismBackbone(model_name=model_name, resolution=resolution, log_fn=log_fn)
        log_fn("VideoPrism backbone loaded successfully")

        head_kwargs = cfg.get("head_kwargs", {}).copy()
        head_kwargs.pop("per_class_query", None)
        head_kwargs_for_metadata = head_kwargs.copy()
        dropout = cfg.get("dropout", 0.1)
        localization_dropout = 0.0

        log_fn("Creating classifier model...")
        if cfg.get("use_temporal_decoder", True):
            log_fn("Using MAP head + temporal decoder")
        else:
            log_fn("Using MAP head + direct per-frame linear classifier")
        log_fn(f"MAP head kwargs: {head_kwargs}")

        use_loc = cfg.get("use_localization", False)
        num_stages = cfg.get("num_stages", 3)
        if use_loc and num_stages > 1:
            num_stages = 1
            log_fn("Localization ON: reducing MS-TCN to 1 stage (no refinement) to prevent overfitting on tight crops.")

        multi_scale = cfg.get("multi_scale", False) and not use_loc
        model = BehaviorClassifier(
            backbone,
            num_classes=len(train_dataset.classes),
            class_names=train_dataset.classes,
            dropout=dropout,
            freeze_backbone=True,
            head_kwargs=head_kwargs,
            use_localization=use_loc,
            localization_hidden_dim=cfg.get("localization_hidden_dim", 256),
            localization_dropout=localization_dropout,
            use_frame_head=True,
            use_temporal_decoder=cfg.get("use_temporal_decoder", True),
            frame_head_temporal_layers=cfg.get("frame_head_temporal_layers", 1),
            temporal_pool_frames=cfg.get("temporal_pool_frames", 1),
            proj_dim=cfg.get("proj_dim", 256),
            num_stages=num_stages,
            multi_scale=multi_scale,
        )
        log_fn("Classifier model created successfully")
        return model, head_kwargs_for_metadata, dropout, localization_dropout, num_stages, multi_scale

    def _build_backend_train_config(
        self,
        cfg,
        output_path,
        classes,
        augmentation_options,
        head_kwargs_for_metadata,
        dropout,
        localization_dropout,
        num_stages,
        multi_scale,
    ):
        return {
            "batch_size": cfg["batch_size"],
            "epochs": cfg["epochs"],
            "lr": cfg.get("classification_lr", cfg["lr"]),
            "localization_lr": cfg.get("localization_lr", cfg["lr"]),
            "classification_lr": cfg.get("classification_lr", cfg["lr"]),
            "use_scheduler": cfg.get("use_scheduler", True),
            "use_ema": cfg.get("use_ema", True),
            "weight_decay": cfg.get("weight_decay", 1e-3),
            "output_path": output_path,
            "save_best": True,
            "use_class_weights": cfg.get("use_class_weights", False),
            "use_focal_loss": cfg.get("use_focal_loss", False),
            "focal_gamma": cfg.get("focal_gamma", 2.0),
            "use_frame_loss": True,
            "use_temporal_decoder": cfg.get("use_temporal_decoder", True),
            "frame_head_temporal_layers": cfg.get("frame_head_temporal_layers", 1),
            "temporal_pool_frames": cfg.get("temporal_pool_frames", 1),
            "num_stages": num_stages,
            "proj_dim": cfg.get("proj_dim", 256),
            "multi_scale": multi_scale,
            "use_frame_bout_balance": cfg.get("use_frame_bout_balance", True),
            "frame_bout_balance_power": cfg.get("frame_bout_balance_power", 1.0),
            "boundary_loss_weight": cfg.get("boundary_loss_weight", 0.3),
            "boundary_tolerance": cfg.get("boundary_tolerance", 2),
            "smoothness_loss_weight": cfg.get("smoothness_loss_weight", 0.05),
            "use_localization": cfg.get("use_localization", False),
            "use_manual_localization_switch": cfg.get("use_manual_localization_switch", False),
            "manual_localization_switch_epoch": cfg.get("manual_localization_switch_epoch", 20),
            "localization_hidden_dim": cfg.get("localization_hidden_dim", 256),
            "classification_crop_padding": cfg.get("classification_crop_padding", 0.35),
            "crop_jitter": cfg.get("crop_jitter", False),
            "crop_jitter_strength": cfg.get("crop_jitter_strength", 0.15),
            "emb_aug_versions": cfg.get("emb_aug_versions", 1),
            "clip_length": cfg.get("clip_length", 8),
            "use_ovr": cfg.get("use_ovr", False),
            "ovr_label_smoothing": cfg.get("ovr_label_smoothing", 0.05),
            "use_asl": cfg.get("use_asl", False),
            "asl_gamma_neg": cfg.get("asl_gamma_neg", 2.0),
            "asl_gamma_pos": cfg.get("asl_gamma_pos", 0.0),
            "asl_clip": cfg.get("asl_clip", 0.05),
            "use_hard_pair_mining": cfg.get("use_hard_pair_mining", False),
            "hard_pairs": cfg.get("hard_pairs", []),
            "hard_pair_margin": cfg.get("hard_pair_margin", 0.5),
            "hard_pair_loss_weight": cfg.get("hard_pair_loss_weight", 0.2),
            "hard_pair_confusion_boost": cfg.get("hard_pair_confusion_boost", 1.5),
            "use_confusion_sampler": cfg.get("use_confusion_sampler", True),
            "confusion_sampler_temperature": cfg.get("confusion_sampler_temperature", 2.0),
            "confusion_sampler_warmup_pct": cfg.get("confusion_sampler_warmup_pct", 0.2),
            "use_weighted_sampler": cfg.get("use_weighted_sampler", False),
            "use_augmentation": cfg.get("use_augmentation", False),
            "augmentation_options": augmentation_options,
            "virtual_expansion": cfg.get("virtual_expansion", 5),
            "stitch_augmentation_prob": cfg.get("stitch_augmentation_prob", 0.0),
            "f1_exclude_classes": cfg.get("f1_exclude_classes", []),
            "ovr_pos_weight_f1_excluded": cfg.get("ovr_pos_weight_f1_excluded", 1.5),
            "val_split": cfg.get("val_split", 0.2),
            "limit_classes": cfg.get("limit_classes", False),
            "selected_classes": cfg.get("selected_classes", []),
            "limit_per_class": cfg.get("limit_per_class", False),
            "per_class_limits": cfg.get("per_class_limits", {}),
            "per_class_val_limits": cfg.get("per_class_val_limits", {}),
            "backbone_model": cfg.get("backbone_model", "videoprism_public_v1_base"),
            "resolution": cfg.get("resolution", 288),
            "use_all_for_training": cfg.get("use_all_for_training", False),
            "use_embedding_diversity": cfg.get("use_embedding_diversity", False),
            "class_names": cfg.get("class_names", classes),
            "pretrained_path": cfg.get("pretrained_path"),
            "head_kwargs": head_kwargs_for_metadata,
            "dropout": dropout,
            "localization_dropout": localization_dropout,
        }

    def _generate_autotune_candidates(self, base_cfg, num_runs: int):
        """Generate a small random search around the current config."""
        def _uniq_float(values):
            out = []
            for v in values:
                v = float(max(1e-6, min(1.0, v)))
                if all(abs(v - prev) > 1e-12 for prev in out):
                    out.append(v)
            return out

        def _uniq_int(values, lo, hi):
            out = []
            for v in values:
                v = int(max(lo, min(hi, int(round(v)))))
                if v not in out:
                    out.append(v)
            return out

        lr0 = float(base_cfg.get("classification_lr", base_cfg.get("lr", 1e-4)))
        wd0 = float(base_cfg.get("weight_decay", 1e-3))
        drop0 = float(base_cfg.get("dropout", 0.3))
        heads0 = int(base_cfg.get("head_kwargs", {}).get("num_heads", 4))
        layers0 = int(base_cfg.get("frame_head_temporal_layers", 4))
        ovr_ls0 = float(base_cfg.get("ovr_label_smoothing", 0.05))
        use_temporal_decoder = bool(base_cfg.get("use_temporal_decoder", True))

        lr_vals = _uniq_float([3e-5, 1e-4, 3e-4, 1e-3, lr0 / 3.0, lr0, lr0 * 3.0])
        wd_vals = _uniq_float([1e-5, 1e-4, 5e-4, 1e-3, wd0 / 10.0, wd0, wd0 * 3.0])
        drop_vals = _uniq_float([0.1, 0.2, 0.3, 0.4, drop0])
        head_vals = _uniq_int([2, 4, 8, heads0, heads0 - 2, heads0 + 2], 1, 16)
        layer_vals = [layers0] if not use_temporal_decoder else _uniq_int([1, 2, 3, 4, layers0, layers0 - 1, layers0 + 1], 1, 8)
        ovr_ls_vals = _uniq_float([0.0, 0.02, 0.05, ovr_ls0])

        rng = random.Random(42)
        candidates = []
        seen = set()

        def _trial_tuple(cfg):
            return (
                round(float(cfg.get("classification_lr", cfg.get("lr", 0.0))), 12),
                round(float(cfg.get("weight_decay", 0.0)), 12),
                round(float(cfg.get("dropout", 0.0)), 12),
                int(cfg.get("head_kwargs", {}).get("num_heads", 0)),
                int(cfg.get("frame_head_temporal_layers", 0)) if cfg.get("use_temporal_decoder", True) else None,
                round(float(cfg.get("ovr_label_smoothing", 0.0)), 12) if cfg.get("use_ovr", False) else None,
            )

        base_copy = copy.deepcopy(base_cfg)
        candidates.append(base_copy)
        seen.add(_trial_tuple(base_copy))

        max_attempts = max(20, num_runs * 20)
        attempts = 0
        while len(candidates) < max(1, int(num_runs)) and attempts < max_attempts:
            attempts += 1
            cfg = copy.deepcopy(base_cfg)
            new_lr = rng.choice(lr_vals)
            cfg["lr"] = new_lr
            cfg["classification_lr"] = new_lr
            cfg["weight_decay"] = rng.choice(wd_vals)
            cfg["dropout"] = rng.choice(drop_vals)
            cfg.setdefault("head_kwargs", {})
            cfg["head_kwargs"]["num_heads"] = rng.choice(head_vals)
            cfg["frame_head_temporal_layers"] = rng.choice(layer_vals)
            if cfg.get("use_ovr", False):
                cfg["ovr_label_smoothing"] = rng.choice(ovr_ls_vals)
            key = _trial_tuple(cfg)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(cfg)
        return candidates[:max(1, int(num_runs))]

    def _reset_runtime_dataset_caches(self, dataset):
        """Clear per-run disk cache flags so a new training run rebuilds them cleanly."""
        if dataset is None:
            return
        if hasattr(dataset, "_emb_cache_mode"):
            dataset._emb_cache_mode = False
        if hasattr(dataset, "_emb_cache_dir"):
            dataset._emb_cache_dir = None
        if hasattr(dataset, "_roi_cache_mode"):
            dataset._roi_cache_mode = False
        if hasattr(dataset, "_roi_cache_dir"):
            dataset._roi_cache_dir = None

    def _cleanup_autotune_trial_outputs(self, trial_output, log_fn=None):
        """Delete temporary files produced by one auto-tune trial."""
        output_dir = os.path.dirname(trial_output)
        basename = os.path.splitext(os.path.basename(trial_output))[0]
        paths = [
            trial_output,
            trial_output + ".meta.json",
            os.path.join(output_dir, f"{basename}_best.pt"),
            os.path.join(output_dir, f"{basename}_best.pt.meta.json"),
            os.path.join(output_dir, f"{basename}_training_config.json"),
            os.path.join(output_dir, f"{basename}_checkpoints"),
            os.path.join(output_dir, f"{basename}_crop_progress"),
        ]
        paths.extend(glob.glob(os.path.join(output_dir, f"{basename}_training_log_*.csv")))
        paths.extend(glob.glob(os.path.join(output_dir, f"{basename}_training_plot_*.pdf")))
        for path in paths:
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path, ignore_errors=True)
                elif os.path.exists(path):
                    os.remove(path)
            except Exception as e:
                if log_fn:
                    log_fn(f"Warning: could not delete autotune temp output {path}: {e}")

    def _run_autotune_search(self, train_dataset, val_dataset, base_cfg, classes, augmentation_options, log_fn, progress_cb, check_stop):
        num_runs = int(max(1, base_cfg.get("auto_tune_runs", 8)))
        search_epochs = int(max(1, base_cfg.get("auto_tune_epochs", min(12, int(base_cfg.get("epochs", 20))))))
        candidates = self._generate_autotune_candidates(base_cfg, num_runs)
        trial_root = tempfile.mkdtemp(prefix="autotune_trials_", dir=os.path.dirname(self.output_path))
        best_cfg = None
        best_result = None
        best_score = float("-inf")

        log_fn(f"Auto-tune enabled: evaluating {len(candidates)} candidate config(s) for {search_epochs} epoch(s) each.")
        log_fn("Search parameters: classification LR, weight decay, dropout, MAP num_heads, frame temporal layers"
               + (", OvR label smoothing" if base_cfg.get("use_ovr", False) else ""))
        try:
            for idx, candidate in enumerate(candidates, start=1):
                if check_stop():
                    return None, None
                self._reset_runtime_dataset_caches(train_dataset)
                self._reset_runtime_dataset_caches(val_dataset)
                trial_cfg = copy.deepcopy(candidate)
                trial_cfg["epochs"] = search_epochs
                trial_output = os.path.join(trial_root, f"trial_{idx:02d}.pt")
                log_fn(
                    f"[Auto-tune {idx}/{len(candidates)}] "
                    f"lr={trial_cfg['classification_lr']:.2e}, "
                    f"wd={trial_cfg['weight_decay']:.2e}, "
                    f"dropout={trial_cfg['dropout']:.2f}, "
                    f"heads={trial_cfg['head_kwargs'].get('num_heads', 4)}, "
                    f"layers={trial_cfg['frame_head_temporal_layers']}"
                    + (f", ovr_ls={trial_cfg.get('ovr_label_smoothing', 0.0):.2f}" if trial_cfg.get("use_ovr", False) else "")
                )
                model, head_kwargs_meta, dropout, localization_dropout, num_stages, multi_scale = self._build_model_for_config(
                    train_dataset, trial_cfg, log_fn
                )
                backend_cfg = self._build_backend_train_config(
                    trial_cfg,
                    trial_output,
                    classes,
                    augmentation_options,
                    head_kwargs_meta,
                    dropout,
                    localization_dropout,
                    num_stages,
                    multi_scale,
                )
                result = train_model(
                    model,
                    train_dataset,
                    val_dataset,
                    backend_cfg,
                    log_fn=log_fn,
                    progress_callback=progress_cb,
                    stop_callback=check_stop,
                    metrics_callback=None,
                )
                score = float(result.get("best_val_f1", 0.0) or 0.0)
                acc = float(result.get("best_val_acc", 0.0) or 0.0)
                log_fn(f"[Auto-tune {idx}/{len(candidates)}] best val F1={score:.2f}, val acc={acc:.2f}")
                if score > best_score:
                    best_score = score
                    best_cfg = copy.deepcopy(trial_cfg)
                    best_result = dict(result)
                self._cleanup_autotune_trial_outputs(trial_output, log_fn=log_fn)
            return best_cfg, best_result
        finally:
            try:
                shutil.rmtree(trial_root)
            except Exception as e:
                logger.debug("Could not remove autotune trial root: %s", e)

    def _save_autotuned_profile(self, tuned_cfg, best_search_result, log_fn):
        """Persist the selected auto-tune winner as a reusable training profile."""
        profiles_path = self.train_config.get("training_profiles_path", "")
        if not profiles_path:
            return
        try:
            os.makedirs(os.path.dirname(profiles_path), exist_ok=True)
            profiles = {}
            if os.path.exists(profiles_path):
                with open(profiles_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    if isinstance(loaded, dict):
                        profiles = loaded

            profile_cfg = copy.deepcopy(tuned_cfg)
            profile_cfg["auto_tune_before_final"] = False
            profile_cfg["auto_tune_selected_from_search"] = True
            profile_cfg["auto_tune_selected_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            if best_search_result:
                profile_cfg["auto_tune_best_search_val_f1"] = float(best_search_result.get("best_val_f1", 0.0) or 0.0)
                profile_cfg["auto_tune_best_search_val_acc"] = float(best_search_result.get("best_val_acc", 0.0) or 0.0)

            base_name = os.path.splitext(os.path.basename(self.output_path))[0] or "training"
            profile_name = f"autotune_{base_name}_{time.strftime('%Y%m%d_%H%M%S')}"
            suffix = 2
            while profile_name in profiles:
                profile_name = f"autotune_{base_name}_{time.strftime('%Y%m%d_%H%M%S')}_{suffix}"
                suffix += 1

            profiles[profile_name] = profile_cfg
            with open(profiles_path, "w", encoding="utf-8") as f:
                json.dump(profiles, f, indent=2)
            log_fn(f"Saved auto-tuned winner as training profile: {profile_name}")
        except Exception as e:
            log_fn(f"Warning: could not save auto-tuned profile: {e}")
    
    def run(self):
        """Run training."""
        import traceback
        try:
            def log_fn(msg):
                self.log_message.emit(msg)
            
            def progress_cb(epoch, total):
                self.progress.emit(epoch, total)
                
            def metrics_cb(metrics):
                self.epoch_complete.emit(metrics)
            
            log_fn("Initializing training...")
            log_fn(f"Annotation file: {self.annotation_file}")
            log_fn(f"Clips directory: {self.clips_dir}")
            
            annotation_manager = AnnotationManager(self.annotation_file)
            labeled_clips = annotation_manager.get_labeled_clips()
            
            if not labeled_clips:
                error_msg = "No labeled clips found. Please label some clips first."
                log_fn(f"ERROR: {error_msg}")
                self.error.emit(error_msg)
                return
            
            classes = annotation_manager.get_classes()
            
            log_fn("Using class-only training pipeline (hierarchical/attribution disabled).")

            if len(classes) < 2:
                error_msg = "Need at least 2 behavior classes for training."
                log_fn(f"ERROR: {error_msg}")
                self.error.emit(error_msg)
                return
            
            # Filter classes if class selection is enabled
            use_ovr = self.train_config.get("use_ovr", False)
            hybrid_ovr_bg = bool(self.train_config.get("ovr_background_as_negative", False) and use_ovr)
            hybrid_bg_classes = set(self.train_config.get("ovr_background_class_names", []))
            selected_classes = None
            if self.train_config.get("limit_classes", False):
                selected_classes = self.train_config.get("selected_classes", [])
                if selected_classes:
                    classes = [c for c in classes if c in selected_classes]
                    log_fn(f"Limiting training to {len(classes)} selected classes: {classes}")
                    if len(classes) < 2:
                        error_msg = "Need at least 2 selected classes for training."
                        log_fn(f"ERROR: {error_msg}")
                        self.error.emit(error_msg)
                        return
                    # Filter clips: keep selected classes + near_negative clips when OvR
                    if use_ovr:
                        allowed = set(selected_classes)
                        if hybrid_ovr_bg:
                            allowed.update(hybrid_bg_classes)
                        labeled_clips = [
                            clip for clip in labeled_clips
                            if clip.get("label") in allowed
                            or clip.get("label", "").startswith("near_negative")
                        ]
                    else:
                        labeled_clips = [clip for clip in labeled_clips if clip.get("label") in classes]
            
            # OvR mode: filter near_negative_* from class list (they become suppression-only)
            if use_ovr:
                real_classes = [c for c in classes if not c.startswith("near_negative")]
                hn_classes = [c for c in classes if c.startswith("near_negative")]
                bg_classes = [c for c in real_classes if c in hybrid_bg_classes] if hybrid_ovr_bg else []
                if hybrid_ovr_bg and bg_classes:
                    real_classes = [c for c in real_classes if c not in bg_classes]
                if len(real_classes) < 2:
                    error_msg = "OvR mode needs at least 2 non-near-negative classes."
                    log_fn(f"ERROR: {error_msg}")
                    self.error.emit(error_msg)
                    return
                log_fn(f"OvR mode: {len(real_classes)} real classes, {len(hn_classes)} near-negative classes (suppression only)")
                if hybrid_ovr_bg and bg_classes:
                    log_fn(
                        f"Hybrid OvR backgrounds: {bg_classes} kept as negative-only clips "
                        f"(not trained as explicit heads)"
                    )
                classes = real_classes
                # Near-negative clips stay in labeled_clips (they get label=-1 in the dataset)

            log_fn(f"Found {len(labeled_clips)} labeled clips")
            log_fn(f"Classes: {classes}")
            
            from collections import Counter
            label_counts = Counter([clip["label"] for clip in labeled_clips])
            log_fn("Class distribution (before limiting):")
            for label, count in sorted(label_counts.items()):
                log_fn(f"  {label}: {count} ({100.0*count/len(labeled_clips):.1f}%)")

            # Multi-class breakdown
            mc_combos: dict[tuple, int] = {}
            mc_per_label: dict[str, int] = {}
            exc_per_label: dict[str, int] = {}
            for clip in labeled_clips:
                lbl_list = clip.get("labels")
                if not isinstance(lbl_list, list) or not lbl_list:
                    lbl_list = [clip.get("label", "")]
                if len(lbl_list) > 1:
                    key = tuple(sorted(lbl_list))
                    mc_combos[key] = mc_combos.get(key, 0) + 1
                    for lbl in lbl_list:
                        mc_per_label[lbl] = mc_per_label.get(lbl, 0) + 1
                else:
                    exc_per_label[lbl_list[0]] = exc_per_label.get(lbl_list[0], 0) + 1
            if mc_combos:
                total_mc = sum(mc_combos.values())
                log_fn(f"Multi-class clips: {total_mc} of {len(labeled_clips)} ({100.0*total_mc/max(1,len(labeled_clips)):.1f}%)")
                for combo, cnt in sorted(mc_combos.items(), key=lambda x: -x[1]):
                    log_fn(f"  {' + '.join(combo)}: {cnt}")
                for lbl in sorted(mc_per_label):
                    exc = exc_per_label.get(lbl, 0)
                    sh = mc_per_label[lbl]
                    log_fn(f"  {lbl}: {exc} exclusive, {sh} in multi-class clips")
            
            use_all_for_training = self.train_config.get("use_all_for_training", False)
            val_split = self.train_config.get("val_split", 0.2)

            def _split_train_val_clip_stratified(clips, split_ratio, seed=42):
                """Split clips by randomly assigning ~split_ratio per class to val (stratified by clip label)."""
                if not clips or split_ratio <= 0:
                    return list(clips), []
                by_label = {}
                for c in clips:
                    lbl = c.get("label")
                    by_label.setdefault(lbl, []).append(c)
                rng = random.Random(seed)
                train_out, val_out = [], []
                for lbl, group in by_label.items():
                    rng.shuffle(group)
                    n_val = max(0, int(round(len(group) * split_ratio)))
                    n_val = min(n_val, len(group) - 1) if len(group) > 1 else 0
                    train_out.extend(group[n_val:])
                    val_out.extend(group[:n_val])
                log_fn(f"Clip-stratified split: train={len(train_out)} clips, val={len(val_out)} clips")
                if val_out:
                    log_fn(f"  Val clip distribution: {dict(Counter(c.get('label') for c in val_out))}")
                return train_out, val_out

            def _split_train_val_frame_stratified(clips, classes, split_ratio, seed=42):
                """Split by randomly selecting frames per class for val, then assign whole clips to val if they contain any selected frame.
                Ensures roughly equal proportion of each class in validation."""
                if not clips or split_ratio <= 0:
                    return list(clips), []
                class_set = set(classes)
                _clip_len = self.train_config.get("clip_length", 8)
                # Pool (clip_idx, frame_idx) for every labeled frame.
                # Clips without per-frame labels contribute _clip_len entries so their
                # weight in stratification matches what validation will actually count.
                pool_by_class = {c: [] for c in classes}
                for clip_idx, clip in enumerate(clips):
                    fl = clip.get("frame_labels")
                    if not isinstance(fl, (list, tuple)) or len(fl) == 0:
                        primary = clip.get("label")
                        if primary in class_set:
                            for fi in range(_clip_len):
                                pool_by_class[primary].append((clip_idx, fi))
                        continue
                    for frame_idx, lbl in enumerate(fl):
                        if lbl in class_set:
                            pool_by_class[lbl].append((clip_idx, frame_idx))
                rng = random.Random(seed)
                val_frames = set()
                for c in classes:
                    lst = pool_by_class[c]
                    if not lst:
                        continue
                    rng.shuffle(lst)
                    n_val = max(0, int(round(len(lst) * split_ratio)))
                    n_val = min(n_val, len(lst))
                    for i in range(n_val):
                        val_frames.add(lst[i])
                val_clip_indices = {clip_idx for (clip_idx, _) in val_frames}
                # Cap val clips so train gets enough data: at most split_ratio of clips in val.
                max_val_clips = max(1, int(round(len(clips) * split_ratio)))
                if len(val_clip_indices) > max_val_clips:
                    val_clip_list = list(val_clip_indices)
                    rng.shuffle(val_clip_list)
                    val_clip_indices = set(val_clip_list[:max_val_clips])
                # Keep only frames belonging to clips that survived the cap.
                val_frames = {f for f in val_frames if f[0] in val_clip_indices}
                train_out = [c for i, c in enumerate(clips) if i not in val_clip_indices]
                val_out = [c for i, c in enumerate(clips) if i in val_clip_indices]
                log_fn(f"Frame-stratified split: {len(val_frames)} val frames across {len(val_clip_indices)} clips (cap ~{split_ratio:.0%} clips) → "
                       f"train={len(train_out)} clips, val={len(val_out)} clips")
                if val_out:
                    val_label_counts = Counter()
                    for c in val_out:
                        fl = c.get("frame_labels")
                        if isinstance(fl, (list, tuple)):
                            for lbl in fl:
                                if lbl in class_set:
                                    val_label_counts[lbl] += 1
                        else:
                            val_label_counts[c.get("label")] += 1
                    log_fn(f"  Val frame distribution: {dict(val_label_counts)}")
                return train_out, val_out

            # Apply per-class limits to TRAINING ONLY (validation uses all remaining clips)
            per_class_limits = self.train_config.get("per_class_limits", {})
            per_class_val_limits = self.train_config.get("per_class_val_limits", {})
            if per_class_limits:
                log_fn("Applying per-class limits to TRAINING set only...")
                log_fn("Validation will use ALL remaining clips (not limited)")
                
                train_clips = []
                val_clips = []
                
                random.seed(42)  # For reproducibility
                
                # Check if embedding-based diversity selection is enabled
                use_embedding_diversity = self.train_config.get("use_embedding_diversity", False)
                backbone_model = self.train_config.get("backbone_model", "videoprism_public_v1_base")
                
                resolution = self.train_config.get("resolution", 288)
                
                # Initialize VideoPrism backbone if needed for diversity selection
                backbone = None
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                if use_embedding_diversity:
                    try:
                        log_fn(f"Initializing VideoPrism for embedding-based diversity selection at {resolution}×{resolution}...")
                        backbone = VideoPrismBackbone(model_name=backbone_model, resolution=resolution)
                        backbone.eval()
                        backbone.to(device)
                        log_fn("VideoPrism loaded successfully")
                    except Exception as e:
                        log_fn(f"Failed to load VideoPrism: {e}")
                        log_fn("   Falling back to random selection")
                        use_embedding_diversity = False

                def farthest_point_sampling(embeddings, n_samples, seed=42):
                    """Select n_samples using Farthest Point Sampling for maximum diversity."""
                    if len(embeddings) <= n_samples:
                        return list(range(len(embeddings)))
                    
                    np.random.seed(seed)
                    embeddings = np.array(embeddings)
                    selected = []
                    
                    # Start with random point
                    current = np.random.randint(0, len(embeddings))
                    selected.append(current)
                    
                    # Track minimum distance to any selected point for each unselected point
                    min_distances = np.full(len(embeddings), np.inf)
                    
                    # Iteratively select farthest point from all selected
                    for _ in range(n_samples - 1):
                        # Update minimum distances to nearest selected point
                        for i in range(len(embeddings)):
                            if i not in selected:
                                # Distance to the most recently selected point
                                dist_to_current = np.linalg.norm(embeddings[i] - embeddings[current])
                                # Keep minimum distance to any selected point
                                min_distances[i] = min(min_distances[i], dist_to_current)
                        
                        # Select point with maximum minimum distance (farthest from all selected)
                        current = np.argmax(min_distances)
                        selected.append(current)
                        min_distances[current] = 0  # Mark as selected
                    
                    return selected
                
                for label in sorted(set([clip["label"] for clip in labeled_clips])):
                    class_clips = [clip for clip in labeled_clips if clip.get("label") == label]
                    
                    if label in per_class_limits:
                        raw_limit = per_class_limits[label]
                        limit = max(1, int(round(float(raw_limit))))
                        raw_val_limit = per_class_val_limits.get(label, float('inf'))
                        val_limit = max(1, int(round(float(raw_val_limit)))) if raw_val_limit != float('inf') else float('inf')
                        if len(class_clips) > limit:
                            log_fn(f"  {label}: {len(class_clips)} total clips")
                            
                            if use_embedding_diversity and backbone is not None:
                                try:
                                    log_fn(f"    Extracting orientation-invariant embeddings for diversity selection...")
                                    embeddings = []
                                    valid_clips = []
                                    
                                    for clip_idx, clip in enumerate(class_clips):
                                        if (clip_idx + 1) % 20 == 0:
                                            log_fn(f"      Processing clip {clip_idx + 1}/{len(class_clips)}...")
                                        
                                        clip_id = clip.get("id", "")
                                        clip_path = os.path.join(self.clips_dir, clip_id)
                                        
                                        # Try to find clip file
                                        if not os.path.exists(clip_path):
                                            base_name, ext = os.path.splitext(clip_id)
                                            for video_ext in ['.mp4', '.avi', '.mov', '.mkv']:
                                                test_path = os.path.join(self.clips_dir, base_name + video_ext)
                                                if os.path.exists(test_path):
                                                    clip_path = test_path
                                                    break
                                        
                                        if not os.path.exists(clip_path):
                                            continue
                                        
                                        try:
                                            # Load frames
                                            frames_bgr = load_clip_frames(clip_path, target_size=(resolution, resolution))
                                            if not frames_bgr:
                                                continue
                                            
                                            # Convert BGR to RGB and normalize
                                            frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]
                                            frames_array = np.stack(frames_rgb).astype(np.float32) / 255.0
                                            
                                            # Extract embeddings with orientation augmentation (original + hflip + vflip + both)
                                            embedding_list = []
                                            
                                            for hflip, vflip in [(False, False), (True, False), (False, True), (True, True)]:
                                                frames_aug = frames_array.copy()
                                                
                                                # Apply flips
                                                if hflip:
                                                    frames_aug = np.flip(frames_aug, axis=2)  # Flip horizontally (width)
                                                if vflip:
                                                    frames_aug = np.flip(frames_aug, axis=1)  # Flip vertically (height)
                                                
                                                # Copy after all flips to avoid negative strides (PyTorch requirement)
                                                frames_aug = frames_aug.copy()
                                                
                                                # Ensure correct shape: (T, H, W, C) -> (1, T, C, H, W) for VideoPrism
                                                T, H, W, C = frames_aug.shape
                                                frames_tensor = torch.from_numpy(frames_aug).permute(0, 3, 1, 2)  # (T, C, H, W)
                                                frames_tensor = frames_tensor.unsqueeze(0)  # (1, T, C, H, W)
                                                
                                                # Extract embedding
                                                with torch.no_grad():
                                                    tokens = backbone(frames_tensor.to(device))
                                                    # Average pool tokens: (1, N, D) -> (D,)
                                                    emb = tokens.mean(dim=1).squeeze(0).cpu().numpy()
                                                    embedding_list.append(emb)
                                                    # Clear GPU tensors immediately
                                                    del tokens, frames_tensor
                                            
                                            # Average embeddings from all orientations
                                            embedding = np.mean(embedding_list, axis=0)
                                            
                                            # L2 normalize for better diversity in angular space
                                            embedding_norm = np.linalg.norm(embedding)
                                            if embedding_norm > 0:
                                                embedding = embedding / embedding_norm
                                            
                                            embeddings.append(embedding)
                                            valid_clips.append(clip)
                                            
                                            # Clear intermediate variables
                                            del embedding_list, frames_array, frames_rgb, frames_bgr
                                            
                                            # Periodically clear CUDA cache to prevent accumulation
                                            if (clip_idx + 1) % 50 == 0 and torch.cuda.is_available():
                                                torch.cuda.empty_cache()
                                        except Exception as e:
                                            log_fn(f"      Failed to extract embedding for {clip_id}: {e}")
                                            continue
                                    
                                    if len(embeddings) >= limit:
                                        embeddings = np.array(embeddings)
                                        log_fn(f"    Using Farthest Point Sampling to select {limit} diverse clips...")
                                        selected_indices = farthest_point_sampling(embeddings, limit, seed=42)
                                        train_class_clips = [valid_clips[i] for i in selected_indices]
                                        train_clip_ids = {id(c) for c in train_class_clips}
                                        val_class_clips = [c for c in class_clips if id(c) not in train_clip_ids]
                                        log_fn(f"    Selected {len(train_class_clips)} diverse clips based on embeddings")
                                    else:
                                        log_fn(f"    Only {len(embeddings)} valid embeddings, falling back to random")
                                        random.shuffle(class_clips)
                                        train_class_clips = class_clips[:limit]
                                        val_class_clips = class_clips[limit:]
                                except Exception as e:
                                    log_fn(f"    Embedding-based selection failed: {e}")
                                    log_fn(f"    Falling back to random selection")
                                    random.shuffle(class_clips)
                                    train_class_clips = class_clips[:limit]
                                    val_class_clips = class_clips[limit:]
                            else:
                                log_fn(f"    Training: randomly selecting {limit} clips")
                                random.shuffle(class_clips)
                                train_class_clips = class_clips[:limit]
                                val_class_clips = class_clips[limit:]
                            
                            # Apply validation limit if set
                            if len(val_class_clips) > val_limit:
                                log_fn(f"    Validation: limiting to {val_limit} clips (from {len(val_class_clips)} available)")
                                random.shuffle(val_class_clips)
                                val_class_clips = val_class_clips[:val_limit]
                            else:
                                log_fn(f"    Validation: using all remaining {len(val_class_clips)} clips")
                            
                            train_clips.extend(train_class_clips)
                            val_clips.extend(val_class_clips)
                        else:
                            # Not enough clips to limit - use all for training, all for validation
                            log_fn(f"  {label}: {len(class_clips)} total clips (below limit of {limit})")
                            log_fn(f"    Training: using all {len(class_clips)} clips (can't limit below available)")
                            if use_all_for_training:
                                train_clips.extend(class_clips)
                            else:
                                # Keep train/val disjoint even for tiny classes.
                                # If there is only 1 clip, we cannot hold out validation without losing
                                # training signal, so we place it in training only.
                                if len(class_clips) <= 1:
                                    train_clips.extend(class_clips)
                                    log_fn("    Validation: 0 clips (class too small to split without overlap)")
                                else:
                                    tmp = list(class_clips)
                                    random.shuffle(tmp)
                                    n = len(tmp)
                                    n_val = int(round(n * float(val_split)))
                                    if n_val <= 0:
                                        n_val = 1
                                    if n_val >= n:
                                        n_val = n - 1
                                    val_class_clips = tmp[:n_val]
                                    train_class_clips = tmp[n_val:]
                                    # Apply validation cap if requested
                                    if val_limit != float('inf') and len(val_class_clips) > int(val_limit):
                                        random.shuffle(val_class_clips)
                                        val_class_clips = val_class_clips[:int(val_limit)]
                                    train_clips.extend(train_class_clips)
                                    val_clips.extend(val_class_clips)
                                    log_fn(f"    Validation: {len(val_class_clips)} clips (held out, no overlap)")
                    else:
                        # No limit for this class, split normally
                        if use_all_for_training:
                            train_clips.extend(class_clips)
                        else:
                            train_class, val_class = _split_train_val_clip_stratified(
                                class_clips,
                                val_split,
                                seed=42,
                            )
                            train_clips.extend(train_class)
                            val_clips.extend(val_class)
                
                # Free GPU memory after diversity selection
                if backbone is not None:
                    log_fn("Freeing VideoPrism backbone from GPU memory...")
                    del backbone
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    log_fn("GPU memory freed")
                
                log_fn(f"Final dataset sizes:")
                log_fn(f"  Training: {len(train_clips)} clips")
                if not use_all_for_training:
                    log_fn(f"  Validation: {len(val_clips)} clips")
                
                def _log_split_distribution(clips, split_name):
                    counts = Counter([c["label"] for c in clips])
                    log_fn(f"{split_name} class distribution:")
                    mc = 0
                    for c in clips:
                        ll = c.get("labels")
                        if isinstance(ll, list) and len(ll) > 1:
                            mc += 1
                    for label, count in sorted(counts.items()):
                        log_fn(f"  {label}: {count}")
                    if mc > 0:
                        log_fn(f"  ({mc} multi-class clips in {split_name.lower()} set)")

                _log_split_distribution(train_clips, "Training")
                if not use_all_for_training:
                    _log_split_distribution(val_clips, "Validation")
            else:
                # No per-class limits, use standard train/val split
                if use_all_for_training:
                    log_fn("Using all data for training (no validation split)")
                    train_clips = labeled_clips
                    val_clips = []
                else:
                    log_fn(f"Splitting dataset into train/val (val_split={val_split:.1%}, stratified when possible)...")
                    has_frame_labels = any(
                        isinstance(c.get("frame_labels"), (list, tuple)) and len(c.get("frame_labels") or []) > 0
                        for c in labeled_clips
                    )
                    if has_frame_labels:
                        log_fn("Using frame-stratified split (equal proportion of each class in validation).")
                        train_clips, val_clips = _split_train_val_frame_stratified(
                            labeled_clips,
                            classes,
                            val_split,
                            seed=42,
                        )
                    else:
                        log_fn("Using clip-stratified split (no per-frame labels).")
                        train_clips, val_clips = _split_train_val_clip_stratified(
                            labeled_clips,
                            val_split,
                            seed=42,
                        )
                    log_fn(f"Train: {len(train_clips)}, Val: {len(val_clips)}")
            
            log_fn(f"Train: {len(train_clips)} clips")
            
            log_fn("Validating clip files exist...")
            log_fn(f"Checking clips directory: {self.clips_dir}")
            
            if not os.path.exists(self.clips_dir):
                error_msg = f"Clips directory does not exist: {self.clips_dir}\n\nPlease check the clips directory path in the Training tab."
                log_fn(f"ERROR: {error_msg}")
                self.error.emit(error_msg)
                return
            
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV', '.MKV']
            all_video_files = []
            for root, dirs, files in os.walk(self.clips_dir):
                for file in files:
                    if any(file.lower().endswith(ext.lower()) for ext in video_extensions):
                        rel_path = os.path.relpath(os.path.join(root, file), self.clips_dir)
                        all_video_files.append(rel_path.replace('\\', '/'))
            
            log_fn(f"Found {len(all_video_files)} video files in directory (including subdirectories)")
            if len(all_video_files) > 0 and len(all_video_files) < 10:
                log_fn(f"Video files found: {', '.join(all_video_files)}")
            
            missing_clips = []
            
            for clip_info in train_clips:
                clip_id = clip_info["id"]
                clip_path = os.path.join(self.clips_dir, clip_id)
                found = False
                
                if os.path.exists(clip_path):
                    found = True
                else:
                    base_name, ext = os.path.splitext(clip_id)
                    clip_basename = os.path.basename(clip_id)
                    clip_dir_part = os.path.dirname(clip_id) if os.path.dirname(clip_id) else None
                    
                    if not ext:
                        for video_ext in video_extensions:
                            test_path = os.path.join(self.clips_dir, clip_id + video_ext)
                            if os.path.exists(test_path):
                                found = True
                                break
                    else:
                        base_name_only = os.path.basename(base_name)
                        
                        for video_ext in video_extensions:
                            test_path = os.path.join(self.clips_dir, base_name + video_ext)
                            if os.path.exists(test_path):
                                found = True
                                break
                            
                            if not found:
                                test_path = os.path.join(self.clips_dir, base_name_only + video_ext)
                                if os.path.exists(test_path):
                                    found = True
                                    break
                        
                        if not found:
                            for root, dirs, files in os.walk(self.clips_dir):
                                for file in files:
                                    file_base, file_ext = os.path.splitext(file)
                                    if file_base == base_name_only or file_base == base_name:
                                        if file_ext.lower() in [e.lower() for e in video_extensions]:
                                            found = True
                                            break
                                if found:
                                    break
                        
                        if not found and clip_dir_part:
                            subdir_path = os.path.join(self.clips_dir, clip_dir_part)
                            if os.path.exists(subdir_path):
                                for video_ext in video_extensions:
                                    test_path = os.path.join(subdir_path, clip_basename)
                                    if os.path.exists(test_path):
                                        found = True
                                        break
                                    test_path = os.path.join(subdir_path, base_name_only + video_ext)
                                    if os.path.exists(test_path):
                                        found = True
                                        break
                    
                    if not found:
                        missing_clips.append(clip_id)
            
            if missing_clips:
                log_fn(f"WARNING: Found {len(missing_clips)} missing clip files")
                log_fn("Searching for files in subdirectories...")
                
                found_in_subdirs = {}
                sample_missing = missing_clips[:5]
                
                for clip_id in sample_missing:
                    base_name, ext = os.path.splitext(clip_id)
                    for root, dirs, files in os.walk(self.clips_dir):
                        for file in files:
                            file_base, file_ext = os.path.splitext(file)
                            if file_base == base_name or file_base == os.path.basename(base_name):
                                if file_ext.lower() in [e.lower() for e in video_extensions]:
                                    rel_path = os.path.relpath(os.path.join(root, file), self.clips_dir)
                                    if clip_id not in found_in_subdirs:
                                        found_in_subdirs[clip_id] = rel_path
                
                if found_in_subdirs:
                    log_fn(f"Found {len(found_in_subdirs)} files in subdirectories:")
                    for clip_id, found_path in list(found_in_subdirs.items())[:3]:
                        log_fn(f"  {clip_id} -> {found_path}")
                    log_fn("SUGGESTION: Your clips are in subdirectories. Update the clip IDs in annotations.json")
                    log_fn("  to include the subdirectory path, or move files to the root clips directory.")
                
                error_msg = f"Found {len(missing_clips)} missing clip files (out of {len(train_clips)} total)\n\n"
                error_msg += f"Clips directory: {self.clips_dir}\n\n"
                error_msg += "Sample missing files:\n"
                for clip_id in missing_clips[:10]:
                    error_msg += f"  - {clip_id}\n"
                if len(missing_clips) > 10:
                    error_msg += f"  ... and {len(missing_clips) - 10} more\n"
                
                if found_in_subdirs:
                    error_msg += f"\nNOTE: Found {len(found_in_subdirs)} of the sample files in subdirectories.\n"
                    error_msg += "Your annotation file may need to include subdirectory paths in clip IDs.\n"
                    error_msg += "Example: 'subfolder/span10.avi' instead of 'span10.avi'\n"
                
                error_msg += f"\nPlease:\n"
                error_msg += f"1. Check that clip files exist in: {self.clips_dir}\n"
                error_msg += f"2. Verify the clips directory path in the Training tab\n"
                error_msg += f"3. Update annotations.json to include correct paths if files are in subdirectories"
                
                log_fn(f"ERROR: {error_msg}")
                self.error.emit(error_msg)
                return
            
            log_fn(f"All {len(train_clips)} clips validated successfully")
            
            log_fn("Creating datasets...")
            try:
                from singlebehaviorlab.backend.augmentations import ClipAugment
                
                use_augmentation = self.train_config.get("use_augmentation", False)
                augmentation_options = self.train_config.get("augmentation_options", None)
                if not isinstance(augmentation_options, dict):
                    augmentation_options = {}
                transform = None
                if use_augmentation:
                    augmentation_defaults = {
                        "use_horizontal_flip": True,
                        "use_vertical_flip": False,
                        "use_color_jitter": True,
                        "use_gaussian_blur": True,
                        "use_random_noise": True,
                        "use_small_rotation": False,
                        "use_speed_perturb": False,
                        "use_random_shapes": False,
                        "use_grayscale": False,
                        "use_lighting_robustness": True,
                    }
                    for key, value in augmentation_defaults.items():
                        if key not in augmentation_options:
                            augmentation_options[key] = value
                    if self.train_config.get("use_localization", False) and augmentation_options.get("use_small_rotation", False):
                        # Rotation changes object coordinates; current spatial-label augmentation
                        # only mirrors bboxes, so disable rotation for localization training.
                        augmentation_options["use_small_rotation"] = False
                        log_fn("Localization is enabled: disabling small rotation to keep bbox supervision aligned.")
                    transform = ClipAugment(
                        use_horizontal_flip=augmentation_options["use_horizontal_flip"],
                        use_vertical_flip=augmentation_options["use_vertical_flip"],
                        use_color_jitter=augmentation_options["use_color_jitter"],
                        use_gaussian_blur=augmentation_options["use_gaussian_blur"],
                        use_random_noise=augmentation_options["use_random_noise"],
                        use_small_rotation=augmentation_options["use_small_rotation"],
                        use_speed_perturb=augmentation_options.get("use_speed_perturb", False),
                        use_random_shapes=augmentation_options.get("use_random_shapes", False),
                        use_grayscale=augmentation_options.get("use_grayscale", False),
                        use_lighting_robustness=augmentation_options.get("use_lighting_robustness", False),
                        gaussian_blur_sigma=(0.1, 0.5),
                        noise_std=0.02,
                        rotation_degrees=5.0,
                    )
                    log_fn("Using selected data augmentation for training:")
                    if augmentation_options["use_horizontal_flip"]:
                        log_fn("  - Random horizontal flip")
                    if augmentation_options["use_vertical_flip"]:
                        log_fn("  - Random vertical flip")
                    if augmentation_options["use_color_jitter"]:
                        log_fn("  - Color jitter (brightness, contrast, saturation, hue)")
                    if augmentation_options["use_gaussian_blur"]:
                        log_fn("  - Gaussian blur (0.1-0.5 sigma)")
                    if augmentation_options["use_random_noise"]:
                        log_fn("  - Random noise (std=0.02)")
                    if augmentation_options.get("use_speed_perturb", False):
                        log_fn("  - Speed perturbation (0.7x-1.3x)")
                    if augmentation_options.get("use_random_shapes", False):
                        log_fn("  - Random shape overlays (occlusion)")
                    if augmentation_options.get("use_grayscale", False):
                        log_fn("  - Random grayscale (50% chance)")
                    if augmentation_options.get("use_lighting_robustness", False):
                        log_fn("  - Lighting/color robustness (gamma + channel gain)")
                    if augmentation_options["use_small_rotation"]:
                        log_fn("  - Small rotation (+/- 5 degrees)")
                    log_fn("  - No cropping (content always preserved)")
                else:
                    log_fn("No augmentation (using raw clips)")
                
                clip_length = self.train_config.get("clip_length", 16)
                resolution = self.train_config.get("resolution", 288)
                log_fn(f"Using clip_length={clip_length} frames (center-cropped if clips are longer)")
                log_fn(f"Using input resolution: {resolution}x{resolution}")
                
                # Virtual Dataset Expansion for small datasets
                num_train = len(train_clips)
                virtual_multiplier = 1
                if num_train > 0 and use_augmentation:
                    virtual_multiplier = int(self.train_config.get("virtual_expansion", 5))
                    log_fn(f"Small dataset detected ({num_train} clips). Using virtual expansion x{virtual_multiplier}")
                    log_fn(f"Effective epoch size: {num_train * virtual_multiplier} samples (unique augmentations)")
                
                stitch_prob = float(self.train_config.get("stitch_augmentation_prob", 0.0))
                emb_aug_versions = int(self.train_config.get("emb_aug_versions", 1))
                _multi_scale = self.train_config.get("multi_scale", False) and not self.train_config.get("use_localization", False)
                if self.train_config.get("use_localization", False):
                    if stitch_prob > 0.0:
                        stitch_prob = 0.0
                        log_fn("Localization is enabled: disabling clip-stitch augmentation for this run.")
                aug_note = f" ({emb_aug_versions} aug version(s) per clip)" if emb_aug_versions > 1 else ""
                ms_note = " + short-scale (multi-scale)" if _multi_scale else ""
                log_fn(f"Embedding cache: always active{aug_note}{ms_note} — backbone skipped every training step")
                if stitch_prob > 0.0:
                    log_fn(
                        f"Clip-stitch augmentation: prob={stitch_prob:.0%}, fixed 50/50 split "
                        f"(applied on cached embeddings during classification training)"
                    )

                use_crop_jitter = bool(self.train_config.get("crop_jitter", False))
                crop_jitter_strength = float(self.train_config.get("crop_jitter_strength", 0.15))
                if use_crop_jitter and self.train_config.get("use_localization", False):
                    log_fn(f"Crop jitter: enabled (strength={crop_jitter_strength:.0%} of bbox size)")

                train_dataset = BehaviorDataset(
                    train_clips,
                    annotation_manager,
                    classes,
                    self.clips_dir,
                    transform=transform,
                    target_size=(resolution, resolution),
                    clip_length=clip_length,
                    virtual_size_multiplier=virtual_multiplier,
                    stitch_prob=stitch_prob,
                    crop_jitter=bool(self.train_config.get("crop_jitter", False)),
                    crop_jitter_strength=float(self.train_config.get("crop_jitter_strength", 0.15)),
                    ovr_background_classes=self.train_config.get("ovr_background_class_names", []) if hybrid_ovr_bg else [],
                )
                log_fn(f"Train dataset created: {len(train_dataset)} virtual samples (from {num_train} unique clips)")
            except Exception as e:
                error_msg = f"Failed to create train dataset: {str(e)}\n{traceback.format_exc()}"
                log_fn(f"ERROR: {error_msg}")
                self.error.emit(error_msg)
                return
            
            val_dataset = None
            if val_clips:
                try:
                    clip_length = self.train_config.get("clip_length", 16)
                    val_dataset = BehaviorDataset(
                        val_clips,
                        annotation_manager,
                        classes,
                        self.clips_dir,
                        target_size=(resolution, resolution),
                        clip_length=clip_length,
                        ovr_background_classes=self.train_config.get("ovr_background_class_names", []) if hybrid_ovr_bg else [],
                    )
                    log_fn(f"Val dataset created: {len(val_dataset)} samples")
                except Exception as e:
                    error_msg = f"Failed to create val dataset: {str(e)}\n{traceback.format_exc()}"
                    log_fn(f"ERROR: {error_msg}")
                    self.error.emit(error_msg)
                    return
            else:
                log_fn("No validation dataset (using all data for training)")

            device = 'cuda' if torch.cuda.is_available() else 'cpu'

            def check_stop():
                return self.should_stop

            if self.train_config.get("auto_tune_before_final", False):
                base_train_cfg = copy.deepcopy(self.train_config)
                if val_dataset is None:
                    error_msg = "Auto-tune requires a validation set. Disable 'Use all data for training' or set a validation split."
                    log_fn(f"ERROR: {error_msg}")
                    self.error.emit(error_msg)
                    return
                best_cfg, best_search_result = self._run_autotune_search(
                    train_dataset,
                    val_dataset,
                    self.train_config,
                    classes,
                    augmentation_options,
                    log_fn,
                    progress_cb,
                    check_stop,
                )
                if self.should_stop:
                    log_fn("Auto-tune stopped.")
                    self.training_complete.emit(0.0, 0.0, 0.0, {})
                    self.finished.emit()
                    return
                if not best_cfg:
                    error_msg = "Auto-tune did not produce a valid candidate."
                    log_fn(f"ERROR: {error_msg}")
                    self.error.emit(error_msg)
                    return
                self._reset_runtime_dataset_caches(train_dataset)
                self._reset_runtime_dataset_caches(val_dataset)
                self.train_config = copy.deepcopy(best_cfg)
                self.train_config["epochs"] = int(base_train_cfg.get("epochs", 1))
                log_fn("Auto-tune selected final config:")
                log_fn(
                    f"  lr={self.train_config.get('classification_lr', self.train_config.get('lr', 0.0)):.2e}, "
                    f"wd={self.train_config.get('weight_decay', 0.0):.2e}, "
                    f"dropout={self.train_config.get('dropout', 0.0):.2f}, "
                    f"heads={self.train_config.get('head_kwargs', {}).get('num_heads', 4)}, "
                    f"layers={self.train_config.get('frame_head_temporal_layers', 1)}"
                    + (
                        f", ovr_ls={self.train_config.get('ovr_label_smoothing', 0.0):.2f}"
                        if self.train_config.get("use_ovr", False) else ""
                    )
                )
                if best_search_result:
                    log_fn(
                        f"  best search val F1={float(best_search_result.get('best_val_f1', 0.0) or 0.0):.2f}, "
                        f"val acc={float(best_search_result.get('best_val_acc', 0.0) or 0.0):.2f}"
                    )
                self._save_autotuned_profile(self.train_config, best_search_result, log_fn)
                log_fn("Starting final retrain from scratch with the selected config...")

            try:
                model, head_kwargs_for_metadata, dropout, localization_dropout, num_stages, multi_scale = self._build_model_for_config(
                    train_dataset, self.train_config, log_fn
                )
            except Exception as e:
                error_msg = f"Failed to create classifier: {str(e)}\n{traceback.format_exc()}"
                log_fn(f"ERROR: {error_msg}")
                self.error.emit(error_msg)
                return

            train_config = self._build_backend_train_config(
                self.train_config,
                self.output_path,
                classes,
                augmentation_options,
                head_kwargs_for_metadata,
                dropout,
                localization_dropout,
                num_stages,
                multi_scale,
            )
            
            log_fn("Starting training loop...")
            
            def check_stop():
                return self.should_stop
            
            try:
                result = train_model(
                    model,
                    train_dataset,
                    val_dataset,
                    train_config,
                    log_fn=log_fn,
                    progress_callback=progress_cb,
                    stop_callback=check_stop,
                    metrics_callback=metrics_cb
                )
                
                if self.should_stop:
                    log_fn("Training stopped.")
                    self.training_complete.emit(0.0, 0.0, 0.0, {})
                else:
                    log_fn("Training completed successfully!")
                    best_val = result.get("best_val_acc", 0.0) if result else 0.0
                    best_f1 = result.get("best_val_f1", 0.0) if result else 0.0
                    final_train = result.get("final_train_acc", 0.0) if result else 0.0
                    per_class_f1 = result.get("per_class_f1", {})
                    
                    self.training_complete.emit(best_val, best_f1, final_train, per_class_f1)
                    
                self.finished.emit()
            except Exception as e:
                error_msg = f"Training failed: {str(e)}\n{traceback.format_exc()}"
                log_fn(f"ERROR: {error_msg}")
                self.error.emit(error_msg)
                return
            
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}\n{traceback.format_exc()}"
            self.log_message.emit(f"ERROR: {error_msg}")
            self.error.emit(error_msg)


class TrainingVisualizationDialog(QDialog):
    """Real-time training visualization with adaptive layout based on active metrics."""

    _COLORS = {
        "train": "#2196F3",
        "val": "#FF9800",
        "train_cls": "#64B5F6",
        "macro_f1": "#4CAF50",
        "grid": "#e0e0e0",
        "bg": "#fafafa",
        "text": "#333333",
        "loc_iou": "#AB47BC",
        "loc_cerr": "#EF5350",
        "loc_vrate": "#26A69A",
    }
    _CLASS_PALETTE = [
        "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
        "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
        "#dcbeff", "#9A6324", "#800000", "#aaffc3", "#808000",
        "#000075", "#a9a9a9",
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Training Monitor")
        self.resize(1150, 850)

        root = QVBoxLayout()
        root.setContentsMargins(6, 6, 6, 6)
        self.setLayout(root)

        top_bar = QHBoxLayout()
        self._status_label = QLabel("Waiting for first epoch...")
        self._status_label.setStyleSheet("font-weight: bold; font-size: 13px; padding: 4px 0;")
        top_bar.addWidget(self._status_label, stretch=1)

        # F1 class filter: toggle which per-class lines are visible
        self._f1_filter_layout = QHBoxLayout()
        self._f1_filter_layout.setContentsMargins(0, 0, 0, 0)
        self._f1_filter_label = QLabel("F1 classes:")
        self._f1_filter_label.setStyleSheet("font-size: 11px; color: #666;")
        self._f1_filter_layout.addWidget(self._f1_filter_label)
        self._f1_filter_checks: dict[str, QCheckBox] = {}
        self._f1_filter_container = QWidget()
        self._f1_filter_container.setLayout(self._f1_filter_layout)
        self._f1_filter_container.setVisible(False)

        root.addLayout(top_bar)
        root.addWidget(self._f1_filter_container)

        # Horizontal splitter: charts left, crop preview right
        from PyQt6.QtWidgets import QSplitter, QScrollArea
        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(self._splitter, stretch=1)

        # Left: matplotlib charts
        chart_widget = QWidget()
        chart_layout = QVBoxLayout()
        chart_layout.setContentsMargins(0, 0, 0, 0)
        chart_widget.setLayout(chart_layout)
        self.figure = Figure(figsize=(10, 8), dpi=100)
        self.figure.patch.set_facecolor(self._COLORS["bg"])
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        chart_layout.addWidget(self.canvas)
        self._splitter.addWidget(chart_widget)

        # Right: crop progress preview (hidden until localization is active)
        self._crop_panel = QWidget()
        crop_layout = QVBoxLayout()
        crop_layout.setContentsMargins(4, 0, 4, 0)
        self._crop_panel.setLayout(crop_layout)

        crop_header = QLabel("Localization Crop Preview")
        crop_header.setStyleSheet("font-weight: bold; font-size: 12px; padding: 2px 0;")
        crop_header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        crop_layout.addWidget(crop_header)

        self._crop_epoch_label = QLabel("")
        self._crop_epoch_label.setStyleSheet("font-size: 11px; color: #666; padding: 0 0 4px 0;")
        self._crop_epoch_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        crop_layout.addWidget(self._crop_epoch_label)

        # Scrollable area for crop images
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; background: #fafafa; }")
        self._crop_container = QWidget()
        self._crop_container_layout = QVBoxLayout()
        self._crop_container_layout.setContentsMargins(0, 0, 0, 0)
        self._crop_container_layout.setSpacing(6)
        self._crop_container.setLayout(self._crop_container_layout)
        scroll.setWidget(self._crop_container)
        crop_layout.addWidget(scroll, stretch=1)

        self._crop_panel.setVisible(False)
        self._splitter.addWidget(self._crop_panel)
        self._splitter.setStretchFactor(0, 3)
        self._splitter.setStretchFactor(1, 1)

        self._axes = {}
        self._crop_labels: list = []
        self._init_data()

    def _init_data(self):
        self.epochs = []
        self._confusion_warmup_epoch = 0
        self.train_acc = []
        self.val_acc = []
        self.train_loss = []
        self.train_loss_class = []
        self.val_loss = []
        self.val_f1 = []
        self.per_class_f1 = {}
        self.loc_iou = []
        self.loc_center_error = []
        self.loc_valid_rate = []
        self._has_localization = False
        self._has_validation = False
        self._phase = "classification"
        self._crop_progress_dir = None
        self._last_crop_epoch = -1
        self._f1_classes_to_show = None  # None = show all; set of str = only these in F1 graph

    def reset(self, f1_classes_to_show=None, confusion_warmup_epoch: int = 0):
        """Clear all data for a new training run.

        f1_classes_to_show: when limit_classes is used, set of class names whose
        checkboxes start checked; None = all checked by default.
        confusion_warmup_epoch: epoch after which confusion sampler activates (0 = off/no line).
        """
        self._init_data()
        self._confusion_warmup_epoch = confusion_warmup_epoch
        self._f1_classes_to_show = f1_classes_to_show
        for cb in self._f1_filter_checks.values():
            cb.setParent(None)
            cb.deleteLater()
        self._f1_filter_checks.clear()
        self._f1_filter_container.setVisible(False)
        self.figure.clear()
        self._axes.clear()
        self.canvas.draw()
        self._status_label.setText("Waiting for first epoch...")
        self._crop_panel.setVisible(False)
        self._crop_epoch_label.setText("")
        self._clear_crop_images()

    def _clear_crop_images(self):
        for lbl in self._crop_labels:
            lbl.setParent(None)
            lbl.deleteLater()
        self._crop_labels.clear()

    # Layout helpers.

    def _rebuild_layout(self):
        """Create subplot grid based on which metrics are active."""
        self.figure.clear()
        self._axes.clear()

        panels = ["loss", "acc", "f1"]
        if self._has_localization:
            panels.append("loc")

        n = len(panels)
        for i, key in enumerate(panels):
            ax = self.figure.add_subplot(n, 1, i + 1)
            ax.set_facecolor(self._COLORS["bg"])
            self._axes[key] = ax

    def _style_ax(self, ax, title, ylabel, xlabel=None):
        ax.set_title(title, fontsize=10, fontweight="bold", color=self._COLORS["text"], pad=6)
        ax.set_ylabel(ylabel, fontsize=9, color=self._COLORS["text"])
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=9, color=self._COLORS["text"])
        ax.grid(True, linewidth=0.5, color=self._COLORS["grid"], alpha=0.7)
        ax.tick_params(labelsize=8, colors=self._COLORS["text"])
        for spine in ax.spines.values():
            spine.set_color(self._COLORS["grid"])

    # Crop preview.

    def _update_crop_preview(self, epoch, crop_dir):
        """Load latest crop progress PNGs from disk and display them."""
        if not crop_dir:
            return
        self._crop_progress_dir = crop_dir

        # Crop progress images are saved every 2 epochs as epoch_NNN_sample_M.png
        pattern = os.path.join(crop_dir, f"epoch_{epoch:03d}_sample_*.png")
        files = sorted(glob.glob(pattern))
        if not files:
            return
        if epoch == self._last_crop_epoch:
            return
        self._last_crop_epoch = epoch

        self._clear_crop_images()
        self._crop_panel.setVisible(True)
        self._crop_epoch_label.setText(f"Epoch {epoch}  ·  {len(files)} sample(s)")

        panel_width = max(280, self._crop_panel.width() - 20)
        for fpath in files:
            try:
                pixmap = QPixmap(fpath)
                if pixmap.isNull():
                    continue
                scaled = pixmap.scaledToWidth(panel_width, Qt.TransformationMode.SmoothTransformation)
                lbl = QLabel()
                lbl.setPixmap(scaled)
                lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
                lbl.setStyleSheet("border: 1px solid #ccc; border-radius: 3px; background: white; padding: 2px;")
                lbl.setToolTip(os.path.basename(fpath))
                self._crop_container_layout.addWidget(lbl)
                self._crop_labels.append(lbl)
            except Exception as e:
                logger.debug("Could not load crop preview image: %s", e)

    # Main update.

    def update_plots(self, metrics):
        """Update plots with new epoch metrics dict."""
        epoch = metrics["epoch"]
        self.epochs.append(epoch)
        self.train_acc.append(metrics["train_acc"])
        # In frame-level training, clip-level val_acc is not meaningful/updated.
        # Prefer val_frame_acc for monitor accuracy visualization.
        self.val_acc.append(metrics.get("val_frame_acc", metrics.get("val_acc", 0.0)))
        self.train_loss.append(metrics["train_loss"])
        self.train_loss_class.append(metrics.get("train_loss_class", 0.0))
        self.val_loss.append(metrics.get("val_loss", 0.0))
        self.val_f1.append(metrics.get("val_f1", 0.0))
        self._phase = metrics.get("training_phase", "classification")

        if not self._has_validation and (
            any(v > 0 for v in self.val_acc)
            or any(v > 0 for v in self.val_loss)
            or any(v > 0 for v in self.val_f1)
        ):
            self._has_validation = True

        # Localization metrics
        iou = metrics.get("loc_val_iou", 0.0)
        cerr = metrics.get("loc_val_center_error", 0.0)
        vrate = metrics.get("loc_val_valid_rate", 0.0)
        self.loc_iou.append(iou)
        self.loc_center_error.append(cerr)
        self.loc_valid_rate.append(vrate)
        if not self._has_localization and (iou > 0 or vrate > 0):
            self._has_localization = True

        # Per-class F1
        for cls, score in metrics.get("per_class_f1", {}).items():
            if cls not in self.per_class_f1:
                self.per_class_f1[cls] = [0.0] * (len(self.epochs) - 1)
            self.per_class_f1[cls].append(score)
            if cls not in self._f1_filter_checks:
                checked = self._f1_classes_to_show is None or cls in self._f1_classes_to_show
                cb = QCheckBox(cls)
                cb.setChecked(checked)
                cb.setStyleSheet("font-size: 11px;")
                cb.stateChanged.connect(lambda _: self._redraw_f1_only())
                self._f1_filter_checks[cls] = cb
                self._f1_filter_layout.addWidget(cb)
                self._f1_filter_container.setVisible(True)
        for cls in self.per_class_f1:
            while len(self.per_class_f1[cls]) < len(self.epochs):
                self.per_class_f1[cls].append(0.0)

        self._rebuild_layout()
        self._draw_loss()
        self._draw_acc()
        self._draw_f1()
        if self._has_localization and "loc" in self._axes:
            self._draw_loc()

        # Only the bottom subplot gets an x-label
        panels = list(self._axes.values())
        if panels:
            panels[-1].set_xlabel("Epoch", fontsize=9, color=self._COLORS["text"])

        self.figure.tight_layout(h_pad=1.2)
        self.canvas.draw()

        # Crop preview: load latest images when available
        crop_dir = metrics.get("crop_progress_dir")
        if crop_dir:
            self._update_crop_preview(epoch, crop_dir)

        # Status bar
        phase_tag = f"[{self._phase}]" if self._phase != "classification" else ""
        best_f1 = max(self.val_f1) if self.val_f1 else 0.0
        self._status_label.setText(
            f"Epoch {epoch} {phase_tag}  |  Train Loss: {self.train_loss[-1]:.4f}  |  "
            f"Val F1: {self.val_f1[-1]:.1f}%  (best {best_f1:.1f}%)  |  "
            f"Val Frame Acc: {self.val_acc[-1]:.1f}%"
        )

    # Individual panel drawers.

    def _draw_loss(self):
        ax = self._axes["loss"]
        ax.plot(self.epochs, self.train_loss, color=self._COLORS["train"],
                linewidth=2, label="Train Loss")
        if any(v > 0 for v in self.train_loss_class):
            ax.plot(self.epochs, self.train_loss_class, color=self._COLORS["train_cls"],
                    linewidth=1.3, linestyle="--", alpha=0.8, label="Classification Loss")
        if self._has_validation:
            ax.plot(self.epochs, self.val_loss, color=self._COLORS["val"],
                    linewidth=2, label="Val Loss")
        self._style_ax(ax, "Loss", "Loss")
        ax.legend(fontsize=8, loc="upper right", framealpha=0.8)

    def _draw_acc(self):
        ax = self._axes["acc"]
        ax.plot(self.epochs, self.train_acc, color=self._COLORS["train"],
                linewidth=2, label="Train Acc")
        if self._has_validation:
            ax.plot(self.epochs, self.val_acc, color=self._COLORS["val"],
                    linewidth=2, label="Val Frame Acc")
        self._style_ax(ax, "Accuracy", "Accuracy (%)")
        ax.legend(fontsize=8, loc="lower right", framealpha=0.8)

    def _redraw_f1_only(self):
        """Redraw just the F1 panel when class visibility toggles change."""
        if "f1" not in self._axes or not self.epochs:
            return
        self._axes["f1"].clear()
        self._draw_f1()
        self.canvas.draw()

    def _draw_f1(self):
        ax = self._axes["f1"]
        ax.plot(self.epochs, self.val_f1, color=self._COLORS["macro_f1"],
                linewidth=2.5, label="Macro F1")
        visible_items = []
        for i, (cls, scores) in enumerate(self.per_class_f1.items()):
            cb = self._f1_filter_checks.get(cls)
            if cb is not None and not cb.isChecked():
                continue
            color = self._CLASS_PALETTE[i % len(self._CLASS_PALETTE)]
            ax.plot(self.epochs, scores, color=color,
                    linewidth=1.2, linestyle="--", alpha=0.7, label=cls)
            visible_items.append(cls)
        if self._confusion_warmup_epoch > 0:
            ax.axvline(x=self._confusion_warmup_epoch, color="#9E9E9E",
                       linestyle=":", linewidth=1.2, alpha=0.8)
            ax.text(self._confusion_warmup_epoch + 0.3, ax.get_ylim()[1] * 0.97,
                    "confusion sampler", fontsize=6, color="#757575",
                    va="top", ha="left", rotation=0)
        self._style_ax(ax, "Val Macro F1 (frame)", "F1 (%)")
        ncol = min(4, max(1, len(visible_items) + 1))
        ax.legend(fontsize=7, ncol=ncol, loc="lower right", framealpha=0.8)

    def _draw_loc(self):
        ax = self._axes["loc"]
        ax.plot(self.epochs, self.loc_iou, color=self._COLORS["loc_iou"],
                linewidth=2, label="IoU")
        ax.plot(self.epochs, self.loc_valid_rate, color=self._COLORS["loc_vrate"],
                linewidth=2, label="Valid Rate")
        ax2 = ax.twinx()
        ax2.plot(self.epochs, self.loc_center_error, color=self._COLORS["loc_cerr"],
                 linewidth=1.5, linestyle="--", alpha=0.8, label="Center Err")
        ax2.set_ylabel("Center Error", fontsize=8, color=self._COLORS["loc_cerr"])
        ax2.tick_params(labelsize=8, colors=self._COLORS["loc_cerr"])
        self._style_ax(ax, "Localization", "IoU / Valid Rate")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="lower right", framealpha=0.8)

class TrainingWidget(QWidget):
    """Widget for training the behavior classifier."""
    
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.augmentation_options = self._default_augmentation_options()
        self.worker = None
        self.annotation_manager = AnnotationManager(
            self.config.get(
                "training_annotation_file",
                self.config.get("annotation_file", "data/annotations/annotations.json"),
            )
        )
        self._config_initialized = False
        self.profile_dialog = None
        self.training_queue = []
        self.is_batch_training = False
        self.batch_results = []
        self.batch_results_path = None
        self.current_profile_name = None
        self.visualization_dialog = None
        self._resolution = int(self.config.get("resolution", 288))
        self._setup_ui()
        self._load_current_config(force=True)
        self.refresh_annotation_info()
    
    def _load_current_config(self, force: bool = False):
        """Apply current config paths to UI."""
        if self._config_initialized and not force:
            return
        
        annotation_file = self.config.get(
            "training_annotation_file",
            self.config.get("annotation_file", "data/annotations/annotations.json"),
        )
        clips_dir = self.config.get(
            "training_clips_dir",
            self.config.get("clips_dir", "data/clips"),
        )
        models_dir = self.config.get("models_dir", "models/behavior_heads")
        
        if hasattr(self, "annotation_file_edit"):
            self.annotation_file_edit.setText(annotation_file)
        if hasattr(self, "clips_dir_edit"):
            self.clips_dir_edit.setText(clips_dir)
        if hasattr(self, "output_path_edit"):
            default_output = os.path.join(models_dir, "head.pt")
            self.output_path_edit.setText(default_output)
        self._resolution = int(self.config.get("resolution", 288))
        if self._resolution % 18 != 0:
            self._resolution = (self._resolution // 18) * 18
        if hasattr(self, "weight_decay_spin"):
            self.weight_decay_spin.setValue(float(self.config.get("default_weight_decay", 0.001)))
        if hasattr(self, "use_supcon_check"):
            default_use_supcon = bool(self.config.get("default_use_supcon_loss", False))
            self.use_supcon_check.setChecked(default_use_supcon)
            if hasattr(self, "supcon_weight_spin"):
                self.supcon_weight_spin.setValue(float(self.config.get("default_supcon_weight", 0.2)))
                self.supcon_weight_spin.setEnabled(default_use_supcon)
            if hasattr(self, "supcon_temp_spin"):
                self.supcon_temp_spin.setValue(float(self.config.get("default_supcon_temperature", 0.1)))
                self.supcon_temp_spin.setEnabled(default_use_supcon)
        
        self._config_initialized = True
    
    def _setup_ui(self):
        """Setup UI components."""
        layout = QVBoxLayout()
        
        # Dataset Info with scrollbar
        info_group = QGroupBox("Dataset info")
        info_layout = QVBoxLayout()
        self.info_label = QLabel("Loading...")
        self.info_label.setWordWrap(True)
        info_layout.addWidget(self.info_label)
        info_group.setLayout(info_layout)
        
        info_scroll = QScrollArea()
        info_scroll.setWidget(info_group)
        info_scroll.setWidgetResizable(True)
        info_scroll.setMinimumHeight(120)
        info_scroll.setMaximumHeight(200)
        
        # Training Configuration inside a scrollable container with grouped sections
        config_container = QWidget()
        config_vbox = QVBoxLayout(config_container)
        config_vbox.setContentsMargins(2, 2, 2, 2)
        config_vbox.setSpacing(6)

        # --- Paths & Files ---
        paths_group = QGroupBox("Paths && Files")
        config_layout = QFormLayout()

        self.annotation_file_edit = QLineEdit()
        self.annotation_file_edit.setText(
            self.config.get(
                "training_annotation_file",
                self.config.get("annotation_file", "data/annotations/annotations.json"),
            )
        )
        self.annotation_browse_btn = QPushButton("Browse...")
        self.annotation_browse_btn.clicked.connect(self._browse_annotation)
        annotation_layout = QHBoxLayout()
        annotation_layout.addWidget(self.annotation_file_edit)
        annotation_layout.addWidget(self.annotation_browse_btn)
        config_layout.addRow("Annotation file:", annotation_layout)
        
        self.clips_dir_edit = QLineEdit()
        self.clips_dir_edit.setText(
            self.config.get(
                "training_clips_dir",
                self.config.get("clips_dir", "data/clips"),
            )
        )
        self.clips_browse_btn = QPushButton("Browse...")
        self.clips_browse_btn.clicked.connect(self._browse_clips_dir)
        clips_layout = QHBoxLayout()
        clips_layout.addWidget(self.clips_dir_edit)
        clips_layout.addWidget(self.clips_browse_btn)
        config_layout.addRow("Clips directory:", clips_layout)
        
        self.output_path_edit = QLineEdit()
        self.output_path_edit.setText(self.config.get("models_dir", "models/behavior_heads") + "/head.pt")
        self.output_browse_btn = QPushButton("Browse...")
        self.output_browse_btn.clicked.connect(self._browse_output)
        output_layout = QHBoxLayout()
        output_layout.addWidget(self.output_path_edit)
        output_layout.addWidget(self.output_browse_btn)
        config_layout.addRow("Output model:", output_layout)

        paths_group.setLayout(config_layout)

        # --- Training Hyperparameters ---
        hyper_group = QGroupBox("Training Hyperparameters")
        config_layout = QFormLayout()

        clip_length_layout = QHBoxLayout()
        self.clip_length_spin = QSpinBox()
        self.clip_length_spin.setRange(1, 64)
        self.clip_length_spin.setValue(int(self.config.get("default_clip_length", 8)))
        self.clip_length_spin.setToolTip(
            "Number of frames to use for training.\n"
            "Can be equal to or less than the actual clip length.\n"
            "If less, the middle N frames are used (temporal center-crop)."
        )
        clip_length_layout.addWidget(self.clip_length_spin)
        
        info_btn = QToolButton()
        info_btn.setText("?")
        info_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        info_btn.setStyleSheet("""
            QToolButton {
                background-color: #4A90E2;
                color: white;
                border: none;
                border-radius: 10px;
                width: 20px;
                height: 20px;
                font-weight: bold;
                font-size: 12px;
            }
            QToolButton:hover {
                background-color: #357ABD;
            }
        """)
        info_btn.setFixedSize(20, 20)
        info_btn.setToolTip("Click for information about frames per clip")
        info_btn.clicked.connect(self._show_clip_length_info)
        clip_length_layout.addWidget(info_btn)
        clip_length_layout.addStretch()
        
        config_layout.addRow("Frames per clip:", clip_length_layout)
        
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 32)
        self.batch_size_spin.setValue(8)
        config_layout.addRow("Batch size:", self.batch_size_spin)
        
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(60)
        config_layout.addRow("Epochs:", self.epochs_spin)
        
        lr_row = QHBoxLayout()
        self.loc_lr_spin = QDoubleSpinBox()
        self.loc_lr_spin.setRange(1e-6, 1.0)
        self.loc_lr_spin.setValue(1e-4)
        self.loc_lr_spin.setDecimals(6)
        self.loc_lr_spin.setSingleStep(1e-4)
        self.loc_lr_spin.setToolTip("Learning rate used during localization phase (localization head only).")
        lr_row.addWidget(self.loc_lr_spin)
        lr_row.addWidget(QLabel("Loc LR"))
        self.class_lr_spin = QDoubleSpinBox()
        self.class_lr_spin.setRange(1e-6, 1.0)
        self.class_lr_spin.setValue(1e-4)
        self.class_lr_spin.setDecimals(6)
        self.class_lr_spin.setSingleStep(1e-4)
        self.class_lr_spin.setToolTip("Learning rate used during classification phase (backbone + MAP head).")
        lr_row.addWidget(self.class_lr_spin)
        lr_row.addWidget(QLabel("Class LR"))
        lr_row.addStretch()
        config_layout.addRow("Learning rates:", lr_row)
        
        self.weight_decay_spin = QDoubleSpinBox()
        self.weight_decay_spin.setRange(0.0, 1.0)
        self.weight_decay_spin.setValue(float(self.config.get("default_weight_decay", 0.001)))
        self.weight_decay_spin.setDecimals(6)
        config_layout.addRow("Weight decay:", self.weight_decay_spin)

        hyper_group.setLayout(config_layout)

        # --- Model Architecture ---
        arch_group = QGroupBox("Model Architecture")
        config_layout = QFormLayout()

        self.map_num_heads_spin = QSpinBox()
        self.map_num_heads_spin.setRange(1, 16)
        self.map_num_heads_spin.setValue(4)
        config_layout.addRow("MAP num_heads:", self.map_num_heads_spin)
        
        self.proj_dim_spin = QSpinBox()
        self.proj_dim_spin.setRange(64, 1024)
        self.proj_dim_spin.setSingleStep(64)
        self.proj_dim_spin.setValue(256)
        self.proj_dim_spin.setToolTip("Projection dimension for spatial attention pool in the frame head.")
        config_layout.addRow("Spatial pool proj dim:", self.proj_dim_spin)

        self.use_multi_scale_check = QCheckBox("Multi-scale temporal context")
        self.use_multi_scale_check.setToolTip(
            "Cache backbone embeddings at two temporal scales: full fps and half fps\n"
            "(same clip duration, fewer frames). The temporal head sees both fine-grained\n"
            "local motion and broader context per frame, improving precision for subtle\n"
            "and short behaviors.\n\n"
            "Doubles backbone precomputation time. Requires clip_length ≥ 4.\n"
            "Disabled when localization is active."
        )
        self.use_multi_scale_check.setChecked(False)
        config_layout.addRow("", self.use_multi_scale_check)
        
        self.map_dropout_spin = QDoubleSpinBox()
        self.map_dropout_spin.setRange(0.0, 0.9)
        self.map_dropout_spin.setValue(0.3)
        self.map_dropout_spin.setDecimals(2)
        self.map_dropout_spin.setToolTip(
            "Dropout for classification head (spatial pool + temporal TCN + boundary head).\n"
            "0.2-0.3 recommended. Higher values may hurt temporal conv layers.\n"
            "Localization head dropout is fixed at 0.0."
        )
        config_layout.addRow("Classification head dropout:", self.map_dropout_spin)
        
        # Class-balanced loss weights
        self.use_class_weights_check = QCheckBox("Use class-balanced loss weights")
        self.use_class_weights_check.setToolTip(
            "Weight the loss by inverse class frequency to compensate for class imbalance.\n"
            "Rare classes get higher loss weight. Recommended when class counts are uneven."
        )
        self.use_class_weights_check.setChecked(False)
        config_layout.addRow("", self.use_class_weights_check)
        
        self.use_supcon_check = QCheckBox("Use supervised contrastive loss on MAP embeddings")
        self.use_supcon_check.setToolTip(
            "Add an auxiliary supervised contrastive loss on the attention-pooled\n"
            "frame embeddings before the final classifier. Can be used with or without\n"
            "the temporal decoder."
        )
        self.use_supcon_check.setChecked(False)
        self.use_supcon_check.stateChanged.connect(
            lambda state: (
                self.supcon_weight_spin.setEnabled(bool(state)),
                self.supcon_temp_spin.setEnabled(bool(state)),
            )
        )
        config_layout.addRow("", self.use_supcon_check)

        self.supcon_weight_spin = QDoubleSpinBox()
        self.supcon_weight_spin.setRange(0.0, 5.0)
        self.supcon_weight_spin.setSingleStep(0.05)
        self.supcon_weight_spin.setValue(0.2)
        self.supcon_weight_spin.setDecimals(2)
        self.supcon_weight_spin.setEnabled(False)
        self.supcon_weight_spin.setToolTip("Weight of the supervised contrastive loss term.")
        config_layout.addRow("SupCon weight:", self.supcon_weight_spin)

        self.supcon_temp_spin = QDoubleSpinBox()
        self.supcon_temp_spin.setRange(0.01, 2.0)
        self.supcon_temp_spin.setSingleStep(0.01)
        self.supcon_temp_spin.setValue(0.10)
        self.supcon_temp_spin.setDecimals(2)
        self.supcon_temp_spin.setEnabled(False)
        self.supcon_temp_spin.setToolTip("Temperature used in the supervised contrastive loss.")
        config_layout.addRow("SupCon temperature:", self.supcon_temp_spin)

        # Frame head settings (always active)
        self.use_temporal_decoder_check = QCheckBox("Use temporal decoder / refinement head")
        self.use_temporal_decoder_check.setToolTip(
            "Enable the temporal MS-TCN decoder after spatial attention pooling.\n"
            "If disabled, training uses the simpler baseline: spatial attention pooling\n"
            "+ direct per-frame classifier, while still producing framewise predictions."
        )
        self.use_temporal_decoder_check.setChecked(True)
        self.use_temporal_decoder_check.stateChanged.connect(self._on_temporal_decoder_toggled)
        config_layout.addRow("", self.use_temporal_decoder_check)

        self.frame_head_layers_spin = QSpinBox()
        self.frame_head_layers_spin.setRange(1, 8)
        self.frame_head_layers_spin.setValue(4)
        self.frame_head_layers_spin.setToolTip(
            "Number of dilated temporal conv layers in the frame head.\n"
            "Higher values increase temporal receptive field."
        )
        config_layout.addRow("Frame head temporal layers:", self.frame_head_layers_spin)

        self.temporal_pool_spin = QSpinBox()
        self.temporal_pool_spin.setRange(1, 4)
        self.temporal_pool_spin.setValue(1)
        self.temporal_pool_spin.setToolTip(
            "Average this many adjacent frames before temporal classification.\n"
            "1 = per-frame (no pooling), 2 = average pairs."
        )
        config_layout.addRow("Temporal pool (frames):", self.temporal_pool_spin)

        # Boundary loss
        self.boundary_loss_weight_spin = QDoubleSpinBox()
        self.boundary_loss_weight_spin.setRange(0.0, 5.0)
        self.boundary_loss_weight_spin.setSingleStep(0.1)
        self.boundary_loss_weight_spin.setValue(0.3)
        self.boundary_loss_weight_spin.setDecimals(2)
        self.boundary_loss_weight_spin.setToolTip("Weight of boundary detection loss (change-point prediction).")
        config_layout.addRow("Boundary loss weight:", self.boundary_loss_weight_spin)

        self.boundary_tolerance_spin = QSpinBox()
        self.boundary_tolerance_spin.setRange(0, 10)
        self.boundary_tolerance_spin.setValue(2)
        self.boundary_tolerance_spin.setToolTip("Tolerance in frames around labeled transition for boundary target.")
        config_layout.addRow("Boundary tolerance:", self.boundary_tolerance_spin)

        # Smoothness loss
        self.smoothness_loss_weight_spin = QDoubleSpinBox()
        self.smoothness_loss_weight_spin.setRange(0.0, 1.0)
        self.smoothness_loss_weight_spin.setSingleStep(0.01)
        self.smoothness_loss_weight_spin.setValue(0.05)
        self.smoothness_loss_weight_spin.setDecimals(3)
        self.smoothness_loss_weight_spin.setToolTip("Weight for temporal smoothness regularization on frame predictions.")
        config_layout.addRow("Smoothness loss weight:", self.smoothness_loss_weight_spin)

        # Bout balance weighting
        self.use_bout_balance_check = QCheckBox("Use bout balance weighting")
        self.use_bout_balance_check.setToolTip(
            "Weight each frame inversely to its contiguous segment length.\n"
            "Prevents long bouts from dominating the loss over short actions."
        )
        self.use_bout_balance_check.setChecked(True)
        config_layout.addRow("", self.use_bout_balance_check)

        self.bout_balance_power_spin = QDoubleSpinBox()
        self.bout_balance_power_spin.setRange(0.1, 2.0)
        self.bout_balance_power_spin.setSingleStep(0.1)
        self.bout_balance_power_spin.setValue(1.0)
        self.bout_balance_power_spin.setDecimals(1)
        self.bout_balance_power_spin.setToolTip(
            "Exponent for bout balance weighting: weight = segment_length ^ (-power).\n"
            "1.0 = linear (default, most aggressive).\n"
            "0.5 = square-root (softer).\n"
            "Higher = shorter segments get relatively more weight."
        )
        config_layout.addRow("Bout balance power:", self.bout_balance_power_spin)
        self.use_bout_balance_check.stateChanged.connect(
            lambda s: self.bout_balance_power_spin.setEnabled(bool(s))
        )

        arch_group.setLayout(config_layout)

        # --- Localization ---
        loc_group = QGroupBox("Localization (requires bbox labels)")
        self.loc_group = loc_group
        loc_group.setEnabled(False)
        loc_group.setToolTip(
            "Optional: localizes individual animals when multiple are in camera view.\n"
            "To activate, draw and save bounding boxes on at least some clips in the Labeling tab."
        )
        config_layout = QFormLayout()

        self.use_localization_check = QCheckBox("Use Localization Supervision (bbox)")
        self.use_localization_check.setToolTip(
            "Autonomous 2-stage training: first learn localization from bbox labels, then train classifier on localized crops."
        )
        self.use_localization_check.setChecked(False)
        config_layout.addRow("", self.use_localization_check)

        self.use_manual_loc_switch_check = QCheckBox("Manual switch to classification epoch")
        self.use_manual_loc_switch_check.setToolTip(
            "If enabled, switch from localization phase to classification phase at the selected epoch."
        )
        self.use_manual_loc_switch_check.setChecked(False)
        self.use_manual_loc_switch_check.stateChanged.connect(
            lambda state: self.manual_loc_switch_epoch_spin.setEnabled(
                bool(state) and self.use_localization_check.isChecked()
            )
        )
        config_layout.addRow("", self.use_manual_loc_switch_check)

        self.manual_loc_switch_epoch_spin = QSpinBox()
        self.manual_loc_switch_epoch_spin.setRange(1, 10000)
        self.manual_loc_switch_epoch_spin.setValue(20)
        self.manual_loc_switch_epoch_spin.setEnabled(False)
        self.manual_loc_switch_epoch_spin.setToolTip(
            "Epoch number at which localization stops and classification starts."
        )
        config_layout.addRow("Switch epoch:", self.manual_loc_switch_epoch_spin)

        self.use_localization_check.stateChanged.connect(
            lambda state: self.manual_loc_switch_epoch_spin.setEnabled(
                bool(state) and self.use_manual_loc_switch_check.isChecked()
            )
        )
        self.use_localization_check.stateChanged.connect(self._on_localization_toggled)

        self.crop_padding_spin = QDoubleSpinBox()
        self.crop_padding_spin.setRange(-0.45, 1.0)
        self.crop_padding_spin.setSingleStep(0.05)
        self.crop_padding_spin.setDecimals(2)
        self.crop_padding_spin.setValue(0.35)
        self.crop_padding_spin.setToolTip(
            "Fractional padding around the predicted bbox for classification crops.\n"
            "Positive: expand crop (0.35 = 70% larger than bbox).\n"
            "Negative: shrink/zoom in (-0.2 = crop is 60% of bbox, zooming into center).\n"
            "Use 0.05-0.10 for tight crops, negative for extreme close-ups."
        )
        config_layout.addRow("Crop padding:", self.crop_padding_spin)

        self.use_crop_jitter_check = QCheckBox("Crop jitter augmentation")
        self.use_crop_jitter_check.setToolTip(
            "Randomly shift the crop center during training to prevent\n"
            "the model from memorizing background/location cues.\n"
            "Only available when localization is enabled."
        )
        self.use_crop_jitter_check.setChecked(False)
        self.use_crop_jitter_check.setEnabled(False)
        config_layout.addRow("", self.use_crop_jitter_check)

        self.crop_jitter_strength_spin = QDoubleSpinBox()
        self.crop_jitter_strength_spin.setRange(0.01, 0.5)
        self.crop_jitter_strength_spin.setSingleStep(0.05)
        self.crop_jitter_strength_spin.setDecimals(2)
        self.crop_jitter_strength_spin.setValue(0.15)
        self.crop_jitter_strength_spin.setEnabled(False)
        self.crop_jitter_strength_spin.setToolTip(
            "Max random shift as a fraction of bbox size.\n"
            "0.15 = shift up to 15% of bbox width/height in any direction."
        )
        config_layout.addRow("Crop jitter strength:", self.crop_jitter_strength_spin)

        self.use_crop_jitter_check.stateChanged.connect(
            lambda s: self.crop_jitter_strength_spin.setEnabled(bool(s))
        )

        loc_group.setLayout(config_layout)

        # --- OvR & Hard Mining ---
        ovr_group = QGroupBox("OvR && Hard Mining")
        config_layout = QFormLayout()

        self.use_ovr_check = QCheckBox("One-vs-Rest heads (OvR)")
        self.use_ovr_check.setChecked(True)
        self.use_ovr_check.setVisible(False)

        self.ovr_background_negative_check = QCheckBox("Treat Other/background as OvR negatives")
        self.ovr_background_negative_check.setToolTip(
            "Hybrid OvR mode: remove helper classes like Other/Background from the trained heads,\n"
            "but keep those clips as all-zero negative supervision for the target heads.\n"
            "Useful when Other is heterogeneous and hurts learning real behaviors."
        )
        self.ovr_background_negative_check.setChecked(False)
        self.ovr_background_negative_check.setEnabled(False)
        config_layout.addRow("", self.ovr_background_negative_check)

        self.ovr_label_smoothing_spin = QDoubleSpinBox()
        self.ovr_label_smoothing_spin.setRange(0.0, 0.3)
        self.ovr_label_smoothing_spin.setSingleStep(0.01)
        self.ovr_label_smoothing_spin.setValue(0.05)
        self.ovr_label_smoothing_spin.setDecimals(2)
        self.ovr_label_smoothing_spin.setToolTip(
            "Smooth binary targets from [0,1] to [eps, 1-eps].\n"
            "Prevents overconfident predictions and improves generalization.\n"
            "0 = no smoothing, 0.05 = recommended default, 0.1+ = strong regularization."
        )
        config_layout.addRow("OvR label smoothing:", self.ovr_label_smoothing_spin)

        self.use_asl_check = QCheckBox("Use Asymmetric Loss (ASL)")
        self.use_asl_check.setToolTip(
            "Asymmetric Loss down-weights easy negatives more aggressively than positives.\n"
            "Recommended for OvR training where negatives dominate.\n"
            "γ- controls negative focusing, γ+ controls positive focusing."
        )
        self.use_asl_check.setChecked(True)
        self.use_asl_check.setEnabled(False)
        self.use_asl_check.stateChanged.connect(self._on_asl_toggled)
        config_layout.addRow("", self.use_asl_check)

        asl_params_layout = QHBoxLayout()
        self.asl_gamma_neg_spin = QDoubleSpinBox()
        self.asl_gamma_neg_spin.setRange(0.0, 10.0)
        self.asl_gamma_neg_spin.setSingleStep(0.5)
        self.asl_gamma_neg_spin.setValue(2.0)
        self.asl_gamma_neg_spin.setDecimals(1)
        self.asl_gamma_neg_spin.setToolTip("Focusing parameter for negative samples (higher = more suppression)")
        asl_params_layout.addWidget(QLabel("γ-:"))
        asl_params_layout.addWidget(self.asl_gamma_neg_spin)

        self.asl_gamma_pos_spin = QDoubleSpinBox()
        self.asl_gamma_pos_spin.setRange(0.0, 10.0)
        self.asl_gamma_pos_spin.setSingleStep(0.5)
        self.asl_gamma_pos_spin.setValue(0.0)
        self.asl_gamma_pos_spin.setDecimals(1)
        self.asl_gamma_pos_spin.setToolTip("Focusing parameter for positive samples (0 = no down-weighting)")
        asl_params_layout.addWidget(QLabel("γ+:"))
        asl_params_layout.addWidget(self.asl_gamma_pos_spin)

        self.asl_clip_spin = QDoubleSpinBox()
        self.asl_clip_spin.setRange(0.0, 0.5)
        self.asl_clip_spin.setSingleStep(0.01)
        self.asl_clip_spin.setValue(0.05)
        self.asl_clip_spin.setDecimals(2)
        self.asl_clip_spin.setToolTip("Probability margin for hard thresholding negatives")
        asl_params_layout.addWidget(QLabel("clip:"))
        asl_params_layout.addWidget(self.asl_clip_spin)

        self.asl_gamma_neg_spin.setEnabled(False)
        self.asl_gamma_pos_spin.setEnabled(False)
        self.asl_clip_spin.setEnabled(False)
        config_layout.addRow("ASL parameters:", asl_params_layout)

        # Confusion-aware sampler (OvR only)
        self.use_confusion_sampler_check = QCheckBox("Confusion-aware hard mining")
        self.use_confusion_sampler_check.setToolTip(
            "After each epoch, clips that trigger the wrong OvR heads (e.g. groom clips\n"
            "that also activate dig's head) are sampled more often in the next epoch.\n"
            "Helps the model learn hard distinctions between visually similar behaviours.\n"
            "Temperature: how sharply to focus on the hardest clips (higher = more aggressive)."
        )
        self.use_confusion_sampler_check.setChecked(True)
        self.use_confusion_sampler_check.setEnabled(False)
        def _on_confusion_sampler_toggled(state):
            on = bool(state) and self.use_ovr_check.isChecked()
            self.confusion_temperature_spin.setEnabled(on)
            self.confusion_warmup_spin.setEnabled(on)
        self.use_confusion_sampler_check.stateChanged.connect(_on_confusion_sampler_toggled)
        config_layout.addRow("", self.use_confusion_sampler_check)

        confusion_temp_layout = QHBoxLayout()
        self.confusion_temperature_spin = QDoubleSpinBox()
        self.confusion_temperature_spin.setRange(0.5, 8.0)
        self.confusion_temperature_spin.setSingleStep(0.5)
        self.confusion_temperature_spin.setValue(2.0)
        self.confusion_temperature_spin.setDecimals(1)
        self.confusion_temperature_spin.setToolTip(
            "Sharpness of the hard-mining distribution.\n"
            "1.0 = mild (all clips get similar weight).\n"
            "2.0 = moderate (default).\n"
            "4.0+ = aggressive (almost only hardest clips sampled)."
        )
        self.confusion_temperature_spin.setEnabled(False)
        confusion_temp_layout.addWidget(self.confusion_temperature_spin)
        config_layout.addRow("Confusion temperature:", confusion_temp_layout)

        confusion_warmup_layout = QHBoxLayout()
        self.confusion_warmup_spin = QSpinBox()
        self.confusion_warmup_spin.setRange(0, 80)
        self.confusion_warmup_spin.setSingleStep(5)
        self.confusion_warmup_spin.setValue(20)
        self.confusion_warmup_spin.setSuffix("%")
        self.confusion_warmup_spin.setToolTip(
            "Percentage of total epochs to use uniform sampling before\n"
            "activating confusion-based hard mining.\n"
            "0% = active from start, 20% = default warmup."
        )
        self.confusion_warmup_spin.setEnabled(False)
        confusion_warmup_layout.addWidget(self.confusion_warmup_spin)
        config_layout.addRow("Confusion warmup:", confusion_warmup_layout)

        self.use_hard_pair_mining_check = QCheckBox("Hard-pair mining")
        self.use_hard_pair_mining_check.setToolTip(
            "Add extra pairwise margin pressure for specific confusing class pairs.\n"
            "Use this for cases like rear vs digg where standard OvR is not enough."
        )
        self.use_hard_pair_mining_check.setChecked(False)
        self.use_hard_pair_mining_check.setEnabled(False)
        self.use_hard_pair_mining_check.stateChanged.connect(self._on_hard_pair_toggled)
        config_layout.addRow("", self.use_hard_pair_mining_check)

        self.hard_pair_edit = QLineEdit()
        self.hard_pair_edit.setPlaceholderText("rear:digg, move:digg")
        self.hard_pair_edit.setEnabled(False)
        self.hard_pair_edit.setToolTip(
            "Comma-separated hard pairs in the form class_a:class_b.\n"
            "Example: rear:digg, move:digg"
        )
        config_layout.addRow("Hard pairs:", self.hard_pair_edit)

        self.hard_pair_loss_weight_spin = QDoubleSpinBox()
        self.hard_pair_loss_weight_spin.setRange(0.0, 5.0)
        self.hard_pair_loss_weight_spin.setSingleStep(0.05)
        self.hard_pair_loss_weight_spin.setValue(0.2)
        self.hard_pair_loss_weight_spin.setDecimals(3)
        self.hard_pair_loss_weight_spin.setEnabled(False)
        self.hard_pair_loss_weight_spin.setToolTip(
            "Weight of the extra pair-margin loss added on top of the main frame loss."
        )
        config_layout.addRow("Hard-pair loss weight:", self.hard_pair_loss_weight_spin)

        self.hard_pair_margin_spin = QDoubleSpinBox()
        self.hard_pair_margin_spin.setRange(0.0, 5.0)
        self.hard_pair_margin_spin.setSingleStep(0.05)
        self.hard_pair_margin_spin.setValue(0.5)
        self.hard_pair_margin_spin.setDecimals(2)
        self.hard_pair_margin_spin.setEnabled(False)
        self.hard_pair_margin_spin.setToolTip(
            "Required logit gap between the true class and its configured rival."
        )
        config_layout.addRow("Hard-pair margin:", self.hard_pair_margin_spin)

        self.hard_pair_confusion_boost_spin = QDoubleSpinBox()
        self.hard_pair_confusion_boost_spin.setRange(1.0, 5.0)
        self.hard_pair_confusion_boost_spin.setSingleStep(0.1)
        self.hard_pair_confusion_boost_spin.setValue(1.5)
        self.hard_pair_confusion_boost_spin.setDecimals(2)
        self.hard_pair_confusion_boost_spin.setEnabled(False)
        self.hard_pair_confusion_boost_spin.setToolTip(
            "Extra multiplier for confusion-sampler scores when the top rival is a configured hard pair."
        )
        config_layout.addRow("Hard-pair sampler boost:", self.hard_pair_confusion_boost_spin)

        self.use_weighted_sampler_check = QCheckBox("Use weighted random sampler")
        self.use_weighted_sampler_check.setToolTip("Oversample rare classes during training")
        self.use_weighted_sampler_check.setChecked(False)
        config_layout.addRow("", self.use_weighted_sampler_check)

        ovr_group.setLayout(config_layout)
        self.use_ovr_check.stateChanged.connect(self._on_ovr_toggled)
        self._on_ovr_toggled(0)

        # --- Augmentation & Data ---
        data_group = QGroupBox("Augmentation && Data")
        config_layout = QFormLayout()

        self.use_augmentation_check = QCheckBox("Use data augmentation")
        self.use_augmentation_check.setToolTip("Apply selected augmentations to training clips")
        self.use_augmentation_check.setChecked(False)
        self.use_augmentation_check.stateChanged.connect(self._on_use_augmentation_changed)
        
        self.augmentation_options_btn = QPushButton("Augmentation options...")
        self.augmentation_options_btn.setToolTip("Choose which augmentations to apply during training")
        self.augmentation_options_btn.setEnabled(False)
        self.augmentation_options_btn.clicked.connect(self._open_augmentation_options_dialog)
        
        augmentation_row = QHBoxLayout()
        augmentation_row.addWidget(self.use_augmentation_check)
        augmentation_row.addWidget(self.augmentation_options_btn)
        config_layout.addRow("", augmentation_row)

        self.virtual_expansion_spin = QSpinBox()
        self.virtual_expansion_spin.setRange(1, 20)
        self.virtual_expansion_spin.setValue(5)
        self.virtual_expansion_spin.setEnabled(False)
        self.virtual_expansion_spin.setToolTip(
            "Virtual dataset expansion multiplier (only active when augmentation is on).\n"
            "Each unique clip is sampled this many times per epoch with different augmentations.\n"
            "Higher = more augmented variety per epoch; useful for small datasets."
        )
        config_layout.addRow("Virtual expansion (x):", self.virtual_expansion_spin)
        self.use_augmentation_check.stateChanged.connect(
            lambda s: self.virtual_expansion_spin.setEnabled(bool(s))
        )

        self.use_stitch_check = QCheckBox("Clip-stitch augmentation")
        self.use_stitch_check.setToolTip(
            "Splice two clips from different classes with a fixed 50/50 split.\n"
            "Teaches the model per-frame behavior regardless of clip-level context,\n"
            "which improves inference on clips containing multiple behaviors."
        )
        self.use_stitch_check.setChecked(False)
        config_layout.addRow("", self.use_stitch_check)

        self.stitch_prob_spin = QDoubleSpinBox()
        self.stitch_prob_spin.setRange(0.0, 1.0)
        self.stitch_prob_spin.setSingleStep(0.05)
        self.stitch_prob_spin.setValue(0.3)
        self.stitch_prob_spin.setDecimals(2)
        self.stitch_prob_spin.setEnabled(False)
        self.stitch_prob_spin.setToolTip(
            "Probability per sample of applying clip-stitch augmentation (0–1).\n"
            "0.3 means ~30% of training samples will be stitched mixed clips."
        )
        config_layout.addRow("Stitch probability:", self.stitch_prob_spin)

        self.emb_aug_versions_spin = QSpinBox()
        self.emb_aug_versions_spin.setRange(1, 20)
        self.emb_aug_versions_spin.setValue(5)
        self.emb_aug_versions_spin.setEnabled(False)
        self.emb_aug_versions_spin.setToolTip(
            "Number of augmented embedding versions to pre-compute per clip.\n"
            "Each version applies a different random augmentation before the backbone.\n"
            "Higher = more diversity (larger cache, slower precompute, same training speed).\n"
            "Requires augmentation to be enabled. Recommended: 3–8."
        )
        config_layout.addRow("Cached aug versions:", self.emb_aug_versions_spin)

        def _sync_emb_cache_controls():
            loc_on = self.use_localization_check.isChecked()
            aug_on = self.use_augmentation_check.isChecked()
            # Cached aug versions only useful when augmentation is on
            self.emb_aug_versions_spin.setEnabled(aug_on and not loc_on)
            # Multi-scale requires no localization
            self.use_multi_scale_check.setEnabled(not loc_on)
            if loc_on:
                self.use_multi_scale_check.setChecked(False)

        self._sync_emb_cache_controls = _sync_emb_cache_controls
        self.use_augmentation_check.stateChanged.connect(lambda _: _sync_emb_cache_controls())

        self.use_stitch_check.stateChanged.connect(
            lambda s: self.stitch_prob_spin.setEnabled(bool(s))
        )
        self.use_stitch_check.stateChanged.connect(lambda _s: self._sync_stitch_controls())
        self.use_localization_check.stateChanged.connect(lambda _s: self._sync_stitch_controls())
        self._sync_stitch_controls()

        self.use_all_for_training_check = QCheckBox("Use all data for training (no validation)")
        self.use_all_for_training_check.setToolTip("Enable this for small datasets. Disables validation split.")
        self.use_all_for_training_check.stateChanged.connect(self._on_use_all_changed)
        config_layout.addRow("", self.use_all_for_training_check)
        
        self.val_split_spin = QDoubleSpinBox()
        self.val_split_spin.setRange(0.0, 0.5)
        self.val_split_spin.setValue(0.2)
        self.val_split_spin.setDecimals(2)
        self.val_split_spin.setSingleStep(0.05)
        self.val_split_spin.setSuffix(" (20% = 0.2)")
        config_layout.addRow("Validation split:", self.val_split_spin)

        self.auto_tune_check = QCheckBox("Auto-tune before final training")
        self.auto_tune_check.setToolTip(
            "Run a small random search over a few important training settings,\n"
            "then retrain once from scratch using the best candidate.\n"
            "Requires a validation split."
        )
        self.auto_tune_check.setChecked(False)
        self.auto_tune_check.stateChanged.connect(self._on_auto_tune_changed)
        config_layout.addRow("", self.auto_tune_check)

        auto_tune_row = QHBoxLayout()
        self.auto_tune_runs_spin = QSpinBox()
        self.auto_tune_runs_spin.setRange(1, 32)
        self.auto_tune_runs_spin.setValue(8)
        self.auto_tune_runs_spin.setToolTip("Number of short candidate runs to evaluate before the final retrain.")
        auto_tune_row.addWidget(self.auto_tune_runs_spin)
        auto_tune_row.addWidget(QLabel("runs"))
        self.auto_tune_epochs_spin = QSpinBox()
        self.auto_tune_epochs_spin.setRange(1, 200)
        self.auto_tune_epochs_spin.setValue(12)
        self.auto_tune_epochs_spin.setToolTip("Epoch budget for each short auto-tune trial.")
        auto_tune_row.addWidget(self.auto_tune_epochs_spin)
        auto_tune_row.addWidget(QLabel("search epochs"))
        auto_tune_row.addStretch()
        config_layout.addRow("Auto-tune search:", auto_tune_row)
        self._on_auto_tune_changed(int(self.auto_tune_check.isChecked()))
        
        self.select_classes_check = QCheckBox("Limit classes for training")
        self.select_classes_check.setToolTip("Select specific classes to use for training (useful for testing minimum examples needed)")
        self.select_classes_check.stateChanged.connect(self._on_select_classes_changed)
        config_layout.addRow("", self.select_classes_check)
        
        self.class_selection_list = QListWidget()
        self.class_selection_list.setMaximumHeight(150)
        self.class_selection_list.setEnabled(False)
        config_layout.addRow("Selected classes:", self.class_selection_list)
        
        self.limit_per_class_check = QCheckBox("Limit annotations per class")
        self.limit_per_class_check.setToolTip("Limit the maximum number of clips used per class for training")
        self.limit_per_class_check.stateChanged.connect(self._on_limit_per_class_changed)
        config_layout.addRow("", self.limit_per_class_check)
        
        self.per_class_limit_table = QTableWidget()
        self.per_class_limit_table.setColumnCount(3)
        self.per_class_limit_table.setHorizontalHeaderLabels(["Class", "Max Train", "Max Val"])
        self.per_class_limit_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.per_class_limit_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.per_class_limit_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.per_class_limit_table.setMaximumHeight(150)
        self.per_class_limit_table.setEnabled(False)
        config_layout.addRow("Per-class limits:", self.per_class_limit_table)
        
        self.use_embedding_diversity_check = QCheckBox("Use embedding-based diversity selection")
        self.use_embedding_diversity_check.setToolTip("When limiting per-class samples, use VideoPrism embeddings to select the most diverse clips (Farthest Point Sampling). Better than random for capturing variety.")
        self.use_embedding_diversity_check.setChecked(False)
        config_layout.addRow("", self.use_embedding_diversity_check)
        
        # Fine-tuning controls
        self.finetune_check = QCheckBox("Fine-tune existing model")
        self.finetune_check.setToolTip("Load weights from an existing model instead of training from scratch")
        self.finetune_check.stateChanged.connect(self._on_finetune_changed)
        config_layout.addRow("", self.finetune_check)
        
        self.pretrained_path_edit = QLineEdit()
        self.pretrained_path_edit.setPlaceholderText("Path to existing .pt model")
        self.pretrained_path_edit.setEnabled(False)
        self.pretrained_browse_btn = QPushButton("Browse...")
        self.pretrained_browse_btn.setEnabled(False)
        self.pretrained_browse_btn.clicked.connect(self._browse_pretrained)
        
        pretrained_layout = QHBoxLayout()
        pretrained_layout.addWidget(self.pretrained_path_edit)
        pretrained_layout.addWidget(self.pretrained_browse_btn)
        config_layout.addRow("Pretrained model:", pretrained_layout)
        
        data_group.setLayout(config_layout)

        row1 = QHBoxLayout()
        row1.addWidget(paths_group, 1)
        row1.addWidget(info_scroll, 1)
        config_vbox.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(hyper_group, 1)
        row2.addWidget(loc_group, 1)
        config_vbox.addLayout(row2)

        row3 = QHBoxLayout()
        row3.addWidget(arch_group, 1)
        row3.addWidget(ovr_group, 1)
        config_vbox.addLayout(row3)

        config_vbox.addWidget(data_group)

        config_scroll = QScrollArea()
        config_scroll.setWidget(config_container)
        config_scroll.setWidgetResizable(True)
        config_scroll.setMinimumHeight(300)
        layout.addWidget(config_scroll, 1)
        
        control_layout = QHBoxLayout()
        self.visualize_btn = QPushButton("Visualize training")
        self.visualize_btn.setToolTip("Open real-time training visualization")
        self.visualize_btn.clicked.connect(self._open_visualization)
        self.visualize_btn.setEnabled(False)
        control_layout.addWidget(self.visualize_btn)
        
        self.advanced_btn = QPushButton("Advanced: Profiles")
        self.advanced_btn.setToolTip("Manage training profiles for batch experiments")
        self.advanced_btn.clicked.connect(self._open_profile_manager)
        control_layout.addWidget(self.advanced_btn)
        
        control_layout.addStretch()
        
        self.batch_train_check = QCheckBox("Batch Train Selected Profiles")
        self.batch_train_check.setToolTip("If checked, training will run sequentially for all profiles selected in the Advanced menu.")
        control_layout.addWidget(self.batch_train_check)
        
        self.train_btn = QPushButton("Start training")
        self.train_btn.clicked.connect(self._start_training)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self._stop_training)
        self.stop_btn.setEnabled(False)
        control_layout.addWidget(self.train_btn)
        control_layout.addWidget(self.stop_btn)
        layout.addLayout(control_layout)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        log_group = QGroupBox("Training logs")
        log_layout = QVBoxLayout()
        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)
        
        self.setLayout(layout)
        self._on_temporal_decoder_toggled(int(self.use_temporal_decoder_check.isChecked()))
    
    def _on_use_all_changed(self, state: int):
        """Enable/disable validation split controls."""
        use_all = self.use_all_for_training_check.isChecked()
        self.val_split_spin.setEnabled(not use_all)
        
        if use_all:
            self.log_text.appendPlainText("Using all data for training - validation disabled")
        else:
            self.log_text.appendPlainText(f"Validation split enabled: {self.val_split_spin.value():.1%}")

    def _on_temporal_decoder_toggled(self, state: int):
        """Enable only the controls that affect the temporal decoder branch."""
        use_decoder = self.use_temporal_decoder_check.isChecked()
        self.frame_head_layers_spin.setEnabled(use_decoder)
        self.temporal_pool_spin.setEnabled(use_decoder)
        self.boundary_loss_weight_spin.setEnabled(use_decoder)
        self.boundary_tolerance_spin.setEnabled(use_decoder)
        if not use_decoder:
            self.boundary_loss_weight_spin.setValue(0.0)
        elif self.boundary_loss_weight_spin.value() == 0.0:
            self.boundary_loss_weight_spin.setValue(0.3)
    
    def _on_ovr_toggled(self, state: int):
        """OvR has built-in per-head class balancing — disable redundant class weights."""
        is_on = self.use_ovr_check.isChecked()
        self.ovr_background_negative_check.setEnabled(is_on)
        self.ovr_label_smoothing_spin.setEnabled(is_on)
        self.use_asl_check.setEnabled(is_on)
        self.asl_gamma_neg_spin.setEnabled(is_on and self.use_asl_check.isChecked())
        self.asl_gamma_pos_spin.setEnabled(is_on and self.use_asl_check.isChecked())
        self.asl_clip_spin.setEnabled(is_on and self.use_asl_check.isChecked())
        self.use_confusion_sampler_check.setEnabled(is_on)
        self.confusion_temperature_spin.setEnabled(
            is_on and self.use_confusion_sampler_check.isChecked()
        )
        self.confusion_warmup_spin.setEnabled(
            is_on and self.use_confusion_sampler_check.isChecked()
        )
        self.use_hard_pair_mining_check.setEnabled(is_on)
        self._on_hard_pair_toggled(state)
        if is_on:
            self.use_class_weights_check.setChecked(False)
            self.use_class_weights_check.setEnabled(False)
        else:
            self.use_class_weights_check.setEnabled(True)
        
    def _on_asl_toggled(self, state: int):
        """Enable/disable ASL parameter controls."""
        is_on = self.use_asl_check.isChecked() and self.use_ovr_check.isChecked()
        self.asl_gamma_neg_spin.setEnabled(is_on)
        self.asl_gamma_pos_spin.setEnabled(is_on)
        self.asl_clip_spin.setEnabled(is_on)

    def _on_hard_pair_toggled(self, state: int):
        """Enable/disable hard-pair controls."""
        is_on = self.use_ovr_check.isChecked() and self.use_hard_pair_mining_check.isChecked()
        self.hard_pair_edit.setEnabled(is_on)
        self.hard_pair_loss_weight_spin.setEnabled(is_on)
        self.hard_pair_margin_spin.setEnabled(is_on)
        self.hard_pair_confusion_boost_spin.setEnabled(is_on)

    def _on_use_augmentation_changed(self, state: int):
        """Enable/disable augmentation options button and aug cache versions."""
        is_on = self.use_augmentation_check.isChecked()
        if hasattr(self, "augmentation_options_btn"):
            self.augmentation_options_btn.setEnabled(is_on)
        if not is_on and hasattr(self, "emb_aug_versions_spin"):
            self.emb_aug_versions_spin.setValue(1)
        if hasattr(self, "_sync_emb_cache_controls"):
            self._sync_emb_cache_controls()

    def _on_localization_toggled(self, state: int):
        """Manage crop jitter controls when localization is toggled."""
        loc_on = bool(state)
        if loc_on:
            self.use_crop_jitter_check.setEnabled(True)
            self.crop_jitter_strength_spin.setEnabled(self.use_crop_jitter_check.isChecked())
        else:
            self.use_crop_jitter_check.setChecked(False)
            self.use_crop_jitter_check.setEnabled(False)
            self.crop_jitter_strength_spin.setEnabled(False)

    def _sync_stitch_controls(self):
        """Disable stitching whenever localization supervision is enabled."""
        if not hasattr(self, "use_stitch_check"):
            return
        localization_on = bool(self.use_localization_check.isChecked()) if hasattr(self, "use_localization_check") else False
        if localization_on:
            self.use_stitch_check.setChecked(False)
            self.use_stitch_check.setEnabled(False)
            self.stitch_prob_spin.setEnabled(False)
        else:
            self.use_stitch_check.setEnabled(True)
            self.stitch_prob_spin.setEnabled(self.use_stitch_check.isChecked())
        if hasattr(self, "_sync_emb_cache_controls"):
            self._sync_emb_cache_controls()

    def _parse_hard_pairs_text(self, text: str) -> list[list[str]]:
        """Parse comma-separated hard-pair text into [[class_a, class_b], ...]."""
        pairs = []
        seen = set()
        for chunk in (text or "").split(","):
            item = chunk.strip()
            if not item:
                continue
            parts = [p.strip() for p in re.split(r"\s*(?::|>|/|\bvs\b)\s*", item, maxsplit=1, flags=re.IGNORECASE) if p.strip()]
            if len(parts) != 2 or parts[0] == parts[1]:
                continue
            key = tuple(sorted(parts))
            if key in seen:
                continue
            seen.add(key)
            pairs.append([parts[0], parts[1]])
        return pairs

    def _format_hard_pairs_text(self, pairs) -> str:
        """Format stored hard-pair config back into the UI text box."""
        if not isinstance(pairs, (list, tuple)):
            return ""
        items = []
        for pair in pairs:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                continue
            a_name = str(pair[0]).strip()
            b_name = str(pair[1]).strip()
            if not a_name or not b_name:
                continue
            items.append(f"{a_name}:{b_name}")
        return ", ".join(items)

    def _default_augmentation_options(self) -> dict:
        return {
            "use_horizontal_flip": True,
            "use_vertical_flip": False,
            "use_color_jitter": True,
            "use_gaussian_blur": True,
            "use_random_noise": True,
            "use_small_rotation": False,
            "use_speed_perturb": False,
            "use_random_shapes": False,
            "use_grayscale": False,
            "use_lighting_robustness": True,
        }

    def _normalize_augmentation_options(self, options: dict) -> dict:
        defaults = self._default_augmentation_options()
        if not isinstance(options, dict):
            return defaults
        for key in defaults:
            defaults[key] = bool(options.get(key, defaults[key]))
        return defaults

    def _open_augmentation_options_dialog(self):
        """Open dialog to select augmentations."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Augmentation Options")
        
        layout = QVBoxLayout(dialog)
        grid = QGridLayout()
        
        options = self._normalize_augmentation_options(self.augmentation_options)
        
        hflip_check = QCheckBox("Random horizontal flip")
        hflip_check.setChecked(options["use_horizontal_flip"])
        grid.addWidget(hflip_check, 0, 0)

        vflip_check = QCheckBox("Random vertical flip")
        vflip_check.setChecked(options["use_vertical_flip"])
        grid.addWidget(vflip_check, 1, 0)
        
        color_check = QCheckBox("Color jitter (brightness/contrast/saturation/hue)")
        color_check.setChecked(options["use_color_jitter"])
        grid.addWidget(color_check, 2, 0)
        
        blur_check = QCheckBox("Gaussian blur (0.1-0.5 sigma)")
        blur_check.setChecked(options["use_gaussian_blur"])
        grid.addWidget(blur_check, 3, 0)
        
        noise_check = QCheckBox("Random noise (std=0.02)")
        noise_check.setChecked(options["use_random_noise"])
        grid.addWidget(noise_check, 4, 0)
        
        rot_check = QCheckBox("Small rotation (+/- 5 degrees)")
        rot_check.setChecked(options["use_small_rotation"])
        grid.addWidget(rot_check, 5, 0)

        speed_check = QCheckBox("Speed perturbation (0.7x - 1.3x)")
        speed_check.setChecked(options.get("use_speed_perturb", False))
        grid.addWidget(speed_check, 6, 0)

        shapes_check = QCheckBox("Random shape overlays (occlusion)")
        shapes_check.setChecked(options.get("use_random_shapes", False))
        grid.addWidget(shapes_check, 7, 0)

        gray_check = QCheckBox("Random grayscale (50% chance)")
        gray_check.setChecked(options.get("use_grayscale", False))
        grid.addWidget(gray_check, 8, 0)

        light_check = QCheckBox("Lighting / color robustness")
        light_check.setToolTip("Clip-consistent gamma and per-channel gain jitter to reduce brightness/color bias.")
        light_check.setChecked(options.get("use_lighting_robustness", False))
        grid.addWidget(light_check, 9, 0)
        
        layout.addLayout(grid)
        
        buttons_layout = QHBoxLayout()
        buttons_layout.addStretch()
        ok_btn = QPushButton("OK")
        cancel_btn = QPushButton("Cancel")
        ok_btn.clicked.connect(dialog.accept)
        cancel_btn.clicked.connect(dialog.reject)
        buttons_layout.addWidget(ok_btn)
        buttons_layout.addWidget(cancel_btn)
        layout.addLayout(buttons_layout)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.augmentation_options = {
                "use_horizontal_flip": hflip_check.isChecked(),
                "use_vertical_flip": vflip_check.isChecked(),
                "use_color_jitter": color_check.isChecked(),
                "use_gaussian_blur": blur_check.isChecked(),
                "use_random_noise": noise_check.isChecked(),
                "use_small_rotation": rot_check.isChecked(),
                "use_speed_perturb": speed_check.isChecked(),
                "use_random_shapes": shapes_check.isChecked(),
                "use_grayscale": gray_check.isChecked(),
                "use_lighting_robustness": light_check.isChecked(),
            }
    
    def _browse_annotation(self):
        """Browse for annotation file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Annotation File",
            self.config.get("data_dir", "data"),
            "JSON Files (*.json);;All Files (*)"
        )
        if file_path:
            self.annotation_file_edit.setText(file_path)
            self.refresh_annotation_info()
    
    def _browse_clips_dir(self):
        """Browse for clips directory."""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Clips Directory",
            self.config.get("clips_dir", "data/clips")
        )
        if dir_path:
            self.clips_dir_edit.setText(dir_path)
            
            # Check for metadata
            import json
            meta_path = os.path.join(dir_path, "clips_metadata.json")
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, 'r') as f:
                        meta = json.load(f)
                    
                    clip_len = meta.get("clip_length")
                    if clip_len:
                        self.clip_length_spin.setValue(int(clip_len))
                        self.log_text.appendPlainText(f"Automatically set 'Frames per clip' to {clip_len} from metadata.")
                except Exception as e:
                    logger.error("Error reading clips metadata: %s", e)
    
    def _show_clip_length_info(self):
        """Show information about frames per clip."""
        QMessageBox.information(
            self,
            "Frames per Clip - Information",
            "Number of frames the model will use for training.\n\n"
            "-Can be equal to or less than the actual clip length\n"
            "-If less, the middle N frames are selected (temporal center-crop)\n"
            "  e.g. 16-frame clips with this set to 8 → frames 4-11 are used\n"
            "-Useful for testing whether shorter temporal context is sufficient\n\n"
            "Important:\n"
            "-Use the same value in the Inference tab when running predictions\n"
            "-Shorter clips = faster training and lower memory usage\n"
            "-If clips have fewer frames than this value, they are padded"
        )
    
    def _browse_output(self):
        """Browse for output model path."""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Model",
            self.config.get("models_dir", "models/behavior_heads"),
            "PyTorch Files (*.pt);;All Files (*)"
        )
        if file_path:
            self.output_path_edit.setText(file_path)

    def _browse_pretrained(self):
        """Browse for pretrained model path."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Pretrained Model",
            self.config.get("models_dir", "models/behavior_heads"),
            "PyTorch Files (*.pt);;All Files (*)"
        )
        if file_path:
            self.pretrained_path_edit.setText(file_path)
    
    def _on_finetune_changed(self, state: int):
        """Enable/disable pretrained path inputs."""
        enabled = self.finetune_check.isChecked()
        self.pretrained_path_edit.setEnabled(enabled)
        self.pretrained_browse_btn.setEnabled(enabled)
    
    def _open_visualization(self):
        """Open the training visualization dialog."""
        if self.visualization_dialog is None:
            self.visualization_dialog = TrainingVisualizationDialog(self)
        
        self.visualization_dialog.show()
        self.visualization_dialog.raise_()
        self.visualization_dialog.activateWindow()
    
    def refresh_annotation_info(self):
        """Refresh dataset info display."""
        try:
            from collections import Counter

            annotation_file = self.annotation_file_edit.text().strip()
            if not annotation_file:
                annotation_file = self.config.get(
                    "training_annotation_file",
                    self.config.get("annotation_file", "data/annotations/annotations.json"),
                )
            
            self.annotation_manager = AnnotationManager(annotation_file)
            
            labeled_clips = self.annotation_manager.get_labeled_clips()
            has_bboxes = any(
                clip.get("spatial_bbox") or clip.get("spatial_bbox_frames")
                for clip in labeled_clips
            )
            self.loc_group.setEnabled(has_bboxes)
            if has_bboxes:
                self.loc_group.setTitle("Localization")
                self.loc_group.setToolTip("")
            else:
                self.use_localization_check.setChecked(False)
                self.use_manual_loc_switch_check.setChecked(False)
                self.use_crop_jitter_check.setChecked(False)
                self.loc_group.setTitle("Localization (requires bbox labels)")
                self.loc_group.setToolTip(
                    "Optional: localizes individual animals when multiple are in camera view.\n"
                    "To activate, draw and save bounding boxes on at least some clips in the Labeling tab."
                )
            classes = sorted(self.annotation_manager.get_classes())
            real_classes = [c for c in classes if not c.startswith("near_negative")]
            hard_negative_classes = [c for c in classes if c.startswith("near_negative")]
            counts = self.annotation_manager.get_clip_count_by_label()
            primary_counts = Counter(
                clip.get("label", "")
                for clip in labeled_clips
                if clip.get("label")
            )
            
            ml_stats = self.annotation_manager.get_multilabel_stats()
            exclusive = ml_stats["exclusive"]
            shared = ml_stats["shared"]
            combos = ml_stats["combos"]

            info_text = f"Labeled clips: {len(labeled_clips)}\n"
            info_text += f"Behavior classes: {len(real_classes)}\n"
            if real_classes:
                info_text += f"Behavior names: {', '.join(real_classes)}\n"
            if hard_negative_classes:
                hn_clip_count = sum(primary_counts.get(label, 0) for label in hard_negative_classes)
                info_text += f"Hard-negative helper labels: {len(hard_negative_classes)} ({hn_clip_count} clips)\n"

            info_text += "\nPrimary-label training counts:\n"
            for label in real_classes:
                primary_count = int(primary_counts.get(label, 0))
                membership_count = int(counts.get(label, 0))
                exc = int(exclusive.get(label, 0))
                sh = int(shared.get(label, 0))
                if sh > 0:
                    info_text += (
                        f"  {label}: {primary_count} primary"
                        f"  ({membership_count} label memberships: {exc} exclusive, {sh} multi-class)\n"
                    )
                else:
                    info_text += f"  {label}: {primary_count} primary\n"

            if hard_negative_classes:
                info_text += "\nHard-negative suppression clips:\n"
                for label in hard_negative_classes:
                    info_text += f"  {label}: {int(primary_counts.get(label, 0))}\n"

            real_combos = {
                combo: cnt
                for combo, cnt in combos.items()
                if combo and all(lbl in real_classes for lbl in combo)
            }
            if real_combos:
                total_mc = sum(real_combos.values())
                info_text += f"\nMulti-class clips: {total_mc}\n"
                for combo, cnt in sorted(real_combos.items(), key=lambda x: -x[1]):
                    info_text += f"  {' + '.join(combo)}: {cnt}\n"
            
            self.info_label.setText(info_text)
            
            # Update class selection list (exclude near_negative_* — they're auto-included in OvR)
            self.class_selection_list.clear()
            for class_name in real_classes:
                item = QListWidgetItem(class_name)
                item.setCheckState(Qt.CheckState.Checked)
                self.class_selection_list.addItem(item)
            
            # Update per-class limit table
            self.per_class_limit_table.setRowCount(len(real_classes))
            for row, class_name in enumerate(real_classes):
                class_item = QTableWidgetItem(class_name)
                class_item.setFlags(class_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.per_class_limit_table.setItem(row, 0, class_item)
                
                count = int(primary_counts.get(class_name, 0))
                
                # Max Train
                train_item = QTableWidgetItem(str(count))
                train_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.per_class_limit_table.setItem(row, 1, train_item)
                
                # Max Val (default to count i.e. unlimited/all)
                val_item = QTableWidgetItem(str(count))
                val_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.per_class_limit_table.setItem(row, 2, val_item)
        except Exception as e:
            self.info_label.setText(f"Error loading info: {e}")
    
    def _on_select_classes_changed(self, state: int):
        """Enable/disable class selection list."""
        self.class_selection_list.setEnabled(self.select_classes_check.isChecked())
    
    def _on_limit_per_class_changed(self, state: int):
        """Enable/disable per-class limit table."""
        self.per_class_limit_table.setEnabled(self.limit_per_class_check.isChecked())

    def _on_auto_tune_changed(self, state: int):
        """Enable/disable auto-tune controls."""
        is_on = self.auto_tune_check.isChecked()
        self.auto_tune_runs_spin.setEnabled(is_on)
        self.auto_tune_epochs_spin.setEnabled(is_on)

    def _get_training_profiles_path(self):
        """Return the profile storage path for the current experiment."""
        config_path = self.config.get("config_path")
        if config_path:
            exp_dir = os.path.dirname(config_path)
            return os.path.join(exp_dir, "training_profiles.json")
        from singlebehaviorlab._paths import get_training_profiles_path
        return str(get_training_profiles_path())
    
    def _open_profile_manager(self):
        """Open the training profiles manager dialog."""
        profiles_path = self._get_training_profiles_path()
            
        if not self.profile_dialog:
            self.profile_dialog = TrainingProfileDialog(self, profiles_file=profiles_path)
        else:
            self.profile_dialog.reload_profiles(profiles_path)
            
        self.profile_dialog.show()
        self.profile_dialog.raise_()
        self.profile_dialog.activateWindow()
    
    def get_training_config(self):
        """Extract current training configuration from UI components."""
        head_kwargs = {
            "num_heads": self.map_num_heads_spin.value(),
        }
        
        selected_classes = []
        if self.select_classes_check.isChecked():
            for i in range(self.class_selection_list.count()):
                item = self.class_selection_list.item(i)
                if item.checkState() == Qt.CheckState.Checked:
                    selected_classes.append(item.text())
        
        per_class_limits = {}
        per_class_val_limits = {}
        if self.limit_per_class_check.isChecked():
            for row in range(self.per_class_limit_table.rowCount()):
                class_item = self.per_class_limit_table.item(row, 0)
                train_item = self.per_class_limit_table.item(row, 1)
                val_item = self.per_class_limit_table.item(row, 2)
                
                if class_item:
                    class_name = class_item.text()
                    # Train limit (allow float e.g. 1.5 → applied as 2)
                    if train_item:
                        try:
                            val = float(train_item.text())
                            if val > 0: per_class_limits[class_name] = val
                        except ValueError: pass
                    
                    # Val limit (allow float)
                    if val_item:
                        try:
                            val = float(val_item.text())
                            if val > 0: per_class_val_limits[class_name] = val
                        except ValueError: pass
        
        # Use selected classes for metadata if class limiting is enabled, otherwise use all classes
        if hasattr(self, 'annotation_manager'):
            all_classes = self.annotation_manager.get_classes()
        else:
            all_classes = []
        classes_for_metadata = selected_classes if self.select_classes_check.isChecked() and selected_classes else all_classes
        background_class_names = [
            c for c in classes_for_metadata
            if c.lower() in ("other", "background", "bg", "none")
        ]
        if self.use_ovr_check.isChecked() and self.ovr_background_negative_check.isChecked():
            classes_for_metadata = [c for c in classes_for_metadata if c not in background_class_names]
        
        # Auto-detect helper classes (Other/Background) to exclude from F1 metrics
        _f1_exclude = [c for c in classes_for_metadata
                       if c.lower() in ("other", "background", "bg", "none")]

        # Get pretrained path if enabled
        pretrained_path = None
        if self.finetune_check.isChecked():
            pretrained_path = self.pretrained_path_edit.text().strip()
        hard_pairs = self._parse_hard_pairs_text(self.hard_pair_edit.text())
        resolution_cfg = int(self.config.get("resolution", self._resolution if hasattr(self, "_resolution") else 288))
        if resolution_cfg % 18 != 0:
            resolution_cfg = max(18, (resolution_cfg // 18) * 18)

        return {
            "batch_size": self.batch_size_spin.value(),
            "epochs": self.epochs_spin.value(),
            "lr": self.class_lr_spin.value(),
            "localization_lr": self.loc_lr_spin.value(),
            "classification_lr": self.class_lr_spin.value(),
            "use_scheduler": bool(self.config.get("default_use_scheduler", True)),
            "use_ema": bool(self.config.get("default_use_ema", True)),
            "weight_decay": self.weight_decay_spin.value(),
            "head_kwargs": head_kwargs,
            "dropout": self.map_dropout_spin.value(),
            "clip_length": self.clip_length_spin.value(),
            "target_fps": int(self.config.get("default_target_fps", 16)),
            "resolution": resolution_cfg,
            "use_all_for_training": self.use_all_for_training_check.isChecked(),
            "val_split": self.val_split_spin.value(),
            "auto_tune_before_final": self.auto_tune_check.isChecked(),
            "auto_tune_runs": self.auto_tune_runs_spin.value(),
            "auto_tune_epochs": self.auto_tune_epochs_spin.value(),
            "use_class_weights": bool(getattr(self, "use_class_weights_check", None) and self.use_class_weights_check.isChecked()),
            "use_focal_loss": False,
            "focal_gamma": 2.0,
            "use_supcon_loss": self.use_supcon_check.isChecked(),
            "supcon_weight": self.supcon_weight_spin.value(),
            "supcon_temperature": self.supcon_temp_spin.value(),
            "use_frame_loss": True,
            "use_temporal_decoder": self.use_temporal_decoder_check.isChecked(),
            "frame_head_temporal_layers": self.frame_head_layers_spin.value(),
            "temporal_pool_frames": self.temporal_pool_spin.value(),
            "proj_dim": self.proj_dim_spin.value(),
            "use_frame_bout_balance": self.use_bout_balance_check.isChecked(),
            "frame_bout_balance_power": self.bout_balance_power_spin.value(),
            "boundary_loss_weight": self.boundary_loss_weight_spin.value() if self.use_temporal_decoder_check.isChecked() else 0.0,
            "boundary_tolerance": self.boundary_tolerance_spin.value(),
            "smoothness_loss_weight": self.smoothness_loss_weight_spin.value(),
            "use_localization": self.use_localization_check.isChecked(),
            "use_manual_localization_switch": self.use_manual_loc_switch_check.isChecked(),
            "manual_localization_switch_epoch": self.manual_loc_switch_epoch_spin.value(),
            "localization_hidden_dim": 256,
            "classification_crop_padding": self.crop_padding_spin.value(),
            "crop_jitter": self.use_crop_jitter_check.isChecked() and self.use_localization_check.isChecked(),
            "crop_jitter_strength": self.crop_jitter_strength_spin.value(),
            "use_ovr": self.use_ovr_check.isChecked(),
            "ovr_background_as_negative": (
                self.ovr_background_negative_check.isChecked()
                and self.use_ovr_check.isChecked()
            ),
            "ovr_background_class_names": background_class_names,
            "ovr_label_smoothing": self.ovr_label_smoothing_spin.value(),
            "use_asl": self.use_asl_check.isChecked() and self.use_ovr_check.isChecked(),
            "asl_gamma_neg": self.asl_gamma_neg_spin.value(),
            "asl_gamma_pos": self.asl_gamma_pos_spin.value(),
            "asl_clip": self.asl_clip_spin.value(),
            "use_hard_pair_mining": (
                self.use_hard_pair_mining_check.isChecked()
                and self.use_ovr_check.isChecked()
                and bool(hard_pairs)
            ),
            "hard_pairs": hard_pairs,
            "hard_pair_loss_weight": self.hard_pair_loss_weight_spin.value(),
            "hard_pair_margin": self.hard_pair_margin_spin.value(),
            "hard_pair_confusion_boost": self.hard_pair_confusion_boost_spin.value(),
            "use_confusion_sampler": (
                self.use_confusion_sampler_check.isChecked()
                and self.use_ovr_check.isChecked()
            ),
            "confusion_sampler_temperature": self.confusion_temperature_spin.value(),
            "confusion_sampler_warmup_pct": self.confusion_warmup_spin.value() / 100.0,
            "use_weighted_sampler": self.use_weighted_sampler_check.isChecked(),
            "use_augmentation": self.use_augmentation_check.isChecked(),
            "virtual_expansion": self.virtual_expansion_spin.value(),
            "stitch_augmentation_prob": self.stitch_prob_spin.value() if self.use_stitch_check.isChecked() else 0.0,
            "emb_aug_versions": self.emb_aug_versions_spin.value() if hasattr(self, "emb_aug_versions_spin") else 1,
            "multi_scale": self.use_multi_scale_check.isChecked() if hasattr(self, "use_multi_scale_check") else False,
            "augmentation_options": dict(self.augmentation_options),
            "limit_classes": self.select_classes_check.isChecked(),
            "selected_classes": selected_classes,
            "limit_per_class": self.limit_per_class_check.isChecked(),
            "per_class_limits": per_class_limits,
            "per_class_val_limits": per_class_val_limits,
            "use_embedding_diversity": self.use_embedding_diversity_check.isChecked(),
            "backbone_model": self.config.get("backbone_model", "videoprism_public_v1_base"),
            "class_names": classes_for_metadata,
            "pretrained_path": pretrained_path,
            "f1_exclude_classes": _f1_exclude,
            "ovr_pos_weight_f1_excluded": getattr(self, "_ovr_pos_weight_f1_excluded", 1.5),
        }

    def apply_training_config(self, config):
        """Apply a configuration dictionary to the UI components."""
        try:
            self.use_localization_check.setChecked(False)
            self.use_manual_loc_switch_check.setChecked(False)
            self.manual_loc_switch_epoch_spin.setValue(20)
            self.crop_padding_spin.setValue(0.35)
            self.use_crop_jitter_check.setChecked(False)
            self.crop_jitter_strength_spin.setValue(0.15)

            if "batch_size" in config: self.batch_size_spin.setValue(config["batch_size"])
            if "epochs" in config: self.epochs_spin.setValue(config["epochs"])
            if "lr" in config:
                self.class_lr_spin.setValue(config["lr"])
            if "classification_lr" in config:
                self.class_lr_spin.setValue(config["classification_lr"])
            if "localization_lr" in config:
                self.loc_lr_spin.setValue(config["localization_lr"])
            elif "classification_lr" in config:
                self.loc_lr_spin.setValue(config["classification_lr"])
            elif "lr" in config:
                self.loc_lr_spin.setValue(config["lr"])
            if "weight_decay" in config: self.weight_decay_spin.setValue(config["weight_decay"])
            if "dropout" in config: self.map_dropout_spin.setValue(config["dropout"])
            if "clip_length" in config: self.clip_length_spin.setValue(config["clip_length"])
            if "use_all_for_training" in config: self.use_all_for_training_check.setChecked(config["use_all_for_training"])
            if "val_split" in config: self.val_split_spin.setValue(config["val_split"])
            if "auto_tune_before_final" in config: self.auto_tune_check.setChecked(bool(config["auto_tune_before_final"]))
            if "auto_tune_runs" in config: self.auto_tune_runs_spin.setValue(int(config["auto_tune_runs"]))
            if "auto_tune_epochs" in config: self.auto_tune_epochs_spin.setValue(int(config["auto_tune_epochs"]))
            
            # Loss settings
            if "use_class_weights" in config: self.use_class_weights_check.setChecked(config["use_class_weights"])
            
            if "use_supcon_loss" in config:
                self.use_supcon_check.setChecked(bool(config["use_supcon_loss"]))
            if "supcon_weight" in config:
                self.supcon_weight_spin.setValue(float(config["supcon_weight"]))
            if "supcon_temperature" in config:
                self.supcon_temp_spin.setValue(float(config["supcon_temperature"]))
            if "use_temporal_decoder" in config:
                self.use_temporal_decoder_check.setChecked(bool(config["use_temporal_decoder"]))

            if "frame_head_temporal_layers" in config:
                self.frame_head_layers_spin.setValue(int(config["frame_head_temporal_layers"]))
            if "temporal_pool_frames" in config:
                self.temporal_pool_spin.setValue(int(config["temporal_pool_frames"]))
            if "proj_dim" in config:
                self.proj_dim_spin.setValue(int(config["proj_dim"]))
            if "multi_scale" in config and hasattr(self, "use_multi_scale_check"):
                self.use_multi_scale_check.setChecked(bool(config["multi_scale"]))
            if "boundary_loss_weight" in config:
                self.boundary_loss_weight_spin.setValue(config["boundary_loss_weight"])
            if "boundary_tolerance" in config:
                self.boundary_tolerance_spin.setValue(int(config["boundary_tolerance"]))
            if "smoothness_loss_weight" in config:
                self.smoothness_loss_weight_spin.setValue(config["smoothness_loss_weight"])
            if "use_frame_bout_balance" in config:
                self.use_bout_balance_check.setChecked(bool(config["use_frame_bout_balance"]))
            if "frame_bout_balance_power" in config and config["frame_bout_balance_power"] is not None:
                self.bout_balance_power_spin.setValue(float(config["frame_bout_balance_power"]))
            if "use_localization" in config:
                self.use_localization_check.setChecked(config["use_localization"])
            
            if "use_manual_localization_switch" in config:
                self.use_manual_loc_switch_check.setChecked(config["use_manual_localization_switch"])
            if "manual_localization_switch_epoch" in config:
                self.manual_loc_switch_epoch_spin.setValue(config["manual_localization_switch_epoch"])
            if "classification_crop_padding" in config:
                self.crop_padding_spin.setValue(float(config["classification_crop_padding"]))
            if "crop_jitter" in config:
                self.use_crop_jitter_check.setChecked(bool(config["crop_jitter"]))
            if "crop_jitter_strength" in config:
                self.crop_jitter_strength_spin.setValue(float(config["crop_jitter_strength"]))

            self.use_ovr_check.setChecked(True)
            if "ovr_background_as_negative" in config:
                self.ovr_background_negative_check.setChecked(bool(config["ovr_background_as_negative"]))
            if "ovr_pos_weight_f1_excluded" in config:
                self._ovr_pos_weight_f1_excluded = float(config["ovr_pos_weight_f1_excluded"])
            if "ovr_label_smoothing" in config: self.ovr_label_smoothing_spin.setValue(config["ovr_label_smoothing"])
            if "use_asl" in config: self.use_asl_check.setChecked(config["use_asl"])
            if "asl_gamma_neg" in config: self.asl_gamma_neg_spin.setValue(float(config["asl_gamma_neg"]))
            if "asl_gamma_pos" in config: self.asl_gamma_pos_spin.setValue(float(config["asl_gamma_pos"]))
            if "asl_clip" in config: self.asl_clip_spin.setValue(float(config["asl_clip"]))
            if "hard_pairs" in config:
                self.hard_pair_edit.setText(self._format_hard_pairs_text(config["hard_pairs"]))
            if "use_hard_pair_mining" in config:
                self.use_hard_pair_mining_check.setChecked(bool(config["use_hard_pair_mining"]))
            if "hard_pair_loss_weight" in config and config["hard_pair_loss_weight"] is not None:
                self.hard_pair_loss_weight_spin.setValue(float(config["hard_pair_loss_weight"]))
            if "hard_pair_margin" in config and config["hard_pair_margin"] is not None:
                self.hard_pair_margin_spin.setValue(float(config["hard_pair_margin"]))
            if "hard_pair_confusion_boost" in config and config["hard_pair_confusion_boost"] is not None:
                self.hard_pair_confusion_boost_spin.setValue(float(config["hard_pair_confusion_boost"]))
            if "use_confusion_sampler" in config:
                self.use_confusion_sampler_check.setChecked(bool(config["use_confusion_sampler"]))
            if "confusion_sampler_temperature" in config:
                self.confusion_temperature_spin.setValue(float(config["confusion_sampler_temperature"]))
            if "confusion_sampler_warmup_pct" in config:
                self.confusion_warmup_spin.setValue(int(float(config["confusion_sampler_warmup_pct"]) * 100))
            if "use_weighted_sampler" in config: self.use_weighted_sampler_check.setChecked(config["use_weighted_sampler"])
            if "use_augmentation" in config: self.use_augmentation_check.setChecked(config["use_augmentation"])
            if "virtual_expansion" in config: self.virtual_expansion_spin.setValue(int(config["virtual_expansion"]))
            if "stitch_augmentation_prob" in config:
                prob = float(config["stitch_augmentation_prob"])
                self.use_stitch_check.setChecked(prob > 0.0)
                self.stitch_prob_spin.setValue(prob)
            if "emb_aug_versions" in config and hasattr(self, "emb_aug_versions_spin"):
                self.emb_aug_versions_spin.setValue(int(config["emb_aug_versions"]))
            if "augmentation_options" in config:
                self.augmentation_options = self._normalize_augmentation_options(config["augmentation_options"])
            
            if "head_kwargs" in config:
                hk = config["head_kwargs"]
                if "num_heads" in hk: self.map_num_heads_spin.setValue(hk["num_heads"])
            
            if "limit_classes" in config: 
                self.select_classes_check.setChecked(config["limit_classes"])
                # Restore selected classes if possible
                if config["limit_classes"] and "selected_classes" in config:
                    selected_set = set(config["selected_classes"])
                    # Ensure list is populated (might need refresh if empty, but usually populated)
                    if self.class_selection_list.count() == 0:
                        self.refresh_annotation_info()
                    
                    for i in range(self.class_selection_list.count()):
                        item = self.class_selection_list.item(i)
                        if item.text() in selected_set:
                            item.setCheckState(Qt.CheckState.Checked)
                        else:
                            item.setCheckState(Qt.CheckState.Unchecked)

            if "limit_per_class" in config:
                self.limit_per_class_check.setChecked(config["limit_per_class"])
                # Restore per-class limits
                if config["limit_per_class"]:
                    limits = config.get("per_class_limits", {})
                    val_limits = config.get("per_class_val_limits", {})
                    
                    if self.per_class_limit_table.rowCount() == 0:
                        self.refresh_annotation_info()
                        
                    for row in range(self.per_class_limit_table.rowCount()):
                        class_item = self.per_class_limit_table.item(row, 0)
                        if class_item:
                            name = class_item.text()
                            if name in limits:
                                self.per_class_limit_table.setItem(row, 1, QTableWidgetItem(str(limits[name])))
                            if name in val_limits:
                                self.per_class_limit_table.setItem(row, 2, QTableWidgetItem(str(val_limits[name])))
            
            if "use_embedding_diversity" in config: self.use_embedding_diversity_check.setChecked(config["use_embedding_diversity"])
            if "pretrained_path" in config:
                if config["pretrained_path"]:
                    self.finetune_check.setChecked(True)
                    self.pretrained_path_edit.setText(config["pretrained_path"])
                else:
                    self.finetune_check.setChecked(False)
            self._sync_stitch_controls()
                    
        except Exception as e:
            logger.error("Error applying config: %s", e)
            raise

    def _run_next_batch_item(self):
        """Run the next profile in the batch queue."""
        if not self.training_queue:
            self.is_batch_training = False
            
            # Show batch summary
            if self.batch_results:
                best_result = max(
                    self.batch_results,
                    key=lambda x: (
                        x.get("best_val_f1", 0.0),
                        x.get("best_val_acc", 0.0)
                    )
                )
                summary = f"Batch training completed!\n\n"
                summary += f"Best Profile: {best_result['profile_name']}\n"
                summary += f"Best Val Macro F1: {best_result.get('best_val_f1', 0.0):.2f}%\n"
                summary += f"Best Val Acc: {best_result.get('best_val_acc', 0.0):.2f}%\n\n"
                summary += f"Results saved to:\n{self.batch_results_path}"
                QMessageBox.information(self, "Batch Training Complete", summary)
            else:
                QMessageBox.information(self, "Batch Training", "Batch training completed!")
            
            self.train_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.progress_bar.setVisible(False)
            return

        profile_name, config = self.training_queue.pop(0)
        self.current_profile_name = profile_name
        self.log_text.appendPlainText(f"\n=== Starting Batch Item: {profile_name} ===")
        
        # Apply config to UI (so it's visible what's running)
        try:
            self.apply_training_config(config)
        except Exception as e:
            self.log_text.appendPlainText(f"Error applying profile {profile_name}: {e}")
            self._run_next_batch_item() # Skip bad profile
            return
        
        # Determine output path with profile suffix
        base_output = self.output_path_edit.text().strip()
        dir_name = os.path.dirname(base_output)
        file_name = os.path.basename(base_output)
        name_root, ext = os.path.splitext(file_name)
        new_output = os.path.join(dir_name, f"{name_root}_{profile_name}{ext}")
        
        self._start_training_internal(override_output_path=new_output, profile_name=profile_name)

    def _start_training_internal(self, override_output_path=None, profile_name=None):
        """Internal method to start a single training run."""
        annotation_file = self.annotation_file_edit.text().strip()
        if not os.path.exists(annotation_file):
            if not self.is_batch_training: QMessageBox.warning(self, "Error", "Annotation file not found.")
            else: self.log_text.appendPlainText("Error: Annotation file not found.")
            return
        
        clips_dir = self.clips_dir_edit.text().strip()
        if not os.path.exists(clips_dir):
            if not self.is_batch_training: QMessageBox.warning(self, "Error", "Clips directory not found.")
            else: self.log_text.appendPlainText("Error: Clips directory not found.")
            return
        
        output_path = override_output_path or self.output_path_edit.text().strip()
        if not output_path:
            if not self.is_batch_training: QMessageBox.warning(self, "Error", "Please specify output path.")
            else: self.log_text.appendPlainText("Error: No output path.")
            return
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Get configuration from UI
        train_config = self.get_training_config()
        if profile_name:
            train_config["profile_name"] = profile_name

        train_config["training_profiles_path"] = self._get_training_profiles_path()

        raw_resolution = int(self.config.get("resolution", 288))
        adjusted_resolution = train_config.get("resolution", raw_resolution)
        if raw_resolution % 18 != 0:
            msg = (
                "Resolution must be a multiple of 18 (patch size).\n\n"
                f"Entered: {raw_resolution}\n"
                f"Adjusted to: {adjusted_resolution}"
            )
            if not self.is_batch_training:
                QMessageBox.information(self, "Resolution adjusted", msg)
            else:
                self.log_text.appendPlainText(f"Resolution adjusted: {msg}")
            self._resolution = adjusted_resolution
        
        # Check pretrained path validity if needed
        if self.finetune_check.isChecked():
            if not train_config["pretrained_path"] or not os.path.exists(train_config["pretrained_path"]):
                if not self.is_batch_training: QMessageBox.warning(self, "Error", "Please select a valid pretrained model file.")
                else: self.log_text.appendPlainText("Error: Invalid pretrained path.")
                return

        if train_config.get("auto_tune_before_final", False):
            if train_config.get("use_all_for_training", False) or float(train_config.get("val_split", 0.0)) <= 0.0:
                msg = "Auto-tune requires a validation split. Disable 'Use all data for training' and set validation split > 0."
                if not self.is_batch_training: QMessageBox.warning(self, "Error", msg)
                else: self.log_text.appendPlainText(f"Error: {msg}")
                return
        
        head_kwargs = train_config["head_kwargs"]

        # Persist actual training parameters to experiment config.yaml
        try:
            config_path = self.config.get("config_path")
            if config_path:
                # Update last-used defaults for convenience
                self.config["default_batch_size"] = train_config["batch_size"]
                self.config["default_epochs"] = train_config["epochs"]
                self.config["default_learning_rate"] = train_config["classification_lr"]
                self.config["default_localization_lr"] = train_config["localization_lr"]
                self.config["default_classification_lr"] = train_config["classification_lr"]
                self.config["default_use_scheduler"] = train_config["use_scheduler"]
                self.config["default_use_ema"] = train_config["use_ema"]
                self.config["default_weight_decay"] = train_config["weight_decay"]
                self.config["default_clip_length"] = train_config["clip_length"]
                self.config["default_use_focal_loss"] = train_config["use_focal_loss"]
                self.config["default_focal_gamma"] = train_config["focal_gamma"]
                self.config["default_use_supcon_loss"] = train_config["use_supcon_loss"]
                self.config["default_supcon_weight"] = train_config["supcon_weight"]
                self.config["default_supcon_temperature"] = train_config["supcon_temperature"]
                self.config["backbone_model"] = train_config["backbone_model"]
                self.config["resolution"] = train_config["resolution"]
                # Save full last training block
                self.config["last_training"] = {
                    "parameters": {
                        "batch_size": train_config["batch_size"],
                        "epochs": train_config["epochs"],
                        "lr": train_config["classification_lr"],
                        "localization_lr": train_config["localization_lr"],
                        "classification_lr": train_config["classification_lr"],
                        "use_scheduler": train_config["use_scheduler"],
                        "use_ema": train_config["use_ema"],
                        "weight_decay": train_config["weight_decay"],
                        "clip_length": train_config["clip_length"],
                        "val_split": train_config["val_split"],
                        "auto_tune_before_final": train_config["auto_tune_before_final"],
                        "auto_tune_runs": train_config["auto_tune_runs"],
                        "auto_tune_epochs": train_config["auto_tune_epochs"],
                        "use_class_weights": train_config["use_class_weights"],
                        "use_focal_loss": train_config["use_focal_loss"],
                        "focal_gamma": train_config["focal_gamma"],
                        "use_weighted_sampler": train_config["use_weighted_sampler"],
                        "use_augmentation": train_config["use_augmentation"],
                        "augmentation_options": train_config.get("augmentation_options", {}),
                        "limit_classes": train_config["limit_classes"],
                        "limit_per_class": train_config["limit_per_class"],
                    },
                    "selected_classes": train_config["selected_classes"],
                    "per_class_limits": train_config["per_class_limits"],
                    "head": {
                        "dropout": train_config["dropout"],
                        "map_head_kwargs": train_config["head_kwargs"],
                    },
                    "pretrained_path": train_config["pretrained_path"],
                    "output_path": output_path
                }
                os.makedirs(os.path.dirname(config_path), exist_ok=True)
                with open(config_path, "w", encoding="utf-8") as f:
                    yaml.safe_dump(dict(self.config), f, sort_keys=False)
                self.log_text.appendPlainText(f"Saved training parameters to: {config_path}")
        except Exception as e:
            self.log_text.appendPlainText(f"Warning: Could not save training parameters to config.yaml: {e}")
        
        self.log_text.appendPlainText(f"Starting training run (Output: {os.path.basename(output_path)})...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.train_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.visualize_btn.setEnabled(True)
        
        # Open visualization automatically on start
        self._open_visualization()
        
        self.worker = TrainingWorker(
            self.config,
            train_config,
            annotation_file,
            clips_dir,
            output_path
        )
        self.worker.log_message.connect(self._on_log)
        self.worker.progress.connect(self._on_progress)
        self.worker.training_complete.connect(self._on_training_complete)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        # Reset and connect visualization update; restrict F1 graph to selected classes when limit_classes is on
        if self.visualization_dialog:
            f1_exclude = set(train_config.get("f1_exclude_classes", []))
            f1_classes = None
            if train_config.get("limit_classes", False):
                sel = train_config.get("selected_classes", [])
                if sel:
                    f1_classes = set(sel) - f1_exclude
            elif f1_exclude:
                f1_classes = set(train_config.get("class_names", [])) - f1_exclude
            confusion_warmup_ep = 0
            if train_config.get("use_confusion_sampler", False):
                pct = float(train_config.get("confusion_sampler_warmup_pct", 0.2))
                confusion_warmup_ep = int(train_config.get("epochs", 60) * pct)
            self.visualization_dialog.reset(
                f1_classes_to_show=f1_classes if f1_classes else None,
                confusion_warmup_epoch=confusion_warmup_ep,
            )
            self.worker.epoch_complete.connect(self.visualization_dialog.update_plots)
            
        self.worker.start()

    def _start_training(self):
        """Start training (single or batch)."""
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "Training", "Training is already running.")
            return
            
        if self.batch_train_check.isChecked():
            if not self.profile_dialog:
                self.profile_dialog = TrainingProfileDialog(self, profiles_file=self._get_training_profiles_path())
            else:
                self.profile_dialog.reload_profiles(self._get_training_profiles_path())
            
            profiles = self.profile_dialog.get_selected_profiles_for_batch()
            if not profiles:
                QMessageBox.warning(self, "Batch Training", "No profiles selected in Advanced > Profiles.\nPlease open Advanced settings and check profiles to run.")
                return
            
            self.training_queue = profiles
            self.is_batch_training = True
            self.batch_results = []
            
            # Initialize batch results CSV path
            from datetime import datetime
            output_dir = os.path.dirname(self.output_path_edit.text().strip())
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.batch_results_path = os.path.join(output_dir, f"batch_training_results_{timestamp}.csv")
            self.log_text.appendPlainText(f"Batch results will be saved to: {self.batch_results_path}")
            
            self.train_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self._run_next_batch_item()
        else:
            self._start_training_internal()
    
    def _stop_training(self):
        """Stop training."""
        if self.worker and self.worker.isRunning():
            self.is_batch_training = False # Stop batch execution
            self.training_queue = []       # Clear queue
            self.worker.stop()
            self.log_text.appendPlainText("Stopping training...")
            self.worker.wait()
            self._on_finished()
    
    def _on_log(self, message: str):
        """Handle log message."""
        self.log_text.appendPlainText(message)
    
    def _on_progress(self, epoch: int, total: int):
        """Update progress bar."""
        if total > 0:
            progress = int(100 * epoch / total)
            self.progress_bar.setValue(progress)
    
    def _on_training_complete(self, best_val_acc: float, best_val_f1: float, final_train_acc: float, per_class_f1: dict):
        """Handle training metrics from completed run."""
        summary_msg = (
            f"Training summary → Best Val Macro F1: {best_val_f1:.2f}%, "
            f"Best Val Acc: {best_val_acc:.2f}%, Final Train Acc: {final_train_acc:.2f}%"
        )
        self.log_text.appendPlainText(summary_msg)
        
        if per_class_f1:
            self.log_text.appendPlainText("Per-class F1 scores (Final Epoch):")
            for cls_name, f1 in sorted(per_class_f1.items()):
                self.log_text.appendPlainText(f"  - {cls_name}: {f1:.2f}%")
        
        if self.is_batch_training and self.current_profile_name:
            # Get current config for this profile
            current_config = self.get_training_config()
            
            result = {
                "profile_name": self.current_profile_name,
                "best_val_acc": round(best_val_acc, 2),
                "best_val_f1": round(best_val_f1, 2),
                "final_train_acc": round(final_train_acc, 2),
                "epochs": current_config.get("epochs", 0),
                "batch_size": current_config.get("batch_size", 0),
                "lr": current_config.get("lr", 0),
                "use_focal_loss": current_config.get("use_focal_loss", False),
            }
            
            # Add per-class F1 columns
            if per_class_f1:
                for cls_name, f1 in per_class_f1.items():
                    # Sanitize column name
                    safe_name = f"F1_{cls_name}".replace(" ", "_")
                    result[safe_name] = round(f1, 2)
            
            self.batch_results.append(result)
            
            # Save/update CSV after each profile
            self._save_batch_results_csv()
    
    def _save_batch_results_csv(self):
        """Save batch results to CSV."""
        if not self.batch_results or not self.batch_results_path:
            return
        try:
            import pandas as pd
            df = pd.DataFrame(self.batch_results)
            # Sort by best_val_f1 (primary) then accuracy
            sort_cols = [col for col in ["best_val_f1", "best_val_acc"] if col in df.columns]
            if sort_cols:
                df = df.sort_values(sort_cols, ascending=False)
            df.to_csv(self.batch_results_path, index=False)
            self.log_text.appendPlainText(f"  → Updated batch results: {self.batch_results_path}")
        except Exception as e:
            self.log_text.appendPlainText(f"  Failed to save batch results: {e}")
    
    def _on_finished(self):
        """Handle training completion."""
        if self.is_batch_training:
            self.refresh_annotation_info()
            self._run_next_batch_item()
            return
            
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.visualize_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        if self.profile_dialog:
            self.profile_dialog.reload_profiles(self._get_training_profiles_path())
        QMessageBox.information(self, "Training", "Training completed!")
        self.refresh_annotation_info()
    
    def _on_error(self, error_msg: str):
        """Handle training error."""
        self.log_text.appendPlainText(f"ERROR: {error_msg}")
        
        if self.is_batch_training:
             QMessageBox.critical(self, "Training Error", f"Batch training failed on current item:\n{error_msg}\n\nBatch execution stopped.")
             # Stop batch on error
             self.is_batch_training = False
             self.train_btn.setEnabled(True)
             self.stop_btn.setEnabled(False)
             self.progress_bar.setVisible(False)
        else:
             self._on_finished()
             QMessageBox.critical(self, "Training Error", f"Training failed:\n{error_msg}")

    def update_config(self, config: dict):
        """Apply a new configuration (experiment management)."""
        self.config = config
        self.annotation_manager = AnnotationManager(
            self.config.get("annotation_file", "data/annotations/annotations.json")
        )
        self._config_initialized = False
        self._load_current_config(force=True)
        self.refresh_annotation_info()
