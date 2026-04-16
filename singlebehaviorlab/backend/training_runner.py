"""Headless training entry point used by the CLI.

Assembles the model, datasets, and backend config from an experiment
directory and drives a single run of ``train_model``. Intentionally avoids
the GUI-only features (auto-tune, multi-run sweeps, embedding-based
diversity selection); those remain available through the Training tab.
"""

from __future__ import annotations

import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Optional

import yaml

from singlebehaviorlab._paths import get_training_profiles_path
from singlebehaviorlab.backend.data_store import AnnotationManager
from singlebehaviorlab.backend.model import BehaviorClassifier, VideoPrismBackbone
from singlebehaviorlab.backend.train import BehaviorDataset, train_model

__all__ = ["run_training_session"]


def _load_profile(profile_name: str) -> dict[str, Any]:
    path = get_training_profiles_path()
    if not path.exists():
        raise FileNotFoundError(f"Training profiles file not found: {path}")
    with open(path, "r") as f:
        profiles = json.load(f)
    if profile_name not in profiles:
        available = ", ".join(sorted(profiles.keys()))
        raise KeyError(
            f"Unknown training profile '{profile_name}'. Available profiles: {available}"
        )
    return dict(profiles[profile_name])


def _load_experiment_config(experiment_dir: Path, override_config: Optional[Path]) -> dict[str, Any]:
    cfg_path = Path(override_config) if override_config else experiment_dir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Experiment config not found: {cfg_path}")
    with open(cfg_path, "r") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Experiment config {cfg_path} did not parse as a dict.")
    return data


def _split_train_val_clip_stratified(
    clips: list[dict[str, Any]],
    val_split: float,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if val_split <= 0.0 or val_split >= 1.0:
        return list(clips), []
    rng = random.Random(seed)
    buckets: dict[Any, list[dict[str, Any]]] = {}
    for c in clips:
        buckets.setdefault(c.get("label"), []).append(c)
    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    for label, bucket in buckets.items():
        shuffled = list(bucket)
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_val = int(round(n * val_split))
        if n <= 1:
            train.extend(shuffled)
            continue
        if n_val <= 0:
            n_val = 1
        if n_val >= n:
            n_val = n - 1
        val.extend(shuffled[:n_val])
        train.extend(shuffled[n_val:])
    return train, val


def _build_backend_train_config(
    cfg: dict[str, Any],
    output_path: str,
    classes: list[str],
) -> dict[str, Any]:
    return {
        "batch_size": int(cfg.get("batch_size", 16)),
        "epochs": int(cfg.get("epochs", 50)),
        "lr": float(cfg.get("classification_lr", cfg.get("lr", 1e-4))),
        "localization_lr": float(cfg.get("localization_lr", cfg.get("lr", 1e-4))),
        "classification_lr": float(cfg.get("classification_lr", cfg.get("lr", 1e-4))),
        "use_scheduler": bool(cfg.get("use_scheduler", True)),
        "use_ema": bool(cfg.get("use_ema", True)),
        "weight_decay": float(cfg.get("weight_decay", 1e-3)),
        "output_path": output_path,
        "save_best": True,
        "use_class_weights": bool(cfg.get("use_class_weights", False)),
        "use_focal_loss": bool(cfg.get("use_focal_loss", False)),
        "focal_gamma": float(cfg.get("focal_gamma", 2.0)),
        "use_frame_loss": True,
        "use_temporal_decoder": bool(cfg.get("use_temporal_decoder", True)),
        "frame_head_temporal_layers": int(cfg.get("frame_head_temporal_layers", 1)),
        "temporal_pool_frames": int(cfg.get("temporal_pool_frames", 1)),
        "num_stages": int(cfg.get("num_stages", 3)),
        "proj_dim": int(cfg.get("proj_dim", 256)),
        "multi_scale": bool(cfg.get("multi_scale", False)),
        "use_frame_bout_balance": bool(cfg.get("use_frame_bout_balance", True)),
        "frame_bout_balance_power": float(cfg.get("frame_bout_balance_power", 1.0)),
        "boundary_loss_weight": float(cfg.get("boundary_loss_weight", 0.3)),
        "boundary_tolerance": int(cfg.get("boundary_tolerance", 2)),
        "smoothness_loss_weight": float(cfg.get("smoothness_loss_weight", 0.05)),
        "use_localization": bool(cfg.get("use_localization", False)),
        "use_manual_localization_switch": bool(cfg.get("use_manual_localization_switch", False)),
        "manual_localization_switch_epoch": int(cfg.get("manual_localization_switch_epoch", 20)),
        "localization_hidden_dim": int(cfg.get("localization_hidden_dim", 256)),
        "classification_crop_padding": float(cfg.get("classification_crop_padding", 0.35)),
        "crop_jitter": bool(cfg.get("crop_jitter", False)),
        "crop_jitter_strength": float(cfg.get("crop_jitter_strength", 0.15)),
        "emb_aug_versions": int(cfg.get("emb_aug_versions", 1)),
        "clip_length": int(cfg.get("clip_length", 16)),
        "use_ovr": bool(cfg.get("use_ovr", True)),
        "ovr_label_smoothing": float(cfg.get("ovr_label_smoothing", 0.05)),
        "use_asl": bool(cfg.get("use_asl", False)),
        "asl_gamma_neg": float(cfg.get("asl_gamma_neg", 2.0)),
        "asl_gamma_pos": float(cfg.get("asl_gamma_pos", 0.0)),
        "asl_clip": float(cfg.get("asl_clip", 0.05)),
        "use_confusion_sampler": bool(cfg.get("use_confusion_sampler", True)),
        "confusion_sampler_temperature": float(cfg.get("confusion_sampler_temperature", 2.0)),
        "confusion_sampler_warmup_pct": float(cfg.get("confusion_sampler_warmup_pct", 0.2)),
        "use_weighted_sampler": bool(cfg.get("use_weighted_sampler", False)),
        "use_augmentation": bool(cfg.get("use_augmentation", False)),
        "augmentation_options": cfg.get("augmentation_options", {}) or {},
        "virtual_expansion": int(cfg.get("virtual_expansion", 5)),
        "stitch_augmentation_prob": float(cfg.get("stitch_augmentation_prob", 0.0)),
        "f1_exclude_classes": list(cfg.get("f1_exclude_classes", [])),
        "ovr_pos_weight_f1_excluded": float(cfg.get("ovr_pos_weight_f1_excluded", 1.5)),
        "val_split": float(cfg.get("val_split", 0.2)),
        "limit_classes": False,
        "selected_classes": list(cfg.get("selected_classes", [])),
        "limit_per_class": False,
        "per_class_limits": {},
        "per_class_val_limits": {},
        "backbone_model": cfg.get("backbone_model", "videoprism_public_v1_base"),
        "resolution": int(cfg.get("resolution", 288)),
        "use_all_for_training": bool(cfg.get("use_all_for_training", False)),
        "use_embedding_diversity": False,
        "class_names": classes,
        "pretrained_path": cfg.get("pretrained_path"),
        "head_kwargs": cfg.get("head_kwargs", {}),
        "dropout": float(cfg.get("dropout", 0.1)),
        "localization_dropout": 0.0,
    }


def run_training_session(
    experiment_dir: str | os.PathLike[str],
    *,
    config_override: Optional[str | os.PathLike[str]] = None,
    profile: Optional[str] = None,
    cli_overrides: Optional[dict[str, Any]] = None,
    output_name: Optional[str] = None,
    log_fn: Optional[Callable[[str], None]] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    stop_callback: Optional[Callable[[], bool]] = None,
) -> dict[str, Any]:
    """Train a single behavior classifier from an experiment directory.

    Returns the dict produced by ``train_model`` plus the output path under
    the ``output_path`` key.
    """
    experiment_dir = Path(experiment_dir).expanduser().resolve()
    if not experiment_dir.is_dir():
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")

    cfg = _load_experiment_config(experiment_dir, config_override)

    if profile:
        profile_cfg = _load_profile(profile)
        cfg.update(profile_cfg)

    if cli_overrides:
        cfg.update({k: v for k, v in cli_overrides.items() if v is not None})

    annotation_file = cfg.get("annotation_file") or str(
        experiment_dir / "data" / "annotations" / "annotations.json"
    )
    clips_dir = cfg.get("clips_dir") or str(experiment_dir / "data" / "clips")
    models_dir = cfg.get("models_dir") or str(experiment_dir / "models" / "behavior_heads")

    for label, path in (("annotations", annotation_file), ("clips directory", clips_dir)):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required {label} not found: {path}")

    os.makedirs(models_dir, exist_ok=True)
    output_filename = f"{output_name or 'model'}.pt"
    output_path = os.path.join(models_dir, output_filename)

    def _log(msg: str) -> None:
        if log_fn:
            log_fn(msg)

    _log(f"Experiment: {experiment_dir}")
    _log(f"Annotations: {annotation_file}")
    _log(f"Clips directory: {clips_dir}")
    _log(f"Output checkpoint: {output_path}")

    annotation_manager = AnnotationManager(annotation_file)
    labeled_clips = annotation_manager.get_labeled_clips()
    if not labeled_clips:
        raise RuntimeError("No labeled clips found in annotation file.")

    classes = annotation_manager.get_classes()
    selected = cfg.get("selected_classes")
    if selected:
        classes = [c for c in classes if c in selected]
        labeled_clips = [clip for clip in labeled_clips if clip.get("label") in classes]
    if not classes:
        raise RuntimeError("No classes available for training after filtering.")

    _log(f"Classes: {classes}")
    _log(f"Labeled clips: {len(labeled_clips)}")
    counts = Counter(clip["label"] for clip in labeled_clips)
    for label_name, count in sorted(counts.items()):
        _log(f"  {label_name}: {count}")

    use_all_for_training = bool(cfg.get("use_all_for_training", False))
    val_split = float(cfg.get("val_split", 0.2))

    if use_all_for_training:
        train_clips = labeled_clips
        val_clips: list[dict[str, Any]] = []
        _log("Using all data for training (no validation split).")
    else:
        train_clips, val_clips = _split_train_val_clip_stratified(
            labeled_clips, val_split, seed=42
        )
        _log(f"Train/val split: {len(train_clips)} train, {len(val_clips)} val")

    resolution = int(cfg.get("resolution", 288))
    clip_length = int(cfg.get("clip_length", 16))

    _log("Creating datasets...")
    train_dataset = BehaviorDataset(
        train_clips,
        annotation_manager,
        classes,
        clips_dir,
        target_size=(resolution, resolution),
        clip_length=clip_length,
    )
    _log(f"Train dataset: {len(train_dataset)} samples")

    val_dataset: Optional[BehaviorDataset] = None
    if val_clips:
        val_dataset = BehaviorDataset(
            val_clips,
            annotation_manager,
            classes,
            clips_dir,
            target_size=(resolution, resolution),
            clip_length=clip_length,
        )
        _log(f"Val dataset: {len(val_dataset)} samples")

    _log("Loading VideoPrism backbone...")
    backbone_model = cfg.get("backbone_model", "videoprism_public_v1_base")
    backbone = VideoPrismBackbone(
        model_name=backbone_model,
        resolution=resolution,
        log_fn=log_fn,
    )

    head_kwargs = dict(cfg.get("head_kwargs", {}) or {})
    head_kwargs.pop("per_class_query", None)
    dropout = float(cfg.get("dropout", 0.1))

    use_loc = bool(cfg.get("use_localization", False))
    num_stages = int(cfg.get("num_stages", 3))
    if use_loc and num_stages > 1:
        num_stages = 1
    multi_scale = bool(cfg.get("multi_scale", False)) and not use_loc

    _log("Building classifier...")
    model = BehaviorClassifier(
        backbone,
        num_classes=len(train_dataset.classes),
        class_names=train_dataset.classes,
        dropout=dropout,
        freeze_backbone=True,
        head_kwargs=head_kwargs,
        use_localization=use_loc,
        localization_hidden_dim=int(cfg.get("localization_hidden_dim", 256)),
        localization_dropout=0.0,
        use_frame_head=True,
        use_temporal_decoder=bool(cfg.get("use_temporal_decoder", True)),
        frame_head_temporal_layers=int(cfg.get("frame_head_temporal_layers", 1)),
        temporal_pool_frames=int(cfg.get("temporal_pool_frames", 1)),
        proj_dim=int(cfg.get("proj_dim", 256)),
        num_stages=num_stages,
        multi_scale=multi_scale,
    )

    backend_cfg = _build_backend_train_config(cfg, output_path, classes)

    _log("Starting training...")
    result = train_model(
        model,
        train_dataset,
        val_dataset,
        backend_cfg,
        log_fn=log_fn,
        progress_callback=progress_callback,
        stop_callback=stop_callback,
    )

    result = dict(result) if isinstance(result, dict) else {}
    result["output_path"] = output_path
    return result
