"""Temporal contrastive projection for behavior-focused embeddings.

Trains a lightweight MLP on pre-computed VideoPrism embeddings using
temporal proximity as the supervision signal: clips close in time within
the same video should map nearby; clips far apart should map far away.
The projected embeddings suppress static visual factors (lighting,
background, camera) and amplify behavioral dynamics.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["learn_behavior_features"]

_DEFAULT_DIM = 128
_DEFAULT_EPOCHS = 30
_DEFAULT_LR = 3e-4
_POSITIVE_WINDOW = 5
_TEMPERATURE = 0.07


class _ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        hidden = max(out_dim, in_dim // 2)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


def _build_pairs(
    metadata: pd.DataFrame,
    n_samples: int,
    positive_window: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    group_col = None
    for col in ("group", "video_id"):
        if col in metadata.columns:
            group_col = col
            break
    snippet_col = "snippet" if "snippet" in metadata.columns else None
    if not group_col or not snippet_col:
        indices = np.arange(len(metadata))
        rng.shuffle(indices)
        anchors = indices[:n_samples]
        positives = np.clip(anchors + rng.integers(-positive_window, positive_window + 1, size=n_samples), 0, len(metadata) - 1)
        negatives = rng.integers(0, len(metadata), size=n_samples)
        return anchors, positives, negatives

    groups = metadata[group_col].values
    unique_groups = np.unique(groups)
    group_indices: dict[Any, np.ndarray] = {}
    for g in unique_groups:
        group_indices[g] = np.where(groups == g)[0]

    anchors = []
    positives = []
    negatives = []
    per_group = max(1, n_samples // len(unique_groups))

    for g in unique_groups:
        idx = group_indices[g]
        if len(idx) < 2:
            continue
        a = rng.choice(idx, size=min(per_group, len(idx)), replace=len(idx) < per_group)
        for ai in a:
            pos_in_group = np.where(idx == ai)[0][0]
            lo = max(0, pos_in_group - positive_window)
            hi = min(len(idx), pos_in_group + positive_window + 1)
            candidates = idx[lo:hi]
            candidates = candidates[candidates != ai]
            if len(candidates) == 0:
                continue
            pi = rng.choice(candidates)

            other_groups = [og for og in unique_groups if og != g]
            if other_groups:
                ng = rng.choice(other_groups)
                ni = rng.choice(group_indices[ng])
            else:
                far_lo = max(0, pos_in_group - 3 * positive_window)
                far_hi = min(len(idx), pos_in_group + 3 * positive_window + 1)
                far_candidates = np.setdiff1d(idx, idx[far_lo:far_hi])
                if len(far_candidates) == 0:
                    far_candidates = idx
                ni = rng.choice(far_candidates)

            anchors.append(ai)
            positives.append(pi)
            negatives.append(ni)

    return np.array(anchors), np.array(positives), np.array(negatives)


def _info_nce_loss(anchor, positive, negative, temperature):
    pos_sim = (anchor * positive).sum(dim=-1) / temperature
    neg_sim = (anchor * negative).sum(dim=-1) / temperature
    logits = torch.stack([pos_sim, neg_sim], dim=-1)
    labels = torch.zeros(len(anchor), dtype=torch.long, device=anchor.device)
    return F.cross_entropy(logits, labels)


def learn_behavior_features(
    matrix_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    *,
    metadata_path: Optional[str | os.PathLike[str]] = None,
    out_dim: int = _DEFAULT_DIM,
    epochs: int = _DEFAULT_EPOCHS,
    lr: float = _DEFAULT_LR,
    positive_window: int = _POSITIVE_WINDOW,
    temperature: float = _TEMPERATURE,
    log_fn: Optional[Callable[[str], None]] = None,
) -> dict[str, str]:
    """Train a contrastive projection and write the projected embedding matrix.

    Returns dict with ``matrix`` and ``metadata`` output paths.
    """
    from singlebehaviorlab.backend.clustering import _load_matrix_metadata

    matrix_path = str(Path(matrix_path).expanduser().resolve())
    output_path_obj = Path(output_path).expanduser().resolve()
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    metadata_path_str = str(Path(metadata_path).expanduser().resolve()) if metadata_path else None

    def _log(msg: str) -> None:
        if log_fn:
            log_fn(msg)

    matrix_df, metadata_df = _load_matrix_metadata(matrix_path, metadata_path_str)
    X = matrix_df.T
    embeddings = X.values.astype(np.float32)
    n_samples, in_dim = embeddings.shape
    _log(f"Loaded {n_samples} embeddings ({in_dim}-dim)")

    if metadata_df is None:
        metadata_df = pd.DataFrame({"snippet": X.index, "group": "video_0"})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _ProjectionHead(in_dim, out_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    all_emb = torch.from_numpy(embeddings).to(device)

    rng = np.random.default_rng(42)
    pairs_per_epoch = max(1024, min(n_samples * 4, 65536))

    _log(f"Training projection head ({in_dim} → {out_dim}) for {epochs} epochs on {device}")
    for epoch in range(epochs):
        anchors, positives, negatives = _build_pairs(metadata_df, pairs_per_epoch, positive_window, rng)
        if len(anchors) == 0:
            _log("No valid pairs found — check metadata has group/video_id column")
            break
        a_emb = model(all_emb[anchors])
        p_emb = model(all_emb[positives])
        n_emb = model(all_emb[negatives])
        loss = _info_nce_loss(a_emb, p_emb, n_emb, temperature)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            _log(f"  epoch {epoch + 1}/{epochs}  loss={loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        projected = model(all_emb).cpu().numpy()
    _log(f"Projected embeddings: {projected.shape}")

    snippet_ids = np.array(X.index.tolist())
    feature_names = np.array([f"behavior_feat_{i}" for i in range(out_dim)])

    out_matrix = str(output_path_obj)
    if out_matrix.endswith("_matrix.npz"):
        out_metadata = out_matrix.replace("_matrix.npz", "_metadata.npz")
    elif out_matrix.endswith(".npz"):
        out_metadata = out_matrix[:-4] + "_metadata.npz"
    else:
        out_metadata = out_matrix + "_metadata.npz"

    np.savez_compressed(out_matrix, matrix=projected.T, feature_names=feature_names, snippet_ids=snippet_ids)
    _log(f"Wrote projected matrix: {out_matrix}")

    if metadata_df is not None:
        np.savez_compressed(out_metadata, metadata=metadata_df.values, columns=np.array(metadata_df.columns))
        _log(f"Wrote metadata: {out_metadata}")

    return {"matrix": out_matrix, "metadata": out_metadata}
