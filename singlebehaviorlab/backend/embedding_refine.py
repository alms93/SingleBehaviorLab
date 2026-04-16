"""Embedding-based timeline refinement.

Uses the per-frame embeddings produced during inference to detect true
behavior boundaries (cosine-distance spikes) and smooth predictions
within each segment by majority vote over the embedding neighborhood.
"""

from __future__ import annotations

import numpy as np
from typing import Optional

__all__ = ["refine_with_embeddings"]


def _cosine_distance_adjacent(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normed = embeddings / norms
    sims = np.sum(normed[:-1] * normed[1:], axis=1)
    return 1.0 - sims


def _detect_boundaries(distances: np.ndarray, threshold_factor: float = 1.5) -> list[int]:
    if len(distances) == 0:
        return []
    median = np.median(distances)
    mad = np.median(np.abs(distances - median))
    threshold = median + threshold_factor * max(mad, 1e-6)
    boundaries = [0]
    for i, d in enumerate(distances):
        if d > threshold:
            boundaries.append(i + 1)
    return boundaries


def _majority_label(labels: np.ndarray, weights: Optional[np.ndarray] = None) -> int:
    valid = labels[labels >= 0]
    if len(valid) == 0:
        return -1
    if weights is not None:
        w = weights[labels >= 0]
        counts: dict[int, float] = {}
        for lbl, wt in zip(valid, w):
            counts[int(lbl)] = counts.get(int(lbl), 0.0) + float(wt)
        return max(counts, key=counts.get)
    vals, cnts = np.unique(valid, return_counts=True)
    return int(vals[np.argmax(cnts)])


def refine_with_embeddings(
    frame_labels: np.ndarray,
    frame_embeddings: np.ndarray,
    frame_confidences: Optional[np.ndarray] = None,
    boundary_sensitivity: float = 1.5,
    min_segment_frames: int = 3,
) -> np.ndarray:
    """Refine per-frame predictions using embedding similarity.

    Args:
        frame_labels: (N,) int array of per-frame class indices (-1 = unlabeled).
        frame_embeddings: (N, D) float array of per-frame embeddings.
        frame_confidences: optional (N,) confidence scores used as weights.
        boundary_sensitivity: lower values detect more boundaries (default 1.5).
        min_segment_frames: segments shorter than this are merged with neighbors.

    Returns:
        (N,) int array of refined per-frame labels.
    """
    n_frames = len(frame_labels)
    if n_frames == 0 or frame_embeddings.shape[0] != n_frames:
        return frame_labels.copy()

    distances = _cosine_distance_adjacent(frame_embeddings)
    boundaries = _detect_boundaries(distances, boundary_sensitivity)
    boundaries.append(n_frames)

    refined = frame_labels.copy()
    segments: list[tuple[int, int]] = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        if end <= start:
            continue
        seg_labels = frame_labels[start:end]
        seg_weights = frame_confidences[start:end] if frame_confidences is not None else None
        majority = _majority_label(seg_labels, seg_weights)
        refined[start:end] = majority
        segments.append((start, end))

    # Merge short segments with the more similar neighbor
    changed = True
    while changed:
        changed = False
        new_segments = []
        i = 0
        while i < len(segments):
            start, end = segments[i]
            if (end - start) < min_segment_frames and len(segments) > 1:
                mean_emb = frame_embeddings[start:end].mean(axis=0)
                mean_emb /= max(np.linalg.norm(mean_emb), 1e-8)
                best_sim = -1.0
                merge_with = -1
                for j in [i - 1, i + 1]:
                    if 0 <= j < len(segments):
                        ns, ne = segments[j]
                        neighbor_emb = frame_embeddings[ns:ne].mean(axis=0)
                        neighbor_emb /= max(np.linalg.norm(neighbor_emb), 1e-8)
                        sim = float(np.dot(mean_emb, neighbor_emb))
                        if sim > best_sim:
                            best_sim = sim
                            merge_with = j
                if merge_with >= 0:
                    ms, me = segments[merge_with]
                    merged_start = min(start, ms)
                    merged_end = max(end, me)
                    majority = _majority_label(
                        frame_labels[merged_start:merged_end],
                        frame_confidences[merged_start:merged_end] if frame_confidences is not None else None,
                    )
                    refined[merged_start:merged_end] = majority
                    if merge_with < i:
                        new_segments[-1] = (merged_start, merged_end)
                    else:
                        new_segments.append((merged_start, merged_end))
                        i += 1
                    changed = True
                    i += 1
                    continue
            new_segments.append((start, end))
            i += 1
        segments = new_segments

    return refined
