"""Embedding-based timeline refinement.

Uses per-frame embeddings from the inference model to correct predictions
via semi-supervised label propagation on a nearest-neighbor graph, then
detects true behavior boundaries from embedding distance spikes.
"""

from __future__ import annotations

import numpy as np
from typing import Optional

__all__ = ["refine_with_embeddings"]


def _cosine_distance_adjacent(embeddings: np.ndarray) -> np.ndarray:
    norms = np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-8)
    normed = embeddings / norms
    return 1.0 - np.sum(normed[:-1] * normed[1:], axis=1)


def _detect_boundaries(distances: np.ndarray, threshold_factor: float) -> list[int]:
    if len(distances) == 0:
        return []
    median = float(np.median(distances))
    mad = float(np.median(np.abs(distances - median)))
    threshold = median + threshold_factor * max(mad, 1e-6)
    boundaries = [0]
    for i, d in enumerate(distances):
        if d > threshold:
            boundaries.append(i + 1)
    return boundaries


def _majority_label(labels: np.ndarray, weights: Optional[np.ndarray] = None) -> int:
    valid_mask = labels >= 0
    valid = labels[valid_mask]
    if len(valid) == 0:
        return -1
    if weights is not None:
        w = weights[valid_mask]
        counts: dict[int, float] = {}
        for lbl, wt in zip(valid, w):
            counts[int(lbl)] = counts.get(int(lbl), 0.0) + float(wt)
        return max(counts, key=counts.get)
    vals, cnts = np.unique(valid, return_counts=True)
    return int(vals[np.argmax(cnts)])


def _label_propagation_correction(
    frame_labels: np.ndarray,
    frame_embeddings: np.ndarray,
    frame_confidences: np.ndarray,
    confidence_threshold: float,
) -> np.ndarray:
    from sklearn.semi_supervised import LabelSpreading

    n_frames = len(frame_labels)
    labels_for_propagation = frame_labels.copy()

    for i in range(n_frames):
        if frame_confidences[i] < confidence_threshold:
            labels_for_propagation[i] = -1

    n_labeled = np.sum(labels_for_propagation >= 0)
    if n_labeled < 2 or n_labeled == n_frames:
        return frame_labels.copy()

    n_neighbors = min(7, n_frames - 1)
    lp = LabelSpreading(kernel="knn", n_neighbors=n_neighbors, max_iter=30, alpha=0.2)
    lp.fit(frame_embeddings, labels_for_propagation)
    propagated = lp.transduction_

    result = frame_labels.copy()
    for i in range(n_frames):
        if frame_confidences[i] < confidence_threshold and propagated[i] >= 0:
            result[i] = int(propagated[i])

    return result


def refine_with_embeddings(
    frame_labels: np.ndarray,
    frame_embeddings: np.ndarray,
    frame_confidences: Optional[np.ndarray] = None,
    n_classes: int = 0,
    boundary_sensitivity: float = 1.5,
    min_segment_frames: int = 3,
    confidence_threshold: float = 0.7,
) -> np.ndarray:
    """Refine per-frame predictions using label propagation and boundary detection.

    High-confidence predictions seed a nearest-neighbor graph. Labels
    propagate to uncertain frames through embedding similarity. Boundary
    detection then snaps segment edges to real embedding transitions.
    """
    n_frames = len(frame_labels)
    if n_frames < 4 or frame_embeddings.shape[0] != n_frames:
        return frame_labels.copy()

    if frame_confidences is None:
        frame_confidences = np.ones(n_frames, dtype=np.float32)

    corrected = _label_propagation_correction(
        frame_labels, frame_embeddings, frame_confidences, confidence_threshold,
    )

    distances = _cosine_distance_adjacent(frame_embeddings)
    boundaries = _detect_boundaries(distances, boundary_sensitivity)
    boundaries.append(n_frames)

    refined = corrected.copy()
    segments: list[tuple[int, int]] = []
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        if end <= start:
            continue
        majority = _majority_label(corrected[start:end], frame_confidences[start:end])
        refined[start:end] = majority
        segments.append((start, end))

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
                best_sim, merge_with = -1.0, -1
                for j in [i - 1, i + 1]:
                    if 0 <= j < len(segments):
                        ns, ne = segments[j]
                        ne_emb = frame_embeddings[ns:ne].mean(axis=0)
                        ne_emb /= max(np.linalg.norm(ne_emb), 1e-8)
                        sim = float(np.dot(mean_emb, ne_emb))
                        if sim > best_sim:
                            best_sim, merge_with = sim, j
                if merge_with >= 0:
                    ms, me = segments[merge_with]
                    ms2, me2 = min(start, ms), max(end, me)
                    majority = _majority_label(corrected[ms2:me2], frame_confidences[ms2:me2])
                    refined[ms2:me2] = majority
                    if merge_with < i:
                        new_segments[-1] = (ms2, me2)
                    else:
                        new_segments.append((ms2, me2))
                        i += 1
                    changed = True
                    i += 1
                    continue
            new_segments.append((start, end))
            i += 1
        segments = new_segments

    return refined
