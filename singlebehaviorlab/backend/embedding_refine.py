"""Embedding-based timeline refinement.

Uses per-frame embeddings from the inference model to:
1. Cluster frames by behavioral similarity (k-means)
2. Correct misclassified frames using cluster consensus
3. Detect true behavior boundaries via embedding distance spikes
4. Smooth and rebuild segments
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


def _cluster_correction(
    frame_labels: np.ndarray,
    frame_embeddings: np.ndarray,
    frame_confidences: np.ndarray,
    n_classes: int,
    confidence_threshold: float = 0.7,
) -> np.ndarray:
    from sklearn.cluster import MiniBatchKMeans

    n_frames = len(frame_labels)
    n_clusters = min(n_classes * 3, max(n_classes, n_frames // 50))
    n_clusters = max(2, min(n_clusters, n_frames - 1))

    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=min(1024, n_frames))
    cluster_ids = kmeans.fit_predict(frame_embeddings)

    cluster_dominant_class: dict[int, int] = {}
    for ci in range(n_clusters):
        mask = cluster_ids == ci
        if not np.any(mask):
            continue
        cluster_dominant_class[ci] = _majority_label(
            frame_labels[mask], frame_confidences[mask]
        )

    corrected = frame_labels.copy()
    for fi in range(n_frames):
        if corrected[fi] < 0:
            continue
        ci = cluster_ids[fi]
        dominant = cluster_dominant_class.get(ci, -1)
        if dominant < 0:
            continue
        if frame_confidences[fi] < confidence_threshold and corrected[fi] != dominant:
            corrected[fi] = dominant

    return corrected


def refine_with_embeddings(
    frame_labels: np.ndarray,
    frame_embeddings: np.ndarray,
    frame_confidences: Optional[np.ndarray] = None,
    n_classes: int = 0,
    boundary_sensitivity: float = 1.5,
    min_segment_frames: int = 3,
    confidence_threshold: float = 0.7,
) -> np.ndarray:
    """Refine per-frame predictions using embedding clustering and boundary detection.

    Args:
        frame_labels: (N,) int array of per-frame class indices (-1 = unlabeled).
        frame_embeddings: (N, D) float array of per-frame embeddings.
        frame_confidences: optional (N,) confidence scores.
        n_classes: number of behavior classes (used to size the clustering).
        boundary_sensitivity: lower values detect more boundaries.
        min_segment_frames: segments shorter than this are merged with neighbors.
        confidence_threshold: frames below this confidence defer to cluster consensus.

    Returns:
        (N,) int array of refined per-frame labels.
    """
    n_frames = len(frame_labels)
    if n_frames < 4 or frame_embeddings.shape[0] != n_frames:
        return frame_labels.copy()

    if frame_confidences is None:
        frame_confidences = np.ones(n_frames, dtype=np.float32)

    if n_classes <= 0:
        n_classes = max(1, int(frame_labels.max()) + 1)

    # Step 1: cluster-based correction of misclassified frames
    corrected = _cluster_correction(
        frame_labels, frame_embeddings, frame_confidences,
        n_classes, confidence_threshold,
    )

    # Step 2: boundary detection from embedding distances
    distances = _cosine_distance_adjacent(frame_embeddings)
    boundaries = _detect_boundaries(distances, boundary_sensitivity)
    boundaries.append(n_frames)

    # Step 3: within each boundary segment, assign majority class
    refined = corrected.copy()
    segments: list[tuple[int, int]] = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        if end <= start:
            continue
        majority = _majority_label(corrected[start:end], frame_confidences[start:end])
        refined[start:end] = majority
        segments.append((start, end))

    # Step 4: merge short segments with the most similar neighbor
    changed = True
    while changed:
        changed = False
        new_segments = []
        i = 0
        while i < len(segments):
            start, end = segments[i]
            if (end - start) < min_segment_frames and len(segments) > 1:
                mean_emb = frame_embeddings[start:end].mean(axis=0)
                norm = max(np.linalg.norm(mean_emb), 1e-8)
                mean_emb /= norm
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
                        corrected[merged_start:merged_end],
                        frame_confidences[merged_start:merged_end],
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
