"""Clip-level embedding refinement via label propagation."""

from __future__ import annotations

import numpy as np
from typing import Optional

__all__ = ["refine_clip_predictions"]


def refine_clip_predictions(
    clip_labels: np.ndarray,
    clip_embeddings: np.ndarray,
    clip_confidences: np.ndarray,
    confidence_threshold: float = 0.7,
) -> np.ndarray:
    """Refine per-clip predictions using label propagation on clip embeddings.

    High-confidence clips seed a k-NN label-spreading graph.
    Low-confidence clips defer to their embedding neighbors.

    Args:
        clip_labels: [N] int array of predicted class indices per clip.
        clip_embeddings: [N, D] float array of clip-level embeddings.
        clip_confidences: [N] float array of prediction confidence per clip.
        confidence_threshold: clips above this are trusted as seed labels.

    Returns:
        [N] int array of refined clip labels.
    """
    from sklearn.semi_supervised import LabelSpreading

    N = len(clip_labels)
    if N < 4 or clip_embeddings.shape[0] != N:
        return clip_labels.copy()

    labels_for_propagation = clip_labels.copy()
    for i in range(N):
        if clip_confidences[i] < confidence_threshold:
            labels_for_propagation[i] = -1

    n_labeled = int(np.sum(labels_for_propagation >= 0))
    if n_labeled < 2 or n_labeled == N:
        return clip_labels.copy()

    n_neighbors = min(7, N - 1)
    lp = LabelSpreading(kernel="knn", n_neighbors=n_neighbors, max_iter=30, alpha=0.2)
    lp.fit(clip_embeddings, labels_for_propagation)
    propagated = lp.transduction_

    result = clip_labels.copy()
    for i in range(N):
        if clip_confidences[i] < confidence_threshold and propagated[i] >= 0:
            result[i] = int(propagated[i])

    return result
