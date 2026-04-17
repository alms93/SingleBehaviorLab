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
    seed_labels: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Refine per-clip predictions using label propagation on clip embeddings.

    When seed_labels is provided, those are used as hard seeds and all other
    clips are unlabeled. Otherwise, high-confidence clips seed the graph.
    """
    from sklearn.semi_supervised import LabelSpreading

    N = len(clip_labels)
    if N < 4 or clip_embeddings.shape[0] != N:
        return clip_labels.copy()

    if seed_labels is not None and len(seed_labels) == N:
        labels_for_propagation = seed_labels.copy()
    else:
        labels_for_propagation = clip_labels.copy()
        labels_for_propagation[clip_confidences < confidence_threshold] = -1

    n_labeled = int(np.sum(labels_for_propagation >= 0))
    if n_labeled < 2 or n_labeled == N:
        return clip_labels.copy()

    n_neighbors = min(7, N - 1)
    lp = LabelSpreading(kernel="knn", n_neighbors=n_neighbors, max_iter=30, alpha=0.2)
    lp.fit(clip_embeddings, labels_for_propagation)
    propagated = lp.transduction_

    result = clip_labels.copy()
    unlabeled = labels_for_propagation < 0
    result[unlabeled] = propagated[unlabeled]

    return result
