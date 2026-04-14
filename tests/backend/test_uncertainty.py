"""Tests for uncertainty ranking (singlebehaviorlab.backend.uncertainty)."""

import json
import os
import numpy as np
import pytest

from singlebehaviorlab.backend.uncertainty import (
    _compute_clip_uncertainty,
    rank_clips_for_review,
    rank_confident_clips_for_review,
    rank_transition_clips_for_review,
    save_uncertainty_report,
)


def _make_results(clip_probs_list, video="video.mp4"):
    """Helper: wrap a list of per-clip probability vectors into an inference results dict."""
    return {
        video: {
            "clip_probabilities": clip_probs_list,
            "clip_starts": list(range(0, len(clip_probs_list) * 8, 8)),
            "frame_interval": 1,
            "total_frames": len(clip_probs_list) * 8,
        }
    }


class TestComputeClipUncertainty:
    def test_certain_clip(self):
        u = _compute_clip_uncertainty([0.99, 0.01])
        assert u["margin"] > 0.9
        assert u["top_class_idx"] == 0

    def test_uncertain_clip(self):
        u = _compute_clip_uncertainty([0.51, 0.49])
        assert u["margin"] < 0.1

    def test_empty_probs(self):
        u = _compute_clip_uncertainty([])
        assert u["margin"] == 1.0


class TestRankClipsForReview:
    def test_most_uncertain_ranked_first(self):
        classes = ["A", "B"]
        probs = [
            [0.99, 0.01],  # very certain
            [0.52, 0.48],  # uncertain
            [0.55, 0.45],  # somewhat uncertain
        ]
        results = _make_results(probs)
        ranked = rank_clips_for_review(results, classes, n_per_class=3)
        a_clips = ranked["A"]
        if len(a_clips) >= 2:
            assert a_clips[0]["uncertainty_score"] >= a_clips[1]["uncertainty_score"]

    def test_per_class_keys(self):
        classes = ["A", "B", "C"]
        probs = [[0.6, 0.3, 0.1], [0.5, 0.3, 0.2]]
        ranked = rank_clips_for_review(_make_results(probs), classes, n_per_class=5)
        for cls in classes:
            assert cls in ranked

    def test_ovr_mode_boundary(self):
        classes = ["A", "B"]
        probs = [[0.5, 0.1], [0.9, 0.1]]
        ranked = rank_clips_for_review(
            _make_results(probs), classes, n_per_class=5, is_ovr=True
        )
        a_clips = ranked["A"]
        assert len(a_clips) >= 1
        assert a_clips[0]["clip_idx"] == 0  # score 0.5 is right at boundary

    def test_empty_results(self):
        ranked = rank_clips_for_review({}, ["A", "B"], n_per_class=5)
        assert ranked == {"A": [], "B": []}


class TestRankConfidentClips:
    def test_highest_confidence_bubbles_up(self):
        classes = ["A", "B"]
        probs = [[0.99, 0.01], [0.6, 0.4], [0.95, 0.05]]
        ranked = rank_confident_clips_for_review(
            _make_results(probs), classes, n_per_class=3
        )
        a_clips = ranked["A"]
        assert len(a_clips) >= 2
        assert a_clips[0]["confidence_score"] >= a_clips[1]["confidence_score"]


class TestRankTransitionClips:
    def test_transition_detected(self):
        classes = ["A", "B"]
        n_frames = 20
        frame_scores = np.zeros((n_frames, 2), dtype=np.float32)
        frame_scores[:10, 0] = 0.9
        frame_scores[:10, 1] = 0.1
        frame_scores[10:, 0] = 0.1
        frame_scores[10:, 1] = 0.9

        results = {
            "video.mp4": {
                "aggregated_frame_probs": frame_scores,
                "total_frames": n_frames,
                "frame_interval": 1,
                "clip_probabilities": [],
                "clip_starts": [],
            }
        }
        ranked = rank_transition_clips_for_review(results, classes, clip_length=8)
        total = sum(len(v) for v in ranked.values())
        assert total >= 1


class TestSaveUncertaintyReport:
    def test_writes_valid_json(self, tmp_path):
        output = str(tmp_path / "report.json")
        classes = ["A", "B"]
        probs = [[0.6, 0.4], [0.9, 0.1]]
        report = save_uncertainty_report(
            _make_results(probs), classes, output, n_per_class=5
        )
        assert os.path.exists(output)
        with open(output) as f:
            data = json.load(f)
        assert data["classes"] == classes
        assert "per_class" in data
