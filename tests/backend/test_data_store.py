"""Tests for AnnotationManager (singlebehaviorlab.backend.data_store)."""

import json
import os
import pytest

from singlebehaviorlab.backend.data_store import AnnotationManager


class TestAnnotationManagerCreate:
    def test_create_empty_store(self, tmp_annotation_file):
        mgr = AnnotationManager(tmp_annotation_file)
        assert mgr.get_classes() == []
        assert mgr.get_all_clips() == []

    def test_create_writes_file_on_save(self, tmp_annotation_file):
        mgr = AnnotationManager(tmp_annotation_file)
        mgr.save()
        assert os.path.exists(tmp_annotation_file)
        with open(tmp_annotation_file) as f:
            data = json.load(f)
        assert data == {"classes": [], "clips": []}


class TestAnnotationManagerClasses:
    def test_add_remove_class(self, tmp_annotation_file):
        mgr = AnnotationManager(tmp_annotation_file)
        mgr.add_class("grooming")
        assert "grooming" in mgr.get_classes()
        mgr.remove_class("grooming")
        assert "grooming" not in mgr.get_classes()

    def test_add_class_idempotent(self, tmp_annotation_file):
        mgr = AnnotationManager(tmp_annotation_file)
        mgr.add_class("grooming")
        mgr.add_class("grooming")
        assert mgr.get_classes().count("grooming") == 1

    def test_rename_class_propagates(self, sample_annotations):
        mgr = AnnotationManager(sample_annotations)
        assert mgr.rename_class("grooming", "digging")
        assert "digging" in mgr.get_classes()
        assert "grooming" not in mgr.get_classes()
        for clip in mgr.get_all_clips():
            if clip["label"]:
                assert clip["label"] != "grooming"
            for lbl in clip.get("labels", []):
                assert lbl != "grooming"


class TestAnnotationManagerClips:
    def test_add_clip_and_label(self, tmp_annotation_file):
        mgr = AnnotationManager(tmp_annotation_file)
        mgr.add_class("grooming")
        mgr.add_clip("clip_000000.mp4", "grooming")
        assert mgr.get_clip_label("clip_000000.mp4") == "grooming"
        assert len(mgr.get_labeled_clips()) == 1

    def test_set_get_frame_labels(self, sample_annotations):
        mgr = AnnotationManager(sample_annotations)
        fl = ["grooming", None, "rearing", "grooming"]
        mgr.set_frame_labels("clip_000000.mp4", fl)
        result = mgr.get_frame_labels("clip_000000.mp4")
        assert result == fl

    def test_set_get_spatial_bbox(self, sample_annotations):
        mgr = AnnotationManager(sample_annotations)
        bbox = [0.1, 0.2, 0.8, 0.9]
        mgr.set_spatial_bbox("clip_000000.mp4", bbox)
        assert mgr.get_spatial_bbox("clip_000000.mp4") == pytest.approx(bbox)

    def test_set_get_spatial_bbox_frames(self, sample_annotations):
        mgr = AnnotationManager(sample_annotations)
        bboxes = [[0.1, 0.2, 0.5, 0.6], None, [0.3, 0.4, 0.7, 0.8]]
        mgr.set_spatial_bbox_frames("clip_000000.mp4", bboxes)
        result = mgr.get_spatial_bbox_frames("clip_000000.mp4")
        assert result[0] == pytest.approx([0.1, 0.2, 0.5, 0.6])
        assert result[1] is None
        assert result[2] == pytest.approx([0.3, 0.4, 0.7, 0.8])

    def test_remove_clip(self, sample_annotations):
        mgr = AnnotationManager(sample_annotations)
        before = len(mgr.get_all_clips())
        mgr.remove_clip("clip_000000.mp4")
        assert len(mgr.get_all_clips()) == before - 1
        assert mgr.get_clip_label("clip_000000.mp4") is None

    def test_clear_all_clips(self, sample_annotations):
        mgr = AnnotationManager(sample_annotations)
        mgr.clear_all_clips()
        assert mgr.get_all_clips() == []


class TestAnnotationManagerNormalization:
    def test_normalize_clip_id_backslash(self, tmp_annotation_file):
        mgr = AnnotationManager(tmp_annotation_file)
        mgr.add_clip("path\\to\\clip.mp4", "test")
        assert mgr.get_clip_label("path/to/clip.mp4") == "test"

    def test_normalize_clip_id_leading_dot_slash(self, tmp_annotation_file):
        mgr = AnnotationManager(tmp_annotation_file)
        mgr.add_clip("./clip_000000.mp4", "test")
        assert mgr.get_clip_label("clip_000000.mp4") == "test"


class TestAnnotationManagerCounts:
    def test_get_clip_count_by_label(self, sample_annotations):
        mgr = AnnotationManager(sample_annotations)
        counts = mgr.get_clip_count_by_label()
        assert counts.get("grooming") == 2
        assert counts.get("rearing") == 1

    def test_multilabel_stats(self, tmp_annotation_file):
        mgr = AnnotationManager(tmp_annotation_file)
        mgr.add_clip("c1.mp4", ["grooming"])
        mgr.add_clip("c2.mp4", ["rearing"])
        mgr.add_clip("c3.mp4", ["grooming", "rearing"])
        stats = mgr.get_multilabel_stats()
        assert stats["exclusive"]["grooming"] == 1
        assert stats["exclusive"]["rearing"] == 1
        assert stats["shared"]["grooming"] == 1
        assert stats["shared"]["rearing"] == 1
        combo_key = ("grooming", "rearing")
        assert stats["combos"][combo_key] == 1


class TestAnnotationManagerRoundTrip:
    def test_save_reload_roundtrip(self, tmp_annotation_file):
        mgr = AnnotationManager(tmp_annotation_file)
        mgr.add_class("grooming")
        mgr.add_class("rearing")
        mgr.add_clip("clip_000000.mp4", "grooming")
        mgr.add_clip("clip_000001.mp4", "rearing")
        mgr.set_spatial_bbox("clip_000000.mp4", [0.1, 0.2, 0.8, 0.9])
        mgr.set_frame_labels("clip_000001.mp4", ["rearing", None, "grooming"])
        mgr.save()

        mgr2 = AnnotationManager(tmp_annotation_file)
        assert mgr2.get_classes() == mgr.get_classes()
        assert len(mgr2.get_all_clips()) == len(mgr.get_all_clips())
        assert mgr2.get_clip_label("clip_000000.mp4") == "grooming"
        assert mgr2.get_spatial_bbox("clip_000000.mp4") == pytest.approx([0.1, 0.2, 0.8, 0.9])
        assert mgr2.get_frame_labels("clip_000001.mp4") == ["rearing", None, "grooming"]
