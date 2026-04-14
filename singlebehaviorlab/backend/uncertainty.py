"""
Uncertainty computation for active learning.

Ranks inference clips by model uncertainty per class to surface candidate
examples for human review and training-set enrichment.
"""

import os
import json
import numpy as np


def _compute_clip_uncertainty(probs: list, is_ovr: bool = False) -> dict:
    """
    Compute uncertainty metrics for a single clip.

    Returns:
        margin          – top_score minus second_score (lower = more uncertain)
        entropy         – Shannon entropy over the probability vector
        top_class_idx   – index of the highest-scoring class
        top_score       – raw score of the top class
        second_score    – raw score of the runner-up
    """
    arr = np.array(probs, dtype=np.float64)
    if arr.size == 0:
        return {"margin": 1.0, "entropy": 0.0, "top_class_idx": 0,
                "top_score": 0.0, "second_score": 0.0}

    sorted_idx = np.argsort(arr)[::-1]
    top_score = float(arr[sorted_idx[0]])
    second_score = float(arr[sorted_idx[1]]) if len(sorted_idx) > 1 else 0.0
    margin = top_score - second_score

    # Normalise for entropy (handles both softmax and independent-sigmoid outputs)
    arr_norm = arr / (arr.sum() + 1e-8)
    arr_clip = np.clip(arr_norm, 1e-10, 1.0)
    entropy = float(-np.sum(arr_clip * np.log(arr_clip)))

    return {
        "margin": float(margin),
        "entropy": float(entropy),
        "top_class_idx": int(sorted_idx[0]),
        "top_score": float(top_score),
        "second_score": float(second_score),
    }


def _build_clip_entries(results: dict, classes: list, is_ovr: bool = False) -> list:
    """Build a flat list of clip candidates from inference results."""
    all_entries = []
    for video_path, res in results.items():
        clip_probs = res.get("clip_probabilities") or []
        clip_starts = res.get("clip_starts") or []
        frame_interval = int(res.get("frame_interval") or 1)
        for clip_idx, probs in enumerate(clip_probs):
            if not probs:
                continue
            u = _compute_clip_uncertainty(probs, is_ovr=is_ovr)
            top_idx = u["top_class_idx"]
            start_frame = int(clip_starts[clip_idx]) if clip_idx < len(clip_starts) else 0
            scores = {
                classes[i]: float(probs[i])
                for i in range(min(len(probs), len(classes)))
            }
            all_entries.append({
                "video": video_path,
                "clip_idx": clip_idx,
                "start_frame": start_frame,
                "frame_interval": frame_interval,
                "predicted_class": classes[top_idx] if top_idx < len(classes) else "Unknown",
                "predicted_class_idx": top_idx,
                "scores": scores,
                "margin": u["margin"],
                "entropy": u["entropy"],
                "top_score": u["top_score"],
            })
    return all_entries


def _build_center_merge_weights(length: int) -> np.ndarray:
    if length <= 1:
        return np.ones((max(1, length),), dtype=np.float32)
    w = np.hanning(length).astype(np.float32)
    if not np.any(w > 0):
        return np.ones((length,), dtype=np.float32)
    return np.clip(0.1 + 0.9 * w, 1e-3, None).astype(np.float32)


def _get_frame_scores_for_result(res: dict, classes: list, is_ovr: bool = False) -> np.ndarray | None:
    scores = res.get("aggregated_frame_probs")
    if isinstance(scores, np.ndarray) and scores.ndim == 2 and scores.shape[1] == len(classes):
        return scores.astype(np.float32, copy=False)
    if isinstance(scores, list) and scores:
        arr = np.asarray(scores, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[1] == len(classes):
            return arr

    clip_frame_probabilities = res.get("clip_frame_probabilities") or []
    clip_starts = res.get("clip_starts") or []
    total_frames = int(res.get("total_frames") or 0)
    frame_interval = max(1, int(res.get("frame_interval") or 1))
    if not clip_frame_probabilities or not clip_starts or total_frames <= 0:
        return None

    num_classes = len(classes)
    agg_probs = np.zeros((total_frames, num_classes), dtype=np.float32)
    agg_counts = np.zeros((total_frames, 1), dtype=np.float32)
    for i, probs in enumerate(clip_frame_probabilities):
        if probs is None or i >= len(clip_starts):
            continue
        probs_arr = np.clip(np.asarray(probs, dtype=np.float32), 0.0, None)
        if probs_arr.ndim != 2 or probs_arr.shape[1] != num_classes:
            continue
        merge_w = _build_center_merge_weights(int(probs_arr.shape[0]))
        start_f = int(clip_starts[i])
        for t in range(int(probs_arr.shape[0])):
            f_start = start_f + t * frame_interval
            f_end = min(f_start + frame_interval, total_frames)
            if f_start >= total_frames:
                break
            if f_end <= f_start:
                continue
            w = float(merge_w[t])
            agg_probs[f_start:f_end] += probs_arr[t][np.newaxis, :] * w
            agg_counts[f_start:f_end] += w
    covered = agg_counts.squeeze(-1) > 0
    if not np.any(covered):
        return None
    agg_probs = agg_probs / np.maximum(agg_counts, 1.0)
    if not is_ovr:
        row_sums = agg_probs[covered].sum(axis=1, keepdims=True)
        agg_probs[covered] = agg_probs[covered] / np.maximum(row_sums, 1e-8)
    last_covered = int(np.max(np.where(covered))) + 1
    return agg_probs[:last_covered]


def rank_clips_for_review(
    results: dict,
    classes: list,
    n_per_class: int = 20,
    is_ovr: bool = False,
    ovr_boundary: tuple = (0.25, 0.75),
    max_margin_softmax: float = 0.45,
) -> dict:
    """
    For each class find the N most uncertain clips from inference results.

    'Uncertain about class C' means one of:
    - OvR: C's sigmoid score is in the ambiguous range [ovr_boundary]
    - OvR/softmax: C was the top prediction but the margin against the
      runner-up is below max_margin_softmax
    - softmax: C has non-trivial probability but is not the top prediction
      (potential false-negative)

    Args:
        results:             Inference result dict  (video_path -> result_entry).
        classes:             Ordered list of class names.
        n_per_class:         How many clips to surface per class.
        is_ovr:              Whether the model uses OvR (independent sigmoids).
        ovr_boundary:        (low, high) sigmoid range considered ambiguous.
        max_margin_softmax:  Clips where top − second < this are considered uncertain.

    Returns:
        Dict  {class_name: [entry, ...]}  where each entry is a plain dict.
    """
    lo, hi = ovr_boundary

    all_entries = _build_clip_entries(results, classes, is_ovr=is_ovr)

    if not all_entries:
        return {class_name: [] for class_name in classes}

    for e in all_entries:
        e["review_kind"] = "uncertain"

    ranked = {}
    for class_idx, class_name in enumerate(classes):
        candidates = []

        for e in all_entries:
            class_score = e["scores"].get(class_name, 0.0)
            is_top = e["predicted_class"] == class_name

            if is_ovr:
                near_boundary = lo <= class_score <= hi
                top_and_uncertain = is_top and e["margin"] < max_margin_softmax
                if near_boundary or top_and_uncertain:
                    # Uncertainty ∝ closeness to 0.5 for OvR score
                    u_score = 1.0 - abs(class_score - 0.5) * 2.0
                    candidates.append({**e, "_u": u_score, "_cs": class_score})
            else:
                top_and_uncertain = is_top and e["margin"] < max_margin_softmax
                fn_candidate = not is_top and class_score > 0.20
                if top_and_uncertain or fn_candidate:
                    u_score = (1.0 - e["margin"]) if top_and_uncertain else e["entropy"]
                    candidates.append({**e, "_u": u_score, "_cs": class_score})

        # Sort by uncertainty descending, deduplicate (same clip, different pass)
        seen = set()
        unique = []
        for e in sorted(candidates, key=lambda x: x["_u"], reverse=True):
            key = (e["video"], e["clip_idx"])
            if key not in seen:
                seen.add(key)
                entry = {k: v for k, v in e.items() if not k.startswith("_")}
                entry["uncertainty_score"] = float(e["_u"])
                entry["class_score"] = float(e["_cs"])
                unique.append(entry)

        ranked[class_name] = unique[:n_per_class]

    return ranked


def rank_clips_per_video_for_review(
    results: dict,
    classes: list,
    n_per_class: int = 20,
    is_ovr: bool = False,
    ovr_boundary: tuple = (0.25, 0.75),
    max_margin_softmax: float = 0.45,
) -> dict:
    """Rank uncertain clips per class within each video."""
    ranked = {class_name: {} for class_name in classes}
    for video_path, res in results.items():
        video_ranked = rank_clips_for_review(
            {video_path: res},
            classes,
            n_per_class=n_per_class,
            is_ovr=is_ovr,
            ovr_boundary=ovr_boundary,
            max_margin_softmax=max_margin_softmax,
        )
        for class_name in classes:
            ranked[class_name][video_path] = list(video_ranked.get(class_name, []) or [])
    return ranked


def rank_transition_clips_for_review(
    results: dict,
    classes: list,
    clip_length: int,
    is_ovr: bool = False,
    n_per_class: int = 50,
) -> dict:
    """Mine transition-window candidates from precise frame timeline outputs."""
    ranked = {class_name: [] for class_name in classes}
    if clip_length <= 0 or not classes:
        return ranked

    for video_path, res in results.items():
        frame_scores = _get_frame_scores_for_result(res, classes, is_ovr=is_ovr)
        if frame_scores is None or frame_scores.shape[0] < 2:
            continue
        total_frames = int(res.get("total_frames") or frame_scores.shape[0])
        frame_interval = max(1, int(res.get("frame_interval") or 1))
        top_idx = np.argmax(frame_scores, axis=1)
        segments = list(res.get("aggregated_segments") or [])
        if len(segments) < 2:
            run_start = 0
            for fi in range(1, frame_scores.shape[0] + 1):
                at_end = fi >= frame_scores.shape[0]
                if at_end or int(top_idx[fi]) != int(top_idx[run_start]):
                    cls_idx = int(top_idx[run_start])
                    if 0 <= cls_idx < len(classes):
                        seg_end = fi - 1
                        segments.append({
                            "class": cls_idx,
                            "start": int(run_start),
                            "end": int(seg_end),
                            "confidence": float(np.mean(frame_scores[run_start:seg_end + 1, cls_idx])),
                        })
                    run_start = fi
        if len(segments) < 2:
            continue
        sorted_scores = np.sort(frame_scores, axis=1)
        if sorted_scores.shape[1] >= 2:
            margins = sorted_scores[:, -1] - sorted_scores[:, -2]
        else:
            margins = sorted_scores[:, -1]
        clip_span_frames = max(frame_interval, int(clip_length) * frame_interval)
        half_span = max(frame_interval, (int(clip_length) // 2) * frame_interval)
        max_start = max(0, total_frames - clip_span_frames)
        seen_keys = set()

        for left_seg, right_seg in zip(segments[:-1], segments[1:]):
            left_cls = int(left_seg.get("class", -1))
            right_cls = int(right_seg.get("class", -1))
            if left_cls == right_cls or left_cls < 0 or right_cls < 0:
                continue
            if left_cls >= len(classes) or right_cls >= len(classes):
                continue
            boundary_frame = int(right_seg.get("start", left_seg.get("end", 0) + 1))
            if boundary_frame <= 0 or boundary_frame >= frame_scores.shape[0]:
                continue
            desired_start = boundary_frame - half_span
            start_frame = max(0, min(max_start, desired_start))
            sampled_frames = [
                min(frame_scores.shape[0] - 1, start_frame + t * frame_interval)
                for t in range(int(clip_length))
            ]
            if not sampled_frames:
                continue
            proposed_frame_labels = [classes[int(top_idx[f])] for f in sampled_frames]
            center_lo = max(0, boundary_frame - frame_interval)
            center_hi = min(frame_scores.shape[0], boundary_frame + frame_interval + 1)
            center_margin = float(np.mean(margins[center_lo:center_hi])) if center_hi > center_lo else float(margins[boundary_frame])
            left_conf = float(left_seg.get("confidence", frame_scores[max(0, boundary_frame - 1), left_cls]))
            right_conf = float(right_seg.get("confidence", frame_scores[min(frame_scores.shape[0] - 1, boundary_frame), right_cls]))
            center_scores = frame_scores[sampled_frames[len(sampled_frames) // 2]]
            transition_score = max(0.0, min(1.0, 0.5 * (left_conf + right_conf) + (1.0 - max(0.0, center_margin))))
            key = (video_path, start_frame, left_cls, right_cls)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            entry = {
                "review_kind": "transition",
                "video": video_path,
                "clip_idx": int(boundary_frame),
                "start_frame": int(start_frame),
                "frame_interval": int(frame_interval),
                "transition_frame": int(boundary_frame),
                "left_class": classes[left_cls],
                "right_class": classes[right_cls],
                "predicted_class": f"{classes[left_cls]} -> {classes[right_cls]}",
                "scores": {
                    classes[i]: float(center_scores[i])
                    for i in range(min(len(classes), center_scores.shape[0]))
                },
                "top_score": float(np.max(center_scores)),
                "transition_score": float(transition_score),
                "proposed_frame_labels": proposed_frame_labels,
            }
            ranked[classes[left_cls]].append(entry)
            ranked[classes[right_cls]].append(entry)

    deduped = {}
    for class_name, entries in ranked.items():
        seen = set()
        unique = []
        for entry in sorted(entries, key=lambda e: float(e.get("transition_score", 0.0)), reverse=True):
            key = (entry.get("video", ""), int(entry.get("start_frame", -1)), entry.get("left_class", ""), entry.get("right_class", ""))
            if key in seen:
                continue
            seen.add(key)
            unique.append(entry)
        deduped[class_name] = unique[:n_per_class]
    return deduped


def rank_transition_clips_per_video_for_review(
    results: dict,
    classes: list,
    clip_length: int,
    is_ovr: bool = False,
    n_per_class: int = 50,
) -> dict:
    ranked = {class_name: {} for class_name in classes}
    for video_path, res in results.items():
        video_ranked = rank_transition_clips_for_review(
            {video_path: res},
            classes,
            clip_length=clip_length,
            is_ovr=is_ovr,
            n_per_class=n_per_class,
        )
        for class_name in classes:
            ranked[class_name][video_path] = list(video_ranked.get(class_name, []) or [])
    return ranked


def rank_confident_clips_for_review(
    results: dict,
    classes: list,
    n_per_class: int = 100,
    is_ovr: bool = False,
    clip_length: int = 8,
    min_gap_multiplier: float = 0.75,
) -> dict:
    """
    Rank top confident clips with lightweight diversity across videos/time.

    The selection is confidence-first, but uses round-robin across videos and
    avoids choosing clips that are too close in time within the same video.
    """
    all_entries = _build_clip_entries(results, classes, is_ovr=is_ovr)
    if not all_entries or n_per_class <= 0:
        return {class_name: [] for class_name in classes}

    for entry in all_entries:
        entry["review_kind"] = "confident"
        entry["confidence_score"] = float(entry.get("top_score", 0.0))

    min_gap_frames = max(
        1,
        int(round(max(1, int(clip_length)) * max(0.1, float(min_gap_multiplier))))
    )

    def is_far_enough(candidate: dict, selected: list) -> bool:
        cand_video = candidate.get("video", "")
        cand_start = int(candidate.get("start_frame", 0))
        cand_interval = max(1, int(candidate.get("frame_interval", 1)))
        gap_frames = int(round(min_gap_frames * cand_interval))
        for prev in selected:
            if prev.get("video", "") != cand_video:
                continue
            if abs(cand_start - int(prev.get("start_frame", 0))) < gap_frames:
                return False
        return True

    def select_diverse_entries(entries: list, limit: int) -> list:
        entries = sorted(
            entries,
            key=lambda e: (
                -float(e.get("confidence_score", 0.0)),
                os.path.basename(e.get("video", "")),
                int(e.get("start_frame", 0)),
            )
        )

        by_video = {}
        for entry in entries:
            by_video.setdefault(entry.get("video", ""), []).append(entry)

        ordered_videos = sorted(
            by_video.keys(),
            key=lambda video: (
                -float(by_video[video][0].get("confidence_score", 0.0)),
                os.path.basename(video),
            )
        )

        selected = []
        selected_keys = set()
        positions = {video: 0 for video in ordered_videos}

        made_progress = True
        while len(selected) < limit and made_progress:
            made_progress = False
            for video in ordered_videos:
                candidates = by_video[video]
                pos = positions[video]
                while pos < len(candidates):
                    entry = candidates[pos]
                    pos += 1
                    key = (entry.get("video", ""), int(entry.get("clip_idx", -1)))
                    if key in selected_keys:
                        continue
                    if not is_far_enough(entry, selected):
                        continue
                    selected.append(entry)
                    selected_keys.add(key)
                    positions[video] = pos
                    made_progress = True
                    break
                else:
                    positions[video] = pos

                if len(selected) >= limit:
                    break

        if len(selected) < limit:
            for entry in entries:
                key = (entry.get("video", ""), int(entry.get("clip_idx", -1)))
                if key in selected_keys:
                    continue
                selected.append(entry)
                selected_keys.add(key)
                if len(selected) >= limit:
                    break

        return selected[:limit]

    ranked = {}
    for class_name in classes:
        class_entries = []
        for entry in all_entries:
            class_score = float(entry.get("scores", {}).get(class_name, 0.0))
            class_entries.append({
                **entry,
                "class_score": class_score,
                "confidence_score": class_score,
                "predicted_class": class_name,
            })
        ranked[class_name] = select_diverse_entries(class_entries, n_per_class)

    return ranked


def rank_confident_clips_per_video_for_review(
    results: dict,
    classes: list,
    n_per_class: int = 100,
    is_ovr: bool = False,
    clip_length: int = 8,
    min_gap_multiplier: float = 0.75,
) -> dict:
    """Rank top confident clips per class within each video."""
    ranked = {}
    for class_name in classes:
        ranked[class_name] = {}

    for video_path, res in results.items():
        video_ranked = rank_confident_clips_for_review(
            {video_path: res},
            classes,
            n_per_class=n_per_class,
            is_ovr=is_ovr,
            clip_length=clip_length,
            min_gap_multiplier=min_gap_multiplier,
        )
        for class_name in classes:
            ranked[class_name][video_path] = list(video_ranked.get(class_name, []) or [])

    return ranked


def save_uncertainty_report(
    results: dict,
    classes: list,
    output_path: str,
    is_ovr: bool = False,
    n_per_class: int = 25,
    clip_length: int = 8,
    target_fps: int = 16,
) -> dict:
    """Compute uncertainty ranking and write it to a JSON file."""
    ranked = rank_clips_for_review(
        results, classes, n_per_class=n_per_class, is_ovr=is_ovr
    )
    ranked_per_video = rank_clips_per_video_for_review(
        results, classes, n_per_class=n_per_class, is_ovr=is_ovr
    )
    transition_per_class = rank_transition_clips_for_review(
        results,
        classes,
        clip_length=clip_length,
        is_ovr=is_ovr,
        n_per_class=max(50, n_per_class),
    )
    transition_per_class_per_video = rank_transition_clips_per_video_for_review(
        results,
        classes,
        clip_length=clip_length,
        is_ovr=is_ovr,
        n_per_class=max(50, n_per_class),
    )
    confident_per_class = rank_confident_clips_for_review(
        results,
        classes,
        n_per_class=max(100, n_per_class),
        is_ovr=is_ovr,
        clip_length=clip_length,
    )
    confident_per_class_per_video = rank_confident_clips_per_video_for_review(
        results,
        classes,
        n_per_class=max(100, n_per_class),
        is_ovr=is_ovr,
        clip_length=clip_length,
    )
    report = {
        "classes": classes,
        "is_ovr": is_ovr,
        "n_per_class": n_per_class,
        "clip_length": clip_length,
        "target_fps": target_fps,
        "per_class": ranked,
        "per_class_per_video": ranked_per_video,
        "transition_per_class": transition_per_class,
        "transition_per_class_per_video": transition_per_class_per_video,
        "confident_per_class": confident_per_class,
        "confident_per_class_per_video": confident_per_class_per_video,
    }
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report
