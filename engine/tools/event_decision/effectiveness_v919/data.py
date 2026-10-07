"""Frozen source-group pilot and dense manifests. No inference or human answers."""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import random
import re

from ..contracts import WindowKey, file_sha256, iter_jsonl, read_json, semantic_sha256, write_json, write_jsonl
from ..b1b4_trial.protocol import immutable

CODES = ("B1", "B2", "B4", "B5", "B6", "G")
REVIEWED_UIDS = {
    "edb6091cadf84583ee6ac99cf67cd7f67d338a6b089cbeb95d90776cc38cea90",
    "01dfb1e8b7208657a6aa60f460fdb448d60afa2c8e33271ceadc63a627d4ef8a",
    "e5896297709fc4b12ed74a3a2fce6a66c8b1841c890620936b8c74c91f1b2067",
    "be18a4a445c66c95a245d94ccbb468f03dc7199ac4c320f4018605dcecca3bae",
}


def codes(video_id):
    return [x for x in CODES if x in re.split(r"[-_.]", video_id.split("_label_", 1)[-1])]


def pure_normal(video_id):
    return "_label_" in video_id and "A" in re.split(r"[-_.]", video_id.split("_label_", 1)[1]) and not codes(video_id)


def group(video_id):
    return video_id.split("__", 1)[0].casefold()


def frame_count(path):
    from ..local_screen import frame_probe
    return int(frame_probe(path))


def dense_spans(n, window=96, stride=48):
    if n < 8 or not 1 <= stride <= window or window < 8:
        raise ValueError("Need >=8 real frames and 1 <= stride <= window")
    if n <= window:
        return [(0, n)]
    starts = list(range(0, n - window + 1, stride))
    if starts[-1] != n - window:
        starts.append(n - window)
    return [(s, s + window) for s in starts]


def row_for(path, n, start, end, partition):
    vid = path.stem
    return {"window_uid": WindowKey(partition, vid, start, end).uid,
            "video_id": vid, "video_path": str(path.resolve()), "source_group": group(vid),
            "dataset_partition": partition, "start_frame": start, "end_frame_exclusive": end,
            "video_frames": n, "sampled_frame_indices": [start + i * (end - start - 1) // 7 for i in range(8)],
            "video_size": path.stat().st_size, "video_mtime_ns": path.stat().st_mtime_ns}


def save_jsonl_new(path, rows):
    if path.exists():
        if semantic_sha256(list(iter_jsonl(path))) != semantic_sha256(rows):
            raise ValueError("Frozen file changed: " + str(path))
    else:
        write_jsonl(path, rows)


TRAIN_PARTS = ("1-1004", "1005-2004", "2005-2804", "2805-3319", "3320-3954")


def inventory(root, official_train=False):
    parts = [root / p for p in TRAIN_PARTS]
    if official_train and any(p.is_dir() for p in parts):
        if not all(p.is_dir() for p in parts):
            raise ValueError("Incomplete five-part training media root")
        videos = sorted(p for part in parts for p in part.rglob("*.mp4"))
    else:
        videos = sorted(root.rglob("*.mp4"))
    by_id = defaultdict(list)
    for p in videos:
        by_id[p.stem].append(p)
    if any(len(v) != 1 for v in by_id.values()):
        raise ValueError("Duplicate video identities in media root; resolve before enrollment")
    if not videos:
        raise ValueError("No mp4 files in " + str(root))
    return {k: v[0] for k, v in by_id.items()}


def reviewed_source_groups(project):
    path = project / "runs/governed_v912_b1b4_minimal_effect_20260915/enrollment/windows.jsonl"
    if not path.exists():
        raise ValueError("Cannot resolve four design-reviewed sources: " + str(path))
    rows = [r for r in iter_jsonl(path) if r.get("window_uid") in REVIEWED_UIDS]
    if {r["window_uid"] for r in rows} != REVIEWED_UIDS:
        raise ValueError("Four reviewed source identities must be resolved, not guessed")
    return {group(r["video_id"]) for r in rows}, {"path": str(path), "sha256": file_sha256(path)}


def build_pilot(project, out, train_root, anchor_root, config):
    excluded, exclusion_source = reviewed_source_groups(project)
    videos = inventory(train_root, official_train=True)
    anchors = {}
    for p in sorted(anchor_root.glob("*/selected_frames.json")):
        obj = read_json(p)
        vid = Path(str(obj.get("video", p.parent.name))).stem
        if vid not in videos or not obj.get("covered_by_72B"):
            continue
        spans = []
        for s in obj.get("selected_segments", []):
            a, b = int(s["segment_start"]), int(s["segment_end_inclusive"]) + 1
            if b - a >= 8 and any(a >= int(x) and b <= int(y) + 1 for x, y in obj.get("positive_intervals", [])):
                spans.append((a, b))
        if spans:
            anchors[vid] = {"spans": sorted(set(spans)), "path": str(p), "sha256": file_sha256(p)}
    if not anchors:
        raise ValueError("No traceable positive anchors; filename anomaly codes alone are not labels")
    rng = random.Random(config["seed"])
    buckets = defaultdict(list)
    for vid, p in videos.items():
        if group(vid) in excluded:
            continue
        if pure_normal(vid):
            bucket = "A"
        elif vid in anchors and codes(vid):
            bucket = codes(vid)[0]
        else:
            continue
        buckets[bucket].append(vid)
    for values in buckets.values():
        rng.shuffle(values)
    # Normal examples occupy half the design, not only a tiny negative control.
    schedule = ["A", "B1", "A", "B2", "A", "B4", "A", "B5", "A", "B6", "A", "G"]
    selected, seen_groups, failures = [], set(), []
    while len(selected) < config["pilot_videos"] and any(buckets.values()):
        for bucket in schedule:
            if len(selected) >= config["pilot_videos"]:
                break
            while buckets[bucket]:
                vid = buckets[bucket].pop()
                if group(vid) in seen_groups:
                    continue
                p = videos[vid]
                try:
                    n = frame_count(p)
                    if n < config["window"]:
                        raise ValueError("pilot_video_shorter_than_frozen_window")
                    span = None
                    if bucket != "A":
                        valid = [(a, b) for a, b in anchors[vid]["spans"] if 0 <= a < b <= n]
                        if not valid:
                            raise ValueError("anchor_outside_video")
                        span = rng.choice(valid)
                    selected.append((vid, bucket, n, span))
                    seen_groups.add(group(vid))
                    break
                except Exception as exc:
                    failures.append({"video_id": vid, "error": type(exc).__name__ + ": " + str(exc)})
    # Allocate entire sources before creating windows. No model outputs guide splitting.
    role_map = {}
    for bucket in ["A", *CODES]:
        items = [x for x in selected if x[1] == bucket]
        rng.shuffle(items)
        ncal = max(1, round(len(items) * .2)) if len(items) >= 3 else 0
        nval = ncal
        for i, item in enumerate(items):
            role_map[item[0]] = "calibration" if i < ncal else ("validation" if i < ncal + nval else "fit")
    inputs, labels = [], []
    w = config["window"]
    for vid, bucket, n, anchor in selected:
        p = videos[vid]
        if anchor:
            start = min(n - w, max(0, (anchor[0] + anchor[1]) // 2 - w // 2))
        else:
            start = rng.randrange(n - w + 1)
        starts = [start]
        # Second window preserves natural context. It is NOT an automatic negative.
        options = [s for s, e in dense_spans(n, w, w) if e <= start or s >= start + w]
        if config["pilot_windows_per_video"] == 2 and options:
            starts.append(rng.choice(options))
        for i, s in enumerate(starts):
            row = row_for(p, n, s, s + w, "train")
            row["role"] = role_map[vid]
            inputs.append(row)
            supervised = bucket == "A" or i == 0
            labels.append({"window_uid": row["window_uid"], "target": (0 if bucket == "A" else 1) if supervised else None,
                           "loss_mask": supervised, "codes": codes(vid), "pure_normal": bucket == "A",
                           "scope": "filename_A_normal" if bucket == "A" else ("72B_positive_anchor_weak" if i == 0 else "unlabelled_context"),
                           "anchor_provenance": anchors.get(vid) if i == 0 and anchor else None})
    if not inputs:
        raise ValueError("Empty pilot; inspect train/anchor paths")
    save_jsonl_new(out / "pilot/inputs.jsonl", inputs)
    save_jsonl_new(out / "private/pilot_labels.jsonl", labels)
    report = {"windows": len(inputs), "source_groups": len(selected), "role_windows": dict(Counter(r["role"] for r in inputs)),
              "inventory_videos": len(videos), "train_media_scope": "official_five_parts_when_present; auxiliary_analysis_copies_excluded",
              "selected_primary_classes": dict(Counter(x[1] for x in selected)), "excluded_design_source_groups": sorted(excluded),
              "exclusion_identity_source": exclusion_source, "probe_failures": failures,
              "label_counts": dict(Counter(r["scope"] for r in labels)), "history_status": "development_exposed_not_certified_unseen",
              "hard_normal_semantics_verified": False, "claim_limit": "source-disjoint weak-anchor pilot, not frame gold or six-class benchmark"}
    report["role_scope"] = {role: {"sources": len({r["source_group"] for r in inputs if r["role"] == role}),
        "classes": dict(Counter(c for r, y in zip(inputs, labels) if r["role"] == role and y["loss_mask"] for c in (y["codes"] or ["A"])))}
        for role in ("fit", "calibration", "validation")}
    report["six_class_scope_all_roles"] = all(all(report["role_scope"][role]["classes"].get(c, 0) for c in ["A", *CODES]) for role in report["role_scope"])
    immutable(out / "pilot/enrollment.json", report)
    return report


def build_dense(out, test_root, config, model_path):
    videos = inventory(test_root)
    if len(videos) != config["expected_test_videos"]:
        raise ValueError(f"Expected {config['expected_test_videos']} complete test videos, found {len(videos)}; check TEST_ROOT")
    rows, errors = [], []
    for vid, p in videos.items():
        try:
            n = frame_count(p)
            rows.extend(row_for(p, n, s, e, "test") for s, e in dense_spans(n, config["window"], config["stride"]))
        except Exception as exc:
            errors.append({"video_id": vid, "error": type(exc).__name__ + ": " + str(exc)})
    if errors:
        write_json(out / "dense/preflight_errors.json", errors)
        raise ValueError("Dense manifest incomplete; inspect dense/preflight_errors.json")
    save_jsonl_new(out / "dense/inputs.jsonl", rows)
    report = {"videos": len(videos), "normal_videos": sum(pure_normal(v) for v in videos), "windows": len(rows),
              "window": config["window"], "stride": config["stride"], "tail_policy": "append_last_full_window",
              "short_video_policy": "one_real_span_at_least_8_frames", "claim_scope": "XD_test_development_exposed",
              "model_sha256": file_sha256(model_path), "no_labels_in_collection": True}
    immutable(out / "dense/enrollment.json", report)
    return report
