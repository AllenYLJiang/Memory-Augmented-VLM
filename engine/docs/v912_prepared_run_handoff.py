#!/usr/bin/env python3
"""Read-only prepared-run audit and explicitly authorized smoke-only budget handoff.

No provider imports, media generation, model changes, review answers or API calls.
Kept outside tools to preserve the existing implementation seal.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "tools"))
from event_decision.contracts import WindowKey, iter_jsonl, read_json, semantic_sha256, write_json
from event_decision.role_scoped import portable
from event_decision.b1b4_trial.enrollment import weak_label
from event_decision.b1b4_trial.protocol import now, run_lock, stable_hash


def index_rows(rows, field):
    result = {r[field]: r for r in rows}
    if len(result) != len(rows):
        raise ValueError("duplicate " + field)
    return result


def probe_clip(path, ffprobe="ffprobe"):
    command = [ffprobe, "-v", "error", "-count_frames", "-show_entries",
               "stream=codec_type,nb_read_frames", "-of", "json", str(path)]
    data = json.loads(subprocess.run(command, check=True, capture_output=True, text=True, timeout=60).stdout)
    video = [s for s in data["streams"] if s.get("codec_type") == "video"]
    if len(video) != 1 or int(video[0].get("nb_read_frames", -1)) != 96:
        raise ValueError("clip must contain exactly 96 readable video frames")
    if any(s.get("codec_type") == "audio" for s in data["streams"]):
        raise ValueError("blind clip contains audio")


def audit(out, probe=False, ffprobe="ffprobe"):
    errors, checks = [], []
    def check(ok, message):
        (checks if ok else errors).append(message)
    def hash_check(path, expected, name):
        try:
            check(stable_hash(path) == expected, name)
        except OSError as exc:
            errors.append(name + ": " + str(exc))

    state = read_json(out / "state.json", {}).get("state", "MISSING")
    config = read_json(out / "protocol/config.json")
    check(config == read_json(PROJECT / "config/b1b4_minimal_effect_v912.yaml"), "registered configuration unchanged")
    preflight = read_json(out / "enrollment/preflight_report.json")
    rows = list(iter_jsonl(out / "enrollment/windows.jsonl"))
    by_uid = index_rows(rows, "window_uid")
    role_rows = index_rows(list(iter_jsonl(out / "enrollment/role_map.jsonl")), "window_uid")
    feature_rows = index_rows(list(iter_jsonl(out / "enrollment/feature_inputs.jsonl")), "window_uid")
    labels = index_rows(list(iter_jsonl(out / "enrollment/adaptation_labels.jsonl")), "window_uid")
    inventory = read_json(out / "seal/legacy_inventory.json")
    hash_check(out / "seal/legacy_inventory.json", read_json(out / "seal/identity.json")["inventory_sha256"], "seal inventory identity")
    for i, item in enumerate(inventory):
        hash_check(portable(item["path"]), item["sha256"], "sealed input " + item["path"])
        if i and i % 1000 == 0:
            print(f"[audit] {i}/{len(inventory)} sealed files; API=0", flush=True)
    history = read_json(out / "history/inventory.json")
    approval = read_json(out / "history/scope_approval.json")
    scope = read_json(out / "enrollment/history_scope.json")
    check(bool(approval.get("reviewer")) and approval.get("inventory_sha256") == semantic_sha256(history)
          and all(approval.get(k) is True for k in ("coverage_complete", "external_use_checked", "no_new_use_since_audit")), "current history approval")
    check(scope.get("ready") is True and scope.get("approval") == approval and not history["issues"], "enrolled history scope")
    for item in history["files"]:
        hash_check(portable(item["path"]), item["sha256"], "history input " + item["path"])
    for item in read_json(out / "enrollment/input_hashes.json"):
        hash_check(portable(item["path"]), item["sha256"], "candidate input " + item["path"])
    for item in read_json(out / "enrollment/negative_anchor_snapshot.json")["files"]:
        hash_check(portable(item["path"]), item["sha256"], "negative-anchor input " + item["path"])
    counts = Counter((r["role"], r["stratum"]) for r in rows)
    expected_counts = {(role, s): n for role, quotas in config["quotas"].items() for s, n in quotas.items()}
    check(dict(counts) == expected_counts and len(rows) == 144, "144 windows and exact 48/96 quotas")
    check(preflight["ready"] is True and preflight["selected_n"] == len(rows) and not preflight["gaps"], "preflight ready without gaps")
    check(set(role_rows) == set(feature_rows) == set(by_uid), "role and feature manifest coverage")
    adaptation = {r["window_uid"] for r in rows if r["role"] == "adaptation"}
    locked = set(by_uid) - adaptation
    check(set(labels) == adaptation, "adaptation-only supervision file")
    sources, videos = defaultdict(list), defaultdict(list)
    for r in rows:
        uid = r["window_uid"]
        sources[r["source_group"]].append(r)
        videos[r["video_id"]].append(r)
        check(uid == WindowKey(r["dataset_partition"], r["video_id"], r["start_frame"], r["end_frame_exclusive"]).uid, "canonical UID " + uid)
        check(r["dataset_partition"] == "train" and r["end_frame_exclusive"] - r["start_frame"] == 96, "train 96-frame scope " + uid)
        target = None if r["stratum"] == "context_unverified" else weak_label(r)
        mask = r["role"] == "adaptation" and target is not None
        check(r["weak_target"] == target and r["training_loss_mask"] is mask, "label scope and loss mask " + uid)
        check(role_rows.get(uid) == {k: r[k] for k in ("window_uid", "source_group", "role")}, "role map " + uid)
        keys = ("window_uid", "video_path", "start_frame", "end_frame_exclusive", "sampled_frame_indices")
        check(feature_rows.get(uid) == {k: r[k] for k in keys}, "feature inputs have no labels " + uid)
        if uid in labels:
            check(labels[uid]["target"] == target and labels[uid]["loss_mask"] is mask and labels[uid]["supervision"] == "weak_only", "adaptation label " + uid)
        if uid in locked:
            check(r["history_status"] != "actual_execution_or_review", "no known exposed locked source " + uid)
        for item in r.get("positive_span_provenance", []):
            hash_check(portable(item["path"]), item["sha256"], "positive-anchor input " + item["path"])
    check(all(len({r["role"] for r in group}) == 1 for group in sources.values()), "source disjointness")
    check(max(map(len, sources.values()), default=0) <= config["max_windows_per_source"], "source capacity")
    check(all(all(a["end_frame_exclusive"] <= b["start_frame"] for a, b in zip(sorted(v, key=lambda r:r["start_frame"]), sorted(v, key=lambda r:r["start_frame"])[1:])) for v in videos.values()), "no overlapping video windows")
    media = read_json(out / "media/manifest.json")
    check(set(media) == set(by_uid), "media manifest coverage")
    image_count = 0
    for i, (uid, r) in enumerate(by_uid.items(), 1):
        m = media.get(uid, {})
        indices = [r["start_frame"] + int(k * 95 / 7) for k in range(8)]
        check(m.get("frame_indices") == indices == r["sampled_frame_indices"], "exact frame indices " + uid)
        check(m.get("decoded_clip_frames") == 96 and len(m.get("image_paths", [])) == len(m.get("image_sha256", [])) == 8, "media counts " + uid)
        expected_files = {f"T{k}.jpg" for k in range(8)} | {"clip.mp4"}
        check(set(m.get("files", {})) == expected_files, "nine media file entries " + uid)
        for name, sha in m.get("files", {}).items():
            check(name in expected_files, "safe media filename " + uid)
            if name in expected_files:
                hash_check(out / "media" / uid / name, sha, "media hash " + uid + "/" + name)
        for k, (path, sha) in enumerate(zip(m.get("image_paths", []), m.get("image_sha256", []))):
            check(portable(path).resolve() == (out / "media" / uid / f"T{k}.jpg").resolve() and sha == m["files"].get(f"T{k}.jpg"), "image path/hash binding " + uid)
            image_count += 1
        if probe:
            try:
                probe_clip(out / "media" / uid / "clip.mp4", ffprobe)
            except (ValueError, OSError, subprocess.SubprocessError) as exc:
                errors.append("clip probe " + uid + ": " + str(exc))
        if i % 24 == 0:
            print(f"[audit] {i}/{len(rows)} saved media bundles checked; API=0", flush=True)
    public = read_json(out / "review/public_manifest.json")
    private = read_json(out / "review/private_map.json")
    public_index = index_rows(public, "blind_id")
    check(len(public) == len(private) == len(rows) and set(public_index) == set(private), "blind packet coverage")
    check({p["window_uid"] for p in private.values()} == set(by_uid), "blind UID coverage")
    page = (out / "review/public/index.html").read_text(encoding="utf-8")
    check(all(r["video_id"] not in page and r["source_group"] not in page for r in rows), "public HTML hides source identifiers")
    for blind, item in public_index.items():
        check(set(item) == {"blind_id", "clip", "clip_sha256", "frames"}, "blind manifest field allow-list " + blind)
        uid = private[blind]["window_uid"]
        check(item["clip"] == f"media/{blind}.mp4" and item["frames"] == 96, "blind clip reference " + blind)
        check(private[blind]["role"] == by_uid[uid]["role"], "blind role map " + blind)
        check(item["clip_sha256"] == media[uid]["files"]["clip.mp4"], "blind-original clip identity " + blind)
        hash_check(out / "review/public/media" / (blind + ".mp4"), item["clip_sha256"], "blind clip hash " + blind)
    # Do not inspect reviewers' answers or model outcomes as part of administrative auditing.
    review_templates = {name: (out / "review/returns" / (name + ".jsonl")).is_file() for name in ("R1", "R2")}
    check(all(review_templates.values()), "both reviewer files present (contents not read)")
    smoke = list(iter_jsonl(out / "enrollment/smoke_manifest.jsonl"))
    smoke_ids = set(index_rows(smoke, "window_uid"))
    check(len(smoke_ids) == 4 and smoke_ids <= adaptation, "four adaptation smoke windows")
    check({by_uid[uid]["stratum"] for uid in smoke_ids} == {"B1_weak_positive", "B4_weak_positive", "hard_label_A", "context_unverified"}, "smoke stratum coverage")
    plans = {}
    for phase, ids in (("smoke", smoke_ids), ("adaptation", adaptation), ("locked_evaluation", locked)):
        path = out / "plans" / (phase + ".json")
        plan = read_json(path)
        check(set(index_rows(plan["windows"], "window_uid")) == ids and plan["phase"] == phase, "plan cohort " + phase)
        check(plan["config_sha256"] == semantic_sha256(config) and plan["implementation_seal"] == stable_hash(out / "seal/legacy_inventory.json"), "plan config/seal " + phase)
        for item in plan["windows"]:
            check(item["evidence_sha256"] == semantic_sha256(media[item["window_uid"]]["image_sha256"]), "plan evidence " + phase + "/" + item["window_uid"])
        for key in ("catalog_sha256", "phase_catalog_sha256"):
            rel = config["graph_catalog" if key == "catalog_sha256" else "phase_catalog"]
            hash_check(PROJECT / rel, plan[key], "plan catalog " + phase)
        logical = len(ids) * (2 + plan["max_independent_union_nodes"] + config["top_k_abnormal"] + config["top_k_normal"] + 3)
        physical = logical * config["transport_attempts"] * (1 + config["schema_repair_attempts"])
        check(plan["logical_requests_upper_bound"] == logical and plan["physical_attempts_upper_bound"] == physical
              and plan["output_tokens_upper_bound"] == physical * config["max_output_tokens"], "plan bound arithmetic " + phase)
        plans[phase] = {"windows": len(ids), "plan_sha256": stable_hash(path), **{k:plan[k] for k in ("logical_requests_upper_bound", "physical_attempts_upper_bound", "output_tokens_upper_bound")}}
    attempt_paths = list((out / "cost/attempts").glob("*.json"))
    attempts = [read_json(p) for p in attempt_paths]
    warnings = ["History coverage is a human attestation; new or external exposure is not automatically disproved.",
                "Label_A motion screening is not semantic hard-normal verification; weak positives are not gold labels."]
    result = {"version": "v912_prepared_handoff_audit_v1", "at": now(), "state": state,
              "errors": errors, "checks_passed": len(checks), "technical_prepared_ready": not errors,
              "windows": len(rows), "videos": len(videos), "source_groups": len(sources),
              "roles": {role: {"windows": sum(r["role"] == role for r in rows), "sources": len({r["source_group"] for r in rows if r["role"] == role}),
                                "videos": len({r["video_id"] for r in rows if r["role"] == role}),
                                "strata": {s:counts[role,s] for s in config["quotas"][role]}} for role in config["quotas"]},
              "supervision_counts": {role:dict(Counter(str(r["weak_target"]) for r in rows if r["role"] == role)) for role in config["quotas"]},
              "training_loss_mask_true": sum(r["training_loss_mask"] for r in rows),
              "context_loss_mask_true": sum(r["training_loss_mask"] for r in rows if r["stratum"] == "context_unverified"),
              "rejected_candidates": dict(Counter(r["reason"] for r in preflight["rejected"])),
              "history_scan_files": len(history["files"]), "sealed_files_verified": len(inventory),
              "image_files_verified": image_count, "clips_hash_verified": len(media), "blind_copies_hash_verified": len(public),
              "saved_clips_probed": len(media) if probe else 0, "original_videos_redecoded": 0,
              "smoke_strata": sorted(by_uid[uid]["stratum"] for uid in smoke_ids), "plans": plans,
              "physical_attempt_receipts": len(attempt_paths), "attempts_by_phase": dict(Counter(r["phase"] for r in attempts)),
              "new_API_calls_by_this_tool": 0, "formal_accuracy": None, "formal_AP": None,
              "model_fitting_performed_by_this_tool": False, "warnings": warnings,
              "next": ("FIX_AUDIT_ERRORS_WITHOUT_UNSEALING" if errors else
                       "REVIEW_COMPLETED_SMOKE_BEFORE_ANY_LARGER_ACQUISITION" if (out / "smoke/summary.json").exists() else
                       "EXPLICIT_SMOKE_BUDGET_THEN_PHASE_RUN")}
    write_json(out / "handoff/audits" / (str(time.time_ns()) + ".json"), result)
    write_json(out / "handoff/prepare_audit.json", result)
    return result


def authorize_smoke(out, approved_by, cap, confirmed, audit_result):
    if not confirmed or not approved_by or approved_by.strip().lower() in ("", "your_name", "name", "...", "pending"):
        raise ValueError("Explicit --confirm-paid-smoke and actual --approved-by are required")
    if type(cap) is not int or cap <= 0:
        raise ValueError("max-physical-attempts must be a positive integer")
    if not audit_result.get("technical_prepared_ready"):
        raise ValueError("Prepared-run audit failed")
    if read_json(out / "state.json")["state"] != "ROLES_RESERVED" or (out / "protocol/frozen.json").exists() or (out / "smoke/summary.json").exists():
        raise ValueError("Smoke handoff is only for the current pre-smoke state; do not re-authorize completed phases")
    bundle = read_json(out / "authorizations/approval.json", {})
    if bundle.get("authorized") is True:
        raise ValueError("Combined approval is active and overrides per-phase approval; review it manually first")
    for phase in ("adaptation", "locked_evaluation"):
        if read_json(out / "authorizations" / (phase + ".json"), {}).get("authorized") is True:
            raise ValueError("Later phase already authorized; this helper refuses to imply smoke-only scope")
    path = out / "authorizations/smoke.json"
    previous = read_json(path)
    if previous.get("resume_uncertain_attempts"):
        raise ValueError("Uncertain billing acknowledgment must be reviewed manually, not reset by this tool")
    attempts = [read_json(p) for p in (out / "cost/attempts").glob("*.json")]
    smoke_attempts = [r for r in attempts if r["phase"] == "smoke"]
    if any(r["status"] == "in_flight" for r in smoke_attempts):
        raise ValueError("Uncertain billing: inspect existing attempt receipts first")
    plan = read_json(out / "plans/smoke.json")
    if cap > plan["physical_attempts_upper_bound"] or cap <= len(smoke_attempts):
        raise ValueError("Cap must exceed used attempts and must not exceed the existing frozen plan upper bound")
    if previous.get("authorized") and cap < previous["max_physical_attempts"]:
        raise ValueError("This helper never silently lowers an existing cap")
    config = read_json(out / "protocol/config.json")
    value = {**previous, "approved_by": approved_by.strip(), "authorized": True,
             "plan_sha256": stable_hash(out / "plans/smoke.json"), "max_physical_attempts": cap,
             "max_output_tokens": cap * config["max_output_tokens"], "resume_uncertain_attempts": False}
    if value == previous:
        return {"authorization": "unchanged", "max_physical_attempts": cap, "API_calls": 0}
    receipt = {"at": now(), "previous": previous, "next": value, "scope": "four_adaptation_smoke_windows_only",
               "helper_sha256": stable_hash(Path(__file__)), "audit_sha256": stable_hash(out / "handoff/prepare_audit.json"),
               "protocol_config_sha256": stable_hash(out / "protocol/config.json"), "API_calls": 0}
    write_json(out / "handoff/authorization_receipts" / (str(time.time_ns()) + ".json"), receipt)
    write_json(path, value)
    return {"authorization": str(path), "max_physical_attempts": cap, "max_output_token_reservation": value["max_output_tokens"],
            "later_phases_authorized": False, "API_calls": 0, "next": "Run original launcher with PHASE=run; stop at smoke/protocol review"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit", "authorize-smoke"))
    parser.add_argument("--run", type=Path, default=PROJECT / "runs/governed_v912_b1b4_minimal_effect_20260915")
    parser.add_argument("--probe-clips", action="store_true", help="count frames in saved clips; no original video re-extraction")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--approved-by")
    parser.add_argument("--max-physical-attempts", type=int)
    parser.add_argument("--confirm-paid-smoke", action="store_true")
    args = parser.parse_args(argv)
    try:
        with run_lock(args.run):
            result = audit(args.run, args.probe_clips, args.ffprobe)
            if args.command == "authorize-smoke":
                result = authorize_smoke(args.run, args.approved_by, args.max_physical_attempts, args.confirm_paid_smoke, result)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if not result.get("errors") else 3
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        print("[pause] " + str(exc), flush=True)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
