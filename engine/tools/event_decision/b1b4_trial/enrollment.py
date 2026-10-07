"""Outcome-blind train-source enrollment. Context is never a negative label."""
from __future__ import annotations

import os
import re
from collections import Counter, defaultdict
from pathlib import Path

from ..contracts import WindowKey, iter_jsonl, read_json, semantic_sha256, write_json, write_jsonl
from ..hard_trial import PRUNE, HISTORY_NAMES, group_id
from ..role_scoped import portable
from .protocol import advance, immutable, stable_hash, verify_manifest

STRATA = ("B1_weak_positive", "B4_weak_positive", "hard_label_A", "context_unverified", "easy_label_A")


def union_spans(spans):
    merged = []
    for a, b in sorted(spans):
        if type(a) is not int or type(b) is not int or b <= a or a < 0:
            raise ValueError("Invalid half-open anchor span")
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(b, merged[-1][1])
        else:
            merged.append([a, b])
    return merged


def weak_label(row):
    if row.get("dataset_partition") != "train":
        raise ValueError("PARTITION_NOT_EXPLICIT_TRAIN")
    a, b = row["start_frame"], row["end_frame_exclusive"]
    spans = union_spans(row.get("positive_spans_half_open", []))
    overlap = max([max(0, min(b, y) - max(a, x)) for x, y in spans] or [0])
    negative = row.get("negative_spans_half_open", [])
    conflict = any(max(a, x, u) < min(b, y, v) for x, y in spans for u, v in negative)
    label_a = (row.get("label_source") == "explicit_filename_label_A" and
               re.search(r"_label_A(?:-0-0)?(?:\.mp4)?$", row["video_id"]) is not None)
    if conflict or (label_a and overlap):
        return None
    if label_a:
        return 0
    if row.get("label_source") == "verified_positive_anchor" and row.get("positive_span_provenance") and overlap >= 8:
        return 1
    return None


def load_candidates(paths):
    rows, identities = [], {}
    # Only field allow-list data can enter this protocol; old scores and answers cannot.
    keys = {"dataset_partition", "video_id", "video_path", "source_group", "start_frame", "end_frame_exclusive",
            "stratum", "label_source", "positive_spans_half_open", "negative_spans_half_open",
            "positive_span_provenance", "media_verified", "evidence_sha256", "perceptual_fingerprint",
            "sampled_frame_indices", "local_motion_score", "source_media_sha256"}
    for path in paths:
        for raw in iter_jsonl(path):
            row = {k: raw[k] for k in keys if k in raw}
            if row.get("stratum") == "hard_postevent_unverified":
                row["stratum"] = "context_unverified"
            if row.get("stratum") not in STRATA:
                continue
            uid = WindowKey(row.get("dataset_partition", "unknown"), row["video_id"],
                            row["start_frame"], row["end_frame_exclusive"]).uid
            if uid in identities:
                if semantic_sha256(row) != identities[uid]:
                    raise ValueError("Conflicting duplicate candidate: " + uid)
                continue
            identities[uid] = semantic_sha256(row)
            rows.append({**row, "window_uid": uid, "source_group": group_id(row["video_id"])})
    return rows


def attach_negative_anchor_audit(project, out, rows):
    """Negative anchors detect conflicts, never establish a normal window target."""
    from ..label_scope import _load_gather_anchors
    root = project.parent / "Transformer_semantic_components_select_anomaly/top_anomalous_frames_72B_negative_segments"
    snapshot_path = out / "enrollment/negative_anchor_snapshot.json"
    snapshot = read_json(snapshot_path)
    if snapshot is None:
        if not root.is_dir():
            raise ValueError("WAITING_FOR_NEGATIVE_ANCHOR_AUDIT: source folder unavailable; absence must not mean no conflicts")
        anchors = _load_gather_anchors(project.parent / "pipeline/tools")(root)
        by_video, files = {}, {}
        for video in sorted({r["video_id"] for r in rows}):
            spans = anchors.get(video, [])
            by_video[video] = [[int(a), int(b) + 1] for a, b, *_ in spans]
            for folder in (root / video, root / (video + ".mp4")):
                for path in folder.glob("*.json") if folder.is_dir() else []:
                    files[str(path.resolve())] = stable_hash(path)
        snapshot = {"root": str(root.resolve()), "by_video": by_video, "files": [{"path": k, "sha256": v} for k, v in files.items()],
                    "policy": "negative_anchor_conflict_audit_only_never_normal_supervision"}
        write_json(snapshot_path, snapshot)
    verify_manifest(snapshot["files"])
    if set(snapshot["by_video"]) != {r["video_id"] for r in rows}:
        raise ValueError("Negative anchor snapshot scope changed; use new TAG")
    return [{**r, "negative_spans_half_open": union_spans(r.get("negative_spans_half_open", []) + snapshot["by_video"][r["video_id"]])} for r in rows]


def history_scope(out, inventory):
    identity = semantic_sha256(inventory)
    approval = read_json(out / "history/scope_approval.json", {})
    if not approval:
        write_json(out / "history/scope_approval.json", {
            "reviewer": "", "inventory_sha256": identity, "coverage_complete": False,
            "external_use_checked": False, "no_new_use_since_audit": False,
            "reserved_final_source_groups": [], "source_aliases": {},
            "comment": "Review inventory issues and external use. No model outcomes may be consulted to release a source."})
    valid = (bool(approval.get("reviewer")) and approval.get("inventory_sha256") == identity and not inventory["issues"] and
             all(approval.get(k) is True for k in ("coverage_complete", "external_use_checked", "no_new_use_since_audit")))
    return {"inventory_sha256": identity, "statuses": inventory["statuses"], "approval": approval,
            "ready": valid, "issues": inventory["issues"], "remote_execution_authorized": False}


def import_history(out, audit, rows):
    inventory = read_json(audit / "history/inventory.json")
    if not inventory or set(inventory["statuses"]) != {r["source_group"] for r in rows}:
        raise ValueError("History snapshot candidate-source scope mismatch: perform a fresh scan")
    verify_manifest(inventory["files"])
    write_json(out / "history/inventory.json", inventory)
    write_jsonl(out / "history/evidence.jsonl", list(iter_jsonl(audit / "history/evidence.jsonl")))
    immutable(out / "history/import_receipt.json", {"source": str(audit.resolve()),
        "inventory_sha256": stable_hash(audit / "history/inventory.json"), "all_recorded_files_rehashed": True,
        "new_unregistered_files_not_discoverable_from_snapshot": True,
        "requires_current_external_and_no_new_use_attestation": True})
    return history_scope(out, inventory)


def history_scan(project, out, rows):
    from audit_v91_history import scan_file, SUPPLEMENTAL, PLANNED
    targets = {r["source_group"] for r in rows}
    names = HISTORY_NAMES | SUPPLEMENTAL | PLANNED | {
        "results.jsonl", "selection.json", "exposure_ledger.jsonl", "review_exposure_manifest.jsonl",
        "predictions_committed.jsonl", "graph_catalog_v2.json"}
    paths = []
    for sibling in sorted(project.parent.iterdir()):
        root = sibling / "runs"
        if not root.is_dir():
            continue
        print("[history-inventory] " + str(root), flush=True)
        for directory, dirs, files in os.walk(root):
            dirs[:] = sorted(d for d in dirs if d not in PRUNE and (Path(directory) / d).resolve() != out.resolve())
            paths.extend(Path(directory) / n for n in sorted(set(files) & names))
    evidence, issues, scanned = [], [], []
    for i, path in enumerate(paths):
        stat = path.stat()
        key = semantic_sha256([str(path.resolve()), stat.st_size, stat.st_mtime_ns, sorted(targets), "v912_history1"])
        cache = out / "history/scan_cache" / (key + ".json")
        data = read_json(cache)
        if data is None:
            ev, err, meta = scan_file(path, targets, 2 * 1024**3)
            after = path.stat()
            if (stat.st_size, stat.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError("WAITING_FOR_STABLE_SOURCE: " + str(path))
            data = {"evidence": ev, "issues": err, "meta": meta}
            write_json(cache, data)
        evidence.extend(data["evidence"]); issues.extend(data["issues"])
        if data["meta"]:
            scanned.append(data["meta"])
        if i % 100 == 0:
            print(f"[history] {i + 1}/{len(paths)} recognized files; API=0", flush=True)
    kinds = defaultdict(set)
    for r in evidence:
        kinds[r["source_group"]].add(r["kind"])
    harmless = {"planned_registration", "mock_config_execution", "mock_marker_execution"}
    statuses = {g: ("actual_execution_or_review" if kinds[g] - harmless else
                    "reserved_not_used" if kinds[g] else "history_unknown") for g in sorted(targets)}
    inventory = {"files": scanned, "issues": issues, "statuses": statuses,
                 "scope": "recognized local records only; absence is not proof of no exposure"}
    write_json(out / "history/inventory.json", inventory)
    write_jsonl(out / "history/evidence.jsonl", evidence)
    return history_scope(out, inventory)


def allocate(rows, history, config):
    eligible, rejected = [], []
    aliases = history.get("approval", {}).get("source_aliases", {})
    reserved = set(history.get("approval", {}).get("reserved_final_source_groups", []))
    if not isinstance(aliases, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in aliases.items()):
        raise ValueError("SOURCE_ALIAS_MAP_INVALID")
    def canonical(g):
        seen = set()
        while g in aliases:
            if g in seen:
                raise ValueError("SOURCE_ALIAS_CYCLE")
            seen.add(g)
            g = aliases[g]
        return g
    if (set(aliases) | set(aliases.values())) - set(history["statuses"]):
        raise ValueError("SOURCE_ALIAS_OUTSIDE_AUDITED_SCOPE: refresh history with all alias sources before enrollment")
    reserved |= {canonical(g) for g in reserved}
    exposed_history = {canonical(g) for g, status in history["statuses"].items() if status == "actual_execution_or_review"}
    for raw in rows:
        try:
            r = dict(raw)
            g = canonical(r["source_group"])
            r["source_group"] = g
            if g in reserved or raw["source_group"] in reserved:
                raise ValueError("RESERVED_FINAL_SOURCE")
            if r["end_frame_exclusive"] - r["start_frame"] != 96 or r.get("media_verified") is not True:
                raise ValueError("MEDIA_OR_WINDOW_NOT_VERIFIED")
            if r.get("sampled_frame_indices") != [r["start_frame"] + int(i * 95 / 7) for i in range(8)]:
                raise ValueError("EIGHT_FRAME_CONTRACT_MISSING")
            if not re.fullmatch("[0-9a-f]{64}", str(r.get("evidence_sha256", ""))):
                raise ValueError("EVIDENCE_FINGERPRINT_MISSING")
            y = weak_label(r)
            s = r["stratum"]
            if s.startswith(("B1_", "B4_")) and (y != 1 or s[:2] not in r["video_id"].split("_label_")[-1].split("-")):
                raise ValueError("POSITIVE_ANCHOR_SCOPE_MISSING")
            if s.endswith("label_A") and y != 0:
                raise ValueError("NOT_EXPLICIT_LABEL_A")
            if s == "context_unverified":
                y = None
            for p in r.get("positive_span_provenance", []) if y == 1 else []:
                if stable_hash(portable(p["path"])) != p["sha256"]:
                    raise ValueError("ANCHOR_PROVENANCE_CHANGED")
            r["weak_target"] = y
            r["history_status"] = ("actual_execution_or_review" if g in exposed_history else
                                   history["statuses"].get(raw["source_group"], "history_unknown"))
            eligible.append(r)
        except (ValueError, KeyError, OSError) as exc:
            rejected.append({"window_uid": raw.get("window_uid"), "reason": str(exc)})
    # Exact content aliases share a role. A perceptual near-match only requests review.
    parent = {r["source_group"]: r["source_group"] for r in eligible}
    def representative(g):
        while parent[g] != g:
            parent[g] = parent[parent[g]]
            g = parent[g]
        return g
    digest_group = {}
    for r in eligible:
        digest = r["evidence_sha256"]
        g = r["source_group"]
        if digest in digest_group:
            parent[representative(g)] = representative(digest_group[digest])
        else:
            digest_group[digest] = g
    for r in eligible:
        r["source_group"] = representative(r["source_group"])
    exposed = {r["source_group"] for r in eligible if r["history_status"] == "actual_execution_or_review"}
    pools = {s: [r for r in eligible if r["stratum"] == s] for s in STRATA}
    selected, counts, roles, used_digests = [], Counter(), {}, set()
    # Reserve locked scarce positive/context capacity before exposed-adaptation and flexible A.
    demands = [(role, s, n) for role, quotas in config["quotas"].items() for s, n in quotas.items()]
    demands.sort(key=lambda d: (d[0] != "locked_evaluation",
                              len({r["source_group"] for r in pools[d[1]] if d[0] != "locked_evaluation" or r["source_group"] not in exposed}) / max(d[2], 1), d[1]))
    for role, s, need in demands:
        pool = sorted(pools[s], key=lambda r: (r["source_group"] not in exposed if role == "adaptation" else False,
                                                semantic_sha256([config["seed"], r["window_uid"]])))
        for cap in (1, config["max_windows_per_source"]):
            for r in pool:
                g = r["source_group"]
                if counts[(role, s)] >= need:
                    break
                if role == "locked_evaluation" and (g in exposed or not history["ready"]):
                    continue
                if counts[g] >= cap or roles.get(g, role) != role or r["evidence_sha256"] in used_digests:
                    continue
                if any(p["video_id"] == r["video_id"] and max(p["start_frame"], r["start_frame"]) < min(p["end_frame_exclusive"], r["end_frame_exclusive"]) for p in selected):
                    continue
                roles[g] = role; counts[g] += 1; counts[(role, s)] += 1
                used_digests.add(r["evidence_sha256"])
                selected.append({**r, "role": role, "training_loss_mask": role == "adaptation" and r["weak_target"] is not None,
                                 "human_labels_use": "audit_only" if role == "adaptation" else "evaluation_only"})
    gaps = [{"role": role, "stratum": s, "requested": n, "selected": counts[(role, s)]}
            for role, quotas in config["quotas"].items() for s, n in quotas.items() if counts[(role, s)] != n]
    near = []
    for i, r in enumerate(selected):
        for p in selected[:i]:
            a, b = r.get("perceptual_fingerprint", []), p.get("perceptual_fingerprint", [])
            if r["source_group"] != p["source_group"] and len(a) == len(b) == 8:
                if sum((int(x, 16) ^ int(y, 16)).bit_count() for x, y in zip(a, b)) <= 32:
                    near.append([r["window_uid"], p["window_uid"]])
    return selected, {"selected_n": len(selected), "gaps": gaps, "rejected": rejected,
                      "near_duplicates_for_review_not_identity_proof": near,
                      "ready": not gaps and history["ready"], "allocation_policy": "locked_scarce_first_then_exposed_adaptation_v1",
                      "status": "WAITING_FOR_HISTORY_SCOPE" if not history["ready"] else "INSUFFICIENT_SOURCE_CAPACITY" if gaps else "READY_FOR_MEDIA_REVIEW",
                      "B5_required": False, "remote_calls": 0, "formal_AP": None}


def prepare_enrollment(out, rows, history, config, inputs):
    selected, report = allocate(rows, history, config)
    write_json(out / "enrollment/preflight_report.json", report)
    write_jsonl(out / "enrollment/preview.jsonl", selected)
    immutable(out / "enrollment/input_hashes.json", [{"path": str(p.resolve()), "sha256": stable_hash(p)} for p in inputs])
    if not report["ready"]:
        return report
    advance(out, "SOURCE_CAPACITY_VERIFIED")
    private = out / "enrollment/windows.jsonl"
    if private.exists():
        if list(iter_jsonl(private)) != selected:
            raise ValueError("Enrollment changed: new TAG required")
    else:
        write_jsonl(private, selected)
        write_json(out / "enrollment/history_scope.json", history)
        write_jsonl(out / "enrollment/adaptation_labels.jsonl", [
            {"window_uid": r["window_uid"], "target": r["weak_target"], "loss_mask": r["training_loss_mask"],
             "label_rule_id": "xd_window_contiguous_overlap_ge8_v1", "supervision": "weak_only"}
            for r in selected if r["role"] == "adaptation"])
        feature_keys = ("window_uid", "video_path", "start_frame", "end_frame_exclusive", "sampled_frame_indices")
        write_jsonl(out / "enrollment/feature_inputs.jsonl", [{k: r[k] for k in feature_keys} for r in selected])
        write_jsonl(out / "enrollment/role_map.jsonl", [{k: r[k] for k in ("window_uid", "source_group", "role")} for r in selected])
        smoke = []
        for stratum in ("B1_weak_positive", "B4_weak_positive", "hard_label_A", "context_unverified"):
            group = [r for r in selected if r["role"] == "adaptation" and r["stratum"] == stratum]
            if group:
                smoke.append({"window_uid": group[0]["window_uid"]})
        write_jsonl(out / "enrollment/smoke_manifest.jsonl", smoke)
    advance(out, "ROLES_RESERVED")
    return report
