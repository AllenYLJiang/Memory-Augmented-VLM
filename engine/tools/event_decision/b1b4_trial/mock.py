"""Synthetic end-to-end integration check; never evidence of scientific improvement."""
import copy

import numpy as np

from ..contracts import WindowKey, read_json, semantic_sha256, write_json, write_jsonl
from .protocol import advance, immutable, stable_hash


def run_mock(out, config):
    from .features import export_store
    from .learning import fit_models, predict_and_commit
    from .evaluation import evaluate
    if (out / "mock_summary.json").exists():
        return read_json(out / "mock_summary.json")
    if (out / "protocol/config.json").exists() and not read_json(out / "protocol/config.json").get("mock"):
        raise ValueError("Mock cannot overwrite a real trial")
    config = {**copy.deepcopy(config), "mock": True, "bootstrap_samples": min(config["bootstrap_samples"], 100)}
    immutable(out / "protocol/config.json", config)
    rng = np.random.default_rng(config["seed"])
    rows, public, private, reviews = [], [], {}, {"R1": [], "R2": []}
    for role, quotas in config["quotas"].items():
        for stratum, count in quotas.items():
            for j in range(count):
                i = len(rows)
                uid = WindowKey("train", f"synthetic_{i}", 0, 96).uid
                y = int(stratum.startswith(("B1", "B4")))
                blind = "SYNTHETIC_" + uid[:12]
                row = {"window_uid": uid, "video_id": "synthetic_" + str(i), "source_group": "synthetic_group_" + str(i),
                       "dataset_partition": "train", "role": role, "stratum": stratum,
                       "weak_target": y if stratum != "context_unverified" else None,
                       "training_loss_mask": role == "adaptation" and stratum != "context_unverified"}
                rows.append(row)
                prob = float(np.clip(.25 + .45 * y + rng.normal(0, .17), .01, .99))
                margin = float(prob - .5 + rng.normal(0, .2))
                competitions = {m: {"margin": margin + k * .01} for k, m in enumerate(("independent_direct_nodes", "conditional_rowmax", "conditional_ot_full"))}
                payload = {"baseline": {"competitions": competitions, "completeness": {"independent": True, "joint": True}},
                           "C0": {"complete": True, "current_window_active_occupancy_probability": prob,
                                  "uncertainty": .2, "phase_nodes": {"active": {"direct_physical_escalation_mechanism": {
                                      "presence_probability": prob, "evidence_quality_by_bin": [1.0] * 8}}}},
                           "C1": {"schema_version": "event_candidates_v1", "window_id": uid, "evidence_signature": "synthetic",
                                  "scan_complete": True, "overflow": False, "observation_sufficient": True, "uncertainty": .2,
                                  "entities": [], "evidence": [], "events": []}, "request_keys": [], "synthetic": True}
                write_json(out / "private_acquisition" / role / (uid + ".json"), payload)
                public.append({"blind_id": blind, "clip": None, "clip_sha256": "synthetic_no_media", "frames": 96})
                private[blind] = {"window_uid": uid, "role": role}
                for reviewer in reviews:
                    reviews[reviewer].append({"reviewer_id": reviewer, "blind_id": blind,
                        "current_window_visual_label": "anomalous" if y else "normal", "direct_mechanism": "synthetic mechanism" if y else "none",
                        "visible_event_intervals_local": [[0, 95]] if y else [], "normal_mechanism": "none" if y else "synthetic assistance",
                        "normal_explains_suspicious_action": "no" if y else "yes", "same_actor_support": "supported", "same_time_support": "supported",
                        "unexplained_anomalous_mechanism": "yes" if y else "no", "review_confidence": "high", "note": "SYNTHETIC NOT HUMAN",
                        "event_classes": [stratum[:2]] if y else [], "context_review": {"context_used": False, "audio_used": False}})
    write_jsonl(out / "enrollment/windows.jsonl", rows)
    write_jsonl(out / "enrollment/role_map.jsonl", [{k: r[k] for k in ("window_uid", "role", "source_group")} for r in rows])
    write_jsonl(out / "enrollment/adaptation_labels.jsonl", [{"window_uid": r["window_uid"], "target": r["weak_target"],
                "loss_mask": r["training_loss_mask"], "supervision": "weak_only"} for r in rows if r["role"] == "adaptation"])
    write_json(out / "review/public_manifest.json", public)
    write_json(out / "review/private_map.json", private)
    for rid, values in reviews.items():
        for r in values:
            r["packet_sha256"] = semantic_sha256(public)
        write_jsonl(out / "review/returns" / (rid + ".jsonl"), values)
    immutable(out / "protocol/frozen.json", {"mock": True, "files": []})
    advance(out, "PROTOCOL_LOCKED")
    export_store(out, "adaptation", [r for r in rows if r["role"] == "adaptation"])
    advance(out, "ADAPTATION_FEATURES_FROZEN")
    models = fit_models(out, config)
    export_store(out, "locked_evaluation", [r for r in rows if r["role"] == "locked_evaluation"])
    advance(out, "LOCKED_FEATURES_COMPLETE_OR_DECLARED_MISSING")
    predict_and_commit(out)
    evaluate(out, config)
    summary = {"synthetic_only": True, "windows": len(rows), "models": models, "state": "BRANCH_CLOSED",
               "API_calls": 0, "new_media_decodes": 0, "formal_accuracy": None, "formal_AP": None,
               "scientific_improvement_demonstrated": False, "deployment_authorized": False}
    write_json(out / "mock_summary.json", summary)
    return summary
