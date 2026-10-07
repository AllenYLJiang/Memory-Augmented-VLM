"""Lazy provider, hash-bound request DAG, resumable physical-attempt accounting."""
from __future__ import annotations

import json
import importlib.util
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from ..binding import binding_prompt, proposal_prompt, validate_binding, validate_proposal
from ..contracts import iter_jsonl, read_json, semantic_sha256, write_json, write_jsonl
from ..role_scoped import portable
from .features import c1_quality
from .protocol import advance, at_least, immutable, now, stable_hash, verify_frozen

_PARSER_LOCK = threading.Lock()


class StopAcquisition(RuntimeError):
    pass


class RequestRefused(Exception):
    """Terminal per-window refusal; not a schema-repair or semantic-retry target."""


def parse_local(raw):
    try:
        return json.loads(raw), False
    except json.JSONDecodeError:
        # Reuse the existing pure JSON/ast.literal_eval repair helper, not a provider.
        name = "_v912_reference_json_parser"
        with _PARSER_LOCK:
            if name not in sys.modules:
                path = Path(__file__).resolve().parents[4] / "Previous_code/structural_vlm_binary_v40_15_qwen36_highres_graph_vs_node_revision/src/structural_vlm_binary/vlm/dashscope_backend.py"
                spec = importlib.util.spec_from_file_location(name, path)
                module = importlib.util.module_from_spec(spec)
                sys.modules[name] = module
                spec.loader.exec_module(module)
        return sys.modules[name].parse_json_like(raw), True


def role_rows(out, phase):
    metadata = {r["window_uid"]: r for r in iter_jsonl(out / "enrollment/role_map.jsonl")}
    rows = [{**r, **metadata[r["window_uid"]]} for r in iter_jsonl(out / "enrollment/feature_inputs.jsonl")]
    if phase == "smoke":
        ids = {r["window_uid"] for r in iter_jsonl(out / "enrollment/smoke_manifest.jsonl")}
        return [r for r in rows if r["window_uid"] in ids and r["role"] == "adaptation"]
    return [r for r in rows if r["role"] == phase]


def prepare_media(out):
    from ..enrollment_review import extract_clip
    from ..local_screen import local_frames
    rows = list(iter_jsonl(out / "enrollment/windows.jsonl"))
    media = {}
    for i, row in enumerate(rows):
        uid = row["window_uid"]
        folder = out / "media" / uid
        folder.mkdir(parents=True, exist_ok=True)
        source = portable(row["video_path"])
        source_hash = stable_hash(source)
        old = read_json(folder / "receipt.json")
        if old is not None:
            if old["source_sha256"] != source_hash or any(stable_hash(folder / n) != h for n, h in old["files"].items()):
                raise ValueError("MEDIA_CHANGED: " + uid)
            media[uid] = old
            continue
        if local_frames(source, row["start_frame"])["evidence_sha256"] != row["evidence_sha256"]:
            raise ValueError("Candidate screening frames changed: " + uid)
        extract_clip(source, row["start_frame"], folder / "clip.mp4")
        indices = row["sampled_frame_indices"]
        expr = "+".join(f"eq(n\\,{n})" for n in indices)
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(source), "-map", "0:v:0", "-map_metadata", "-1",
                        "-vf", "select=" + expr, "-fps_mode", "passthrough", "-frames:v", "8", "-q:v", "2",
                        "-start_number", "0", str(folder / "T%d.jpg")], check=True, capture_output=True, timeout=900)
        paths = [folder / f"T{k}.jpg" for k in range(8)]
        value = {"source_sha256": source_hash, "frame_indices": indices, "decoded_clip_frames": 96,
                 "image_paths": [str(p.resolve()) for p in paths], "image_sha256": [stable_hash(p) for p in paths],
                 "files": {p.name: stable_hash(p) for p in paths + [folder / "clip.mp4"]},
                 "image_contract": "original_exact_frames_ffmpeg_jpeg_q2_eight_v1",
                 "evidence_mode": "frames", "mode": "frames", "bin_labels": [f"T{k}" for k in range(8)], "frame_count": 8}
        immutable(folder / "receipt.json", value)
        media[uid] = value
        print(f"[media] {i + 1}/{len(rows)} complete clips; API=0", flush=True)
    immutable(out / "media/manifest.json", media)
    return media


def make_plan(project, out, phase, config):
    from graph_catalog import read_catalog_json
    rows = role_rows(out, phase)
    if not rows:
        raise ValueError("Empty planned cohort")
    media = read_json(out / "media/manifest.json")
    catalog = read_catalog_json(project / config["graph_catalog"])
    phases = read_catalog_json(project / config["phase_catalog"])
    phase_keys = ("crowd_active_or_ongoing_physical_escalation_v4", "crowd_causally_linked_aftermath_v4")
    phase_nodes = {n.key for key in phase_keys for n in phases[key].nodes}
    max_nodes = len({n.key for graph in catalog.values() for n in graph.nodes} | phase_nodes)
    logical_per_window = 2 + max_nodes + config["top_k_abnormal"] + config["top_k_normal"] + 3
    plan = {"version": "v912_acquisition_dag_plan1", "phase": phase, "config_sha256": semantic_sha256(config),
            "implementation_seal": stable_hash(out / "seal/legacy_inventory.json"),
            "catalog_sha256": stable_hash(project / config["graph_catalog"]),
            "phase_catalog_sha256": stable_hash(project / config["phase_catalog"]),
            "windows": [{"window_uid": r["window_uid"], "evidence_sha256": semantic_sha256(media[r["window_uid"]]["image_sha256"])} for r in rows],
            "DAG": {"selector": [], "independent_nodes": ["selector"], "conditional_OT": ["independent_nodes"],
                    "C0_dependencies": [], "C0": ["C0_dependencies"], "C1": [], "C2": ["C1.canonical_sha256"]},
            "max_independent_union_nodes": max_nodes, "C0_phase_dependency_nodes": sorted(phase_nodes),
            "logical_requests_upper_bound": len(rows) * logical_per_window,
            "physical_attempts_upper_bound": len(rows) * logical_per_window * config["transport_attempts"] * (1 + config["schema_repair_attempts"]),
            "schema_repairs_per_logical_request_max": config["schema_repair_attempts"],
            "semantic_retries": 0, "legacy_caches_reused": 0,
            "existing_raw_cache_files": len(list((out / "cache/raw").glob("*.json"))),
            "exact_missing_requests": None,
            "missing_count_reason": "selector and C1-dependent prompts resolve at execution; conservative DAG upper bound, not fabricated exact keys",
            "input_tokens": None, "token_estimate_reason": "provider image token accounting unavailable before upload",
            "output_tokens_upper_bound": len(rows) * logical_per_window * config["transport_attempts"] * (1 + config["schema_repair_attempts"]) * config["max_output_tokens"],
            "parents": "resolved canonical response hashes are included in each final raw key and receipt",
            "authorization": "external JSON signed for exact plan_sha256; no null budget calls"}
    # A plan's cache inventory is a snapshot, not a reason to invalidate resume authorization.
    path = out / "plans" / f"{phase}.json"
    old = read_json(path)
    if old:
        comparable = {k: v for k, v in plan.items() if k != "existing_raw_cache_files"}
        if comparable != {k: v for k, v in old.items() if k != "existing_raw_cache_files"}:
            raise ValueError("Existing phase plan changed")
        plan = old
    else:
        write_json(path, plan)
    auth_path = out / "authorizations" / f"{phase}.json"
    if not auth_path.exists():
        write_json(auth_path, {"approved_by": "", "plan_sha256": stable_hash(path),
                              "authorized": False, "max_physical_attempts": None,
                              "max_output_tokens": None, "resume_uncertain_attempts": False,
                              "note": "Fill caps after inspecting plans. Do not copy upper bounds without reviewing cost."})
    bundle_path = out / "authorizations/approval.json"
    bundle = read_json(bundle_path, {"approved_by": "", "authorized": False, "phases": {}})
    if phase not in bundle["phases"]:
        bundle["phases"][phase] = {"plan_sha256": stable_hash(path), "max_physical_attempts": None,
                                    "max_output_tokens": None, "resume_uncertain_attempts": False}
        write_json(bundle_path, bundle)
    return plan


class Budget:
    def __init__(self, out, phase, config):
        self.out, self.phase, self.config = out, phase, config
        self.plan = read_json(out / "plans" / f"{phase}.json")
        self.auth = read_json(out / "authorizations" / f"{phase}.json", {})
        bundle = read_json(out / "authorizations/approval.json", {})
        if bundle.get("authorized") is True:
            self.auth = {**bundle.get("phases", {}).get(phase, {}), "authorized": True, "approved_by": bundle.get("approved_by")}
        if (not self.auth.get("approved_by") or self.auth.get("authorized") is not True or
                self.auth.get("plan_sha256") != stable_hash(out / "plans" / f"{phase}.json") or
                type(self.auth.get("max_physical_attempts")) is not int or self.auth["max_physical_attempts"] <= 0 or
                type(self.auth.get("max_output_tokens")) is not int or self.auth["max_output_tokens"] <= 0):
            raise StopAcquisition("WAITING_FOR_BUDGET_AUTHORIZATION: authorizations/" + phase + ".json")
        self.lock = threading.Lock()
        self.stopped = False
        self.root = out / "cost/attempts"
        self.root.mkdir(parents=True, exist_ok=True)
        attempts = [read_json(p) for p in self.root.glob("*.json")]
        self.used = sum(r["phase"] == phase for r in attempts)
        self.key_attempts = {}
        for r in attempts:
            self.key_attempts[r["request_key"]] = self.key_attempts.get(r["request_key"], 0) + 1
        uncertain = [r for r in attempts if r["phase"] == phase and r["status"] == "in_flight"]
        if uncertain and not self.auth.get("resume_uncertain_attempts"):
            raise StopAcquisition("WAITING_FOR_UNCERTAIN_BILLING_ACK: previous in-flight attempts retained in cost/attempts")
        self.authorized_uids = {r["window_uid"] for r in self.plan["windows"]}
        role_map = out / "enrollment/role_map.jsonl"
        self.source_groups = {r["window_uid"]: r["source_group"] for r in iter_jsonl(role_map)} if role_map.exists() else {}

    def begin(self, uid, key, stage):
        with self.lock:
            if self.stopped:
                raise StopAcquisition("provider stop flag is set")
            if uid not in self.authorized_uids:
                raise StopAcquisition("request outside authorized cohort")
            if self.key_attempts.get(key, 0) >= self.config["transport_attempts"]:
                raise RuntimeError("TRANSPORT_ATTEMPTS_EXHAUSTED_FOR_REQUEST: no automatic rebuy on resume")
            if self.used >= min(self.auth["max_physical_attempts"], self.plan["physical_attempts_upper_bound"]):
                raise StopAcquisition("BUDGET_EXHAUSTED")
            if (self.used + 1) * self.config["max_output_tokens"] > self.auth["max_output_tokens"]:
                raise StopAcquisition("OUTPUT_TOKEN_RESERVATION_EXHAUSTED")
            self.used += 1
            self.key_attempts[key] = self.key_attempts.get(key, 0) + 1
            attempt_id = semantic_sha256([self.phase, self.used, uid, key, time.time_ns()])
            path = self.root / (attempt_id + ".json")
            write_json(path, {"phase": self.phase, "window_uid": uid, "request_key": key, "stage": stage,
                              "status": "in_flight", "started_at": now(), "usage": None})
            if uid in self.source_groups:
                ledger = self.out / "exposure_ledger.jsonl"
                events = list(iter_jsonl(ledger)) if ledger.exists() else []
                if not any(r["window_uid"] == uid and r["phase"] == self.phase and r["kind"] == "request_started" for r in events):
                    events.append({"window_uid": uid, "source_group": self.source_groups[uid], "phase": self.phase,
                                   "kind": "request_started", "at": now()})
                    write_jsonl(ledger, events)
            return path


def dashscope_once(config, evidence, prompt):
    # Import/credential access happens only after an authorized physical attempt receipt.
    from dashscope import MultiModalConversation
    key = os.environ.get("DASHSCOPE_API_KEY")
    if not key:
        raise StopAcquisition("Missing DASHSCOPE_API_KEY")
    content = []
    for i, path in enumerate(evidence["image_paths"]):
        content.extend([{"text": f"T{i}"}, {"image": portable(path).resolve().as_uri(), "max_pixels": config["image_max_pixels"]}])
    content.append({"text": prompt})
    response = MultiModalConversation.call(api_key=key, model=config["model"],
               messages=[{"role": "user", "content": content}], temperature=config["temperature"],
               max_tokens=config["max_output_tokens"], vl_high_resolution_images=True)
    status = getattr(response, "status_code", None)
    if status != 200:
        code = str(getattr(response, "code", status))
        if any(t in code.lower() for t in ("inspection", "contentfilter", "inappropriate", "safety")):
            raise RequestRefused("PROVIDER_REFUSAL: " + code)
        if status in (401, 402, 403, 429) or any(t in code.lower() for t in ("balance", "quota", "rate", "arrear")):
            raise StopAcquisition("PROVIDER_ACCOUNT_OR_RATE_STOP: " + code)
        raise RuntimeError("PROVIDER_ERROR: " + code)
    blocks = response.output.choices[0].message.content
    raw = "\n".join(b["text"] for b in blocks if isinstance(b, dict) and "text" in b)
    usage = getattr(response, "usage", None)
    return raw, dict(usage) if usage is not None else None


class TrialRuntime:
    def __init__(self, out, uid, evidence, config, budget, provider=dashscope_once):
        self.out, self.uid, self.evidence, self.settings, self.budget = out, uid, evidence, config, budget
        self.provider = provider
        self.config = SimpleNamespace(allow_legacy_temporal_cache=False)
        self.keys = []
        self.parents = []

    def evidence_metadata(self, case):
        return self.evidence

    def request_json(self, *, case, prompt, namespace, mock_spec=None, legacy_prompt=None, validator=None, **unused):
        from temporal_contract import response_errors
        kind = (mock_spec or {}).get("kind", "")
        original = prompt
        for repair in range(self.settings["schema_repair_attempts"] + 1):
            identity = {"window_uid": self.uid, "stage": namespace, "model": self.settings["model"],
                        "provider_revision": self.settings["provider_revision"], "temperature": self.settings["temperature"],
                        "max_tokens": self.settings["max_output_tokens"], "max_pixels": self.settings["image_max_pixels"],
                        "images": self.evidence["image_sha256"], "frame_indices": self.evidence["frame_indices"],
                        "image_contract": self.evidence["image_contract"], "prompt": prompt,
                        "schema": kind or namespace, "parents": list(self.parents), "repair": repair}
            key = semantic_sha256(identity)
            self.keys.append(key)
            raw_path = self.out / "cache/raw" / (key + ".json")
            lock = raw_path.with_suffix(".lock")
            lock.parent.mkdir(parents=True, exist_ok=True)
            # Identical in-run C0/graph-node requests share one raw response.
            while True:
                try:
                    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                    os.close(fd)
                    break
                except FileExistsError:
                    raise StopAcquisition("REQUEST_LOCK_PRESENT: inspect stale lock " + str(lock))
            try:
                saved = read_json(raw_path)
                cache_hit = saved is not None
                if saved is not None and saved["identity"] != identity:
                    raise ValueError("Raw identity mismatch")
                if saved is not None:
                    receipt = portable(saved["receipt"])
                    info = read_json(receipt, {})
                    digest = stable_hash(raw_path)
                    if info.get("raw_file_sha256") not in (None, digest):
                        raise ValueError("RAW_CACHE_CHANGED")
                    if info.get("raw_file_sha256") is None:
                        if info.get("status") != "in_flight" or not self.budget.auth.get("resume_uncertain_attempts"):
                            raise StopAcquisition("RAW_WITHOUT_FINISHED_RECEIPT: acknowledge uncertain billing before recovery")
                        write_json(receipt, {**info, "status": "recovered_response_billing_unknown", "raw_file_sha256": digest})
                if saved is None:
                    if self.provider is dashscope_once:
                        if not os.environ.get("DASHSCOPE_API_KEY"):
                            raise StopAcquisition("Missing DASHSCOPE_API_KEY; no request started")
                        import dashscope  # noqa: F401 -- verify installed SDK before billing reservation
                    for attempt in range(self.settings["transport_attempts"]):
                        receipt = self.budget.begin(self.uid, key, namespace)
                        started = time.monotonic()
                        try:
                            raw, usage = self.provider(self.settings, self.evidence, prompt)
                            saved = {"identity": identity, "raw": raw, "usage": usage, "receipt": str(receipt)}
                            write_json(raw_path, saved)
                            write_json(receipt, {**read_json(receipt), "status": "success", "latency_seconds": time.monotonic() - started,
                                                "usage": usage, "raw_file_sha256": stable_hash(raw_path)})
                            break
                        except Exception as exc:
                            write_json(receipt, {**read_json(receipt), "status": "provider_failure", "latency_seconds": time.monotonic() - started,
                                                "error_type": type(exc).__name__, "error": str(exc)[:300], "billing_unknown": True})
                            if isinstance(exc, StopAcquisition):
                                self.budget.stopped = True
                                raise
                            if isinstance(exc, RequestRefused):
                                raise
                            if attempt + 1 == self.settings["transport_attempts"]:
                                raise
                            time.sleep(2 ** attempt)
                # Local parser version is separate from the raw request identity.
                try:
                    raw = saved["raw"].strip()
                    if raw.lower().startswith(("i cannot assist", "i can't assist", "i'm sorry", "i am sorry", "sorry,")) or raw.startswith(("\u62b1\u6b49", "\u5bf9\u4e0d\u8d77")):
                        raise RequestRefused("saved provider refusal; no format repair")
                    if raw.startswith("```"):
                        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
                    parsed, local_repaired = parse_local(raw)
                    if not isinstance(parsed, dict):
                        raise ValueError("Response is not a JSON object")
                    errors = response_errors(parsed, kind)
                    if errors:
                        raise ValueError("; ".join(errors))
                    if kind in ("independent", "joint"):
                        from ..binding import probability
                        nodes = [parsed] if kind == "independent" else list(parsed["nodes"].values())
                        for node in nodes:
                            probability(node.get("uncertainty"))
                            probability(node.get("null_probability"))
                    if validator:
                        validator(parsed)
                    write_json(self.out / "cache/parsed" / (semantic_sha256([key, "v912_parser2"]) + ".json"),
                               {"raw_sha256": stable_hash(raw_path), "parsed": parsed})
                    receipt_key = semantic_sha256([self.budget.phase, self.uid, key, time.time_ns()])
                    write_json(self.out / "cost/logical_requests" / (receipt_key + ".json"),
                               {"phase": self.budget.phase, "stage": namespace, "request_key": key, "cache_hit": cache_hit,
                                "parser_version": "v912_parser2", "local_parser_repair": local_repaired, "remote_schema_repair_index": repair})
                    return {"parsed": parsed, "raw": saved["raw"], "cache_hit": cache_hit, "cache_path": str(raw_path)}
                except (ValueError, TypeError, KeyError) as exc:
                    write_json(self.out / "cache/invalid" / (key + ".json"), {"reason": str(exc), "raw_path": str(raw_path)})
                    if repair == self.settings["schema_repair_attempts"]:
                        raise ValueError("SCHEMA_INVALID_AFTER_BOUNDED_REPAIR: " + str(exc)) from exc
                    prompt = original + "\nReturn the same visual assessment in valid schema. Fix only these format/reference errors: " + str(exc)[:1500]
            finally:
                lock.unlink(missing_ok=True)
        raise AssertionError("unreachable")


def match_core(runtime, case, catalog, config):
    """Reuse the existing scores, but retain M0 if later graph refinement is missing."""
    from live_matching import WindowMatcher, _union_nodes, _parse_independent, _unary, _evidence_layout, _parse_joint, _compete
    from prompts import independent_node_prompt, conditional_refinement_prompt
    import matching
    matcher = WindowMatcher(runtime, catalog, top_k_abnormal=config["top_k_abnormal"], top_k_normal=config["top_k_normal"])
    graphs, shortlist = matcher._shortlist(case)
    nodes = _union_nodes(graphs)
    independent, joint, errors = {}, {}, []
    methods = {m: {} for m in ("independent_direct_nodes", "conditional_rowmax", "conditional_ot_full")}
    for node in nodes:
        try:
            response = runtime.request_json(case=case, prompt=independent_node_prompt(node), namespace="independent/" + node.key,
                                            mock_spec={"kind": "independent", "node_key": node.key})
            independent[node.key] = _parse_independent(response, node.key)
        except StopAcquisition:
            raise
        except (ValueError, RuntimeError) as exc:
            errors.append({"stage": "independent/" + node.key, "reason": str(exc)[:500]})
    ids, centers = _evidence_layout(2)
    for graph in graphs:
        if not all(n.key in independent for n in graph.nodes):
            continue
        presence = {n.key: independent[n.key]["presence"] for n in graph.nodes}
        temporal = {n.key: float(independent[n.key].get("best_bin") or 0) / 7 for n in graph.nodes}
        methods["independent_direct_nodes"][graph.key] = matching.match_graph_independent_direct(graph, presence, temporal)
        unary = _unary(graph.nodes, independent, 2)
        initial = matching.match_graph_unary_ot(graph, unary, centers)
        prompt, mapping = conditional_refinement_prompt(graph, independent, initial.to_dict())
        runtime.parents = [semantic_sha256(independent[n.key].get("raw", "")) for n in graph.nodes]
        try:
            response = runtime.request_json(case=case, prompt=prompt, namespace="conditional_refinement/" + graph.key,
                     mock_spec={"kind": "joint", "graph_key": graph.key, "node_keys": graph.node_keys, "initial_presence": initial.node_presence})
            cond, trace = _parse_joint(response, graph, mapping, ids, 2)
            if not trace["complete"]:
                raise ValueError("Incomplete joint contract")
            joint[graph.key] = trace
            methods["conditional_rowmax"][graph.key] = matching.match_graph_conditional_rowmax(graph, unary, cond, centers)
            methods["conditional_ot_full"][graph.key] = matching.match_graph_conditional_ot(graph, unary, cond, centers,
                use_coherence=True, coherence_weight=config["coherence_weight"], method_name="conditional_ot_full")
        except StopAcquisition:
            raise
        except (ValueError, RuntimeError) as exc:
            errors.append({"stage": "conditional_refinement/" + graph.key, "reason": str(exc)[:500]})
        finally:
            runtime.parents = []
    competitions = {}
    for method, values in methods.items():
        if len(values) == len(graphs):
            competitions[method] = _compete(method, [values[g.key] for g in graphs if g.polarity == "abnormal"],
                [values[g.key] for g in graphs if g.polarity == "normal"], .03, "logmeanexp", config["competition_temperature"])
    return {"competitions": competitions, "independent_node_calls": independent, "joint_graph_calls": joint,
            "graph_candidates": shortlist, "errors": errors,
            "completeness": {"independent": len(independent) == len(nodes), "joint": len(joint) == len(graphs)}}


def finish_result(out, target, result, runtime):
    result["request_keys"] = list(dict.fromkeys(runtime.keys))
    result["raw_request_manifest"] = [{"path": str((out / "cache/raw" / (key + ".json")).resolve()),
        "sha256": stable_hash(out / "cache/raw" / (key + ".json"))} for key in result["request_keys"] if (out / "cache/raw" / (key + ".json")).exists()]
    write_json(target, result)
    return result


def collect_one(project, out, row, phase, config, budget):
    from graph_catalog import read_catalog_json
    from live_matching import WindowMatcher, _evidence_layout, _parse_independent, _unary, _result_dict
    from prompts import independent_node_prompt, crowd_event_state_v5_prompt
    from crowd_event_state_v4 import parse_event_state_v5_response
    from schemas import WindowCase
    import matching
    uid = row["window_uid"]
    target_role = "adaptation" if phase == "smoke" else phase
    target = out / "private_acquisition" / target_role / (uid + ".json")
    if target.exists():
        return read_json(target)
    media = read_json(out / "media/manifest.json")[uid]
    if any(stable_hash(portable(p)) != h for p, h in zip(media["image_paths"], media["image_sha256"])):
        raise StopAcquisition("MEDIA_HASH_CHANGED")
    runtime = TrialRuntime(out, uid, media, config, budget)
    case = WindowCase(uid, uid, "", 0, 95, None, {})
    result = {"window_uid": uid, "errors": [], "request_keys": [], "human_layers_loaded": False}
    catalog = read_catalog_json(project / config["graph_catalog"])
    independent = {}
    try:
        base = match_core(runtime, case, catalog, config)
        independent.update(base["independent_node_calls"])
        result["errors"].extend(base["errors"])
        result["baseline"] = {k: base[k] for k in ("competitions", "completeness", "independent_node_calls", "joint_graph_calls", "graph_candidates")}
    except StopAcquisition:
        raise
    except RequestRefused as exc:
        result["errors"].append({"stage": "graph", "reason": str(exc), "terminal_refusal": True})
        return finish_result(out, target, result, runtime)
    except Exception as exc:
        result["errors"].append({"stage": "graph", "type": type(exc).__name__, "reason": str(exc)[:500]})
    try:
        phases = read_catalog_json(project / config["phase_catalog"])
        active = phases["crowd_active_or_ongoing_physical_escalation_v4"]
        after = phases["crowd_causally_linked_aftermath_v4"]
        for graph in (active, after):
            for node in graph.nodes:
                if node.key not in independent:
                    response = runtime.request_json(case=case, prompt=independent_node_prompt(node),
                        namespace="independent/" + node.key, mock_spec={"kind": "independent"})
                    independent[node.key] = _parse_independent(response, node.key)
        ids, centers = _evidence_layout(2)
        initial = {g.key: _result_dict(matching.match_graph_unary_ot(g, _unary(g.nodes, independent, 2), centers)) for g in (active, after)}
        prompt, mappings = crowd_event_state_v5_prompt(active, after, independent, initial, media, {"available": False, "reason": "disabled_frozen_v912"})
        runtime.parents = [semantic_sha256(independent[key].get("raw", "")) for key in sorted({n.key for g in (active, after) for n in g.nodes})]
        def validate_c0(value):
            from temporal_contract import node_errors
            from crowd_event_state_v4 import V5_CONTEXT_STATES, V5_NORMAL_CONFOUND_STATES
            from ..binding import probability
            for field in ("current_window_active_occupancy_probability", "uncertainty", "current_window_normal_confound_probability"):
                probability(value.get(field))
            for field, names in (("event_context_state_probabilities", V5_CONTEXT_STATES), ("normal_confound_probabilities", V5_NORMAL_CONFOUND_STATES)):
                if not isinstance(value.get(field), dict):
                    raise ValueError("missing distribution " + field)
                if sum(probability(value[field].get(name)) for name in names) <= 0:
                    raise ValueError("zero mass distribution " + field)
            for phase_name, mapping in mappings.items():
                for node_id in mapping:
                    errors = node_errors(value.get("phase_nodes", {}).get(phase_name, {}).get(node_id), phase_name + "/" + node_id)
                    if errors:
                        raise ValueError("; ".join(errors))
        response = runtime.request_json(case=case, prompt=prompt, namespace="C0", mock_spec={"kind": "event_state_v5",
            "active_ids": list(mappings["active"]), "aftermath_ids": list(mappings["aftermath"])}, validator=validate_c0)
        state, _ = parse_event_state_v5_response(response, active, after, mappings, ids, 2)
        result["C0"] = state
    except StopAcquisition:
        raise
    except RequestRefused as exc:
        result["errors"].append({"stage": "C0", "reason": str(exc), "terminal_refusal": True})
        return finish_result(out, target, result, runtime)
    except Exception as exc:
        result["errors"].append({"stage": "C0", "type": type(exc).__name__, "reason": str(exc)[:500]})
    try:
        runtime.parents = []
        signature = semantic_sha256(media["image_sha256"])
        response = runtime.request_json(case=case, prompt=proposal_prompt(uid, signature), namespace="C1",
                    validator=lambda p: validate_proposal(p, window_id=uid, evidence_signature=signature))
        c1 = result["C1"] = response["parsed"]
        if c1_quality(c1) is not None and not c1["events"]:
            result["C2_skipped"] = "complete_empty_C1"
        else:
            runtime.parents = [semantic_sha256(c1)]
            response = runtime.request_json(case=case, prompt=binding_prompt(c1), namespace="C2",
                                           validator=lambda p: validate_binding(c1, p))
            result["C2"] = response["parsed"]
    except StopAcquisition:
        raise
    except RequestRefused as exc:
        result["errors"].append({"stage": "C1_or_C2", "reason": str(exc), "terminal_refusal": True})
        return finish_result(out, target, result, runtime)
    except Exception as exc:
        result["errors"].append({"stage": "C1_or_C2", "type": type(exc).__name__, "reason": str(exc)[:500]})
    return finish_result(out, target, result, runtime)


def collect(project, out, phase, config):
    if phase != "smoke":
        verify_frozen(out)
    if phase == "locked_evaluation" and not at_least(out, "MODELS_FROZEN"):
        raise ValueError("Freeze models before locked acquisition")
    budget = Budget(out, phase, config)
    rows = role_rows(out, phase)
    # Source exposure persists even after interrupted/failed requests.
    ledger = out / "exposure_ledger.jsonl"
    old = list(iter_jsonl(ledger)) if ledger.exists() else []
    existing = {(r["window_uid"], r["phase"]) for r in old}
    old.extend({"window_uid": r["window_uid"], "source_group": r["source_group"], "phase": phase,
                "kind": "reserved_for_authorized_acquisition", "at": now()} for r in rows if (r["window_uid"], phase) not in existing)
    write_jsonl(ledger, old)
    with ThreadPoolExecutor(max_workers=config["workers"]) as pool:
        futures = [pool.submit(collect_one, project, out, r, phase, config, budget) for r in rows]
        for i, future in enumerate(futures):
            try:
                future.result()
            except BaseException:
                budget.stopped = True
                for f in futures:
                    f.cancel()
                raise
            print(f"[collect/{phase}] {i + 1}/{len(rows)} stored; values hidden until unblind" if phase == "locked_evaluation" else
                  f"[collect/{phase}] {i + 1}/{len(rows)} stored", flush=True)
    if phase == "smoke":
        from .features import feature_record
        features = [feature_record(r["window_uid"], read_json(out / "private_acquisition/adaptation" / (r["window_uid"] + ".json"))) for r in rows]
        summary = {"windows": len(rows), "observed_counts": {k: sum(r["values"][k] is not None for r in features) for k in features[0]["values"]},
                   "semantic_accuracy": None, "developer_schema_repair_cycles_remaining": 1,
                   "same_evidence_semantic_retry_allowed": False}
        summary["core_ready_for_lock"] = all(summary["observed_counts"][k] == len(rows) for k in ("m0_margin", "o_active", "q_direct"))
        summary["binding_missing_is_not_negative"] = True
        immutable(out / "smoke/summary.json", summary)
        advance(out, "DEV_SMOKE_COMPLETE")
    return {"phase": phase, "completed_windows": len(rows), "physical_attempts_this_phase_total": budget.used}
