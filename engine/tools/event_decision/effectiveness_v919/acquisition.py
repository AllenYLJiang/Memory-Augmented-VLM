"""One hash-bound evidence DAG per window, no GT in prompts, no semantic retries."""
from __future__ import annotations

from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import math
import random
import threading
import time

from ..contracts import file_sha256, read_json, semantic_sha256, write_json
from ..b1b4_trial.protocol import immutable, now
from ..binding import proposal_prompt, validate_proposal
from ..contrast_c2_v917 import prompt_for as contrast_prompt, validate as validate_contrast
from .scoring import BASES


class BudgetStop(RuntimeError):
    pass


class TerminalRefusal(Exception):
    """A recorded provider refusal: missing evidence, never a negative prediction."""
    def __init__(self, key, stage):
        self.request_key, self.stage = key, stage
        super().__init__("Provider refused request; terminal missing evidence: " + key)


def refusal_attempt(attempt):
    return (attempt.get("status") in {"provider_failed", "provider_refused", "transport_failed"} and
            attempt.get("error_type") == "RequestRefused" and not attempt.get("retryable_transport", False))


def read_refusal(out, key, identity=None):
    path = out / "terminal_refusals" / (key + ".json")
    envelope = read_json(path)
    if not envelope:
        raise ValueError("Missing terminal refusal receipt: " + key)
    body = envelope.get("body", {})
    if (body.get("version") != "v919_terminal_refusal_1" or semantic_sha256(body) != envelope.get("sha256") or
            body.get("request_key") != key or semantic_sha256(body.get("identity")) != key or
            body.get("outcome") != "provider_refused" or body.get("retryable") is not False):
        raise ValueError("Terminal refusal identity/hash mismatch: " + key)
    if identity is not None and body["identity"] != identity:
        raise ValueError("Refusal belongs to another exact request: " + key)
    if (out / "cache" / (key + ".json")).exists():
        raise ValueError("Both response and refusal exist for one request: " + key)
    if not body.get("source_attempts"):
        raise ValueError("Refusal has no source attempt: " + key)
    for source in body["source_attempts"]:
        rel = Path(source["path"])
        if (rel.is_absolute() or len(rel.parts) != 4 or rel.parts[0] not in ("pilot", "dense") or
                rel.parts[1:3] != ("cost", "attempts") or not rel.stem.isdigit() or rel.suffix != ".json"):
            raise ValueError("Invalid refusal attempt path")
        if file_sha256(out / rel) != source["sha256"]:
            raise ValueError("Refusal source attempt changed: " + key)
        attempt = read_json(out / rel)
        if (not refusal_attempt(attempt) or attempt["request_key"] != key or
                attempt["stage"] != body["identity"]["stage"] or rel.name != f"{attempt['index']:08d}.json"):
            raise ValueError("Refusal source is not an explicit matching provider refusal")
    return body


def verify_refusals(out, result):
    for key, digest in result.get("refusal_sha256", {}).items():
        if file_sha256(out / "terminal_refusals" / (key + ".json")) != digest:
            raise ValueError("Window refusal receipt changed: " + key)
        body = read_refusal(out, key)
        if (body["identity"]["window_uid"] != result["window_uid"] or
                semantic_sha256(body["identity"]["configuration"]) != result["config_sha256"]):
            raise ValueError("Window/refusal configuration mismatch")


def transient_transport(exc):
    from requests.exceptions import ConnectionError as RequestsConnectionError, SSLError, Timeout
    return not isinstance(exc, SSLError) and isinstance(exc, (Timeout, RequestsConnectionError, TimeoutError, ConnectionError))


def retryable_receipt(attempt):
    # Legacy V919 used transport_failed for *all* exceptions; only known network types qualify.
    if attempt.get("status") != "transport_failed":
        return False
    if "retryable_transport" in attempt:
        return attempt["retryable_transport"] is True
    return attempt.get("error_type") in {"ReadTimeout", "ConnectTimeout", "Timeout", "ConnectionError", "ProxyError", "TimeoutError"}


def strict_json(raw):
    def pairs(items):
        value = {}
        for k, v in items:
            if k in value:
                raise ValueError("Duplicate JSON key: " + k)
            value[k] = v
        return value
    def bad_constant(v):
        raise ValueError("Nonfinite JSON constant: " + v)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=bad_constant)


def prepare_video_media(out, rows):
    """Sequentially decode each video once; cache selected frames shared by its windows."""
    import cv2
    path = Path(rows[0]["video_path"])
    if any(r["video_path"] != str(path) for r in rows):
        raise ValueError("Mixed video media task")
    if path.stat().st_size != rows[0]["video_size"] or path.stat().st_mtime_ns != rows[0]["video_mtime_ns"]:
        raise ValueError("Video changed since manifest freeze")
    folder = out / "media" / semantic_sha256(rows[0]["video_id"])[:24]
    source_hash = file_sha256(path)
    indices = sorted({i for r in rows for i in r["sampled_frame_indices"]})
    receipt = folder / "receipt.json"
    old = read_json(receipt)
    if old:
        if old["source_sha256"] != source_hash or old["indices"] != indices or any(file_sha256(folder / k) != v for k, v in old["files"].items()):
            raise ValueError("Frozen image evidence changed")
    else:
        folder.mkdir(parents=True, exist_ok=True)
        cap, got = cv2.VideoCapture(str(path)), set()
        wanted = set(indices)
        try:
            if not cap.isOpened():
                raise ValueError("Video cannot be opened")
            for i in range(indices[-1] + 1):
                ok, frame = cap.read()
                if not ok:
                    raise ValueError("Premature end of video at frame " + str(i))
                if i in wanted:
                    target = folder / f"frame_{i:09d}.jpg"
                    if not cv2.imwrite(str(target), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                        raise ValueError("Image write failed")
                    got.add(i)
        finally:
            cap.release()
        if got != wanted:
            raise ValueError("Incomplete exact-frame extraction")
        old = {"source_sha256": source_hash, "indices": indices, "decoder": "opencv_sequential_no_seek_jpeg95",
               "files": {f"frame_{i:09d}.jpg": file_sha256(folder / f"frame_{i:09d}.jpg") for i in indices}}
        immutable(receipt, old)
    return {r["window_uid"]: {"image_paths": [str((folder / f"frame_{i:09d}.jpg").resolve()) for i in r["sampled_frame_indices"]],
        "image_sha256": [old["files"][f"frame_{i:09d}.jpg"] for i in r["sampled_frame_indices"]],
        "frame_indices": r["sampled_frame_indices"], "evidence_mode": "frames"} for r in rows}


class Budget:
    def __init__(self, out, phase, plan_hash, config, approved_by, max_attempts, max_output_tokens,
                 retry_transport=False, transport_max_attempts=3, retry_base_seconds=10., retry_jitter_seconds=3.):
        if not approved_by or max_attempts <= 0 or max_output_tokens < config["max_output_tokens"]:
            raise ValueError("Explicit reviewer name and positive physical/output-token caps required")
        if not 1 <= transport_max_attempts <= 5 or not math.isfinite(retry_base_seconds) or not 0 <= retry_base_seconds <= 300:
            raise ValueError("Transport attempts must be 1..5; base delay must be finite and 0..300 seconds")
        if not math.isfinite(retry_jitter_seconds) or not 0 <= retry_jitter_seconds <= 30:
            raise ValueError("Retry jitter must be finite and 0..30 seconds")
        self.out, self.phase, self.config = out, phase, config
        self.lock, self.stopped = threading.Lock(), False
        self.stop_event, self.stop_reason = threading.Event(), None
        self.folder = out / phase / "cost/attempts"
        self.folder.mkdir(parents=True, exist_ok=True)
        self.attempts = [read_json(p) for p in sorted(self.folder.glob("*.json"))]
        if [a.get("index") for a in self.attempts] != list(range(1, len(self.attempts) + 1)):
            raise BudgetStop("Attempt ledger has missing/duplicate indices; inspect before resuming")
        for a in self.attempts:
            if a["status"] == "in_flight" and not (out / "cache" / (a["request_key"] + ".json")).exists():
                raise BudgetStop("Unresolved in-flight receipt; inspect provider billing before any retry: " + a["request_key"])
        self.max_attempts, self.max_output_tokens = max_attempts, max_output_tokens
        self.retry_transport = retry_transport
        self.transport_max_attempts = transport_max_attempts if retry_transport else 1
        self.retry_base_seconds, self.retry_jitter_seconds = retry_base_seconds, retry_jitter_seconds
        self.reservation = config["max_output_tokens"]
        auth = {"plan_sha256": plan_hash, "phase": phase, "approved_by": approved_by, "max_attempts": max_attempts,
                "max_reserved_output_tokens": max_output_tokens, "semantic_retries": 0,
                "retry_confirmed_transport_failures": retry_transport,
                "transport_policy": "v919_bounded_network_retry_1",
                "max_attempts_per_request_lifetime": self.transport_max_attempts,
                "backoff_base_seconds": retry_base_seconds, "backoff_multiplier": 3,
                "backoff_cap_seconds": 300, "jitter_seconds": retry_jitter_seconds,
                "retry_billing_may_be_unknown": True}
        immutable(out / phase / "authorizations" / (semantic_sha256(auth) + ".json"), auth)
        self.authorization_sha256 = semantic_sha256(auth)

    def _stop_locked(self, code, message, **details):
        if not self.stopped:
            self.stopped = True
            self.stop_reason = {"code": code, "message": message, "phase": self.phase, "at": now(), **details}
            self.stop_event.set()
            try:
                write_json(self.out / self.phase / "last_pause.json", self.stop_reason)
            except OSError:
                print("[pause] Could not persist pause receipt; inspect local storage before resuming", flush=True)
        return (self.stop_reason or {}).get("message", "Acquisition stopped")

    def stop(self, code, message, **details):
        with self.lock:
            return self._stop_locked(code, message, **details)

    def check_running(self):
        with self.lock:
            if self.stopped:
                raise BudgetStop((self.stop_reason or {}).get("message", "Acquisition stopped"))

    def prior(self, key):
        with self.lock:
            return [dict(a) for a in self.attempts if a["request_key"] == key]

    def finish(self, attempt, path, **details):
        with self.lock:
            attempt.update(details)
            try:
                write_json(path, attempt)
            except Exception as exc:
                raise BudgetStop(self._stop_locked("attempt_persistence_failure", "Attempt receipt persistence failed; inspect local storage before resuming", request_key=attempt["request_key"])) from exc

    def backoff(self, key, failed_count):
        delay = min(300., self.retry_base_seconds * 3 ** (failed_count - 1)) + random.uniform(0., self.retry_jitter_seconds)
        print(f"[retry] network failure key={key[:12]} attempt={failed_count}/{self.transport_max_attempts}; wait={delay:.1f}s", flush=True)
        self.stop_event.wait(delay)
        self.check_running()

    def reserve(self, key, stage):
        with self.lock:
            if self.stopped:
                raise BudgetStop((self.stop_reason or {}).get("message", "Acquisition stopped"))
            if len(self.attempts) >= self.max_attempts or (len(self.attempts) + 1) * self.reservation > self.max_output_tokens:
                raise BudgetStop(self._stop_locked("budget_exhausted", "Physical attempt/output reservation cap reached; resume only with explicitly reviewed caps"))
            prior = [a for a in self.attempts if a["request_key"] == key]
            if prior and (not self.retry_transport or not all(retryable_receipt(a) for a in prior)):
                raise BudgetStop(self._stop_locked("request_not_retryable", "Existing terminal/uncertain request cannot be retried: " + key))
            if len(prior) >= self.transport_max_attempts:
                raise BudgetStop(self._stop_locked("transport_attempts_exhausted", "Network attempts exhausted for request " + key + "; inspect service/network before any new authorization", request_key=key))
            a = {"index": len(self.attempts) + 1, "request_key": key, "stage": stage, "status": "in_flight", "at": now(),
                 "request_attempt_number": len(prior) + 1, "authorization_sha256": self.authorization_sha256}
            self.attempts.append(a)
            p = self.folder / f"{a['index']:08d}.json"
            try:
                write_json(p, a)
            except Exception as exc:
                raise BudgetStop(self._stop_locked("reservation_persistence_failure", "Cannot save request reservation; no provider call was made", request_key=key)) from exc
            return a, p


class Runtime:
    def __init__(self, out, uid, media, config, budget, provider=None):
        self.out, self.uid, self.media, self.config, self.budget = out, uid, media, config, budget
        if provider is None:
            from ..b1b4_trial.evidence import dashscope_once
            provider = dashscope_once
        self.provider, self.parents, self.keys = provider, [], []
        self.refusals = {}

    def terminal_refusal(self, key, identity):
        path = self.out / "terminal_refusals" / (key + ".json")
        try:
            if not path.exists():
                prior = [a for a in self.budget.prior(key) if refusal_attempt(a)]
                if not prior:
                    raise ValueError("No refusal attempt to preserve")
                body = {"version": "v919_terminal_refusal_1", "request_key": key, "identity": identity,
                        "outcome": "provider_refused", "retryable": False, "billing_unknown": True,
                        "source_attempts": [{"path": f"{self.budget.phase}/cost/attempts/{a['index']:08d}.json",
                                             "sha256": file_sha256(self.budget.folder / f"{a['index']:08d}.json")} for a in prior]}
                immutable(path, {"body": body, "sha256": semantic_sha256(body)})
            read_refusal(self.out, key, identity)
            self.refusals[key] = file_sha256(path)
        except Exception as exc:
            raise BudgetStop(self.budget.stop("refusal_receipt_failure", "Cannot verify/persist terminal refusal; inspect local evidence", request_key=key)) from exc
        print(f"[refused] window={self.uid[:12]} stage={identity['stage']}; retained as missing; no retry", flush=True)
        raise TerminalRefusal(key, identity["stage"])

    def request_json(self, *, prompt, namespace, validator=None, **unused):
        if namespace == "candidate_selector_repair":
            raise ValueError("No automatic selector/schema repair in frozen V919")
        identity = {"contract": "v919_exact_request_1", "window_uid": self.uid, "stage": namespace,
                    "images": self.media["image_sha256"], "frame_indices": self.media["frame_indices"],
                    "prompt": prompt, "parents": self.parents, "configuration": self.config}
        key = semantic_sha256(identity)
        self.keys.append(key)
        path = self.out / "cache" / (key + ".json")
        envelope = read_json(path)
        cache_hit = envelope is not None
        if ((self.out / "terminal_refusals" / (key + ".json")).exists() or
                any(refusal_attempt(a) for a in self.budget.prior(key))):
            self.terminal_refusal(key, identity)
        if envelope is None:
            while envelope is None:
                prior = self.budget.prior(key)
                self.budget.check_running()
                if prior and self.budget.retry_transport and all(retryable_receipt(a) for a in prior) and len(prior) < self.budget.transport_max_attempts:
                    self.budget.backoff(key, len(prior))
                attempt, ap = self.budget.reserve(key, namespace)
                started = time.monotonic()
                from ..b1b4_trial.evidence import RequestRefused
                try:
                    raw, usage = self.provider(self.config, self.media, prompt)
                except RequestRefused:
                    self.budget.finish(attempt, ap, status="provider_refused", elapsed_seconds=time.monotonic() - started,
                        error_type="RequestRefused", retryable_transport=False, billing_unknown=True)
                    self.terminal_refusal(key, identity)
                except Exception as exc:
                    retryable = transient_transport(exc)
                    self.budget.finish(attempt, ap, status="transport_failed" if retryable else "provider_failed",
                        elapsed_seconds=time.monotonic() - started, error_type=type(exc).__name__,
                        retryable_transport=retryable, billing_unknown=True)
                    if retryable and self.budget.retry_transport and attempt["request_attempt_number"] < self.budget.transport_max_attempts:
                        continue
                    code = "transport_attempts_exhausted" if retryable else "nonretryable_provider_failure"
                    raise BudgetStop(self.budget.stop(code, f"Provider stopped: {type(exc).__name__}; request attempts={attempt['request_attempt_number']}/{self.budget.transport_max_attempts}; billing may be unknown",
                        request_key=key, error_type=type(exc).__name__)) from exc
                # A local persistence error after a response must never trigger another paid call.
                try:
                    body = {"identity": identity, "raw": raw, "usage": usage, "latency_seconds": time.monotonic() - started}
                    envelope = {"body": body, "sha256": semantic_sha256(body)}
                    write_json(path, envelope)
                    self.budget.finish(attempt, ap, status="response_received", elapsed_seconds=body["latency_seconds"], usage=usage)
                except Exception as exc:
                    raise BudgetStop(self.budget.stop("response_persistence_failure", "Response persistence failed; inspect cache/receipt before resuming", request_key=key, error_type=type(exc).__name__)) from exc
        body = envelope["body"]
        if semantic_sha256(body) != envelope["sha256"] or body["identity"] != identity:
            raise ValueError("Raw cache identity/hash mismatch")
        raw = body["raw"].strip()
        if raw.startswith("```json") and raw.endswith("```"):
            raw = raw[7:-3].strip()
        value = strict_json(raw)
        if not isinstance(value, dict):
            raise ValueError("JSON root must be an object")
        if namespace == "candidate_selector":
            for field in ("abnormal_candidates", "normal_candidates"):
                candidates = value.get(field)
                if not isinstance(candidates, list) or not candidates:
                    raise ValueError("Missing candidate list")
                seen = set()
                for c in candidates:
                    v = c.get("visual_support") if isinstance(c, dict) else None
                    if type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 or type(c.get("id")) is not str or c["id"] in seen:
                        raise ValueError("Invalid selector identity/probability")
                    seen.add(c["id"])
        if validator:
            validator(value)
        return {"parsed": value, "raw": body["raw"], "cache_path": str(path), "cache_hit": cache_hit,
                "evidence": {"image_sha256": self.media["image_sha256"]}}


def match_graphs(runtime, case, catalog, config):
    import matching
    from live_matching import WindowMatcher, _union_nodes, _parse_independent, _evidence_layout, _unary, _parse_joint, _compete, _result_dict
    from prompts import independent_node_prompt, conditional_refinement_prompt
    matcher = WindowMatcher(runtime, catalog, top_k_abnormal=config["top_k_abnormal"], top_k_normal=config["top_k_normal"])
    graphs, shortlist = matcher._shortlist(case)
    independent, joint, errors = {}, {}, []
    for node in _union_nodes(graphs):
        key = node.key
        try:
            response = runtime.request_json(case=case, prompt=independent_node_prompt(node), namespace="independent/" + key,
                                            mock_spec={"kind": "independent", "node_key": key})
            independent[key] = _parse_independent(response, key)
        except (BudgetStop, TerminalRefusal):
            raise
        except Exception as exc:
            errors.append({"stage": "independent/" + key, "error": str(exc)[:400]})
    ids, centers = _evidence_layout(2)
    values = {k: {} for k in BASES}
    for g in graphs:
        if not all(n.key in independent for n in g.nodes):
            continue
        unary = _unary(g.nodes, independent, 2)
        initial = matching.match_graph_unary_ot(g, unary, centers)
        values["independent_direct_nodes"][g.key] = matching.match_graph_independent_direct(g, {k: v["presence"] for k, v in independent.items()}, {k: v["best_bin"] for k, v in independent.items()})
        values["shared_unary_rowmax"][g.key] = matching.match_graph_shared_rowmax(g, unary, centers)
        values["unary_ot"][g.key] = initial
        prompt, mapping = conditional_refinement_prompt(g, independent, initial.to_dict())
        runtime.parents = [semantic_sha256(independent[n.key]["raw"]) for n in g.nodes]
        try:
            response = runtime.request_json(case=case, prompt=prompt, namespace="conditional/" + g.key,
                                            mock_spec={"kind": "joint", "node_keys": g.node_keys, "graph_key": g.key})
            cond, trace = _parse_joint(response, g, mapping, ids, 2)
            if not trace["complete"]:
                raise ValueError("Incomplete joint response")
            joint[g.key] = trace
            values["conditional_rowmax"][g.key] = matching.match_graph_conditional_rowmax(g, unary, cond, centers)
            for name, coherence in (("conditional_ot_no_coherence", False), ("conditional_ot_full", True)):
                values[name][g.key] = matching.match_graph_conditional_ot(g, unary, cond, centers, use_coherence=coherence,
                    coherence_weight=config["coherence_weight"], method_name=name)
        except (BudgetStop, TerminalRefusal):
            raise
        except Exception as exc:
            errors.append({"stage": "conditional/" + g.key, "error": str(exc)[:400]})
        finally:
            runtime.parents = []
    comps = {k: _compete(k, [v[g.key] for g in graphs if g.polarity == "abnormal"], [v[g.key] for g in graphs if g.polarity == "normal"],
                       .03, "logmeanexp", config["competition_temperature"]) for k, v in values.items() if len(v) == len(graphs)}
    return {"competitions": comps, "graph_scores": {k: {g: _result_dict(s) for g, s in v.items()} for k, v in values.items()},
            "independent_node_calls": independent, "joint_graph_calls": joint, "graph_candidates": shortlist, "errors": errors}


def collect_window(out, phase, row, media, catalog, config, budget, provider=None):
    from schemas import WindowCase
    uid = row["window_uid"]
    target = out / phase / "results" / (uid + ".json")
    result = read_json(target)
    if result is not None:
        if result["input_sha256"] != semantic_sha256(row) or result["config_sha256"] != semantic_sha256(config):
            raise ValueError("Completed window identity changed")
        for k, h in result["cache_sha256"].items():
            if file_sha256(out / "cache" / (k + ".json")) != h:
                raise ValueError("Completed evidence cache changed")
        verify_refusals(out, result)
        return result
    runtime = Runtime(out, uid, media, config, budget, provider)
    # Anonymous identity only; no labels, titles, source groups, or absolute frame indices in prompts.
    case = WindowCase(uid, uid, "", 0, row["end_frame_exclusive"] - row["start_frame"] - 1, None, {})
    result = {"window_uid": uid, "input_sha256": semantic_sha256(row), "config_sha256": semantic_sha256(config), "errors": []}
    graph_refused = False
    try:
        result["baseline"] = match_graphs(runtime, case, catalog, config)
    except BudgetStop:
        raise
    except TerminalRefusal as exc:
        graph_refused = True
        result["errors"].append({"stage": exc.stage, "error_type": "RequestRefused", "request_key": exc.request_key,
                                 "outcome": "missing_evidence"})
        result["C1_status"] = result["C2_status"] = "not_requested_after_graph_refusal"
    except Exception as exc:
        result["errors"].append({"stage": "graphs", "error": str(exc)[:400]})
    if not graph_refused:
        try:
            signature = semantic_sha256(media["image_sha256"])
            response = runtime.request_json(prompt=proposal_prompt(uid, signature), namespace="C1",
                validator=lambda p: validate_proposal(p, window_id=uid, evidence_signature=signature))
            c1 = result["C1"] = response["parsed"]
            if c1["scan_complete"] and not c1["overflow"] and c1["observation_sufficient"] and not c1["events"]:
                result["C2_skipped"] = "complete_empty_C1"
            else:
                runtime.parents = [semantic_sha256(c1)]
                prompt = ("Scope clarification: harmful describes visible physical harm/danger, not criminal intent. "
                          "An accidental fire or self-inflicted injury is not a benign mechanism merely because intent is unknown. "
                          "If identity is unclear, retain visible action but do not invent a person or purpose. "
                          "Unknown intent must not erase observed contact/fire. Unknown is allowed.\n" + contrast_prompt(c1))
                response = runtime.request_json(prompt=prompt, namespace="C2_scoped_v919", validator=lambda p: validate_contrast(c1, p))
                result["C2"] = response["parsed"]
        except BudgetStop:
            raise
        except TerminalRefusal as exc:
            result["errors"].append({"stage": exc.stage, "error_type": "RequestRefused", "request_key": exc.request_key,
                                     "outcome": "missing_evidence"})
            if exc.stage == "C1":
                result["C1_status"], result["C2_status"] = "provider_refused", "not_requested_parent_refused"
            else:
                result["C2_status"] = "provider_refused"
        except Exception as exc:
            result["errors"].append({"stage": "C1_C2", "error": str(exc)[:400]})
    result["request_keys"] = runtime.keys
    result["cache_sha256"] = {k: file_sha256(out / "cache" / (k + ".json")) for k in runtime.keys if (out / "cache" / (k + ".json")).exists()}
    if runtime.refusals:
        result["refusal_sha256"] = runtime.refusals
        result["acquisition_status"] = "terminal_with_missing_evidence"
        verify_refusals(out, result)
    result["record_sha256"] = semantic_sha256(result)
    write_json(target, result)
    return result


def collect(out, phase, rows, catalog, config, budget):
    videos = defaultdict(list)
    for row in rows:
        videos[row["video_id"]].append(row)
    def one_video(rs):
        budget.check_running()
        media = prepare_video_media(out, rs)
        for row in sorted(rs, key=lambda r: r["start_frame"]):
            budget.check_running()
            reused = (out / phase / "results" / (row["window_uid"] + ".json")).exists()
            collect_window(out, phase, row, media[row["window_uid"]], catalog, config, budget)
            action = "cached" if reused else "saved"
            print(f"[{phase}] {row['video_id']} [{row['start_frame']},{row['end_frame_exclusive']}) {action}", flush=True)
    # Bound the queue by videos; each worker completes one video in temporal order.
    with ThreadPoolExecutor(max_workers=config["workers"]) as pool:
        futures = [pool.submit(one_video, rs) for rs in videos.values()]
        try:
            for f in futures:
                f.result()
        except BaseException as exc:
            budget.stop("collector_stopped", "Collector stopped: " + type(exc).__name__)
            for f in futures:
                f.cancel()
            raise


def cost_summary(out, phase):
    import numpy as np
    attempts = [read_json(p) for p in sorted((out / phase / "cost/attempts").glob("*.json"))]
    tokens = Counter()
    for a in attempts:
        for k, v in (a.get("usage") or {}).items():
            if isinstance(v, (int, float)):
                tokens[k] += v
    latencies = [a["elapsed_seconds"] for a in attempts if "elapsed_seconds" in a]
    return {"physical_attempts": len(attempts), "statuses": dict(Counter(a["status"] for a in attempts)),
            "unique_request_keys": len({a["request_key"] for a in attempts}),
            "additional_attempts_same_key": len(attempts) - len({a["request_key"] for a in attempts}),
            "failure_types": dict(Counter(a["error_type"] for a in attempts if a.get("error_type"))),
            "provider_refusal_attempts": sum(refusal_attempt(a) for a in attempts),
            "refused_request_keys": sorted({a["request_key"] for a in attempts if refusal_attempt(a)}),
            "last_pause_historical": read_json(out / phase / "last_pause.json"),
            "stage_calls": dict(Counter(a["stage"].split("/")[0] for a in attempts)),
            "known_usage": dict(tokens), "unknown_usage_attempts": sum(not a.get("usage") for a in attempts),
            "sum_request_latency_seconds": sum(latencies), "mean_request_latency_seconds": sum(latencies) / len(latencies) if latencies else None,
            "request_latency_p50_p95_seconds": np.quantile(latencies, [.5, .95]).tolist() if latencies else None,
            "new_DeepSeek_calls": 0, "not_a_currency_bill": True}
