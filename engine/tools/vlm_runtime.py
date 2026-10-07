#!/usr/bin/env python3
"""Cached DashScope calls with deterministic eight-frame evidence locking.

All candidate selection, independent-node, conditional-refinement and verifier calls for a
window can see the exact same T0..T7 images.  ``evidence_mode=video`` remains available for
backward compatibility.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from common import read_json, stable_sha1, write_json
from cache_recovery import archive_file
from temporal_contract import CONTRACT_VERSION, response_errors
from schemas import WindowCase


@dataclass(frozen=True)
class RuntimeConfig:
    code_dir: Path
    cache_dir: Path
    model: str = "qwen3.6-plus"
    fps: float = 2.0
    key_env: str = "DASHSCOPE_API_KEY"
    mock: bool = False
    json_retries: int = 3
    evidence_mode: str = "frames"          # frames | paired16 | video
    evidence_frames: int = 8
    pair_gap: int = 2
    image_jpeg_quality: int = 92
    cache_salt: str = ""
    allow_legacy_temporal_cache: bool = False
    retry_backoff_seconds: float = 2.0


class CachedVideoVLM:
    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        if config.evidence_mode not in {"frames", "paired16", "video"}:
            raise ValueError("evidence_mode must be frames, paired16, or video")
        if config.evidence_frames != 8:
            raise ValueError("temporal scoring requires exactly 8 bins (paired16 also has 8 bins)")
        self._vlm = None
        self._segment_cache = None
        self._ffprobe_meta = None
        self._parse_json_like = None
        if not config.mock:
            src = config.code_dir / "src"
            if str(src) not in sys.path:
                sys.path.insert(0, str(src))
            from structural_vlm_binary.utils.video import SegmentCache, ffprobe_meta
            from structural_vlm_binary.vlm.dashscope_backend import DashScopeVLM, DashScopeVLMConfig, parse_json_like
            self._segment_cache = SegmentCache(config.cache_dir / "segments")
            self._ffprobe_meta = ffprobe_meta
            self._parse_json_like = parse_json_like
            self._vlm = DashScopeVLM(DashScopeVLMConfig(
                model=config.model,
                fps=float(config.fps),
                api_key_env=config.key_env,
                vl_high_resolution_images=True,
                max_retries=6,
                json_parse_retries=1,
            ))

    @staticmethod
    def _provider_error(value: Mapping[str, Any]) -> Optional[str]:
        """Recognize provider error envelopes that are syntactically valid JSON."""
        status = value.get("status_code")
        try:
            failed_status = int(status) >= 400
        except (TypeError, ValueError):
            failed_status = False
        code = str(value.get("code", "") or "")
        message = str(value.get("message", "") or "")
        if failed_status or (code and value.get("output") is None and message):
            return f"status={status} code={code or 'unknown'} message={message or 'provider request failed'}"
        return None

    @staticmethod
    def _retryable_provider_error(value: Mapping[str, Any]) -> bool:
        # A content rejection is terminal even if a provider gives it a bad status.
        code = str(value.get("code", "")).lower()
        if any(token in code for token in ("datainspection", "content", "invalidapikey", "unauthorized")):
            return False
        try:
            status = int(value.get("status_code", 0))
        except (TypeError, ValueError):
            status = 0
        return status in {408, 429} or 500 <= status < 600 or code in {"internalerror", "throttling", "serviceunavailable"}

    @classmethod
    def _recoverable_cached_provider_error(cls, value: Mapping[str, Any]) -> bool:
        """Errors worth retrying after a later process restart.

        Arrearage is deliberately excluded from immediate exponential retries: an account
        cannot normally be repaired within seconds. Once the user restores the account,
        however, a new invocation must not be pinned to the old response forever.
        """
        code = str(value.get("code", "") or "").strip().casefold()
        return cls._retryable_provider_error(value) or code == "arrearage"

    def _segment(self, case: WindowCase) -> Path:
        source_segment = str(case.source_record.get("segment_path", "") or "")
        if source_segment and Path(source_segment).is_file():
            return Path(source_segment)
        video = Path(case.video_path)
        if not video.is_file():
            raise FileNotFoundError(f"video unavailable for {case.segment_key}: {video}")
        assert self._segment_cache is not None and self._ffprobe_meta is not None
        meta = self._ffprobe_meta(video)
        return self._segment_cache.get_segment(
            video_path=video,
            start_frame=int(case.start_frame),
            end_frame=int(case.end_frame),
            fps=float(meta.fps),
        )

    @staticmethod
    def _frame_indices(start: int, end: int, count: int) -> list[int]:
        count = max(1, int(count))
        if end <= start:
            return [int(start)] * count
        return [
            int(round(start + (end - start) * ((index + 0.5) / count)))
            for index in range(count)
        ]

    @classmethod
    def _paired_indices(cls, start: int, end: int, count: int, gap: int) -> tuple[list[int], list[dict]]:
        centers = cls._frame_indices(start, end, count)
        gap = max(1, int(gap))
        indices, pair_map = [], []
        for bin_index, center in enumerate(centers):
            left = max(int(start), min(int(end), int(center) - gap // 2))
            right = max(int(start), min(int(end), left + gap))
            if right == left and end > start:
                left = max(int(start), right - gap)
            pair_map.append({
                "bin": bin_index, "labels": [f"T{bin_index}a", f"T{bin_index}b"],
                "frame_indices": [left, right], "gap": right - left,
            })
            indices.extend((left, right))
        return indices, pair_map

    def _extract_frame_panel(self, case: WindowCase) -> dict:
        import cv2  # type: ignore

        panel_signature = stable_sha1(
            case.segment_key, self.config.evidence_mode, self.config.evidence_frames,
            self.config.pair_gap, size=20,
        )
        output_dir = self.config.cache_dir / "evidence_frames" / panel_signature
        metadata_path = output_dir / "metadata.json"
        cached = read_json(metadata_path, None)
        if isinstance(cached, dict):
            paths = [Path(value) for value in cached.get("image_paths", [])]
            expected = self.config.evidence_frames * (2 if self.config.evidence_mode == "paired16" else 1)
            if len(paths) == expected and all(path.is_file() for path in paths):
                return cached

        video = Path(case.video_path)
        absolute = True
        start, end = int(case.start_frame), int(case.end_frame)
        if not video.is_file():
            video = self._segment(case)
            absolute = False
            capture_probe = cv2.VideoCapture(str(video))
            frame_count = int(capture_probe.get(cv2.CAP_PROP_FRAME_COUNT)) if capture_probe.isOpened() else 0
            capture_probe.release()
            start, end = 0, max(0, frame_count - 1)
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            raise RuntimeError(f"cannot open evidence video: {video}")
        if self.config.evidence_mode == "paired16":
            indices, pair_map = self._paired_indices(
                start, end, self.config.evidence_frames, self.config.pair_gap,
            )
            labels = [label for pair in pair_map for label in pair["labels"]]
        else:
            indices = self._frame_indices(start, end, self.config.evidence_frames)
            pair_map = []
            labels = [f"T{index}" for index in range(len(indices))]
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        try:
            for item_index, frame_index in enumerate(indices):
                path = output_dir / f"{labels[item_index]}_f{frame_index}.jpg"
                if not path.is_file():
                    capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
                    ok, frame = capture.read()
                    if not ok or frame is None:
                        raise RuntimeError(f"failed to extract frame {frame_index} from {video}")
                    if not cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(self.config.image_jpeg_quality)]):
                        raise RuntimeError(f"failed to write evidence frame: {path}")
                paths.append(path)
        finally:
            capture.release()
        metadata = {
            "mode": self.config.evidence_mode,
            "source_video": str(video),
            "absolute_source_frame_indices": bool(absolute),
            "frame_indices": indices,
            "image_paths": [str(path) for path in paths],
            "bin_labels": labels,
            "pair_map": pair_map,
            "pair_gap": int(self.config.pair_gap) if pair_map else 0,
            "temporal_bins": int(self.config.evidence_frames),
            "visual_token_image_count": len(paths),
        }
        write_json(metadata_path, metadata)
        return metadata

    def evidence_metadata(self, case: WindowCase) -> dict:
        if self.config.mock:
            if self.config.evidence_mode == "paired16":
                indices, pair_map = self._paired_indices(
                    case.start_frame, case.end_frame, self.config.evidence_frames, self.config.pair_gap,
                )
                labels = [label for pair in pair_map for label in pair["labels"]]
                return {
                    "mode": "paired16", "frame_indices": indices, "image_paths": [],
                    "bin_labels": labels, "pair_map": pair_map, "pair_gap": self.config.pair_gap,
                    "temporal_bins": self.config.evidence_frames,
                    "visual_token_image_count": len(indices),
                }
            return {
                "mode": self.config.evidence_mode,
                "frame_indices": self._frame_indices(case.start_frame, case.end_frame, self.config.evidence_frames),
                "image_paths": [],
                "bin_labels": [f"T{index}" for index in range(self.config.evidence_frames)],
            }
        if self.config.evidence_mode in {"frames", "paired16"}:
            return self._extract_frame_panel(case)
        return {"mode": "video", "frame_indices": [], "image_paths": [], "bin_labels": []}

    def _call_media(self, case: WindowCase, prompt: str, evidence: Mapping[str, Any]) -> str:
        assert self._vlm is not None
        if evidence.get("mode") in {"frames", "paired16"}:
            content = []
            labels = list(evidence.get("bin_labels", []))
            for index, image_path in enumerate(evidence.get("image_paths", [])):
                content.append({"text": str(labels[index] if index < len(labels) else f"T{index}")})
                content.append({"image": Path(str(image_path)).resolve().as_uri()})
            content.append({"text": prompt})
            messages = [{"role": "user", "content": content}]
            call_messages = getattr(self._vlm, "_call_messages", None)
            if call_messages is None:
                raise RuntimeError("installed DashScope backend does not expose multi-image _call_messages")
            return call_messages(messages, vl_high_resolution_images=True)
        return self._vlm.call_video(video_path=self._segment(case), prompt=prompt)

    def request_json(
        self,
        *,
        case: WindowCase,
        prompt: str,
        namespace: str,
        mock_spec: Optional[Mapping[str, Any]] = None,
        evidence_override: Optional[Mapping[str, Any]] = None,
        legacy_prompt: Optional[str] = None,
    ) -> dict:
        evidence = dict(evidence_override) if evidence_override is not None else self.evidence_metadata(case)
        evidence_signature = stable_sha1(
            evidence.get("mode"),
            evidence.get("frame_indices"),
            evidence.get("image_paths"),
            evidence.get("pair_map"),
            size=40,
        )
        def cache_for(request_prompt: str) -> Path:
            digest = hashlib.sha1(
                (
                    f"{case.segment_key}\0{self.config.model}\0{self.config.fps}\0"
                    f"{self.config.cache_salt}\0{evidence_signature}\0{request_prompt}"
                ).encode("utf-8")
            ).hexdigest()
            cache_root = self.config.cache_dir / "responses"
            if self.config.cache_salt:
                cache_root = cache_root / f"replicate_{stable_sha1(self.config.cache_salt, size=16)}"
            return cache_root / namespace / f"{digest}.json"

        cache_path = cache_for(prompt)
        spec = mock_spec or {}
        kind = str(spec.get("kind", ""))
        node_ids = [f"N{i}" for i in range(len(spec["node_keys"]))] if kind == "joint" and "node_keys" in spec else None
        candidates = [cache_path]
        if legacy_prompt and self.config.allow_legacy_temporal_cache:
            legacy_path = cache_for(legacy_prompt)
            if legacy_path != cache_path:
                candidates.append(legacy_path)
        for candidate_path in candidates:
            cached = read_json(candidate_path, None)
            if not isinstance(cached, dict) or not isinstance(cached.get("parsed"), dict):
                if candidate_path.is_file():
                    archive_file(candidate_path, self.config.cache_dir, "unreadable or non-object response cache", remove=True)
                continue
            provider_error = self._provider_error(cached["parsed"])
            if provider_error:
                if self._recoverable_cached_provider_error(cached["parsed"]):
                    archive_file(candidate_path, self.config.cache_dir, provider_error, remove=True)
                    print(f"[cache-resume] archived recoverable provider failure: {candidate_path}", flush=True)
                    continue
                raise RuntimeError(
                    f"VLM provider rejected {case.segment_key}: {provider_error}; cached={candidate_path}"
                )
            issues = response_errors(cached["parsed"], kind, node_ids)
            if issues:
                archive_file(candidate_path, self.config.cache_dir, "; ".join(issues), remove=True)
                print(f"[cache-repair] invalid temporal response: {candidate_path}", flush=True)
                continue
            result = dict(cached)
            result["cache_hit"] = True
            result["cache_path"] = str(candidate_path)
            result["response_validation_version"] = CONTRACT_VERSION
            if candidate_path != cache_path:
                result["cache_reuse"] = {
                    "policy": "explicit_validated_legacy_temporal_reuse",
                    "source_prompt_sha1": cached.get("prompt_sha1"),
                    "requested_prompt_sha1": stable_sha1(prompt, size=40),
                }
            return result

        if self.config.mock:
            parsed = self._mock_response(case.segment_key, mock_spec or {})
            raw = json.dumps(parsed, ensure_ascii=False)
            raw_attempts = [raw]
        else:
            assert self._parse_json_like is not None
            raw_attempts = []
            parsed = None
            last_error = None
            request_prompt = prompt
            attempts = max(1, int(self.config.json_retries))
            for attempt in range(attempts):
                raw = self._call_media(case, request_prompt, evidence)
                raw_attempts.append(raw)
                try:
                    candidate = self._parse_json_like(raw)
                    if not isinstance(candidate, dict):
                        raise ValueError("VLM returned non-object JSON")
                except Exception as exc:
                    last_error = exc
                    write_json(self.config.cache_dir / "request_failures" / namespace / f"{cache_path.stem}_{time.time_ns()}.json", {
                        "segment_key": case.segment_key, "raw": raw, "parse_error": str(exc),
                    })
                    continue
                provider_error = self._provider_error(candidate)
                if provider_error:
                    failure = {"segment_key": case.segment_key, "namespace": namespace,
                               "parsed": candidate, "raw": raw, "evidence": evidence,
                               "prompt_sha1": stable_sha1(prompt, size=40)}
                    failure_path = self.config.cache_dir / "request_failures" / namespace / f"{cache_path.stem}_{time.time_ns()}.json"
                    write_json(failure_path, failure)
                    if not self._retryable_provider_error(candidate):
                        write_json(cache_path, failure)  # Explicit terminal rejection, never scored.
                        raise RuntimeError(f"VLM provider rejected {case.segment_key}: {provider_error}; cached={cache_path}")
                    last_error = RuntimeError(provider_error)
                    if attempt + 1 < attempts:
                        time.sleep(min(30.0, self.config.retry_backoff_seconds * 2 ** attempt))
                    continue
                issues = response_errors(candidate, kind, node_ids)
                if issues:
                    last_error = ValueError("; ".join(issues))
                    write_json(self.config.cache_dir / "request_failures" / namespace / f"{cache_path.stem}_{time.time_ns()}.json", {
                        "segment_key": case.segment_key, "parsed": candidate, "raw": raw,
                        "validation_errors": issues, "contract": CONTRACT_VERSION,
                    })
                    request_prompt = prompt + "\nFORMAT REPAIR: Return exactly eight entries per temporal array (T0..T7), never 16 OT slots. best_bin and episode_span_bins must use 0..7. Return all required nodes.\nErrors: " + "; ".join(issues)
                    continue
                parsed = candidate
                break
            if parsed is None:
                raise ValueError(f"VLM response failed after {attempts} attempts for {case.segment_key}: {last_error}")
            raw = raw_attempts[-1]

        record = {
            "version": "cached_multimodal_vlm_v3",
            "segment_key": case.segment_key,
            "namespace": namespace,
            "model": self.config.model,
            "fps": float(self.config.fps),
            "evidence": evidence,
            "evidence_signature": evidence_signature,
            "cache_salt": self.config.cache_salt,
            "prompt_sha1": stable_sha1(prompt, size=40),
            "prompt": prompt,
            "parsed": parsed,
            "raw": raw,
            "raw_attempts": raw_attempts,
            "cache_hit": False,
            "cache_path": str(cache_path),
            "response_validation_version": CONTRACT_VERSION,
            "format_repaired": not self.config.mock and request_prompt != prompt,
            "effective_prompt": request_prompt if not self.config.mock else prompt,
        }
        issues = response_errors(parsed, kind, node_ids)
        if issues:
            raise ValueError("; ".join(issues))
        provider_error = self._provider_error(parsed)
        if provider_error:
            raise RuntimeError(
                f"VLM provider rejected {case.segment_key}: {provider_error}"
            )
        write_json(cache_path, record)
        return record

    @staticmethod
    def _mock_response(segment_key: str, spec: Mapping[str, Any]) -> dict:
        kind = str(spec.get("kind", ""))

        def unit(*parts: str) -> float:
            value = int(hashlib.sha1("\0".join((segment_key,) + parts).encode()).hexdigest()[:12], 16)
            return value / float(16 ** 12 - 1)

        def distribution(peak: int) -> list[float]:
            values = [math.exp(-0.7 * abs(index - peak)) for index in range(8)]
            total = sum(values)
            return [round(value / total, 5) for value in values]

        if kind == "independent":
            node_key = str(spec.get("node_key", "node"))
            presence = 0.2 + 0.7 * unit(node_key, "presence")
            peak = int(unit(node_key, "peak") * 8) % 8
            quality = [max(0.05, 0.25 + 0.7 * unit(node_key, str(index), "quality")) for index in range(8)]
            return {
                "presence_probability": round(presence, 4),
                "null_probability": round(1.0 - presence, 4),
                "location_distribution_given_present": distribution(peak),
                "evidence_quality_by_bin": [round(value, 4) for value in quality],
                "best_bin": peak,
                "region": "mock region",
                "visible_evidence": f"mock independent evidence for {node_key}",
                "uncertainty": 0.15,
            }
        if kind in {"selector", "selector_repair"}:
            abnormal_ids = [str(value) for value in spec.get("abnormal_ids", [])]
            normal_ids = [str(value) for value in spec.get("normal_ids", [])]
            top_k_abnormal = max(1, int(spec.get("top_k_abnormal", 1)))
            top_k_normal = max(2, int(spec.get("top_k_normal", 2)))

            def ranked(values: list[str], group: str, limit: int) -> list[dict]:
                ordered = sorted(values, key=lambda value: unit(group, value), reverse=True)[:limit]
                return [{
                    "id": value,
                    "visual_support": round(0.4 + 0.55 * unit(group, value), 4),
                    "visible_reason": f"mock shortlist support for {value}",
                } for value in ordered]

            return {
                "abnormal_candidates": ranked(abnormal_ids, "abnormal", top_k_abnormal),
                "normal_candidates": ranked(normal_ids, "normal", top_k_normal),
            }
        if kind in {"joint", "joint_subset"}:
            node_keys = [str(value) for value in spec.get("node_keys", [])]
            graph_key = str(spec.get("graph_key", "graph"))
            initial_presence = spec.get("initial_presence", {}) if isinstance(spec.get("initial_presence"), Mapping) else {}
            nodes = {}
            for index, node_key in enumerate(node_keys):
                anonymous_id = f"N{index}"
                base = float(initial_presence.get(node_key, 0.2 + 0.7 * unit(node_key, "presence")))
                delta = 0.22 * (unit(graph_key, node_key, "delta") - 0.5)
                presence = max(0.05, min(0.95, base + delta))
                peak = (int(unit(graph_key, node_key, "peak") * 8) + index) % 8
                quality = [max(0.05, 0.25 + 0.7 * unit(graph_key, node_key, str(i), "quality")) for i in range(8)]
                nodes[anonymous_id] = {
                    "presence_probability": round(presence, 4),
                    "null_probability": round(1.0 - presence, 4),
                    "location_distribution_given_present": distribution(peak),
                    "evidence_quality_by_bin": [round(value, 4) for value in quality],
                    "best_bin": peak,
                    "region": "mock joint region",
                    "visible_evidence": f"mock joint evidence for {anonymous_id}",
                    "context_increases_from": [f"N{(index + 1) % len(node_keys)}"] if len(node_keys) > 1 else [],
                    "context_decreases_from": [],
                    "probability_update_reason": "mock conditional update from initial OT",
                    "uncertainty": 0.12,
                }
            return {
                "graph_coherence": round(0.45 + 0.5 * unit(graph_key, "coherence"), 4),
                "episode_span_bins": [0, 7],
                "episode_summary": "mock coherent episode",
                "nodes": nodes,
            }
        if kind == "crowd_event_state":
            raw_states = {
                "active_escalation": 0.1 + unit("event_state", "active"),
                "causally_linked_aftermath": 0.1 + unit("event_state", "aftermath"),
                "benign_or_pre_event_context": 0.1 + unit("event_state", "benign"),
                "none_or_unobservable": 0.1 + unit("event_state", "none"),
            }
            total = sum(raw_states.values())
            states = {key: round(value / total, 6) for key, value in raw_states.items()}
            return {
                "state_probabilities": states,
                "transition_observed_probability": round(0.25 + 0.7 * unit("event_state", "transition"), 4),
                "aftermath_causal_link_probability": round(0.25 + 0.7 * unit("event_state", "causal_link"), 4),
                "same_episode_probability": round(0.35 + 0.6 * unit("event_state", "same_episode"), 4),
                "active_transition_bins": [2, 3],
                "aftermath_evidence_bins": [5, 6],
                "benign_context_bins": [0, 1],
                "observed_transition_evidence": "mock active transition evidence",
                "aftermath_causal_evidence": "mock aftermath causal-link evidence",
                "benign_counterfactual_evidence": "mock benign context evidence",
                "decision_reason": "mock mutually exclusive event-state audit",
                "uncertainty": 0.15,
            }
        if kind in {"crowd_event_state_v4", "event_state_v4", "event_state_v5"}:
            raw_states = {
                "active_or_ongoing_physical_escalation": 0.1 + unit("event_state_v4", "active"),
                "causally_linked_aftermath": 0.1 + unit("event_state_v4", "aftermath"),
                "pre_event_tension_or_flight": 0.1 + unit("event_state_v4", "pre_event"),
                "benign_collective_activity": 0.1 + unit("event_state_v4", "benign"),
                "none_or_unobservable": 0.1 + unit("event_state_v4", "none"),
            }
            total = sum(raw_states.values())
            states = {key: round(value / total, 6) for key, value in raw_states.items()}

            def phase_nodes(ids: list[str], phase: str) -> dict:
                output = {}
                initial = spec.get(f"{phase}_initial_presence", {})
                for index, anonymous_id in enumerate(ids):
                    base = float(initial.get(anonymous_id, 0.2 + 0.7 * unit(phase, anonymous_id, "presence")))
                    presence = max(0.05, min(0.95, base + 0.15 * (unit(phase, anonymous_id, "delta") - 0.5)))
                    peak = (int(unit(phase, anonymous_id, "peak") * 8) + index) % 8
                    quality = [max(0.05, 0.25 + 0.7 * unit(phase, anonymous_id, str(i), "quality")) for i in range(8)]
                    output[anonymous_id] = {
                        "presence_probability": round(presence, 4),
                        "null_probability": round(1.0 - presence, 4),
                        "location_distribution_given_present": distribution(peak),
                        "evidence_quality_by_bin": [round(value, 4) for value in quality],
                        "best_bin": peak,
                        "visible_evidence": f"mock V4 {phase} evidence for {anonymous_id}",
                        "probability_update_reason": "mock shared phase-conditional update",
                        "uncertainty": 0.12,
                    }
                return output

            active_ids = [str(value) for value in spec.get("active_ids", [])]
            aftermath_ids = [str(value) for value in spec.get("aftermath_ids", [])]
            output = {
                "state_probabilities": states,
                "phase_coherence": {
                    "active": round(0.45 + 0.5 * unit("event_state_v4", "active_coherence"), 4),
                    "aftermath": round(0.45 + 0.5 * unit("event_state_v4", "aftermath_coherence"), 4),
                },
                "phase_nodes": {
                    "active": phase_nodes(active_ids, "active"),
                    "aftermath": phase_nodes(aftermath_ids, "aftermath"),
                },
                "active_evidence_bins": [2, 3], "aftermath_evidence_bins": [5, 6],
                "pre_event_bins": [0, 1], "benign_bins": [7],
                "visible_evidence": {"active": "mock ongoing mechanism", "aftermath": "mock linked aftermath"},
                "counterfactual_evidence": {"pre_event": "mock pre-event", "benign": "mock benign"},
                "decision_reason": "mock V4 equal-contract state audit", "uncertainty": 0.15,
            }
            if kind == "event_state_v5":
                context = {
                    "active": states["active_or_ongoing_physical_escalation"],
                    "aftermath": states["causally_linked_aftermath"],
                    "pre": states["pre_event_tension_or_flight"],
                    "benign": states["benign_collective_activity"],
                    "none": states["none_or_unobservable"],
                }
                output.update({
                    "event_context_state_probabilities": context,
                    "current_window_active_occupancy_probability": round(
                        0.05 + 0.9 * unit("event_state_v5", "occupancy"), 4
                    ),
                    "current_window_normal_confound_probability": round(
                        0.05 + 0.9 * unit("event_state_v5", "normal_confound"), 4
                    ),
                    "normal_confound_probabilities": {
                        "structured_sport_or_play": 0.1,
                        "peaceful_protest_or_ceremony": 0.15,
                        "assistance_or_rescue": 0.1,
                        "ordinary_object_interaction": 0.15,
                        "staged_or_performed_action": 0.1,
                        "none": 0.4,
                    },
                    "aftermath_context_probability": context["aftermath"],
                    "aftermath_current_window_occupancy_support_probability": round(
                        0.05 + 0.4 * unit("event_state_v5", "aftermath_support"), 4
                    ),
                    "direct_active_bins": [2, 3], "context_only_bins": [5, 6],
                    "normal_confound_bins": [0, 1],
                    "normal_confound_evidence": {
                        "type": "ordinary_object_interaction",
                        "visible_evidence": "mock visible normal mechanism",
                        "reason": "mock signed-confound audit",
                    },
                    "decision_reason": "mock V5 context and occupancy audit",
                })
            return output
        if kind == "verifier":
            return {
                "preferred_method": str(spec.get("preferred_method", "Y")),
                "confidence": 0.8,
                "visual_reason": "mock verifier preference",
                "method_x_failure": "",
                "method_y_failure": "",
            }
        raise ValueError(f"unknown mock response kind: {kind}")
