#!/usr/bin/env python3
"""Freeze GT-blind DashScope observations for episodic failures."""
from __future__ import annotations

import argparse
import json
from concurrent import futures
from pathlib import Path
from typing import Mapping

from common import iter_jsonl, write_json, write_jsonl
from prompts import assert_blind
from schemas import WindowCase
from vlm_runtime import CachedVideoVLM, RuntimeConfig


OBSERVER_PROMPT = """TASK: BLIND_EVENT_STATE_OBSERVATION
The supplied frames T0..T7 are chronological samples from one anonymous video window.
Do not classify the window. Do not infer a dataset category. Record only visible facts,
including uncertainty and facts that are not visibly established.

Return JSON only:
{
  "entities": [{"id": "E0", "type": "person|group|vehicle|object|scene_region", "visible_role": ""}],
  "visible_actions_states": [{"subject": "E0", "description": "", "bins": [0], "confidence": 0.0}],
  "phase": "prelude|onset|active|aftermath|unclear",
  "state_changes": [{"entity": "E0", "before": "", "after": "", "visible_mechanism": "", "confidence": 0.0}],
  "audio_motion_cues": [],
  "not_visible_or_unresolved": [],
  "global_uncertainty": 0.0,
  "neutral_summary": ""
}

Never use filenames, identities, source style, subtitles, graph names, labels, or an expected answer."""
assert_blind(OBSERVER_PROMPT)


def _mock_observation(row: Mapping) -> dict:
    trace = row.get("observed_event_state", {})
    return {
        "entities": [], "visible_actions_states": [], "phase": "unclear", "state_changes": [],
        "audio_motion_cues": [],
        "not_visible_or_unresolved": ["mock observer has no image content"],
        "global_uncertainty": 1.0,
        "neutral_summary": "mock blind observation",
        "trace_context": trace.get("visible_descriptions", []) if isinstance(trace, Mapping) else [],
    }


def observe(input_path: Path, out_dir: Path, *, code_dir: Path, cache_dir: Path, model: str,
            fps: float, key_env: str, workers: int, mock: bool, evidence_frames: int) -> dict:
    rows = list(iter_jsonl(input_path))
    out_dir.mkdir(parents=True, exist_ok=True)

    def process(row: dict) -> dict:
        case_id = str(row.get("case_id", ""))
        output = out_dir / "observer_cases" / f"{case_id}.json"
        if output.is_file():
            observation = json.loads(output.read_text(encoding="utf-8"))
        elif mock:
            observation = _mock_observation(row)
            write_json(output, observation)
        else:
            window = row.get("window", {})
            case = WindowCase(
                segment_key=str(window.get("segment_key", "")),
                video_id=str(window.get("video_id", "")),
                video_path=str(window.get("video_path", "")),
                start_frame=int(window.get("start_frame", 0)),
                end_frame=int(window.get("end_frame", 0)),
                y_true=None,
                source_record={},
            )
            runtime = CachedVideoVLM(RuntimeConfig(
                code_dir=code_dir, cache_dir=cache_dir, model=model, fps=fps,
                key_env=key_env, mock=False, evidence_mode="frames", evidence_frames=evidence_frames,
            ))
            response = runtime.request_json(case=case, prompt=OBSERVER_PROMPT, namespace="blind_observer")
            observation = response.get("parsed", {})
            write_json(output, observation)
        updated = dict(row)
        updated["observed_event_state"] = {
            "status": "frozen_blind_observer",
            "prompt_version": "event_state_observer_v1",
            "observation": observation,
        }
        updated["status"] = "observed"
        return updated

    if max(1, workers) == 1:
        observed = [process(row) for row in rows]
    else:
        with futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            observed = list(pool.map(process, rows))
    write_jsonl(out_dir / "episodic_failures_observed.jsonl", observed)
    summary = {
        "version": "blind_failure_observer_v1", "input": len(rows), "observed": len(observed),
        "model": "mock" if mock else model, "gt_exposed_to_observer": False,
        "output": str(out_dir / "episodic_failures_observed.jsonl"),
    }
    write_json(out_dir / "observer_summary.json", summary)
    return summary


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--code-dir", required=True, type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--model", default="qwen3.6-plus")
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--evidence-frames", type=int, default=8)
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()
    cache = args.cache_dir or args.out_dir / "observer_cache"
    print(json.dumps(observe(args.input, args.out_dir, code_dir=args.code_dir, cache_dir=cache,
        model=args.model, fps=args.fps, key_env=args.key_env, workers=args.workers,
        mock=args.mock, evidence_frames=args.evidence_frames), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
