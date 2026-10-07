#!/usr/bin/env python3
"""Zero-API semantic audit, targeted review, and non-scoring diagnostic sidecar."""
import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "docs"))
from event_decision.safety import OfflineGuard


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "review", "report"), default="prepare")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--review-file", type=Path)
    args = parser.parse_args()
    guard = OfflineGuard()
    guard.install()
    from event_decision import contrast_semantics_v918 as audit
    from event_decision.b1b4_trial.protocol import run_lock
    source = (args.source or PROJECT / "runs" / audit.DEFAULT_SOURCE).resolve()
    out = (args.out or PROJECT / "runs" / audit.DEFAULT_TAG).resolve()
    try:
        audit.check_destination(source, out)
        if args.phase != "prepare" and not (out / "seal.json").exists():
            raise ValueError("prepare this audit first")
        with run_lock(out):
            if args.phase == "prepare":
                result = audit.prepare(PROJECT, source, out)
            elif args.phase == "review":
                result = audit.import_review(out, args.review_file or out / "review/review.json")
            else:
                result = audit.report(out)
        guard.assert_no_remote_calls()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print("[review] " + str(out / "index.html"))
        return 0
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
        print("[pause] " + str(exc))
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
