#!/usr/bin/env python3
"""Build a hashed union of discovery, previous-packet, and calibration source groups."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import file_sha256, write_json


def build(registry: Path, previous_manifests: list[Path], extra_files: list[Path], out: Path) -> dict:
    groups = set()
    inputs = []
    if registry.is_file():
        value = json.loads(registry.read_text(encoding="utf-8"))
        for row in value.get("candidate_graphs", []):
            groups.update(str(item) for item in row.get("support_source_groups", []) if item)
        inputs.append(registry)
    for path in previous_manifests:
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                value = json.loads(line)
                if value.get("source_group"):
                    groups.add(str(value["source_group"]))
        inputs.append(path)
    for path in extra_files:
        if not path.is_file():
            continue
        groups.update(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
        inputs.append(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(f"{value}\n" for value in sorted(groups)), encoding="utf-8")
    summary = {
        "version": "validation_source_group_exclusions_v1",
        "source_groups": len(groups), "out": str(out), "out_sha256": file_sha256(out),
        "inputs": {str(path): file_sha256(path) for path in inputs},
    }
    write_json(out.with_suffix(".summary.json"), summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--previous-manifest", action="append", type=Path, default=[])
    parser.add_argument("--extra-source-groups-file", action="append", type=Path, default=[])
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.registry, args.previous_manifest, args.extra_source_groups_file, args.out), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
