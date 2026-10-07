#!/usr/bin/env python3
"""Development-split provenance and final-holdout safety checks."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


FINAL_ROLES = {"final", "final_evaluation", "final_holdout"}
NONFINAL_STATUSES = {"development", "development_used_for_model_selection", "calibration"}


def load_split_registry(path: Path | None) -> dict:
    if path is None:
        return {}
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"split registry must contain a JSON object: {path}")
    return value


def resolve_split_provenance(
    seed: int,
    role: str,
    registry: Mapping[str, Any] | None,
    registry_path: Path | None = None,
) -> dict:
    registry = dict(registry or {})
    statuses = registry.get("status", {})
    status = str(statuses.get(str(int(seed)), "unregistered")) if isinstance(statuses, Mapping) else "unregistered"
    role = str(role or "development")
    if role in FINAL_ROLES and status in NONFINAL_STATUSES:
        raise ValueError(
            f"seed {seed} is marked {status!r} and cannot be used as a final evaluation split"
        )
    future = registry.get("future_final_holdout", {})
    frozen_seed = future.get("seed") if isinstance(future, Mapping) else None
    if role in FINAL_ROLES and frozen_seed is not None and int(seed) != int(frozen_seed):
        raise ValueError(
            f"seed {seed} does not match frozen final holdout seed {frozen_seed}"
        )
    return {
        "registry": str(registry_path or ""),
        "registry_version": registry.get("version", "unregistered"),
        "seed": int(seed),
        "declared_role": role,
        "registered_status": status,
        "final_evaluation_allowed": status not in NONFINAL_STATUSES,
    }
