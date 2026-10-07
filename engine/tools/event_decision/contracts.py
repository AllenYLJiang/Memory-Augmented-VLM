from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

CONTRACT_VERSION = "governed_v9_scope_aligned_v1"
FEATURE_SCHEMA_ID = "event_decision_core8_v1"
LABEL_RULE_ID = "xd_window_contiguous_overlap_ge8_v1"
FEATURE_NAMES = (
    "m0_margin",
    "m3a_margin",
    "m3c_margin",
    "o_active",
    "q_direct",
    "state_uncertainty",
    "normal_bound_probability",
    "unexplained_direct_evidence",
)
STATUS_VALUES = {
    "WAITING_FOR_LOCAL_MEDIA",
    "WAITING_FOR_HUMAN_REVIEW",
    "INSUFFICIENT_SCOPE_ALIGNED_DATA",
    "BINDING_NOT_IDENTIFIABLE_FROM_CURRENT_CACHE",
    "DEVELOPMENT_ONLY",
    "NO_CANDIDATE",
}


class EventDecisionError(RuntimeError):
    exit_code = 2


class MissingLocalData(EventDecisionError):
    exit_code = 3


class WaitingForHumanReview(EventDecisionError):
    exit_code = 4


class OfflineNetworkViolation(EventDecisionError):
    exit_code = 5


class SealViolation(EventDecisionError):
    exit_code = 6


@dataclass(frozen=True)
class WindowKey:
    dataset_partition: str
    video_id: str
    start_frame: int
    end_frame_exclusive: int

    def __post_init__(self) -> None:
        if self.start_frame < 0 or self.end_frame_exclusive <= self.start_frame:
            raise EventDecisionError(f"invalid half-open window: {self}")

    @property
    def uid(self) -> str:
        return sha256_parts(
            self.dataset_partition,
            self.video_id,
            self.start_frame,
            self.end_frame_exclusive,
        )


@dataclass(frozen=True)
class FeatureRow:
    window_key: WindowKey
    feature_contract_id: str
    baseline_origin: str
    values: np.ndarray
    observed: np.ndarray
    provenance: dict[str, Any]


@dataclass(frozen=True)
class LabelRecord:
    window_key: WindowKey
    original_label: int | None
    label_scope: str
    evidence_level: str
    source_spans: tuple[tuple[int, int], ...]
    window_target: int | None
    window_loss_mask: bool
    interval_labels: np.ndarray
    interval_observed: np.ndarray
    review_use_policy: str = "audit_only"


def sha256_parts(*parts: Any) -> str:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def semantic_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default


def iter_jsonl(path: Path):
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise EventDecisionError(f"invalid JSONL {path}:{line_no}: {exc}") from exc
            if not isinstance(value, dict):
                raise EventDecisionError(f"JSONL row is not an object: {path}:{line_no}")
            yield value


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(_jsonable(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]] | Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_jsonable(row), ensure_ascii=False) + "\n")
    os.replace(temp, path)


def json_pointer(value: Mapping[str, Any], pointer: str, default: Any = None) -> Any:
    current: Any = value
    for token in pointer.strip("/").split("/") if pointer.strip("/") else []:
        token = token.replace("~1", "/").replace("~0", "~")
        if not isinstance(current, Mapping) or token not in current:
            return default
        current = current[token]
    return current


def canonical_window(record: Mapping[str, Any], default_partition: str = "train") -> WindowKey:
    video_id = str(record.get("video_id") or "")
    start = int(record.get("start_frame", 0))
    if record.get("end_frame_exclusive") is not None:
        end_exclusive = int(record["end_frame_exclusive"])
    else:
        end_exclusive = int(record.get("end_frame", start)) + 1
    partition = str(record.get("dataset_partition") or record.get("dataset_split") or default_partition)
    return WindowKey(partition, video_id, start, end_exclusive)


def source_group(record: Mapping[str, Any]) -> str:
    explicit = record.get("source_group") or record.get("source_group_id")
    if explicit:
        return str(explicit)
    return str(record.get("video_id") or "").split("__#", 1)[0]


_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def resolve_variables(value: Any, variables: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        def replace(match: re.Match[str]) -> str:
            key = match.group(1)
            if key not in variables:
                raise EventDecisionError(f"unknown path variable: {key}")
            return str(variables[key])
        previous = None
        while previous != value:
            previous = value
            value = _VAR.sub(replace, value)
        return value
    if isinstance(value, list):
        return [resolve_variables(v, variables) for v in value]
    if isinstance(value, dict):
        return {k: resolve_variables(v, variables) for k, v in value.items()}
    return value

