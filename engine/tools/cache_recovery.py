"""Non-destructive archives for targeted inference recovery."""
from __future__ import annotations

import shutil
from pathlib import Path

from common import file_sha256, write_json


def archive_file(path: Path, root: Path, reason: str, *, remove: bool = False) -> Path:
    path, root = path.resolve(), root.resolve()
    relative = path.relative_to(root)  # Refuse paths outside the intended run/cache.
    digest = file_sha256(path)
    target = root / "recovery_archive" / digest[:16] / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copy2(path, target)
    if file_sha256(target) != digest:
        raise RuntimeError(f"archive verification failed: {target}")
    write_json(target.with_suffix(target.suffix + ".reason.json"), {
        "source": str(path), "sha256": digest, "reason": reason,
    })
    if remove:
        path.unlink()
    return target
