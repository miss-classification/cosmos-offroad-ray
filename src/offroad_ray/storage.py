"""Storage helpers: atomic writes, per-record JSON files, guarded cleanup.

Storage semantics on this Anyscale workspace (verified 2026-11-18):
  /mnt/cluster_storage  NFS shared by head and all workers; scoped to this
                        workspace; survives node replacement and workspace restarts,
                        deleted with the workspace.
  /mnt/user_storage     NFS scoped to the user across workspaces.
  /mnt/local_storage    node-local NVMe on workers (369 GB); lost on node replacement.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any

PROJECT_MARKER = ".cosmos-offroad-owned"


def atomic_write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def atomic_write_json(path: str | Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=1, sort_keys=True, default=str))


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def unique_attempt_dir(base: str | Path, request_id: str) -> Path:
    """A fresh directory for one attempt; never shared between attempts."""
    d = Path(base) / ".attempts" / f"{request_id}.{uuid.uuid4().hex[:8]}"
    d.mkdir(parents=True, exist_ok=False)
    (d / PROJECT_MARKER).touch()
    return d


def make_owned_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    (p / PROJECT_MARKER).touch()
    return p


def safe_rmtree(path: str | Path, allowed_roots: list[str | Path]) -> bool:
    """Delete a project-owned directory only.

    Refuses unless the resolved path is strictly inside an allowed root and
    contains the ownership marker. Returns True if deleted.
    """
    p = Path(path).resolve()
    roots = [Path(r).resolve() for r in allowed_roots]
    if not any(p != r and r in p.parents for r in roots):
        raise ValueError(f"refusing to delete {p}: not strictly inside {roots}")
    if not (p / PROJECT_MARKER).exists():
        raise ValueError(f"refusing to delete {p}: missing ownership marker")
    shutil.rmtree(p)
    return True


def dir_size_gb(path: str | Path) -> float:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.lstat(os.path.join(root, f)).st_size
            except FileNotFoundError:
                pass
    return total / 1e9
