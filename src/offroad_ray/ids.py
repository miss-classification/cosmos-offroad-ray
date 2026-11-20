"""Stable content-derived identities for clips, preprocessing and requests."""
from __future__ import annotations

import hashlib
import json
from typing import Any


def stable_hash(obj: Any, length: int = 16) -> str:
    """SHA-256 of canonical JSON. Key order and float formatting are normalized."""
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:length]


def sha256_file(path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def prep_config_hash(clip_cfg: dict, control_cfg: dict, prep_code_version: str) -> str:
    """Identity of the preprocessing policy. Changing any field invalidates the cache."""
    return stable_hash({"clips": clip_cfg, "control": control_cfg, "code": prep_code_version})


def clip_id(dataset_id: str, run_id: str, camera: str, src_t0: float, num_frames: int, fps: float) -> str:
    return f"{run_id}__{camera}__t{src_t0:.3f}__{stable_hash([dataset_id, run_id, camera, round(src_t0, 6), num_frames, fps], 8)}"


def request_id(*, clip_checksum: str, prep_hash: str, model: str, cosmos_commit: str,
               checkpoint_revision: str, prompt: str, seed: int, control_weight: float,
               num_steps: int, resolution: str, guardrails: bool, code_version: str) -> str:
    """Immutable identity for one generation request.

    Anything that can change output pixels is included; two requests with the
    same id are interchangeable and may be served from a validated cache.
    """
    return stable_hash({
        "clip_checksum": clip_checksum, "prep": prep_hash, "model": model,
        "cosmos_commit": cosmos_commit, "ckpt_rev": checkpoint_revision,
        "prompt": prompt, "seed": int(seed), "control_weight": round(float(control_weight), 4),
        "num_steps": int(num_steps), "resolution": resolution, "guardrails": bool(guardrails),
        "code": code_version,
    }, 20)
