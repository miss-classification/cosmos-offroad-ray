"""Connect to the existing Ray cluster with a reproducible runtime environment.

Ray tasks and actors run in the Anyscale Python (Ray 2.60.0, Python 3.13) plus
a pinned pip layer derived from pyproject.toml. Cosmos itself never enters this
environment; it runs in a separate interpreter launched as a subprocess.
"""
from __future__ import annotations

import os
import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

# Native thread caps. Ray CPU reservations do not cap OpenMP/OpenCV/FFmpeg threads.
THREAD_ENV = {
    "OMP_NUM_THREADS": "2",
    "OPENBLAS_NUM_THREADS": "2",
    "MKL_NUM_THREADS": "2",
    "OPENCV_NUM_THREADS": "2",
    "OFFROAD_FFMPEG_THREADS": "2",
}

# Custom resource present only on GPU worker nodes; requesting a tiny fraction
# keeps CPU work off the head (which also advertises 0 CPUs to Ray here).
WORKER_ONLY = {"anyscale/accelerator_shape:1xA100-80G": 0.001}


def pinned_requirements() -> list[str]:
    data = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())
    return list(data["project"]["dependencies"])


def runtime_env(extra_env: dict | None = None) -> dict:
    import offroad_ray  # noqa: F401  (ensure importable before shipping)
    return {
        "py_modules": [str(SRC_DIR / "offroad_ray")],
        "pip": {"packages": pinned_requirements(), "pip_check": False},
        "env_vars": {**THREAD_ENV, **(extra_env or {})},
    }


def init_ray(extra_env: dict | None = None, **kwargs):
    import ray
    os.environ.setdefault("RAY_OVERRIDE_JOB_RUNTIME_ENV", "1")
    if ray.is_initialized():
        return ray
    ray.init(address="auto", runtime_env=runtime_env(extra_env), logging_level="ERROR", **kwargs)
    return ray
