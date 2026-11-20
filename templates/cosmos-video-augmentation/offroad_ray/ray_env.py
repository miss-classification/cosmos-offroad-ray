"""Connect to the existing Ray cluster with a reproducible runtime environment.

Ray tasks and actors run in the Anyscale Python (Ray 2.60.0, Python 3.13) plus
a pinned pip layer derived from pyproject.toml. Cosmos itself never enters this
environment; it runs in a separate interpreter launched as a subprocess.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]   # the template folder

# Native thread caps. Ray CPU reservations do not cap OpenMP/OpenCV/FFmpeg threads.
THREAD_ENV = {
    "OMP_NUM_THREADS": "2",
    "OPENBLAS_NUM_THREADS": "2",
    "MKL_NUM_THREADS": "2",
    "OPENCV_NUM_THREADS": "2",
    "OFFROAD_FFMPEG_THREADS": "2",
}

# Optional custom resource that pins CPU-heavy work to worker nodes, as JSON, for example
# OFFROAD_WORKER_RESOURCE='{"anyscale/accelerator_shape:1xA100-80G": 0.001}'. Empty by default,
# so CPU stages run on any node with free CPUs.
WORKER_ONLY = json.loads(os.environ.get("OFFROAD_WORKER_RESOURCE", "{}"))


def runtime_env(extra_env: dict | None = None) -> dict:
    """Ship this package to every worker. Python dependencies come from the template image
    (Dockerfile) or from requirements.txt installed in the Workspace."""
    import offroad_ray  # noqa: F401  (ensure importable before shipping)
    env = {**THREAD_ENV, **(extra_env or {})}
    for k in ("COSMOS_SHARED", "COSMOS_ENV_ROOT", "OFFROAD_WORKER_RESOURCE", "HF_TOKEN"):
        if os.environ.get(k):
            env.setdefault(k, os.environ[k])
    rt = {"py_modules": [str(PROJECT_ROOT / "offroad_ray")], "env_vars": env}
    # The template image sets OFFROAD_IMAGE_DEPS=1 because it already contains requirements.txt.
    # On any other image, Ray installs them on the workers (once per node, then cached).
    if os.environ.get("OFFROAD_IMAGE_DEPS") != "1":
        reqs = [l.strip() for l in (PROJECT_ROOT / "requirements.txt").read_text().splitlines()
                if l.strip() and not l.startswith("#")]
        rt["pip"] = {"packages": reqs, "pip_check": False}
    return rt


def init_ray(extra_env: dict | None = None, **kwargs):
    import ray
    os.environ.setdefault("RAY_OVERRIDE_JOB_RUNTIME_ENV", "1")
    if ray.is_initialized():
        return ray
    ray.init(address="auto", runtime_env=runtime_env(extra_env), logging_level="ERROR", **kwargs)
    return ray
