"""Scheduler tests with a mocked inference actor.

Connects to the existing Ray cluster (CPU-only actors, no GPUs reserved). Skipped
when no cluster is reachable. Never starts a separate local Ray instance.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

ray = pytest.importorskip("ray")

from offroad_ray.config import ExperimentConfig  # noqa: E402
from offroad_ray.inference.mock_actor import MockActor  # noqa: E402
from offroad_ray.inference.scheduler import run_pool, validated_existing  # noqa: E402


@pytest.fixture(scope="module")
def cluster():
    if not ray.is_initialized():
        try:
            from offroad_ray.ray_env import init_ray
            init_ray()
        except Exception as e:  # pragma: no cover
            pytest.skip(f"no Ray cluster: {e}")
    yield


@pytest.fixture
def shared_tmp():
    """Worker-visible scratch (pytest's tmp_path is head-local and invisible to actors)."""
    import uuid
    d = Path("/mnt/cluster_storage/cosmos-offroad/tmp/pytest") / uuid.uuid4().hex[:10]
    d.mkdir(parents=True)
    yield d


def _cfg(tmp_path, max_attempts=500):
    return ExperimentConfig.model_validate({
        "name": "t", "paths": {"root": str(tmp_path / "root")}, "dataset": {"runs": ["r"]},
        "generation": {"conditions": [{"name": "a", "prompt": "a"}]},
        "scheduler": {"max_attempts_per_request": 2, "max_actor_restarts": 3},
        "budget": {"max_generation_attempts": max_attempts}})


def _req(ns, i, prompt="ok", sleep=0.05):
    return {"request_id": f"req{i:03d}", "ns_dir": str(ns), "clip_id": f"c{i}", "run_id": "r", "condition": "a",
            "seed": i, "control_weight": 1.0, "clip_path": "", "control_path": "",
            "sample": {"prompt": prompt, "sleep": sleep}, "expect": {}, "identity": {"v": 1}}


def _run(cfg, reqs, ns, n=3, resume=True):
    return run_pool(cfg, reqs, str(ns), num_actors=n, resume=resume, log=lambda *_: None,
                    actor_cls=ray.remote(MockActor), settings={"mock": True},
                    actor_options={"num_cpus": 0.01, "max_restarts": 0, "max_concurrency": 2,
                                   "resources": {"anyscale/accelerator_shape:1xA100-80G": 0.001}})


def test_dynamic_dispatch_all_complete(cluster, shared_tmp):
    tmp_path = shared_tmp
    cfg = _cfg(tmp_path)
    ns = tmp_path / "ns"
    # one slow request must not block the others (dynamic, not static partition)
    reqs = [_req(ns, 0, sleep=2.0)] + [_req(ns, i) for i in range(1, 12)]
    s = _run(cfg, reqs, ns, n=3)
    assert s["status_counts"] == {"generated_unreviewed": 12}
    by_actor = {}
    for r in s["records"]:
        by_actor.setdefault(r["actor_index"], []).append(r["request_id"])
    slow_actor = next(a for a, ids in by_actor.items() if "req000" in ids)
    assert len(by_actor[slow_actor]) < max(len(v) for v in by_actor.values())
    assert (ns / "manifest.jsonl").exists()
    assert len((ns / "manifest.jsonl").read_text().splitlines()) == 12
    for r in s["records"]:
        assert "queue_wait_s" in r


def test_infra_failure_retried_guardrail_not(cluster, shared_tmp):
    tmp_path = shared_tmp
    cfg = _cfg(tmp_path)
    ns = tmp_path / "ns"
    reqs = [_req(ns, 0, "infra_once"), _req(ns, 1, "guardrail"), _req(ns, 2, "infra_always"), _req(ns, 3)]
    s = _run(cfg, reqs, ns, n=2)
    recs = {r["request_id"]: r for r in s["records"]}
    assert recs["req000"]["status"] == "generated_unreviewed" and recs["req000"]["attempts"] == 2
    assert recs["req001"]["status"] == "guardrail_blocked" and recs["req001"]["attempts"] == 1
    assert recs["req002"]["status"] == "infra_failure" and recs["req002"]["attempts"] == 2
    assert recs["req003"]["status"] == "generated_unreviewed"
    attempts = sorted(p.name for p in (ns / "attempts").glob("*.json"))
    assert "req000.1.json" in attempts and "req000.2.json" in attempts
    assert s["actor_replacements"] >= 2


def test_resume_reuses_only_validated(cluster, shared_tmp):
    tmp_path = shared_tmp
    cfg = _cfg(tmp_path)
    ns = tmp_path / "ns"
    reqs = [_req(ns, i) for i in range(4)]
    _run(cfg, reqs, ns, n=2)
    # corrupt one output and change identity of another -> both must rerun
    rec1 = json.loads((ns / "records" / "req001.json").read_text())
    Path(rec1["output_path"]).unlink()
    reqs[2] = {**reqs[2], "identity": {"v": 2}}
    assert validated_existing(str(ns), reqs[0]) is not None
    assert validated_existing(str(ns), reqs[1]) is None
    assert validated_existing(str(ns), reqs[2]) is None
    s = _run(cfg, reqs, ns, n=2)
    assert s["skipped"] == 2 and s["ran"] == 2
    s2 = _run(cfg, reqs, ns, n=2, resume=False)
    assert s2["skipped"] == 0 and s2["ran"] == 4


def test_budget_enforced_before_launch(cluster, shared_tmp):
    tmp_path = shared_tmp
    cfg = _cfg(tmp_path, max_attempts=3)
    ns = tmp_path / "ns"
    with pytest.raises(RuntimeError, match="budget"):
        _run(cfg, [_req(ns, i) for i in range(5)], ns)
