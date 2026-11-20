"""Driver-side dynamic scheduler over a pool of persistent CosmosActors.

* One request in flight per actor; the next pending request goes to whichever
  actor frees up first (no static partitioning).
* Infrastructure failures (subprocess death, timeout, actor loss) are retried a
  bounded number of times, on a replacement actor. Guardrail blocks and
  generation failures are terminal and never retried with altered seeds or
  settings.
* Every attempt writes its own JSON record; the request record is written
  atomically after the attempt finishes. No shared append-only file is written
  concurrently. Consolidation into manifest.jsonl/parquet happens afterwards.
"""
from __future__ import annotations

import json
import os
import time
from collections import deque
from pathlib import Path

from offroad_ray.ids import request_id as make_request_id
from offroad_ray.storage import atomic_write_json, read_json

CODE_VERSION = "gen-v1"
TERMINAL_OK = {"generated_unreviewed"}
TERMINAL = {"generated_unreviewed", "guardrail_blocked", "generation_failure"}


# ----------------------------------------------------------------- requests
def build_requests(cfg, catalog: list[dict], ns_dir: str) -> list[dict]:
    g = cfg.generation
    reqs = []
    clips = [c for c in catalog if c["status"] == "ok"]
    clips.sort(key=lambda c: (c["run_id"], c["clip_index"]))
    if cfg.max_clips:
        clips = _spread(clips, cfg.max_clips)
    for c in clips:
        for cond in g.conditions:
            for seed in g.seeds:
                for cw in g.control_weights:
                    rid = make_request_id(
                        clip_checksum=c["clip_sha256"], prep_hash=c["prep_hash"], model=g.model,
                        cosmos_commit=g.cosmos_commit, checkpoint_revision=g.checkpoint_revision,
                        prompt=cond.prompt, seed=seed, control_weight=cw, num_steps=g.num_steps,
                        resolution=g.resolution, guardrails=g.guardrails, code_version=CODE_VERSION)
                    reqs.append({
                        "request_id": rid, "ns_dir": ns_dir, "clip_id": c["clip_id"], "run_id": c["run_id"],
                        "condition": cond.name, "seed": seed, "control_weight": cw,
                        "clip_path": c["clip_path"], "control_path": c["control_path"],
                        "sample": {
                            "prompt": cond.prompt, "video_path": c["clip_path"], "seed": seed,
                            "num_steps": g.num_steps, "resolution": g.resolution,
                            "max_frames": cfg.clips.num_frames, "num_video_frames_per_chunk": cfg.clips.num_frames,
                            "keep_input_resolution": g.keep_input_resolution,
                            "edge": {"control_path": c["control_path"], "control_weight": cw,
                                     "preset_edge_threshold": cfg.control.preset_edge_threshold},
                        },
                        "expect": {"frames": int(c["model_input_frames"]), "width": int(c["model_input_width"]),
                                   "height": int(c["model_input_height"]), "fps": float(c["model_input_fps"])},
                        "identity": {"model": g.model, "cosmos_commit": g.cosmos_commit,
                                     "checkpoint_revision": g.checkpoint_revision, "num_steps": g.num_steps,
                                     "guardrails": g.guardrails, "prep_hash": c["prep_hash"],
                                     "clip_sha256": c["clip_sha256"], "code_version": CODE_VERSION},
                    })
    ids = [r["request_id"] for r in reqs]
    assert len(ids) == len(set(ids)), "duplicate request ids"
    return reqs


def _spread(clips: list[dict], n: int) -> list[dict]:
    """Pick n clips round-robin across runs so a small subset still spans runs."""
    by_run: dict[str, list[dict]] = {}
    for c in clips:
        by_run.setdefault(c["run_id"], []).append(c)
    out, i = [], 0
    while len(out) < n and any(by_run.values()):
        for r in sorted(by_run):
            if by_run[r] and len(out) < n:
                out.append(by_run[r].pop(0))
        i += 1
    return out


# ------------------------------------------------------------------- resume
def validated_existing(ns_dir: str, req: dict) -> dict | None:
    """Return the existing terminal record only if it matches this request and its output exists."""
    p = Path(ns_dir) / "records" / f"{req['request_id']}.json"
    if not p.exists():
        return None
    rec = read_json(p)
    if rec.get("identity") != req["identity"]:
        return None
    if rec.get("status") in TERMINAL_OK:
        out = rec.get("output_path")
        if not out or not Path(out).exists() or Path(out).stat().st_size == 0:
            return None
        return rec
    if rec.get("status") == "guardrail_blocked":
        return rec  # deterministic policy outcome; do not reroll
    return None     # failures are retried on resume


# ------------------------------------------------------------------ ledger
def ledger_dir(root: str | Path) -> Path:
    d = Path(root) / "runs" / "_attempt_ledger"
    d.mkdir(parents=True, exist_ok=True)
    return d


def attempts_used(root) -> int:
    return sum(1 for _ in ledger_dir(root).glob("*.json"))


# ---------------------------------------------------------------- settings
def actor_settings(cfg, ns_dir: str, *, hf_offline: bool) -> dict:
    # The template image builds Cosmos under /opt/cosmos; otherwise it lives on shared storage.
    cosmos_root = Path(os.environ.get("COSMOS_ENV_ROOT", "/opt/cosmos" if Path("/opt/cosmos/envs").exists() else cfg.paths.root))
    env_dir = cosmos_root / "envs" / os.environ.get(
        "OFFROAD_COSMOS_ENV_ID", "ct25-v1.5.4-45c6f0fb-cu130-py3.13")
    ready = env_dir / "READY.json"
    if not ready.exists():
        raise RuntimeError(f"Cosmos environment not READY: {ready}")
    meta = read_json(ready)
    if meta["cosmos_commit"] != cfg.generation.cosmos_commit:
        raise RuntimeError(f"env commit {meta['cosmos_commit']} != config {cfg.generation.cosmos_commit}")
    return {
        "cosmos_python": str(env_dir / "venv" / "bin" / "python"),
        "cosmos_src": str(cosmos_root / "src" / f"cosmos-transfer2.5-{meta['cosmos_commit'][:12]}"),
        "hf_home": str(cfg.paths.hf_home), "hf_offline": hf_offline,
        "scratch": cfg.paths.scratch, "log_dir": str(Path(ns_dir) / "logs"),
        "model": cfg.generation.model, "guardrails": cfg.generation.guardrails,
        "init_timeout_s": cfg.scheduler.init_timeout_s, "request_timeout_s": cfg.scheduler.request_timeout_s,
        "env_id": meta["env_id"],
    }


# --------------------------------------------------------------- execution
def run_pool(cfg, requests: list[dict], ns_dir: str, *, num_actors: int, resume: bool,
             hf_offline: bool = True, telemetry_every_s: float = 10.0, log=print,
             actor_cls=None, settings: dict | None = None, actor_options: dict | None = None) -> dict:
    """Generate all requests with a pool of ``num_actors`` persistent actors."""
    import ray
    from ray.exceptions import GetTimeoutError, RayActorError, RayTaskError

    ns = Path(ns_dir)
    for sub in ("records", "attempts", "logs", "telemetry"):
        (ns / sub).mkdir(parents=True, exist_ok=True)
    settings = settings or actor_settings(cfg, ns_dir, hf_offline=hf_offline)
    if actor_cls is None:
        from offroad_ray.inference.actor import CosmosActor
        actor_cls = ray.remote(CosmosActor)

    t_run0 = time.time()
    skipped, pending = [], deque()
    for r in requests:
        existing = validated_existing(ns_dir, r) if resume else None
        if existing is not None:
            skipped.append(existing)
        else:
            pending.append(r)
    budget_left = cfg.budget.max_generation_attempts - attempts_used(cfg.paths.root)
    if len(pending) > budget_left:
        raise RuntimeError(f"{len(pending)} pending requests exceed remaining attempt budget {budget_left}")
    log(f"[pool] {len(requests)} requests: {len(skipped)} reused from validated records, {len(pending)} to run; "
        f"attempt budget left {budget_left}")
    if not pending:
        return {"requests": len(requests), "skipped": len(skipped), "ran": 0, "records": skipped}

    sched = cfg.scheduler
    actor_opts = actor_options or dict(num_gpus=1, num_cpus=sched.cpus_per_actor, max_restarts=0,
                                       max_concurrency=2)
    actors: dict[int, object] = {}
    actor_info: dict[int, dict] = {}
    init_refs = {}
    replacements = 0
    next_index = 0
    events: list[dict] = []

    def spawn() -> None:
        nonlocal next_index
        i = next_index
        next_index += 1
        a = actor_cls.options(**actor_opts, name=None).remote(i, settings)
        actors[i] = a
        init_refs[a.info.remote()] = (i, time.time())
        events.append({"event": "actor_spawn", "actor_index": i, "t": time.time()})

    for _ in range(min(num_actors, len(pending))):
        spawn()

    attempts: dict[str, int] = {}
    t_enqueued = {r["request_id"]: t_run0 for r in pending}
    in_flight: dict = {}
    free: deque[int] = deque()
    results: list[dict] = []
    tel_refs: dict = {}
    last_tel = time.time()
    infra_failures = 0

    def finalize(req: dict, rec: dict) -> None:
        rec.update({k: req[k] for k in ("clip_id", "run_id", "condition", "seed", "control_weight",
                                        "clip_path", "control_path")})
        rec["identity"] = req["identity"]
        rec["prompt"] = req["sample"]["prompt"]
        rec["attempts"] = attempts.get(req["request_id"], 0)
        atomic_write_json(ns / "records" / f"{req['request_id']}.json", rec)
        results.append(rec)

    def record_attempt(req: dict, rec: dict) -> None:
        n = attempts.get(req["request_id"], 0)
        atomic_write_json(ns / "attempts" / f"{req['request_id']}.{n}.json", rec)
        atomic_write_json(ledger_dir(cfg.paths.root) / f"{int(time.time()*1000)}_{req['request_id']}_{n}.json",
                          {"ns": str(ns), "request_id": req["request_id"], "attempt": n,
                           "status": rec.get("status"), "t": time.time()})

    def replace(i: int, reason: str) -> None:
        nonlocal replacements
        a = actors.pop(i, None)
        if a is not None:
            try:
                ray.kill(a, no_restart=True)
            except Exception:
                pass
        events.append({"event": "actor_lost", "actor_index": i, "t": time.time(), "reason": reason[:500]})
        if replacements < sched.max_actor_restarts and pending:
            replacements += 1
            spawn()

    while pending or in_flight or (init_refs and pending):
        waitables = list(in_flight) + list(init_refs) + list(tel_refs)
        if not waitables:
            break
        done, _ = ray.wait(waitables, num_returns=1, timeout=5.0)
        now = time.time()
        for ref in done:
            if ref in init_refs:
                i, t_spawn = init_refs.pop(ref)
                try:
                    info = ray.get(ref)
                    actor_info[i] = info
                    events.append({"event": "actor_ready", "actor_index": i, "t": now, "init_s": info["init_s"],
                                   "host": info["host"], "cvd": info["cuda_visible_devices"],
                                   "gpu_uuid": info["child_info"]["device_uuid"]})
                    log(f"[pool] actor {i} ready on {info['host']} CVD={info['cuda_visible_devices']} "
                        f"uuid={info['child_info']['device_uuid'][:12]} init={info['init_s']:.1f}s "
                        f"(load {info['child_info']['load_s']}s, mem {info['child_info']['mem_after_load_gb']} GB)")
                    free.append(i)
                except Exception as e:
                    infra_failures += 1
                    log(f"[pool] actor {i} failed to initialise: {str(e)[:1500]}")
                    replace(i, f"init: {e}")
            elif ref in tel_refs:
                i = tel_refs.pop(ref)
                try:
                    t = ray.get(ref)
                    with open(ns / "telemetry" / f"actor{i}.jsonl", "a") as f:
                        for s in t["samples"]:
                            f.write(json.dumps(s) + "\n")
                except Exception:
                    pass
            else:
                i, req, t_dispatch = in_flight.pop(ref)
                rid = req["request_id"]
                try:
                    rec = ray.get(ref)
                    rec["queue_wait_s"] = round(t_dispatch - t_enqueued[rid], 3)
                    rec["t_dispatch"] = t_dispatch
                    record_attempt(req, rec)
                    finalize(req, rec)
                    events.append({"event": "request", "actor_index": i, "request_id": rid,
                                   "t0": rec["t_start"], "t1": rec["t_end"], "status": rec["status"]})
                    log(f"[pool] {len(results)}/{len(results) + len(pending) + len(in_flight)} {rec['status']:<21} "
                        f"actor={i} {req['clip_id'][:40]} {req['condition']} seed={req['seed']} "
                        f"gen={rec.get('child_seconds')}s peak={rec.get('peak_mem_gb')}GB")
                    free.append(i)
                except (RayActorError, RayTaskError, GetTimeoutError) as e:
                    infra_failures += 1
                    fail = {"request_id": rid, "status": "infra_failure", "actor_index": i,
                            "error": str(e)[-4000:], "t_dispatch": t_dispatch, "t_end": now}
                    record_attempt(req, fail)
                    events.append({"event": "request", "actor_index": i, "request_id": rid,
                                   "t0": t_dispatch, "t1": now, "status": "infra_failure"})
                    log(f"[pool] infra failure on actor {i} for {rid}: {str(e)[:300]}")
                    if attempts[rid] < sched.max_attempts_per_request:
                        pending.appendleft(req)
                    else:
                        finalize(req, fail)
                    replace(i, str(e))
        # dispatch
        while free and pending:
            i = free.popleft()
            if i not in actors:
                continue
            req = pending.popleft()
            attempts[req["request_id"]] = attempts.get(req["request_id"], 0) + 1
            in_flight[actors[i].generate.remote(req)] = (i, req, time.time())
        # telemetry
        if now - last_tel >= telemetry_every_s:
            last_tel = now
            for i, a in actors.items():
                if i in actor_info and i not in tel_refs.values():
                    tel_refs[a.telemetry.remote()] = i
        if not actors and pending and not init_refs:
            log("[pool] no live actors and no replacements left; stopping")
            break

    # final telemetry + shutdown
    shutdown = {}
    for i, a in list(actors.items()):
        try:
            t = ray.get(a.telemetry.remote(), timeout=60)
            with open(ns / "telemetry" / f"actor{i}.jsonl", "a") as f:
                for s in t["samples"]:
                    f.write(json.dumps(s) + "\n")
            shutdown[i] = ray.get(a.shutdown.remote(), timeout=180)
        except Exception as e:
            shutdown[i] = {"error": str(e)[:500]}
        try:
            ray.kill(a, no_restart=True)
        except Exception:
            pass
    for r in list(pending):
        finalize(r, {"request_id": r["request_id"], "status": "pending", "error": "not executed: pool exhausted"})
    t_run1 = time.time()
    summary = {
        "ns_dir": str(ns), "requests": len(requests), "skipped": len(skipped), "ran": len(results),
        "status_counts": _counts(results + skipped), "infra_failures": infra_failures,
        "actor_replacements": replacements, "wall_s": round(t_run1 - t_run0, 2),
        "t0": t_run0, "t1": t_run1, "num_actors": num_actors, "actor_info": actor_info,
        "actor_shutdown": shutdown, "settings": {k: v for k, v in settings.items()},
    }
    atomic_write_json(ns / "pool_summary.json", summary)
    atomic_write_json(ns / "events.json", events)
    consolidate(ns)
    return {**summary, "records": results + skipped}


def _counts(recs: list[dict]) -> dict:
    out: dict[str, int] = {}
    for r in recs:
        out[r.get("status", "?")] = out.get(r.get("status", "?"), 0) + 1
    return out


def consolidate(ns_dir: str | Path) -> Path:
    """Merge per-request records into manifest.jsonl (single writer, after the run)."""
    ns = Path(ns_dir)
    recs = [read_json(p) for p in sorted((ns / "records").glob("*.json"))]
    tmp = ns / ".manifest.jsonl.tmp"
    with open(tmp, "w") as f:
        for r in recs:
            r = {k: v for k, v in r.items() if k != "traceback"}
            f.write(json.dumps(r, default=str) + "\n")
    os.replace(tmp, ns / "manifest.jsonl")
    return ns / "manifest.jsonl"
