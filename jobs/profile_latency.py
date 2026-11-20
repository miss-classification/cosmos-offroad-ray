"""Latency profile for the live Ray Serve demo (guardrails OFF, timing only, never evaluated).

Runs four variants at the same time on the 8 single-GPU nodes so the GPU-phase
wall time is paid once:

  base     1 GPU   720p x2 (first is warm-up), then 480p x2
  compiled 1 GPU   torch.compile on the DiT + tokenizer compile 'aggressive', 720p x3
  cp2      2 GPUs  context parallel across 2 nodes (NCCL over the network), 720p x2
  cp4      4 GPUs  context parallel across 4 nodes, 720p x2

Writes runs/latency_profile/<ns>/{results.json,pool_summary.json}, counts one
ledger attempt per logical request, and refuses to start if the GPU-phase
budget would be exceeded.

Usage: .venv/bin/python jobs/profile_latency.py [--variants base compiled cp2 cp4]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from offroad_ray.cli import _catalog  # noqa: E402
from offroad_ray.config import load_config  # noqa: E402
from offroad_ray.inference.scheduler import actor_settings, build_requests, ledger_dir  # noqa: E402
from offroad_ray.ray_env import init_ray  # noqa: E402
from offroad_ray.reporting.benchmark import budget_status  # noqa: E402
from offroad_ray.storage import atomic_write_json  # noqa: E402

PROMPT = ("A violent snowstorm in the backcountry. Strong wind drives thick blowing snow sideways across "
          "the trail, snow streaks and flakes hit the camera lens, low visibility, drifting snow covers "
          "the ground and vegetation, flat grey light, cold blue tint, realistic dashcam footage.")

VARIANTS = {
    "base": {"gpus": 1, "options": {}, "plan": ["720", "720", "480", "480"]},
    "compiled": {"gpus": 1, "options": {"compile_net": True, "compile_tokenizer": "aggressive"},
                 "plan": ["720", "720", "720"]},
    "cp2": {"gpus": 2, "options": {}, "plan": ["720", "720"]},
    "cp4": {"gpus": 4, "options": {}, "plan": ["720", "720"]},
}
EST_H = 0.25  # conservative estimate of GPU-phase wall time for one profile run
LIMIT_H = 4.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    args = ap.parse_args()

    b = budget_status()
    used_h = b["gpu_phase_wall_time_h"]["used"]
    used_att = b["generation_attempts"]["used"]
    n_att = sum(len(VARIANTS[v]["plan"]) for v in args.variants)
    if used_h + EST_H > LIMIT_H or used_att + n_att > 600:
        print(f"refusing: budget {used_h:.2f} h + {EST_H} h > {LIMIT_H} h or attempts {used_att}+{n_att} > 600")
        return 2

    cfg = load_config("configs/pilot.yaml")
    ns = Path(cfg.paths.root) / "runs" / "latency_profile" / time.strftime("p%Y%m%d_%H%M%S")
    (ns / "logs").mkdir(parents=True, exist_ok=True)
    req0 = build_requests(cfg, _catalog(cfg), str(ns))[0]

    ray = init_ray()
    from ray.util.placement_group import placement_group
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
    from offroad_ray.inference.actor import CosmosActor
    Actor = ray.remote(CosmosActor)

    base_settings = actor_settings(cfg, str(ns), hf_offline=True)
    base_settings["guardrails"] = False
    base_settings["request_timeout_s"] = 1800

    @ray.remote(num_cpus=0.01)
    def where() -> tuple[str, int]:
        import socket
        s = socket.socket()
        s.bind(("", 0))
        port = s.getsockname()[1]
        s.close()
        return ray.util.get_node_ip_address(), port

    t0 = time.time()
    groups = {}
    actor_idx = 0
    for v in args.variants:
        spec = VARIANTS[v]
        n = spec["gpus"]
        pg = placement_group([{"GPU": 1, "CPU": 4}] * n, strategy="STRICT_SPREAD" if n > 1 else "PACK")
        ray.get(pg.ready(), timeout=600)
        extra = {}
        if n > 1:
            ip, port = ray.get(where.options(scheduling_strategy=PlacementGroupSchedulingStrategy(
                pg, placement_group_bundle_index=0)).remote())
            extra = {"MASTER_ADDR": ip, "MASTER_PORT": port, "WORLD_SIZE": n, "LOCAL_RANK": 0,
                     "NCCL_DEBUG": "WARN"}
        actors = []
        for r in range(n):
            st = dict(base_settings, engine_options=spec["options"],
                      child_extra_env={**extra, "RANK": r} if n > 1 else {})
            a = Actor.options(num_gpus=1, num_cpus=4, max_restarts=0, max_concurrency=2,
                              scheduling_strategy=PlacementGroupSchedulingStrategy(
                                  pg, placement_group_bundle_index=r)).remote(actor_idx, st)
            actor_idx += 1
            actors.append(a)
        groups[v] = {"actors": actors, "pg": pg, "info_refs": [a.info.remote() for a in actors]}
        print(f"[{time.time()-t0:6.1f}s] {v}: spawned {n} actor(s)", flush=True)

    results = {v: {"init": None, "runs": [], "error": None} for v in groups}

    def run_variant(v: str) -> None:
        g = groups[v]
        try:
            infos = ray.get(g["info_refs"], timeout=2400)
            results[v]["init"] = [{"init_s": i["init_s"], "host": i["host"],
                                   "load_s": i["child_info"]["load_s"]} for i in infos]
            print(f"[{time.time()-t0:6.1f}s] {v}: ready {[i['init_s'] for i in infos]}", flush=True)
            for k, res in enumerate(VARIANTS[v]["plan"]):
                req = json.loads(json.dumps(req0))
                req["request_id"] = f"{v}_{k}_{res}"
                req["sample"]["prompt"] = PROMPT
                req["sample"]["resolution"] = res
                atomic_write_json(ledger_dir(cfg.paths.root) / f"{int(time.time()*1000)}_latency_{v}_{k}.json",
                                  {"ns": str(ns), "request_id": req["request_id"], "attempt": 0,
                                   "status": "started", "t": time.time()})
                ts = time.time()
                recs = ray.get([a.generate.remote(req) for a in g["actors"]], timeout=3600)
                lead = recs[0]
                row = {"k": k, "resolution": res, "wall_s": round(time.time() - ts, 2),
                       "child_seconds": lead.get("child_seconds"), "status": lead.get("status"),
                       "stage_times": lead.get("stage_times"), "peak_mem_gb": lead.get("peak_mem_gb"),
                       "output_path": lead.get("output_path"), "error": lead.get("error"),
                       "follower_status": [r.get("status") for r in recs[1:]]}
                results[v]["runs"].append(row)
                print(f"[{time.time()-t0:6.1f}s] {v} #{k} {res}p: {row['child_seconds']} s status={row['status']} "
                      f"stages={json.dumps(row['stage_times'])}", flush=True)
        except Exception as e:
            results[v]["error"] = f"{type(e).__name__}: {str(e)[-3000:]}"
            print(f"[{time.time()-t0:6.1f}s] {v}: ERROR {results[v]['error'][:800]}", flush=True)

    import threading
    th = [threading.Thread(target=run_variant, args=(v,)) for v in groups]
    for t in th:
        t.start()
    for t in th:
        t.join()
    t1 = time.time()

    for v, g in groups.items():
        for a in g["actors"]:
            try:
                ray.get(a.shutdown.remote(), timeout=120)
            except Exception:
                pass
            ray.kill(a)
        ray.util.remove_placement_group(g["pg"])

    n_actors = sum(VARIANTS[v]["gpus"] for v in groups)
    atomic_write_json(ns / "results.json", {"prompt": PROMPT, "variants": results, "t0": t0, "t1": t1,
                                            "guardrails": False, "purpose": "latency profile, not evaluated"})
    counts: dict[str, int] = {}
    for r in results.values():
        for row in r["runs"]:
            counts[row["status"] or "none"] = counts.get(row["status"] or "none", 0) + 1
    atomic_write_json(ns / "pool_summary.json", {"num_actors": n_actors, "wall_s": round(t1 - t0, 2),
                                                 "t0": t0, "t1": t1, "settings": {"guardrails": False},
                                                 "status_counts": counts, "kind": "latency_profile"})
    print(f"done in {t1-t0:.1f}s -> {ns}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
