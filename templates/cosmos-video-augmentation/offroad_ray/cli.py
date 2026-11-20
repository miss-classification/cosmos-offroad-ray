"""offroad-ray command line interface.

    python -m offroad_ray.cli preflight
    python -m offroad_ray.cli env-check  configs/pilot.yaml
    python -m offroad_ray.cli fetch      configs/pilot.yaml
    python -m offroad_ray.cli prepare    configs/pilot.yaml
    python -m offroad_ray.cli generate   configs/pilot.yaml [--actors N] [--namespace NAME] [--no-resume]
    python -m offroad_ray.cli review     configs/pilot.yaml [--namespace NAME]
    python -m offroad_ray.cli benchmark  configs/benchmark.yaml --actors 1 2 4 8
    python -m offroad_ray.cli budget
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from offroad_ray.config import DEFAULT_ROOT
RESULTS = Path(DEFAULT_ROOT) / "results"   # next to the stored results from the data bundle


def _log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def _catalog(cfg) -> list[dict]:
    import pandas as pd
    from offroad_ray.preprocessing import compute_prep_hash
    path = cfg.paths.prepared_dir / compute_prep_hash(cfg) / "catalog"
    if not path.exists():
        raise SystemExit(f"no prepared catalog at {path}; run `prepare` first")
    df = pd.read_parquet(path)
    keep = set(cfg.dataset.runs)
    return [r for r in df.to_dict("records") if r["run_id"] in keep]


def _ns_dir(cfg, namespace: str | None) -> str:
    return str(cfg.paths.runs_dir / cfg.name / (namespace or "main"))


def cmd_preflight(args) -> int:
    from offroad_ray.preflight import cluster_inventory
    from offroad_ray.ray_env import init_ray
    init_ray()
    inv = cluster_inventory()
    out = RESULTS / "preflight" / "cluster_inventory.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(inv, indent=1, default=str))
    _log(f"ray {inv['ray_version']}: {len(inv['nodes'])} nodes, {inv['num_gpu_nodes']} GPU nodes, "
         f"{inv['total_gpus']:.0f} GPUs, {inv['distinct_gpu_uuids']} distinct GPU UUIDs")
    for p in inv["gpu_probes"]:
        _log(f"  {p['node_ip']:<12} CVD={p['cuda_visible_devices']} {p['nvidia_smi']}")
    _log(f"wrote {out}")
    return 0


def cmd_env_check(args) -> int:
    from offroad_ray.config import load_config
    from offroad_ray.inference.scheduler import actor_settings
    from offroad_ray.preflight import env_check
    from offroad_ray.ray_env import init_ray
    cfg = load_config(args.config)
    init_ray()
    s = actor_settings(cfg, "/tmp", hf_offline=False)
    clip = _catalog(cfg)[0]
    res = env_check(s["cosmos_python"], s["cosmos_src"], clip["clip_path"], clip["control_path"],
                    str(RESULTS / "preflight" / "env_check.json"))
    for k, v in res.items():
        if isinstance(v, dict) and "ok" in v:
            extra = {kk: vv for kk, vv in v.items() if kk not in ("ok", "traceback")}
            _log(f"  {'PASS' if v['ok'] else 'FAIL'} {k}: {json.dumps(extra, default=str)[:300]}")
    _log(f"all_ok={res.get('all_ok')} on {res.get('host')}")
    return 0 if res.get("all_ok") else 1


def cmd_hf_check(args) -> int:
    from offroad_ray.config import load_config
    from offroad_ray.inference.scheduler import actor_settings
    from offroad_ray.preflight import hf_access_check
    from offroad_ray.ray_env import init_ray
    cfg = load_config(args.config)
    init_ray()
    s = actor_settings(cfg, "/tmp", hf_offline=False)
    res = hf_access_check(s["cosmos_python"], s["hf_home"], str(RESULTS / "preflight" / "hf_access.json"))
    _log(f"token present: {res.get('token_present')} (checked on {res.get('host')})")
    for repo, v in (res.get("repos") or {}).items():
        _log(f"  {'PASS' if v['ok'] else 'FAIL'} {repo} {v.get('resolved_sha') or v.get('error')}")
    return 0 if res["all_ok"] else 1


def cmd_fetch(args) -> int:
    from offroad_ray.config import load_config
    from offroad_ray.dataset.download import fetch_runs
    from offroad_ray.ray_env import init_ray
    cfg = load_config(args.config)
    init_ray()
    t0 = time.time()
    res = fetch_runs(cfg.dataset.runs, cfg.dataset.camera, str(cfg.paths.raw_dir), cfg.dataset.max_download_gb)
    wall = time.time() - t0
    new_bytes = sum(r["tar"]["bytes"] for r in res if not r["tar"].get("skipped"))
    out = RESULTS / "dataset" / f"fetch_{cfg.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"wall_s": wall, "new_bytes": new_bytes, "runs": res}, indent=1, default=str))
    for r in res:
        _log(f"  {r['run_id']:<45} {r['tar']['bytes']/1e9:6.2f} GB  {r['extract']['num_png']} frames  "
             f"dl={r['download_s']}s{' (cached)' if r['tar'].get('skipped') else ''} on {r['host'].split('.')[0]}")
    _log(f"fetched {len(res)} runs, {new_bytes/1e9:.2f} GB new, wall {wall:.1f}s -> {out}")
    return 0


def cmd_prepare(args) -> int:
    from offroad_ray.config import load_config
    from offroad_ray.preprocessing import run_prepare
    from offroad_ray.ray_env import init_ray
    cfg = load_config(args.config)
    init_ray()
    s = run_prepare(cfg, concurrency=args.concurrency)
    out = RESULTS / "dataset" / f"prepare_{cfg.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(s, indent=1, default=str))
    _log(json.dumps(s, indent=1, default=str))
    return 0 if s["ok"] > 0 else 1


def cmd_generate(args) -> int:
    from offroad_ray.config import load_config
    from offroad_ray.inference.scheduler import build_requests, run_pool
    from offroad_ray.ray_env import init_ray
    cfg = load_config(args.config)
    init_ray()
    ns = _ns_dir(cfg, args.namespace)
    reqs = build_requests(cfg, _catalog(cfg), ns)
    if args.limit:
        reqs = reqs[: args.limit]
    _log(f"{len(reqs)} requests -> {ns}")
    s = run_pool(cfg, reqs, ns, num_actors=args.actors or cfg.scheduler.num_actors,
                 resume=not args.no_resume, hf_offline=not args.allow_download, log=_log)
    _log(f"status: {s.get('status_counts')} wall={s.get('wall_s')}s infra_failures={s.get('infra_failures')}")
    bad = {k: v for k, v in (s.get("status_counts") or {}).items()
           if k not in ("generated_unreviewed", "guardrail_blocked")}
    return 1 if bad else 0


def cmd_review(args) -> int:
    from offroad_ray.config import load_config
    from offroad_ray.quality.review import run_quality
    from offroad_ray.ray_env import init_ray
    cfg = load_config(args.config)
    init_ray()
    s = run_quality(cfg, _ns_dir(cfg, args.namespace))
    _log(json.dumps(s, indent=1, default=str))
    return 0


def cmd_benchmark(args) -> int:
    from offroad_ray.reporting.benchmark import run_benchmark
    from offroad_ray.config import load_config
    from offroad_ray.ray_env import init_ray
    cfg = load_config(args.config)
    init_ray()
    s = run_benchmark(cfg, _catalog(cfg), args.actors, log=_log)
    _log(json.dumps(s, indent=1, default=str))
    return 0


def cmd_judge(args) -> int:
    """Blind VLM judge (Cosmos-Reason1-7B) over guardrails-ON namespaces plus real source clips."""
    import pandas as pd
    from offroad_ray.config import load_config
    from offroad_ray.inference.scheduler import actor_settings
    from offroad_ray.quality.vlm_judge import items_from_namespaces, items_from_sources, run_judge
    from offroad_ray.ray_env import init_ray
    cfg = load_config(args.config)
    init_ray()
    nss = {}
    for spec in args.ns:
        label, _, path = spec.partition("=")
        nss[label] = path
    items = items_from_namespaces(nss)
    if args.sources:
        items += items_from_sources(_catalog(cfg))
    if args.limit:
        items = items[: args.limit]
    s = actor_settings(cfg, "/tmp", hf_offline=True)
    _log(f"judging {len(items)} clips on {args.gpus} GPUs -> {args.out}")
    df = run_judge(items, args.out, cosmos_python=s["cosmos_python"], hf_home=s["hf_home"], num_gpus=args.gpus, log=_log)
    _log(f"judged {int(df['judged'].sum())}/{len(df)}, parsed ok {int((df.get('ok') == True).sum())}")
    return 0


def cmd_budget(args) -> int:
    from offroad_ray.reporting.benchmark import budget_status
    print(json.dumps(budget_status(), indent=1, default=str))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="offroad-ray")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("preflight").set_defaults(fn=cmd_preflight)
    for name, fn in [("env-check", cmd_env_check), ("hf-check", cmd_hf_check), ("fetch", cmd_fetch), ("prepare", cmd_prepare),
                     ("review", cmd_review)]:
        sp = sub.add_parser(name)
        sp.add_argument("config")
        sp.add_argument("--namespace", default=None)
        sp.add_argument("--concurrency", type=int, default=16)
        sp.set_defaults(fn=fn)
    g = sub.add_parser("generate")
    g.add_argument("config")
    g.add_argument("--actors", type=int, default=None)
    g.add_argument("--namespace", default=None)
    g.add_argument("--limit", type=int, default=None)
    g.add_argument("--no-resume", action="store_true")
    g.add_argument("--allow-download", action="store_true",
                   help="let actors fetch checkpoints (use for the single-actor staging run only)")
    g.set_defaults(fn=cmd_generate)
    b = sub.add_parser("benchmark")
    b.add_argument("config")
    b.add_argument("--actors", type=int, nargs="+", default=[1, 2, 4, 8])
    b.set_defaults(fn=cmd_benchmark)
    j = sub.add_parser("judge")
    j.add_argument("config", help="config whose catalog supplies the real source clips")
    j.add_argument("--ns", nargs="+", required=True, help="label=/path/to/namespace (guardrails ON only)")
    j.add_argument("--sources", action="store_true", help="also judge the real source clips (calibration)")
    j.add_argument("--out", required=True)
    j.add_argument("--gpus", type=int, default=8)
    j.add_argument("--limit", type=int, default=None)
    j.set_defaults(fn=cmd_judge)
    sub.add_parser("budget").set_defaults(fn=cmd_budget)
    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
