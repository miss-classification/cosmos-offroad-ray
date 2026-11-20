"""Phase C strong-scaling benchmark and resource accounting.

Every actor count N runs the IDENTICAL request list in its own fresh namespace with
resume disabled, so no output is reused across points. For each point we separate:

* cold start: run start -> last actor ready (subprocess spawn + checkpoint load);
* generation window: first dispatch -> last request end (warm, steady-state);
* end-to-end wall: run start -> pool shutdown.

Speedup S(N) = T(1) / T(N) and efficiency E(N) = S(N) / N are reported for both the
generation window and end-to-end wall. Resource use is reported as node-hours of the
whole GPU worker group (all GPU nodes stay allocated even when N < 8) and as active
GPU-hours (sum of per-request generation time). No dollar figures: pricing was not verified.
"""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

from offroad_ray.storage import atomic_write_json, read_json

from offroad_ray.config import DEFAULT_ROOT
RESULTS = Path(DEFAULT_ROOT) / "results"   # next to the stored results from the data bundle

# chart palette (dataviz reference instance, light mode)
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
STATUS = {"generated_unreviewed": "#2a78d6", "guardrail_blocked": "#eda100",
          "generation_failure": "#e34948", "infra_failure": "#e34948"}


# ------------------------------------------------------------------ analysis
def analyse_namespace(ns_dir: str | Path, gpu_nodes: int = 8) -> dict:
    """Timing and resource metrics for one completed pool run."""
    ns = Path(ns_dir)
    summ = read_json(ns / "pool_summary.json")
    events = read_json(ns / "events.json")
    recs = [json.loads(line) for line in (ns / "manifest.jsonl").read_text().splitlines() if line.strip()]
    t0, t1 = summ["t0"], summ["t1"]
    ready = [e for e in events if e["event"] == "actor_ready"]
    reqs = [e for e in events if e["event"] == "request"]
    gen = [r for r in recs if r.get("status") in ("generated_unreviewed", "guardrail_blocked")]
    child_s = [r["child_seconds"] for r in gen if r.get("child_seconds") is not None]
    first_dispatch = min((r["t_dispatch"] for r in recs if r.get("t_dispatch")), default=None)
    last_end = max((e["t1"] for e in reqs), default=None)
    wall = t1 - t0
    gen_window = (last_end - first_dispatch) if first_dispatch and last_end else None
    busy = sum(e["t1"] - e["t0"] for e in reqs)
    peak = [r.get("peak_mem_gb") for r in gen if r.get("peak_mem_gb") is not None]
    return {
        "ns_dir": str(ns), "num_actors": summ["num_actors"], "requests": summ["requests"],
        "status_counts": summ["status_counts"], "infra_failures": summ["infra_failures"],
        "actor_replacements": summ["actor_replacements"],
        "wall_s": round(wall, 2),
        "cold_start_s": round(max(e["t"] for e in ready) - t0, 2) if ready else None,
        "actor_init_s": sorted(round(e["init_s"], 1) for e in ready),
        "generation_window_s": round(gen_window, 2) if gen_window else None,
        "per_request_child_s_median": round(statistics.median(child_s), 2) if child_s else None,
        "per_request_child_s_min_max": [round(min(child_s), 2), round(max(child_s), 2)] if child_s else None,
        "peak_mem_gb_max": max(peak) if peak else None,
        "distinct_hosts": len({e["host"] for e in ready}),
        "distinct_gpu_uuids": len({e["gpu_uuid"] for e in ready}),
        "requests_per_min_generation_window": round(len(gen) / (gen_window / 60), 3) if gen_window else None,
        "node_hours_gpu_group": round(gpu_nodes * wall / 3600, 4),
        "allocated_actor_gpu_hours": round(summ["num_actors"] * wall / 3600, 4),
        "active_gpu_hours": round(busy / 3600, 4),
        "active_fraction_of_node_hours": round(busy / (gpu_nodes * wall), 4) if wall else None,
    }


def scaling_table(points: list[dict]) -> list[dict]:
    base = next((p for p in points if p["num_actors"] == 1), None)
    rows = []
    for p in sorted(points, key=lambda p: p["num_actors"]):
        n = p["num_actors"]
        row = {"num_actors": n, "wall_s": p["wall_s"], "generation_window_s": p["generation_window_s"],
               "cold_start_s": p["cold_start_s"]}
        if base:
            for key, label in (("generation_window_s", "gen"), ("wall_s", "e2e")):
                if base[key] and p[key]:
                    s = base[key] / p[key]
                    row[f"speedup_{label}"] = round(s, 3)
                    row[f"efficiency_{label}"] = round(s / n, 3)
        rows.append(row)
    return rows


# ------------------------------------------------------------------- charts
def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_scaling(rows: list[dict], out: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ns = [r["num_actors"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor=SURFACE)
    ax = axes[0]
    _style(ax)
    ax.plot(ns, ns, color=INK2, linestyle=(0, (3, 3)), linewidth=1.5)
    ax.annotate("ideal", (ns[-1], ns[-1]), textcoords="offset points", xytext=(-30, 4), color=INK2, fontsize=9)
    for key, col, lab in (("speedup_gen", SERIES[0], "generation window"), ("speedup_e2e", SERIES[1], "end to end")):
        ys = [r.get(key) for r in rows]
        if all(y is not None for y in ys):
            ax.plot(ns, ys, color=col, linewidth=2, marker="o", markersize=7,
                    markeredgecolor=SURFACE, markeredgewidth=2, label=lab)
            ax.annotate(f"{ys[-1]:.2f}x", (ns[-1], ys[-1]), textcoords="offset points", xytext=(6, -4),
                        color=INK, fontsize=9)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xticks(ns, [str(n) for n in ns])
    ax.set_yticks(ns, [str(n) for n in ns])
    ax.set_xlabel("actors (one A100 each)", color=INK2)
    ax.set_title("Speedup T(1)/T(N)", color=INK, loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2)
    ax = axes[1]
    _style(ax)
    for key, col, lab in (("efficiency_gen", SERIES[0], "generation window"), ("efficiency_e2e", SERIES[1], "end to end")):
        ys = [r.get(key) for r in rows]
        if all(y is not None for y in ys):
            ax.plot(ns, [100 * y for y in ys], color=col, linewidth=2, marker="o", markersize=7,
                    markeredgecolor=SURFACE, markeredgewidth=2, label=lab)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ns, [str(n) for n in ns])
    ax.set_ylim(0, 110)
    ax.set_xlabel("actors (one A100 each)", color=INK2)
    ax.set_title("Parallel efficiency, % of ideal", color=INK, loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return out


def plot_timeline(ns_dir: str | Path, out: Path, title: str = "") -> Path:
    """Per-actor Gantt: cold start (gray) then each request, coloured by outcome."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    ns = Path(ns_dir)
    summ = read_json(ns / "pool_summary.json")
    events = read_json(ns / "events.json")
    t0 = summ["t0"]
    spawn = {e["actor_index"]: e["t"] for e in events if e["event"] == "actor_spawn"}
    ready = {e["actor_index"]: e["t"] for e in events if e["event"] == "actor_ready"}
    actors = sorted(spawn)
    fig, ax = plt.subplots(figsize=(10, 0.45 * len(actors) + 1.9), facecolor=SURFACE)
    _style(ax)
    ax.grid(True, axis="x", color=GRID)
    ax.grid(False, axis="y")
    for row, a in enumerate(actors):
        if a in ready:
            ax.barh(row, ready[a] - spawn[a], left=spawn[a] - t0, height=0.6, color="#c3c2b7")
        for e in events:
            if e["event"] == "request" and e["actor_index"] == a:
                ax.barh(row, e["t1"] - e["t0"], left=e["t0"] - t0, height=0.6,
                        color=STATUS.get(e["status"], INK2), edgecolor=SURFACE, linewidth=1.5)
    ax.set_yticks(range(len(actors)), [f"actor {a}" for a in actors])
    ax.invert_yaxis()
    ax.set_xlabel("seconds since run start", color=INK2)
    ax.set_title(title or f"{summ['num_actors']} actors, {summ['requests']} requests", color=INK, loc="left", fontsize=11)
    present = {e["status"] for e in events if e["event"] == "request"}
    handles = [Patch(color="#c3c2b7", label="model load (cold start)")] + \
              [Patch(color=STATUS[s], label=s.replace("_", " ")) for s in STATUS if s in present]
    ax.legend(handles=handles, frameon=False, fontsize=8, labelcolor=INK2, loc="upper left",
              bbox_to_anchor=(0, -0.45 if len(actors) < 3 else -0.12), ncol=len(handles))
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return out


# ---------------------------------------------------------------- benchmark
def run_benchmark(cfg, catalog: list[dict], actor_counts: list[int], *, log=print) -> dict:
    """Strong scaling: identical requests, one fresh namespace per actor count, no resume."""
    from offroad_ray.inference.scheduler import attempts_used, build_requests, run_pool
    stamp = time.strftime("%Y%m%d-%H%M%S")
    base = Path(cfg.paths.runs_dir) / cfg.name
    probe = build_requests(cfg, catalog, str(base / "probe"))
    total = len(probe) * len(actor_counts)
    left = cfg.budget.max_generation_attempts - attempts_used(cfg.paths.root)
    if total > left:
        raise RuntimeError(f"benchmark needs {total} attempts, budget has {left} left; not starting")
    log(f"[bench] {len(probe)} requests x actor counts {actor_counts} = {total} attempts ({left} left)")
    points = []
    for n in actor_counts:
        ns = str(base / f"bench-{stamp}-a{n}")
        reqs = build_requests(cfg, catalog, ns)
        assert [r["request_id"] for r in reqs] == [r["request_id"] for r in probe]
        log(f"[bench] N={n} -> {ns}")
        run_pool(cfg, reqs, ns, num_actors=n, resume=False, log=log)
        p = analyse_namespace(ns)
        points.append(p)
        plot_timeline(ns, RESULTS / "benchmark" / f"timeline_a{n}.png",
                      title=f"{n} actor{'s' if n > 1 else ''}: {p['requests']} identical requests")
        log(f"[bench] N={n}: wall {p['wall_s']}s, cold {p['cold_start_s']}s, gen window {p['generation_window_s']}s")
    rows = scaling_table(points)
    out = {"stamp": stamp, "config": cfg.name, "requests_per_point": len(probe),
           "request_ids": [r["request_id"] for r in probe], "points": points, "scaling": rows,
           "note": "node_hours count every GPU worker node (8) for the run wall time, since the "
                   "cluster keeps them allocated; no cost figures because pricing was not verified."}
    atomic_write_json(RESULTS / "benchmark" / "scaling.json", out)
    plot_scaling(rows, RESULTS / "benchmark" / "scaling.png")
    return {"scaling": rows, "json": str(RESULTS / "benchmark" / "scaling.json")}


# ------------------------------------------------------------------- budget
def budget_status(root: str | None = None) -> dict:
    from offroad_ray.config import DEFAULT_ROOT
    root = root or DEFAULT_ROOT
    """Usage against the brief's ceilings, derived from on-disk records only."""
    from offroad_ray.inference.scheduler import attempts_used
    pools = []
    for p in sorted(Path(root, "runs").glob("*/*/pool_summary.json")):
        s = read_json(p)
        pools.append({"ns": str(p.parent.relative_to(Path(root, "runs"))), "num_actors": s["num_actors"],
                      "wall_s": s["wall_s"], "t0": s.get("t0"), "t1": s.get("t1"), "guardrails": s.get("settings", {}).get("guardrails"),
                      "status_counts": s["status_counts"]})
    dl = 0
    for f in (RESULTS / "dataset").glob("fetch_*.json"):
        dl = max(dl, read_json(f).get("new_bytes", 0))
    raw = Path(root, "data", "tartandrive2", "raw")
    tar_bytes = sum(m.stat().st_size for m in raw.glob("*/*.tar")) if raw.exists() else 0
    # elapsed time while any GPU phase ran: union of [t0, t1] intervals (concurrent phases count once)
    iv = sorted((p["t0"], p["t1"]) for p in pools if p.get("t0") and p.get("t1"))
    union, cur = 0.0, None
    for a, b in iv:
        if cur and a <= cur[1]:
            cur[1] = max(cur[1], b)
        else:
            if cur:
                union += cur[1] - cur[0]
            cur = [a, b]
    if cur:
        union += cur[1] - cur[0]
    gpu_wall = sum(p["wall_s"] for p in pools)
    return {
        "generation_attempts": {"used": attempts_used(root), "limit": 1000000, "note": "raised 500 -> 600 (phase 3); lifted by the user 2026-11-19 ('there is no quota'); still counted"},
        "gpu_phase_wall_time_h": {"used": round(union / 3600, 3), "limit": 1000.0, "sum_of_pool_walls_h": round(gpu_wall / 3600, 3),
                                  "note": "elapsed union of GPU pool intervals; limit raised 2 h -> 2.5 h -> 3.0 h -> 3.5 h -> 4.0 h, then lifted by the user on 2026-11-19 (still counted); "
                                          "env-check and failed smoke inits (about 0.12 h) not included"},
        "source_download_gb": {"used": round(tar_bytes / 1e9, 2), "limit": 100},
        "pools": pools,
    }
