"""Build animated showcase GIFs from a guardrails-ON namespace.

    .venv/bin/python jobs/build_showcase.py /mnt/cluster_storage/cosmos-offroad/runs/pilot/main [--seed 101]

Writes small GIFs (and the MP4s they were made from) to results/showcase/<ns name>/.
Every generated panel is labelled SYNTHETIC; sources are labelled as real TartanDrive 2.0 footage.
Refuses guardrails-OFF namespaces (smoke).
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

CONDS = ["fog", "rain", "night", "snow"]


def scene(clip_id: str) -> str:
    return clip_id.split("__")[0].split("_20")[0].replace("_", " ") or clip_id[:20]


def job(kind: str, args: dict) -> str:
    from offroad_ray.reporting import montage as m
    out = Path(args.pop("out"))
    mp4 = str(out.with_suffix(".mp4"))
    if kind == "grid":
        m.condition_grid(args["source"], args["generated"], mp4, w=384, h=204, title=args["title"], seed=args["seed"])
        width = 1152
    elif kind == "before_after":
        m.before_after(args["source"], args["generated"], mp4, condition=args["condition"], w=480, h=255,
                       title=args["title"])
        width = 960
    elif kind == "wipe":
        m.wipe(args["source"], args["generated"], mp4, condition=args["condition"], w=768, h=408)
        width = 768
    elif kind == "triptych":
        m.triptych(args["source"], args["control"], args["generated"], mp4, condition=args["condition"],
                   seed=args["seed"], w=384, h=204, title=args["title"])
        width = 1152
    m.mp4_to_gif(mp4, str(out), width=width, fps=12)
    return str(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ns")
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--max-scenes", type=int, default=12)
    a = ap.parse_args()
    ns = Path(a.ns)
    summ = json.loads((ns / "pool_summary.json").read_text())
    if not summ.get("settings", {}).get("guardrails", False):
        raise SystemExit("refusing: guardrails-OFF namespace (never shown)")
    recs = [json.loads(l) for l in (ns / "manifest.jsonl").read_text().splitlines() if l.strip()]
    recs = [r for r in recs if r.get("status") == "generated_unreviewed" and r["seed"] == a.seed]
    conds = list(dict.fromkeys(r["condition"] for r in recs))  # conditions present, in config order
    by_clip: dict[str, dict] = {}
    for r in recs:
        by_clip.setdefault(r["clip_id"], {"source": r["clip_path"], "control": r["control_path"], "gen": {}})
        by_clip[r["clip_id"]]["gen"][r["condition"]] = r["output_path"]
    # one clip per scene (first by id), complete condition sets only
    chosen: dict[str, tuple[str, dict]] = {}
    for cid in sorted(by_clip):
        c = by_clip[cid]
        if all(k in c["gen"] for k in conds) and scene(cid) not in chosen:
            chosen[scene(cid)] = (cid, c)
    chosen = dict(list(chosen.items())[: a.max_scenes])
    out = PROJECT / "results" / "showcase" / ns.parent.name
    out.mkdir(parents=True, exist_ok=True)
    attribution = "Source: TartanDrive 2.0 (CC BY 4.0)"
    jobs = []
    for i, (sc, (cid, c)) in enumerate(chosen.items()):
        slug = sc.replace(" ", "_")
        jobs.append(("grid", {"out": out / f"grid_{slug}.gif", "source": c["source"],
                              "generated": {k: c["gen"][k] for k in conds}, "seed": a.seed,
                              "title": f"{sc}: one real clip, {len(conds)} synthetic conditions. {attribution}"}))
        cond = conds[i % len(conds)]  # rotate so every condition gets a before/after
        jobs.append(("before_after", {"out": out / f"before_after_{slug}_{cond}.gif", "source": c["source"],
                                      "generated": c["gen"][cond], "condition": cond,
                                      "title": f"{sc}. {attribution}"}))
        wc = conds[(i + 2) % len(conds)]
        jobs.append(("wipe", {"out": out / f"wipe_{slug}_{wc}.gif", "source": c["source"],
                              "generated": c["gen"][wc], "condition": wc}))
        if i == 0 and "night" in c["gen"]:
            jobs.append(("triptych", {"out": out / f"triptych_{slug}_night.gif", "source": c["source"],
                                      "control": c["control"], "generated": c["gen"]["night"], "condition": "night",
                                      "seed": a.seed, "title": f"{sc}: what the model sees. {attribution}"}))
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(job, k, dict(v)) for k, v in jobs]
        for f in futs:
            p = Path(f.result())
            print(f"{p.stat().st_size/1e6:6.1f} MB  {p.relative_to(PROJECT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
