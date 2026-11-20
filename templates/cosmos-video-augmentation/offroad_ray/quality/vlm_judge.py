"""Automated VLM judge (Cosmos-Reason1-7B) over generated and real clips.

Blind protocol: the judge sees one clip at a time and is not told the requested condition or whether
the clip is real. Real TartanDrive source clips are judged with the same prompt as a calibration
baseline. Judgments are AUTOMATED scores, reported separately from human review labels.

One Ray GPU task per shard; each loads the model once in the Cosmos interpreter and appends JSON
lines to a shard file on shared storage, so a rerun resumes where it stopped.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import time
from pathlib import Path

import pandas as pd

from offroad_ray.storage import atomic_write_json, read_json

REASON1 = ("nvidia/Cosmos-Reason1-7B", "3210bec0495fdc7a8d3dbb8d58da5711eab4b423")
SCORES = ["photorealism", "temporal_stability", "physical_plausibility", "condition_intensity"]


def model_dir(hf_home: str) -> str:
    repo, rev = REASON1
    return str(Path(hf_home) / "hub" / f"models--{repo.replace('/', '--')}" / "snapshots" / rev)


def items_from_namespaces(namespaces: dict[str, str]) -> list[dict]:
    """namespaces: {label: ns_dir}. Only guardrails-ON namespaces with generated outputs."""
    items = []
    for label, ns in namespaces.items():
        summ = read_json(Path(ns) / "pool_summary.json")
        if not summ.get("settings", {}).get("guardrails", False):
            raise ValueError(f"{ns} ran with guardrails OFF; never judged or reported")
        for line in (Path(ns) / "manifest.jsonl").read_text().splitlines():
            r = json.loads(line)
            if r.get("status") == "generated_unreviewed" and r.get("output_path"):
                items.append({"judge_id": f"{label}/{r['request_id']}", "path": r["output_path"], "kind": "synthetic",
                              "set": label, "request_id": r["request_id"], "clip_id": r["clip_id"],
                              "run_id": r["run_id"], "requested_condition": r["condition"], "seed": r["seed"],
                              "control_weight": r.get("control_weight")})
    return items


def items_from_sources(catalog: list[dict]) -> list[dict]:
    return [{"judge_id": f"source/{c['clip_id']}", "path": c["clip_path"], "kind": "real", "set": "source",
             "clip_id": c["clip_id"], "run_id": c["run_id"], "requested_condition": None}
            for c in catalog if c.get("status") == "ok"]


def run_judge(items: list[dict], out_dir: str, *, cosmos_python: str, hf_home: str, num_gpus: int = 8,
              log=print) -> pd.DataFrame:
    import ray
    from offroad_ray.ray_env import WORKER_ONLY
    out = Path(out_dir)
    (out / "shards").mkdir(parents=True, exist_ok=True)
    worker_src = (Path(__file__).parent / "vlm_judge_worker.py").read_text()
    mdir = model_dir(hf_home)
    # skip anything already judged in ANY shard file of this out dir (resume across reshards)
    already = set()
    for f in (out / "shards").glob("shard_*.jsonl"):
        for line in f.read_text().splitlines():
            try:
                r = json.loads(line)
                if _reparse(r).get("ok"):
                    already.add(r["judge_id"])
            except Exception:
                pass
    todo = [it for it in items if it["judge_id"] not in already]
    log(f"[judge] {len(items)} items, {len(items) - len(todo)} already judged, {len(todo)} to run")
    if not todo:
        return merge(items, out)
    # interleave so every shard gets a mix of sets (balanced runtime); new shard files per pass
    k0 = len(list((out / "shards").glob("shard_*.jsonl")))
    shards = [todo[i::num_gpus] for i in range(num_gpus)]

    @ray.remote(num_gpus=1, num_cpus=4, resources=WORKER_ONLY)
    def judge_shard(idx: int, shard: list[dict]) -> dict:
        d = Path("/mnt/local_storage/vlm_judge")
        d.mkdir(parents=True, exist_ok=True)
        script = d / "vlm_judge_worker.py"
        script.write_text(worker_src)
        items_path = d / f"items_{idx}.json"
        items_path.write_text(json.dumps([{"judge_id": it["judge_id"], "path": it["path"]} for it in shard]))
        env = {"PATH": f"{os.path.dirname(cosmos_python)}:/usr/bin:/bin", "HOME": os.environ.get("HOME", "/home/ray"),
               "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""), "PYTHONNOUSERSITE": "1",
               "HF_HOME": hf_home, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
               "OMP_NUM_THREADS": "4", "TOKENIZERS_PARALLELISM": "false"}
        t0 = time.time()
        p = subprocess.run([cosmos_python, "-I", str(script), mdir, str(items_path),
                            str(out / "shards" / f"shard_{idx}.jsonl")],
                           env=env, capture_output=True, text=True, timeout=3600)
        return {"shard": idx, "rc": p.returncode, "host": socket.gethostname(), "wall_s": round(time.time() - t0, 1),
                "stdout": p.stdout[-1500:], "stderr": p.stderr[-3000:] if p.returncode else ""}

    t0 = time.time()
    refs = [judge_shard.remote(k0 + i, s) for i, s in enumerate(shards) if s]
    results = ray.get(refs)
    wall = time.time() - t0
    for r in results:
        log(f"[judge] shard {r['shard']} rc={r['rc']} {r['host'].split('.')[0]} {r['wall_s']}s {r['stdout'].strip()[-200:]}")
        if r["rc"]:
            log(r["stderr"][-2000:])
    atomic_write_json(out / "run.json", {"wall_s": round(wall, 1), "num_gpus": num_gpus, "items": len(items),
                                         "model": REASON1, "shards": results, "t_end": time.time()})
    # GPU-phase accounting (budget_status globs runs/*/*/pool_summary.json)
    atomic_write_json(out / "pool_summary.json", {"num_actors": num_gpus, "wall_s": round(wall, 2), "t0": t0,
                                                  "t1": t0 + wall, "requests": len(items), "kind": "vlm_judge",
                                                  "status_counts": {"judged": len(items)},
                                                  "settings": {"guardrails": None, "model": REASON1[0]}})
    return merge(items, out)


def _reparse(r: dict) -> dict:
    """Re-parse a stored record from its raw text with the current (lenient) parser."""
    from offroad_ray.quality.vlm_judge_worker import parse
    if r.get("raw") and not r.get("error"):
        parsed, err = parse(r["raw"])
        r = {**r, "ok": parsed is not None, "parse_error": err, **(parsed or {})}
    return r


def merge(items: list[dict], out_dir: str | Path) -> pd.DataFrame:
    out = Path(out_dir)
    judged = {}
    for f in sorted((out / "shards").glob("shard_*.jsonl")):
        for line in f.read_text().splitlines():
            try:
                r = _reparse(json.loads(line))
                if r.get("ok") or r["judge_id"] not in judged:
                    judged[r["judge_id"]] = r
            except Exception:
                pass
    rows = []
    for it in items:
        j = judged.get(it["judge_id"], {})
        row = {**it, **{k: v for k, v in j.items() if k not in ("judge_id", "path", "raw")}}
        row["judged"] = bool(j)
        row["observed_condition"] = j.get("condition")
        rc = it.get("requested_condition")
        row["condition_match"] = (j.get("condition") == rc) if (rc and j.get("ok")) else None
        rows.append(row)
    df = pd.DataFrame(rows)
    df.drop(columns=[c for c in ("condition",) if c in df], inplace=True)
    df.to_parquet(out / "judgments.parquet")
    with open(out / "judgments.jsonl", "w") as f:
        for r in df.to_dict("records"):
            f.write(json.dumps(r, default=str) + "\n")
    summarise(df, out)
    return df


def summarise(df: pd.DataFrame, out: Path) -> dict:
    ok = df[df.get("ok", False) == True]  # noqa: E712
    by_set = ok.groupby("set")[SCORES].mean().round(2)
    by_set["n"] = ok.groupby("set").size()
    syn = ok[ok.kind == "synthetic"]
    by_cond = syn.groupby(["set", "requested_condition"]).agg(
        n=("judge_id", "size"), photorealism=("photorealism", "mean"),
        temporal_stability=("temporal_stability", "mean"), physical_plausibility=("physical_plausibility", "mean"),
        condition_intensity=("condition_intensity", "mean"), condition_match=("condition_match", "mean")).round(2)
    real_cond = ok[ok.kind == "real"]["observed_condition"].value_counts().to_dict()
    summary = {
        "judged": int(df["judged"].sum()), "parsed_ok": int(len(ok)), "total": int(len(df)),
        "by_set": by_set.reset_index().to_dict("records"),
        "by_set_condition": by_cond.reset_index().to_dict("records"),
        "real_source_observed_conditions": real_cond,
        "confusion": syn.groupby(["requested_condition", "observed_condition"]).size().unstack(fill_value=0).to_dict(),
        "note": "Automated, blind scores from Cosmos-Reason1-7B (greedy decoding). Not human review.",
    }
    atomic_write_json(out / "summary.json", summary)
    by_cond.to_csv(out / "by_set_condition.csv")
    return summary
