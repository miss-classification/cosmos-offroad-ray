"""Phase E probe: time ONE depth-controlled generation on one A100 before deciding on a depth study.

Upstream defaults for the non-distilled depth model are kept (35 steps, guidance 3); the depth map
is computed on the fly by Video-Depth-Anything-Small (Apache-2.0). Guardrails ON. Downloads the
depth checkpoint on first use (HF_HUB_OFFLINE not set). Counts as one generation attempt.

    .venv/bin/python jobs/depth_probe.py [--condition snow] [--seed 101]
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

CHILD = r'''
import json, os, sys, time
from pathlib import Path
spec = json.loads(sys.argv[1])
t0 = time.time()
from cosmos_oss.init import init_environment, init_output_dir
init_environment()
import torch
from cosmos_transfer2.config import SetupArguments, InferenceArguments
from cosmos_transfer2.inference import Control2WorldInference
out = Path(spec["out_dir"]); out.mkdir(parents=True, exist_ok=True)
init_output_dir(out)
setup = SetupArguments(model="depth", output_dir=out, disable_guardrails=False, keep_going=True, benchmark=True)
inf = Control2WorldInference(setup, batch_hint_keys=["depth"])
torch.cuda.synchronize(); load_s = time.time() - t0
sample = InferenceArguments(**spec["sample"])
torch.cuda.reset_peak_memory_stats(); t1 = time.time()
path = inf._generate_sample(sample, out, sample_id=1)
torch.cuda.synchronize()
res = {"load_s": round(load_s, 1), "generate_s": round(time.time() - t1, 1), "output_path": str(path) if path else None,
       "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2), "num_steps": sample.num_steps,
       "guidance": sample.guidance, "device": torch.cuda.get_device_name(0)}
print("RESULT " + json.dumps(res), flush=True)
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default="snow")
    ap.add_argument("--seed", type=int, default=101)
    a = ap.parse_args()
    import ray
    import yaml
    from offroad_ray.config import load_config
    from offroad_ray.inference.scheduler import actor_settings, attempts_used, ledger_dir
    from offroad_ray.ray_env import WORKER_ONLY, init_ray
    from offroad_ray.storage import atomic_write_json
    from offroad_ray.cli import _catalog

    cfg = load_config(PROJECT / "configs/pilot.yaml")
    if attempts_used(cfg.paths.root) + 1 > cfg.budget.max_generation_attempts:
        raise SystemExit("attempt budget exhausted")
    init_ray()
    s = actor_settings(cfg, "/tmp", hf_offline=False)
    clip = next(c for c in _catalog(cfg) if c["run_id"].startswith("meadows"))
    prompt = next(c["prompt"] for c in yaml.safe_load((PROJECT / "configs/conditions_v2.yaml").read_text())["conditions"]
                  if c["name"] == a.condition)
    ns = Path(cfg.paths.runs_dir) / "depth_probe" / "main"
    out_dir = ns / "outputs" / f"{clip['clip_id'][:40]}_{a.condition}_{a.seed}"
    spec = {"out_dir": str(out_dir), "sample": {
        "name": "output", "prompt": prompt, "video_path": clip["clip_path"], "seed": a.seed,
        "resolution": "720", "max_frames": 93, "num_video_frames_per_chunk": 93, "keep_input_resolution": True,
        "depth": {"control_weight": 1.0}}}

    @ray.remote(num_gpus=1, num_cpus=4, resources=WORKER_ONLY)
    def run(spec: dict) -> dict:
        env = {"PATH": f"{os.path.dirname(s['cosmos_python'])}:/usr/bin:/bin:{os.path.expanduser('~/.local/bin')}",
               "HOME": os.environ.get("HOME", "/home/ray"), "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
               "PYTHONNOUSERSITE": "1", "HF_HOME": s["hf_home"], "COSMOS_EXPERIMENTAL_CHECKPOINTS": "1",
               "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "PYTHONPATH": s["cosmos_src"]}
        uvx = subprocess.run(["bash", "-lc", "command -v uvx"], capture_output=True, text=True).stdout.strip()
        if uvx:
            env["PATH"] += ":" + os.path.dirname(uvx)
        t0 = time.time()
        p = subprocess.run([s["cosmos_python"], "-c", CHILD, json.dumps(spec)], cwd=s["cosmos_src"], env=env,
                           capture_output=True, text=True, timeout=5400)
        res = {}
        for line in p.stdout.splitlines():
            if line.startswith("RESULT "):
                res = json.loads(line[7:])
        res.update(rc=p.returncode, wall_s=round(time.time() - t0, 1), host=socket.gethostname(),
                   stderr_tail=p.stderr[-4000:] if p.returncode else "")
        return res

    t0 = time.time()
    res = ray.get(run.remote(spec))
    t1 = time.time()
    rec = {"probe": "depth", "clip_id": clip["clip_id"], "condition": a.condition, "seed": a.seed,
           "prompt_set": "conditions_v2", "guardrails": True, **res, "t0": t0, "t1": t1}
    ns.mkdir(parents=True, exist_ok=True)
    atomic_write_json(ns / "probe_result.json", rec)
    atomic_write_json(ns / "pool_summary.json", {"num_actors": 1, "wall_s": round(t1 - t0, 2), "requests": 1,
                                                 "status_counts": {"generated_unreviewed" if res.get("output_path") else "failed": 1},
                                                 "settings": {"guardrails": True, "model": "depth"}, "t0": t0, "t1": t1,
                                                 "infra_failures": 0, "actor_replacements": 0})
    atomic_write_json(ledger_dir(cfg.paths.root) / f"{int(t0*1000)}_depth_probe_1.json",
                      {"ns": str(ns), "request_id": "depth_probe", "attempt": 1, "status": rec.get("output_path") and "generated", "t": t1})
    print(json.dumps({k: v for k, v in rec.items() if k != "stderr_tail"}, indent=1))
    if res.get("rc"):
        print(res.get("stderr_tail", "")[-3000:])
    return 0 if res.get("output_path") else 1


if __name__ == "__main__":
    sys.exit(main())
