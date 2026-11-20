"""Unattended phase 2 (user approved 2026-11-18 ~16:00: GPU limit 3.0 h, realism sweep, VLM judge,
showcase regeneration with the winning setting, one depth timing probe).

Waits for jobs/run_remaining.py, then:
  1. realism_a, realism_b generation (12 requests each, 8 actors)
  2. VLM judge (7 GPUs) on pilot + realism_a + realism_b + real sources, CONCURRENT with the
     depth probe (1 GPU)
  3. pick winner among {pilot (seed 101 subset), realism_a, realism_b}
  4. if a realism setting wins: prepare + generate configs/showcase_v2.yaml (9 scenes x 4 conditions)
  5. VLM judge on main (+ showcase_v2), if the budget allows
Each GPU step is skipped (never truncated mid-run) when its estimate would pass the 3.0 h limit.
Progress: /mnt/cluster_storage/cosmos-offroad/logs/run_phase2.log, run_phase2_state.json
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
PY = str(PROJECT / ".venv" / "bin" / "python")
ROOT = Path("/mnt/cluster_storage/cosmos-offroad")
RUNS = ROOT / "runs"
LOGS = ROOT / "logs"
STATE = LOGS / "run_phase2_state.json"
LIMIT_H = 4.0  # raised 3.0 -> 3.5 -> 4.0 by the user (2026-11-18 16:40)
UNCOUNTED_H = 0.12
state: dict = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "steps": []}


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def save(step, **kw):
    state["steps"].append({"step": step, "t": time.strftime("%H:%M:%S"), **kw})
    STATE.write_text(json.dumps(state, indent=1, default=str))


def cli(*args, logname):
    log(f"run: offroad_ray.cli {' '.join(args)}  (log {LOGS / logname})")
    with open(LOGS / logname, "w") as f:
        rc = subprocess.call([PY, "-m", "offroad_ray.cli", *args], stdout=f, stderr=subprocess.STDOUT, cwd=str(PROJECT))
    log(f"  rc={rc}")
    return rc


def used_h():
    from offroad_ray.reporting.benchmark import budget_status
    return budget_status()["gpu_phase_wall_time_h"]["used"] + UNCOUNTED_H


def fits(step, est_min):
    u = used_h()
    ok = u + est_min / 60 <= LIMIT_H
    log(f"budget: used {u:.2f} h, {step} est {est_min:.0f} min -> {'run' if ok else 'SKIP'} (limit {LIMIT_H} h)")
    save(f"budget:{step}", used_h=round(u, 3), est_min=est_min, run=ok)
    return ok


def judge(label, nss, out, gpus, sources=False):
    args = ["judge", "configs/main.yaml", "--out", str(out), "--gpus", str(gpus), "--ns"] + [f"{k}={v}" for k, v in nss.items()]
    if sources:
        args.append("--sources")
    return cli(*args, logname=f"judge_{label}.log")


def scores(out, sets):
    import pandas as pd
    df = pd.read_parquet(Path(out) / "judgments.parquet")
    df = df[(df["ok"] == True) & (df["kind"] == "synthetic")]  # noqa: E712
    res = {}
    for s in sets:
        d = df[df["set"] == s]
        if s == "pilot":
            d = d[d["seed"] == 101]  # same 12 clip x condition pairs as the realism sets
        if len(d) == 0:
            continue
        res[s] = {"n": int(len(d)), **{k: round(float(d[k].mean()), 3) for k in
                  ("photorealism", "physical_plausibility", "condition_intensity", "temporal_stability")},
                  "condition_match": round(float(d["condition_match"].mean()), 3)}
        res[s]["quality"] = round((res[s]["photorealism"] + res[s]["physical_plausibility"] + res[s]["condition_intensity"]) / 3, 3)
    return res


def main():
    # also wait for any generation started by a previous instance of this driver (restart-safe;
    # the next generate call then resumes and reuses validated outputs)
    while any(subprocess.run(["pgrep", "-f", pat], capture_output=True).returncode == 0
              for pat in ("jobs/run_remaining.py", "offroad_ray.cli generate configs/realism")):
        time.sleep(15)
    log("run_remaining finished")
    save("start_after_run_remaining")

    # 1. realism sweep
    for v in ("a", "b"):
        if fits(f"realism_{v}", 6):
            cli("generate", f"configs/realism_{v}.yaml", logname=f"realism_{v}.log")
            save(f"realism_{v}")

    # 2. judge (7 GPUs) || depth probe (1 GPU)
    j1 = RUNS / "judge" / "sweep"
    nss = {"pilot": RUNS / "pilot/main"}
    for v in ("a", "b"):
        if (RUNS / f"realism_{v}/main/pool_summary.json").exists():
            nss[f"realism_{v}"] = RUNS / f"realism_{v}/main"
    depth = None
    if fits("judge_sweep+depth_probe", 25):
        depth = subprocess.Popen([PY, "jobs/depth_probe.py", "--condition", "snow"], cwd=str(PROJECT),
                                 stdout=open(LOGS / "depth_probe.log", "w"), stderr=subprocess.STDOUT)
        log("depth probe started on 1 GPU")
        judge("sweep", nss, j1, gpus=7, sources=True)
        save("judge_sweep")
        depth.wait()
        log(f"depth probe rc={depth.returncode}")
        save("depth_probe", rc=depth.returncode)

    # 3. winner
    winner = "pilot"
    if (j1 / "judgments.parquet").exists():
        sc = scores(j1, list(nss))
        log(f"sweep scores: {json.dumps(sc)}")
        base = sc.get("pilot")
        cands = [s for s in sc if s != "pilot" and base and sc[s]["condition_match"] >= base["condition_match"] - 0.1
                 and sc[s]["temporal_stability"] >= base["temporal_stability"] - 0.5]
        best = max(cands, key=lambda s: sc[s]["quality"], default=None)
        if best and base and sc[best]["quality"] > base["quality"]:
            winner = best
        (PROJECT / "results" / "realism").mkdir(parents=True, exist_ok=True)
        (PROJECT / "results" / "realism" / "sweep_scores.json").write_text(json.dumps(
            {"scores": sc, "winner": winner, "rule": "max mean(photorealism, physical_plausibility, condition_intensity) "
             "subject to condition_match >= pilot - 0.1 and temporal_stability >= pilot - 0.5; pilot wins ties",
             "judge": "Cosmos-Reason1-7B, blind, automated"}, indent=1))
        save("winner", winner=winner, scores=sc)
    log(f"winner: {winner}")

    # 4. showcase regeneration with the winning setting
    nss2 = {"main": RUNS / "main/main"}
    if winner != "pilot":
        import yaml
        wcfg = yaml.safe_load((PROJECT / f"configs/{winner}.yaml").read_text())
        main_cfg = yaml.safe_load((PROJECT / "configs/main.yaml").read_text())
        show = {"name": "showcase_v2", "dataset": main_cfg["dataset"], "clips": main_cfg.get("clips", {"clips_per_run": 3}),
                "max_clips": 9, "control": wcfg["control"],
                "generation": {"conditions_file": "conditions_v2.yaml", "seeds": [101],
                               "control_weights": wcfg["generation"]["control_weights"]},
                "scheduler": {"num_actors": 8}, "budget": {"max_generation_attempts": 500}}
        (PROJECT / "configs/showcase_v2.yaml").write_text(
            f"# Generated by jobs/run_phase2.py: winning realism setting ({winner}) on one clip per main run.\n"
            + yaml.safe_dump(show, sort_keys=False))
        cli("prepare", "configs/showcase_v2.yaml", logname="prepare_showcase_v2.log")
        if fits("showcase_v2", 11):
            cli("generate", "configs/showcase_v2.yaml", logname="showcase_v2.log")
            save("showcase_v2")
            if (RUNS / "showcase_v2/main/pool_summary.json").exists():
                nss2["showcase_v2"] = RUNS / "showcase_v2/main"

    # 5. judge main (+ showcase_v2)
    if fits("judge_main", 22):
        judge("main", nss2, RUNS / "judge" / "main", gpus=8)
        save("judge_main")
    from offroad_ray.reporting.benchmark import budget_status
    save("budget_final", **budget_status())
    log("phase 2 done")
    save("done")


if __name__ == "__main__":
    main()
