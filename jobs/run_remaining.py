"""Unattended driver for the remaining GPU phases (started 2026-11-18 while the user was away).

Order: wait for the running pilot -> review pilot (CPU) -> Phase C benchmark 1/2/4/8 ->
budget check -> Phase D main (full 288 if it fits the 2.5 h GPU-phase limit, else whole clips
that fit; the rest stays pending for a later resume) -> review main (CPU).

Never deletes anything, never changes seeds/resolution/steps, never terminates the cluster.
Progress: /mnt/cluster_storage/cosmos-offroad/logs/run_remaining.log and run_remaining_state.json
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
PY = str(PROJECT / ".venv" / "bin" / "python")
LOGS = Path("/mnt/cluster_storage/cosmos-offroad/logs")
STATE = LOGS / "run_remaining_state.json"
LIMIT_H = 2.5            # user-approved GPU-phase wall-time limit (elapsed time while GPU pools run)
UNCOUNTED_H = 0.12       # env-check + failed smoke inits whose pool_summary was overwritten
REQS_PER_CLIP = 12       # 4 conditions x 3 seeds in main.yaml

state: dict = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "steps": []}


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)


def save(step: str, **kw) -> None:
    state["steps"].append({"step": step, "t": time.strftime("%H:%M:%S"), **kw})
    STATE.write_text(json.dumps(state, indent=1, default=str))


def cli(*args: str, logname: str) -> int:
    log(f"run: offroad_ray.cli {' '.join(args)}  (log {LOGS / logname})")
    with open(LOGS / logname, "w") as f:
        rc = subprocess.call([PY, "-m", "offroad_ray.cli", *args], stdout=f, stderr=subprocess.STDOUT,
                             cwd=str(PROJECT))
    log(f"  rc={rc}")
    return rc


def main() -> int:
    # 1. wait for the pilot started interactively
    while subprocess.run(["pgrep", "-f", "offroad_ray.cli generate configs/pilot.yaml"],
                         capture_output=True).returncode == 0:
        time.sleep(30)
    pilot = Path("/mnt/cluster_storage/cosmos-offroad/runs/pilot/main/pool_summary.json")
    ps = json.loads(pilot.read_text()) if pilot.exists() else {}
    log(f"pilot finished: {ps.get('status_counts')} wall={ps.get('wall_s')}")
    save("pilot", status_counts=ps.get("status_counts"), wall_s=ps.get("wall_s"))
    sc = ps.get("status_counts") or {}
    if not ps or sc.get("generated_unreviewed", 0) + sc.get("guardrail_blocked", 0) < 24:
        # e.g. the interactive session that launched it ended; resume reuses validated outputs only
        log("pilot incomplete; resuming it once")
        cli("generate", "configs/pilot.yaml", logname="pilot_resume.log")
        ps = json.loads(pilot.read_text()) if pilot.exists() else {}
        sc = ps.get("status_counts") or {}
        save("pilot_resume", status_counts=sc)
    if sc.get("generated_unreviewed", 0) + sc.get("guardrail_blocked", 0) < 24:
        log("pilot did not complete cleanly; stopping before benchmark")
        save("stopped", reason="pilot incomplete")
        return 1

    # 2. pilot quality review (CPU only; failure does not block GPU phases)
    rc = cli("review", "configs/pilot.yaml", logname="review_pilot.log")
    save("review_pilot", rc=rc)

    # 3. Phase C strong scaling
    rc = cli("benchmark", "configs/benchmark.yaml", "--actors", "1", "2", "4", "8", logname="benchmark.log")
    save("benchmark", rc=rc)
    if rc != 0:
        log("benchmark failed; stopping before main")
        save("stopped", reason="benchmark failed")
        return 1

    # 4. budget check for main from measured timings
    from offroad_ray.reporting.benchmark import budget_status
    b = budget_status()
    used_h = b["gpu_phase_wall_time_h"]["used"] + UNCOUNTED_H
    scaling = json.loads((PROJECT / "results/benchmark/scaling.json").read_text())
    p8 = next(p for p in scaling["points"] if p["num_actors"] == 8)
    per_req = p8["per_request_child_s_median"] * 1.08        # dispatch + validation overhead
    cold = p8["cold_start_s"] + 30
    remaining_s = (LIMIT_H - used_h) * 3600
    full = 24 * REQS_PER_CLIP
    projected_full = cold + math.ceil(full / 8) * per_req
    if projected_full <= remaining_s:
        limit = full
    else:
        fit = int((remaining_s - cold) / per_req) * 8
        limit = max(0, (fit // REQS_PER_CLIP) * REQS_PER_CLIP)
    log(f"budget: used {used_h:.2f} h of {LIMIT_H} h, attempts {b['generation_attempts']['used']}/500; "
        f"main full projected {projected_full/60:.1f} min vs {remaining_s/60:.1f} min left -> running {limit} of {full}")
    save("budget_check", used_h=used_h, projected_full_min=projected_full / 60, remaining_min=remaining_s / 60,
         main_limit=limit)
    if limit == 0:
        log("no GPU budget left for main; stopping")
        save("stopped", reason="budget")
        return 0

    # 5. Phase D main (resume on, so a later run completes any remainder)
    args = ["generate", "configs/main.yaml"] + (["--limit", str(limit)] if limit < full else [])
    rc = cli(*args, logname="main.log")
    save("main", rc=rc, limit=limit)

    # 6. main quality review
    rc = cli("review", "configs/main.yaml", logname="review_main.log")
    save("review_main", rc=rc)
    save("budget_final", **budget_status())
    log("done")
    save("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
