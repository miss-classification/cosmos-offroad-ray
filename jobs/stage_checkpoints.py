"""Driver: run stage_checkpoints.py in the Cosmos env on one worker via a Ray task."""
import json, os, socket, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from offroad_ray.ray_env import init_ray, WORKER_ONLY
import ray

ROOT = "/mnt/cluster_storage/cosmos-offroad"
PY = f"{ROOT}/envs/ct25-v1.5.4-45c6f0fb-cu130-py3.13/venv/bin/python"
OUT = f"{ROOT}/logs/stage_checkpoints.json"


@ray.remote(num_cpus=4, resources=WORKER_ONLY)
def stage(src):
    script = "/mnt/local_storage/stage_checkpoints.py"
    open(script, "w").write(src)
    env = {"PATH": f"{os.path.dirname(PY)}:/usr/bin:/bin", "HOME": os.environ.get("HOME", "/home/ray"),
           "HF_HOME": f"{ROOT}/hf_home", "PYTHONNOUSERSITE": "1", "HF_HUB_DISABLE_TELEMETRY": "1"}
    t0 = time.time()
    p = subprocess.run([PY, "-I", script, OUT], env=env, capture_output=True, text=True)
    return {"rc": p.returncode, "host": socket.gethostname(), "seconds": round(time.time() - t0, 1),
            "stderr": p.stderr[-3000:]}


if __name__ == "__main__":
    init_ray()
    src = (Path(__file__).resolve().parents[1] / "src/offroad_ray/stage_checkpoints.py").read_text()
    r = ray.get(stage.remote(src))
    print(json.dumps({k: v for k, v in r.items() if k != "stderr"}))
    if Path(OUT).exists():
        for i in json.load(open(OUT))["items"]:
            print(("PASS " if i["ok"] else "FAIL ") + i["repo"], round(i.get("bytes", 0) / 1e9, 2), "GB",
                  i.get("seconds", ""), i.get("error", ""))
