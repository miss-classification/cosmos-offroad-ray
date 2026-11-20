"""Deploy (or tear down) the live "Paint the Trail" Ray Serve demo.

  .venv/bin/python jobs/serve_demo.py up   [--replicas 8] [--guardrails] [--compile] [--autoscale]
  .venv/bin/python jobs/serve_demo.py down
  .venv/bin/python jobs/serve_demo.py status

Guardrails are OFF by default (demo only; the namespace runs/serve_demo/* is never
evaluated). GPU time is booked by the API's heartbeat; ``up`` refuses to start when
the GPU-phase budget has less than --min-left-h remaining, and the API refuses new
jobs once the limit is reached. ``down`` always works.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from offroad_ray.cli import _catalog  # noqa: E402
from offroad_ray.config import load_config  # noqa: E402
from offroad_ray.inference.scheduler import actor_settings  # noqa: E402
from offroad_ray.ray_env import init_ray, runtime_env  # noqa: E402
from offroad_ray.reporting.benchmark import budget_status  # noqa: E402

APP = "paint"


def _label(c: dict) -> str:
    import re
    run = re.sub(r"\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}", "", c["run_id"])
    name = " ".join(t for t in run.replace("-", "_").split("_") if t) or c["run_id"]
    return f"{name.replace('figure 8', 'figure-8')} {int(float(c['clip_id'].split('__t')[1].split('__')[0]))}s"


def build_scenes(cfg, thumbs: Path) -> list[dict]:
    thumbs.mkdir(parents=True, exist_ok=True)
    scenes = []
    for c in sorted((c for c in _catalog(cfg) if c["status"] == "ok"), key=lambda c: (c["run_id"], c["clip_index"])):
        sid = c["clip_sha256"][:10]
        th = thumbs / f"{sid}.jpg"
        if not th.exists():
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", "2", "-i", c["clip_path"], "-frames:v", "1",
                            "-vf", "scale=480:-2", "-q:v", "4", str(th)], check=True)
        scenes.append({"scene_id": sid, "label": _label(c), "run_id": c["run_id"], "clip_id": c["clip_id"],
                       "clip_sha256": c["clip_sha256"], "clip_path": c["clip_path"],
                       "control_path": c["control_path"], "thumb_path": str(th),
                       "expect": {"frames": int(c["model_input_frames"]), "width": int(c["model_input_width"]),
                                  "height": int(c["model_input_height"]), "fps": float(c["model_input_fps"])}})
    return scenes


def up(a) -> int:
    b = budget_status()
    used, limit = b["gpu_phase_wall_time_h"]["used"], b["gpu_phase_wall_time_h"]["limit"]
    if limit - used < a.min_left_h:
        print(f"refusing: GPU-phase budget {used:.2f} of {limit} h used; less than {a.min_left_h} h left")
        return 2
    cfg = load_config("configs/main.yaml")
    root = Path(cfg.paths.root) / "runs" / "serve_demo"
    session = time.strftime("s%Y%m%d_%H%M%S")
    ns = root / session
    ns.mkdir(parents=True, exist_ok=True)
    scenes = build_scenes(cfg, root / "thumbs")
    settings = actor_settings(cfg, str(ns), hf_offline=True)
    settings["guardrails"] = bool(a.guardrails)
    settings["request_timeout_s"] = 1800
    settings["engine_options"] = {"encode_cache": True, "encode_cache_verify": False,  # verified exact (max diff 0.0, 16 hits, s20261119_153944)
                                  **({"compile_net": True, "compile_tokenizer": "aggressive"} if a.compile else {})}
    tag = f"g{int(a.guardrails)}c{int(a.compile)}"  # encode cache is exact (verified), so not in the tag
    meta = {"t0": time.time(), "replicas": a.replicas, "guardrails": bool(a.guardrails), "tag": tag,
            "session": session, "expected_s": a.expected_s,
            "ui_html": (Path(__file__).resolve().parents[1] / "src/offroad_ray/serve/ui.html").read_text(), "gpu_limit_h": limit, "attempt_limit": b["generation_attempts"]["limit"], "compile": a.compile}

    ray = init_ray()
    from ray import serve
    from offroad_ray.serve.app import PaintAPI, Painter, build_app
    # Serve deployment runtime_envs need remote URIs, so ship code as an immutable
    # snapshot on shared storage (never overwritten) and point PYTHONPATH at it.
    import shutil
    snap = root / "code" / session
    shutil.copytree(Path(__file__).resolve().parents[1] / "src" / "offroad_ray", snap / "offroad_ray",
                    ignore=shutil.ignore_patterns("__pycache__"))
    base = runtime_env()
    rt = {"pip": base["pip"], "env_vars": {**base["env_vars"], "PYTHONPATH": str(snap)}}
    import offroad_ray.serve.app as appmod
    appmod.Painter = Painter.options(ray_actor_options={"num_gpus": 1, "num_cpus": 4, "runtime_env": rt})
    appmod.PaintAPI = PaintAPI.options(ray_actor_options={"num_cpus": 1, "runtime_env": rt})
    app = build_app(settings, scenes, str(ns), meta, autoscale=a.autoscale)
    print(f"deploying {APP}: {a.replicas} GPU replicas, guardrails={'ON' if a.guardrails else 'OFF'}, "
          f"compile={a.compile}, namespace {ns}")
    t = time.time()
    serve.run(app, name=APP, route_prefix="/", blocking=False)
    while True:
        st = serve.status().applications.get(APP)
        s = st.status if st else "missing"
        print(f"  [{time.time()-t:5.0f}s] {s}", flush=True)
        if s in ("RUNNING", "DEPLOY_FAILED", "UNHEALTHY"):
            break
        time.sleep(15)
    (root / "CURRENT").write_text(json.dumps({"session": session, "ns": str(ns), "t_up": t,
                                              "ready_s": round(time.time() - t, 1)}))
    print(f"ready in {time.time()-t:.0f}s. Open the workspace Serve URL (port 8000, route /).")
    return 0 if s == "RUNNING" else 1


def ui_only(a) -> int:
    """Serve just the UI at /ui (CPU) next to a running 'paint' app; used to hot-fix the page."""
    init_ray()
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse
    from ray import serve
    html = (Path(__file__).resolve().parents[1] / "src/offroad_ray/serve/ui.html").read_text()
    html = html.replace("<head>", '<head>\n<base href="/">', 1).replace("__PEEK_BASE__", "ui/peek/")
    from fastapi import HTTPException
    from fastapi.responses import FileResponse
    peek_dir = Path(load_config("configs/main.yaml").paths.root) / "runs" / "serve_demo" / "peek"
    ui = FastAPI()

    @serve.deployment(ray_actor_options={"num_cpus": 0.1})
    @serve.ingress(ui)
    class UI:
        @ui.get("/", response_class=HTMLResponse)
        def page(self) -> str:
            return html

        @ui.get("/peek/{name}.mp4")
        def peek(self, name: str):
            if name not in {"real", "golden", "snow", "storm"}:
                raise HTTPException(404, "unknown preview")
            return FileResponse(str(peek_dir / f"{name}.mp4"), media_type="video/mp4")
    serve.run(UI.bind(), name="paintui", route_prefix="/ui", blocking=False)
    print("UI at /ui")
    return 0


def down(a) -> int:
    init_ray()
    from ray import serve
    serve.delete(APP)
    if "paintui" in serve.status().applications:
        serve.delete("paintui")
    print("deleted", APP)
    return 0


def status(a) -> int:
    init_ray()
    from ray import serve
    st = serve.status().applications.get(APP)
    print(st.status if st else "not deployed")
    if st:
        for name, d in st.deployments.items():
            print(" ", name, d.status, dict(d.replica_states))
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["up", "down", "status", "ui"])
    p.add_argument("--replicas", type=int, default=8)
    p.add_argument("--guardrails", action="store_true")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--autoscale", action="store_true")
    p.add_argument("--expected-s", type=float, default=75.0)
    p.add_argument("--min-left-h", type=float, default=0.1)
    a = p.parse_args()
    return {"up": up, "down": down, "status": status, "ui": ui_only}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
