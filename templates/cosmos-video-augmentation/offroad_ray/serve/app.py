"""Ray Serve application: pick a real drive, choose a condition, generate on GPUs.

Two tiers, composed with ``.bind()``:

* ``Painter`` (1 GPU per replica, ``max_ongoing_requests=1``) owns one persistent Cosmos
  Transfer 2.5 subprocess through ``CosmosActor``, the same class the batch pipeline uses,
  so the model loads once per replica. ``ReplayPainter`` has the same interface and runs on
  CPU: it returns stored outputs for the preset conditions, so the whole application can
  be tried without GPUs.
* ``PaintAPI`` (FastAPI ingress, CPU) serves the page and a job API. ``POST /api/paint``
  returns job IDs at once and the browser polls ``GET /api/jobs/<id>``, so a request that
  takes 80 seconds never holds a connection open.

Scene IDs are used as multiplexed model IDs (``@serve.multiplexed``): Serve sends repeat
requests for a drive to a replica that already holds it, where the drive's video encodes
are cached on the GPU.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
import time
import uuid
from pathlib import Path

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from ray import serve

from offroad_ray.config import DEFAULT_ROOT

UI_HTML = Path(__file__).parent / "ui.html"
PEEK = {"real": ("5d20270fcc", "src"), "golden": ("5d20270fcc", "golden_hour_glare"),
        "nightrain": ("5d20270fcc", "night_rain"), "autumn": ("5d20270fcc", "autumn")}
BEST = {"golden_hour_glare", "night_rain", "autumn", "night", "blue_hour_dusk"}


def _rid(scene: dict, prompt: str, seed: int, resolution: str, tag: str) -> str:
    h = hashlib.sha256(json.dumps([scene["clip_sha256"], prompt.strip(), seed, resolution, tag]).encode())
    return "s" + h.hexdigest()[:19]


def load_presets(config_dir: str | Path) -> list[dict]:
    """The exact prompts used by the batch runs (configs/conditions*.yaml)."""
    out, seen = [], set()
    for name in ("conditions_v3.yaml", "conditions.yaml"):
        f = Path(config_dir) / name
        if not f.exists():
            continue
        for c in yaml.safe_load(f.read_text())["conditions"]:
            if c["name"] in seen:
                continue
            seen.add(c["name"])
            out.append({"name": c["name"], "label": c["name"].replace("_", " ").replace(" glare", "").capitalize(),
                        "prompt": " ".join(c["prompt"].split()), "best": c["name"] in BEST})
    out.sort(key=lambda p: (not p["best"], p["label"]))
    return out


# ----------------------------------------------------------------- GPU tier
@serve.deployment(max_ongoing_requests=1, ray_actor_options={"num_gpus": 1, "num_cpus": 4},
                  health_check_period_s=30, health_check_timeout_s=120, graceful_shutdown_timeout_s=120)
class Painter:
    def __init__(self, settings: dict):
        from offroad_ray.inference.actor import CosmosActor
        idx = abs(hash(serve.get_replica_context().replica_id.unique_id)) % 10_000
        self.engine = CosmosActor(idx, settings)
        self.info = self.engine.info()
        self.served = 0

    @serve.multiplexed(max_num_models_per_replica=12)
    async def _scene(self, scene_id: str) -> str:
        return scene_id

    async def paint(self, req: dict) -> dict:
        await self._scene(serve.get_multiplexed_model_id())
        t0 = time.time()
        rec = await asyncio.to_thread(self.engine.generate, req)
        self.served += 1
        rec.update(replica_host=self.engine.host, replica_served=self.served, replica_wall_s=round(time.time() - t0, 2))
        return rec

    def check_health(self) -> None:
        if self.engine.proc.poll() is not None:
            raise RuntimeError("cosmos subprocess exited")


@serve.deployment(max_ongoing_requests=1, ray_actor_options={"num_cpus": 0.5})
class ReplayPainter:
    """CPU stand-in for Painter: returns the stored output for a preset condition."""

    def __init__(self, media_dir: str, delay_s: float = 3.0):
        self.media = Path(media_dir)
        self.delay_s = delay_s
        self.served = 0

    @serve.multiplexed(max_num_models_per_replica=12)
    async def _scene(self, scene_id: str) -> str:
        return scene_id

    async def paint(self, req: dict) -> dict:
        sid = await self._scene(serve.get_multiplexed_model_id())
        t0 = time.time()
        src = self.media / sid / f"{req.get('condition')}.mp4"
        rec = {"request_id": req["request_id"], "replica_host": "replay"}
        if not src.exists():
            rec.update(status="failed", error="no stored output for this drive and condition")
        else:
            await asyncio.sleep(self.delay_s)   # stands in for generation time; the live model takes about 80 s
            out = Path(req["ns_dir"]) / "outputs" / req["request_id"] / "output.mp4"
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, out)
            rec.update(status="generated_unreviewed", output_path=str(out))
        self.served += 1
        rec.update(replica_served=self.served, replica_wall_s=round(time.time() - t0, 2))
        return rec


# ----------------------------------------------------------------- API tier
class PaintIn(BaseModel):
    prompt: str
    scenes: list[str]
    seed: int = 101
    resolution: str = "720"


api = FastAPI(title="Every Season on Every Trail")


@serve.deployment(max_ongoing_requests=256, ray_actor_options={"num_cpus": 1})
@serve.ingress(api)
class PaintAPI:
    def __init__(self, painter, scenes: list[dict], meta: dict):
        self.painter = painter
        self.scenes = {s["scene_id"]: s for s in scenes}
        self.order = [s["scene_id"] for s in scenes]
        self.meta = meta
        self.gallery = Path(meta["gallery"])
        (self.gallery / "outputs").mkdir(parents=True, exist_ok=True)
        self.by_prompt = {p["prompt"]: p["name"] for p in meta["presets"]}
        self.jobs: dict[str, dict] = {}
        self.durations: list[float] = []
        self.t0 = time.time()

    # -- routes ---------------------------------------------------------------
    @api.get("/", response_class=HTMLResponse)
    def index(self) -> str:
        return self.meta["ui_html"].replace("__PEEK_BASE__", "api/peek/")

    @api.get("/api/peek/{name}.mp4")
    def peek(self, name: str):
        if name not in PEEK:
            raise HTTPException(404, "unknown preview")
        sid, cond = PEEK[name]
        f = Path(self.meta["media_dir"]) / sid / f"{cond}.mp4"
        if not f.exists():
            raise HTTPException(404, "preview not downloaded")
        return FileResponse(f, media_type="video/mp4")

    @api.get("/api/scenes")
    def list_scenes(self) -> dict:
        return {"scenes": [{k: self.scenes[s][k] for k in ("scene_id", "label", "run_id")} for s in self.order],
                "replicas": self.meta["replicas"], "guardrails": self.meta["guardrails"], "mode": self.meta["mode"],
                "median_s": self._median(),
                "presets": [[p["label"], p["best"], p["prompt"]] for p in self.meta["presets"]]}

    @api.get("/api/scenes/{scene_id}/thumb.jpg")
    def thumb(self, scene_id: str):
        return FileResponse(self._scene(scene_id)["thumb_path"], media_type="image/jpeg")

    @api.get("/api/scenes/{scene_id}/source.mp4")
    def source(self, scene_id: str):
        return FileResponse(self._scene(scene_id)["clip_path"], media_type="video/mp4")

    @api.post("/api/paint")
    async def paint(self, body: PaintIn) -> dict:
        prompt = " ".join(body.prompt.split())[:1200]
        if len(prompt) < 3:
            raise HTTPException(400, "prompt is empty")
        condition = self.by_prompt.get(prompt)
        if self.meta["mode"] == "replay" and condition is None:
            raise HTTPException(400, "Replay mode returns stored results for the preset conditions only. "
                                     "Set USE_GPUS = True to generate your own prompts.")
        if not body.scenes or len(body.scenes) > 16:
            raise HTTPException(400, "pick 1 to 16 scenes")
        out = []
        for sid in body.scenes:
            scene = self._scene(sid)
            rid = _rid(scene, prompt, body.seed, body.resolution, self.meta["tag"])
            job = {"job_id": uuid.uuid4().hex[:12], "request_id": rid, "scene_id": sid, "prompt": prompt,
                   "condition": condition, "seed": body.seed, "resolution": body.resolution,
                   "t_submit": time.time(), "status": "queued", "t_start": None, "t_end": None,
                   "seconds": None, "cached": False, "error": None}
            if (self.gallery / "outputs" / rid / "output.mp4").exists():
                job.update(status="done", cached=True, t_end=time.time(), seconds=0.0)
            else:
                asyncio.get_event_loop().create_task(self._run(job, scene))
            self.jobs[job["job_id"]] = job
            out.append(self._public(job))
        return {"jobs": out}

    @api.get("/api/jobs/{job_id}")
    def job(self, job_id: str) -> dict:
        if job_id not in self.jobs:
            raise HTTPException(404, "unknown job")
        return self._public(self.jobs[job_id])

    @api.get("/api/jobs/{job_id}/video.mp4")
    def video(self, job_id: str):
        j = self.jobs.get(job_id)
        if not j or j["status"] != "done":
            raise HTTPException(404, "not ready")
        return FileResponse(self.gallery / "outputs" / j["request_id"] / "output.mp4", media_type="video/mp4")

    @api.get("/api/status")
    def status(self) -> dict:
        active = [j for j in self.jobs.values() if j["status"] in ("queued", "running")]
        return {"replicas": self.meta["replicas"], "active": len(active), "mode": self.meta["mode"],
                "done": sum(j["status"] == "done" for j in self.jobs.values()),
                "median_s": self._median(), "blocked": None, "uptime_s": round(time.time() - self.t0, 1)}

    # -- internals ------------------------------------------------------------
    def _scene(self, sid: str) -> dict:
        if sid not in self.scenes:
            raise HTTPException(404, "unknown scene")
        return self.scenes[sid]

    def _median(self) -> float | None:
        d = sorted(self.durations[-50:])
        return round(d[len(d) // 2], 1) if d else self.meta.get("expected_s")

    def _queue_position(self, job: dict) -> int:
        if job["status"] != "queued":
            return 0
        ahead = sum(1 for j in self.jobs.values() if j["status"] == "queued" and j["t_submit"] < job["t_submit"])
        running = sum(1 for j in self.jobs.values() if j["status"] == "running")
        return max(0, ahead - max(0, self.meta["replicas"] - running) + 1)

    def _public(self, j: dict) -> dict:
        p = {k: j[k] for k in ("job_id", "scene_id", "prompt", "seed", "resolution", "status", "cached", "seconds", "error")}
        now = time.time()
        p["elapsed_s"] = round((j["t_end"] or now) - j["t_submit"], 1)
        p["running_s"] = round((j["t_end"] or now) - j["t_start"], 1) if j["t_start"] else None
        p["queue_position"] = self._queue_position(j)
        p["expected_s"] = self._median()
        p["video_url"] = f"api/jobs/{j['job_id']}/video.mp4" if j["status"] == "done" else None
        return p

    async def _run(self, job: dict, scene: dict) -> None:
        req = {"request_id": job["request_id"], "ns_dir": str(self.gallery), "condition": job["condition"],
               "sample": {"prompt": job["prompt"], "video_path": scene["clip_path"], "seed": job["seed"],
                          "num_steps": 4, "resolution": job["resolution"], "max_frames": 93,
                          "num_video_frames_per_chunk": 93, "keep_input_resolution": True,
                          "edge": {"control_path": scene.get("control_path"), "control_weight": 1.0,
                                   "preset_edge_threshold": "medium"}},
               "expect": scene.get("expect")}
        try:
            task = asyncio.ensure_future(self.painter.options(multiplexed_model_id=job["scene_id"]).paint.remote(req))
            while not task.done():
                await asyncio.sleep(0.5)
                if job["t_start"] is None and self._looks_assigned(job):
                    job.update(t_start=time.time(), status="running")
            rec = task.result()
            job["t_end"] = time.time()
            job["t_start"] = job["t_start"] or job["t_end"] - (rec.get("replica_wall_s") or 0)
            job["seconds"] = rec.get("replica_wall_s")
            if rec.get("status") == "generated_unreviewed":
                job["status"] = "done"
                self.durations.append(rec["replica_wall_s"])
            else:
                job.update(status="failed", error=(rec.get("error") or rec.get("status") or "")[:500])
        except Exception as e:
            job.update(status="failed", error=f"{type(e).__name__}: {str(e)[:400]}", t_end=time.time())

    def _looks_assigned(self, job: dict) -> bool:
        # Serve does not expose per-request assignment; approximate FIFO over replicas.
        if sum(1 for j in self.jobs.values() if j["status"] == "running") >= self.meta["replicas"]:
            return False
        return not [j for j in self.jobs.values() if j["status"] == "queued" and j["t_submit"] < job["t_submit"]]


# ----------------------------------------------------------------- assembly
def bundle_scenes(root: str | Path) -> list[dict]:
    """The 9 showcase drives shipped in the data bundle (replay mode)."""
    root = Path(root)
    idx = json.loads((root / "media" / "index.json").read_text())
    return [{"scene_id": sid, "label": s["label"], "run_id": s["run_id"], "clip_sha256": s["clip_sha256"],
             "clip_path": str(root / "media" / sid / "src.mp4"), "thumb_path": str(root / "media" / sid / "thumb.jpg")}
            for sid, s in idx["scenes"].items()]


def catalog_scenes(cfg) -> list[dict]:
    """Every prepared clip in the Ray Data catalog (GPU mode; run the preparation step first)."""
    import subprocess
    from offroad_ray.cli import _catalog
    thumbs = Path(cfg.paths.root) / "serve" / "thumbs"
    thumbs.mkdir(parents=True, exist_ok=True)
    out = []
    for c in sorted((c for c in _catalog(cfg) if c["status"] == "ok"), key=lambda c: (c["run_id"], c["clip_index"])):
        sid = c["clip_sha256"][:10]
        th = thumbs / f"{sid}.jpg"
        if not th.exists():
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", "2", "-i", c["clip_path"], "-frames:v", "1",
                            "-vf", "scale=480:-2", "-q:v", "4", str(th)], check=True)
        out.append({"scene_id": sid, "label": f"{c['run_id']} {c['clip_id'].split('__t')[1].split('__')[0]}s",
                    "run_id": c["run_id"], "clip_sha256": c["clip_sha256"], "clip_path": c["clip_path"],
                    "control_path": c["control_path"], "thumb_path": str(th),
                    "expect": {"frames": int(c["model_input_frames"]), "width": int(c["model_input_width"]),
                               "height": int(c["model_input_height"]), "fps": float(c["model_input_fps"])}})
    return out


def build_app(*, use_gpus: bool, replicas: int = 8, guardrails: bool = True, root: str | None = None,
              config_path: str = "configs/main.yaml"):
    """Return a bound Serve application. GPU mode needs the Cosmos image, weights and prepared clips."""
    root = Path(root or DEFAULT_ROOT)
    template_dir = Path(__file__).resolve().parents[2]
    presets = load_presets(template_dir / "configs")
    meta = {"mode": "gpu" if use_gpus else "replay", "replicas": replicas, "guardrails": guardrails,
            "presets": presets, "ui_html": UI_HTML.read_text(), "media_dir": str(root / "media"),
            "gallery": str(root / "serve" / ("gallery" if use_gpus else "replay_gallery")),
            "tag": f"g{int(guardrails)}", "expected_s": 80.0 if use_gpus else 3.0}
    if use_gpus:
        from offroad_ray.config import load_config
        from offroad_ray.inference.scheduler import actor_settings
        cfg = load_config(template_dir / config_path)
        ns = root / "serve" / time.strftime("s%Y%m%d_%H%M%S")
        (ns / "logs").mkdir(parents=True, exist_ok=True)
        settings = actor_settings(cfg, str(ns), hf_offline=True)
        settings.update(guardrails=guardrails, request_timeout_s=1800,
                        engine_options={"encode_cache": True, "encode_cache_verify": False})
        painter = Painter.options(num_replicas=replicas).bind(settings)
        scenes = catalog_scenes(cfg)
    else:
        painter = ReplayPainter.options(num_replicas=min(replicas, 4)).bind(str(root / "media"))
        meta["replicas"] = min(replicas, 4)
        scenes = bundle_scenes(root)
    return PaintAPI.bind(painter, scenes, meta)
