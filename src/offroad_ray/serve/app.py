"""Live "paint the trail" demo: Ray Serve in front of persistent Cosmos Transfer 2.5 replicas.

Two deployments:

* ``Painter`` (1 GPU each, ``max_ongoing_requests=1``): owns one persistent Cosmos
  subprocess through ``CosmosActor`` (the same class the batch pipeline uses), so
  the model loads once per replica and every request pays only generation time.
* ``PaintAPI`` (FastAPI ingress, CPU): job API + the single-page UI. ``POST /api/paint``
  returns a job id immediately; the browser polls ``GET /api/jobs/<id>``. Ray Serve's
  router queues work across Painter replicas, so a multi-scene request fans out to
  all GPUs at once.

Demo only: guardrails are configurable and OFF by default for this namespace, which
is never evaluated or reported as a result. GPU time is booked in
``runs/serve_demo/<session>/pool_summary.json`` (heartbeat) so ``offroad_ray.cli budget``
counts it, and the API refuses new work once the GPU-phase or attempt limit is reached.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from ray import serve

ROOT = "/mnt/cluster_storage/cosmos-offroad"
UI_HTML = (Path(__file__).parent / "ui.html")
PEEK_DIR = Path(ROOT) / "runs" / "serve_demo" / "peek"  # small preview clips for the page header
PEEK_FILES = {"real", "golden", "snow", "storm"}


def _rid(scene: dict, prompt: str, seed: int, resolution: str, tag: str) -> str:
    h = hashlib.sha256(json.dumps([scene["clip_sha256"], prompt.strip(), seed, resolution, tag]).encode())
    return "s" + h.hexdigest()[:19]


# ----------------------------------------------------------------- GPU tier
@serve.deployment(max_ongoing_requests=1, ray_actor_options={"num_gpus": 1, "num_cpus": 4},
                  health_check_period_s=30, health_check_timeout_s=120,
                  graceful_shutdown_timeout_s=120)
class Painter:
    def __init__(self, settings: dict):
        from offroad_ray.inference.actor import CosmosActor
        ctx = serve.get_replica_context()
        idx = abs(hash(ctx.replica_id.unique_id)) % 10_000
        self.engine = CosmosActor(idx, settings)
        self.info = self.engine.info()
        self.served = 0

    @serve.multiplexed(max_num_models_per_replica=12)
    async def _scene(self, scene_id: str) -> str:
        # Scene affinity: Serve routes a scene's requests to replicas that already
        # "hold" it, so the per-replica VAE latent cache for that scene is hit.
        return scene_id

    async def paint(self, req: dict) -> dict:
        await self._scene(serve.get_multiplexed_model_id())
        t0 = time.time()
        rec = await asyncio.to_thread(self.engine.generate, req)
        self.served += 1
        rec["replica_host"] = self.engine.host
        rec["replica_served"] = self.served
        rec["replica_wall_s"] = round(time.time() - t0, 2)
        return rec

    def stats(self) -> dict:
        return {"host": self.engine.host, "served": self.served, "init_s": self.info["init_s"]}

    def check_health(self) -> None:
        if self.engine.proc.poll() is not None:
            raise RuntimeError("cosmos subprocess exited")


# ----------------------------------------------------------------- API tier
class PaintIn(BaseModel):
    prompt: str
    scenes: list[str]
    seed: int = 101
    resolution: str = "720"


api = FastAPI(title="Paint the trail")


@serve.deployment(max_ongoing_requests=256, ray_actor_options={"num_cpus": 1})
@serve.ingress(api)
class PaintAPI:
    def __init__(self, painter, scenes: list[dict], ns_dir: str, meta: dict):
        self.painter = painter
        self.scenes = {s["scene_id"]: s for s in scenes}
        self.order = [s["scene_id"] for s in scenes]
        self.ns = Path(ns_dir)
        self.gallery = self.ns.parent / "gallery"   # outputs shared across sessions (instant replays)
        for sub in ("records", "logs", "jobs"):
            (self.ns / sub).mkdir(parents=True, exist_ok=True)
        (self.gallery / "outputs").mkdir(parents=True, exist_ok=True)
        self.meta = meta
        self.jobs: dict[str, dict] = {}
        self.durations: list[float] = []
        self.t0 = meta["t0"]
        self.attempts = 0
        self.blocked_reason: str | None = None
        self._hb = asyncio.get_event_loop().create_task(self._heartbeat())

    # -- budget + heartbeat -------------------------------------------------
    async def _heartbeat(self) -> None:
        while True:
            try:
                self._write_pool_summary()
                await asyncio.to_thread(self._check_budget)
            except Exception as e:  # never kill the API over bookkeeping
                print("heartbeat error", repr(e))
            await asyncio.sleep(30)

    def _write_pool_summary(self) -> None:
        counts: dict[str, int] = {}
        for j in self.jobs.values():
            counts[j["status"]] = counts.get(j["status"], 0) + 1
        t1 = time.time()
        from offroad_ray.storage import atomic_write_json
        atomic_write_json(self.ns / "pool_summary.json", {
            "num_actors": self.meta["replicas"], "wall_s": round(t1 - self.t0, 2), "t0": self.t0, "t1": t1,
            "settings": {"guardrails": self.meta["guardrails"]}, "status_counts": counts,
            "kind": "serve_demo", "meta": self.meta})

    def _check_budget(self) -> None:
        from offroad_ray.reporting.benchmark import budget_status
        b = budget_status(ROOT)
        g = b["gpu_phase_wall_time_h"]
        # Limits come from the launcher (meta): replicas can import an older offroad_ray copy that
        # the Serve controller's job shipped earlier, so its hard-coded limits may be stale.
        lim_h = self.meta.get("gpu_limit_h", g["limit"])
        a = {**b["generation_attempts"], "limit": self.meta.get("attempt_limit", b["generation_attempts"]["limit"])}
        if g["used"] >= lim_h:
            self.blocked_reason = f"GPU-phase limit reached ({g['used']:.2f} of {lim_h} h)"
        elif a["used"] >= a["limit"]:
            self.blocked_reason = f"attempt limit reached ({a['used']} of {a['limit']})"
        else:
            self.blocked_reason = None
        self.budget = {"gpu_h": g["used"], "gpu_limit_h": lim_h, "attempts": a["used"], "attempt_limit": a["limit"]}

    # -- routes ---------------------------------------------------------------
    @api.get("/", response_class=HTMLResponse)
    def index(self) -> str:
        # The UI ships inside meta (read by the driver): replicas may import this module from a
        # pickled copy whose __file__ points at the driver's checkout, not at the code snapshot.
        html = self.meta.get("ui_html") or UI_HTML.read_text()
        return html.replace("__PEEK_BASE__", "api/peek/")

    @api.get("/api/peek/{name}.mp4")
    def peek(self, name: str):
        if name not in PEEK_FILES:
            raise HTTPException(404, "unknown preview")
        return FileResponse(PEEK_DIR / f"{name}.mp4", media_type="video/mp4")

    @api.get("/api/scenes")
    def list_scenes(self) -> dict:
        return {"scenes": [{k: self.scenes[s][k] for k in ("scene_id", "label", "run_id")} for s in self.order],
                "replicas": self.meta["replicas"], "guardrails": self.meta["guardrails"],
                "median_s": self._median(), "presets": self.meta.get("presets", [])}

    @api.get("/api/scenes/{scene_id}/thumb.jpg")
    def thumb(self, scene_id: str):
        s = self._scene(scene_id)
        return FileResponse(s["thumb_path"], media_type="image/jpeg")

    @api.get("/api/scenes/{scene_id}/source.mp4")
    def source(self, scene_id: str):
        s = self._scene(scene_id)
        return FileResponse(s["clip_path"], media_type="video/mp4")

    @api.post("/api/paint")
    async def paint(self, body: PaintIn) -> dict:
        prompt = " ".join(body.prompt.split())[:1200]
        if len(prompt) < 3:
            raise HTTPException(400, "prompt is empty")
        if body.resolution != "720":
            raise HTTPException(400, "only 720 is supported (480 output from the distilled model is unusable)")
        if not body.scenes or len(body.scenes) > 16:
            raise HTTPException(400, "pick 1 to 16 scenes")
        out = []
        for sid in body.scenes:
            scene = self._scene(sid)
            rid = _rid(scene, prompt, body.seed, body.resolution, self.meta["tag"])
            jid = uuid.uuid4().hex[:12]
            job = {"job_id": jid, "request_id": rid, "scene_id": sid, "prompt": prompt, "seed": body.seed,
                   "resolution": body.resolution, "t_submit": time.time(), "status": "queued",
                   "t_start": None, "t_end": None, "seconds": None, "cached": False, "error": None}
            cached = self.gallery / "outputs" / rid / "output.mp4"
            if cached.exists():
                job.update(status="done", cached=True, t_end=time.time(), seconds=0.0)
            elif self.blocked_reason:
                job.update(status="refused", error=self.blocked_reason)
            else:
                self.attempts += 1
                asyncio.get_event_loop().create_task(self._run(job, scene))
            self.jobs[jid] = job
            out.append(self._public(job))
        return {"jobs": out}

    @api.get("/api/jobs/{job_id}")
    def job(self, job_id: str) -> dict:
        j = self.jobs.get(job_id)
        if not j:
            raise HTTPException(404, "unknown job")
        return self._public(j)

    @api.get("/api/jobs/{job_id}/video.mp4")
    def video(self, job_id: str):
        j = self.jobs.get(job_id)
        if not j or j["status"] != "done":
            raise HTTPException(404, "not ready")
        return FileResponse(self.gallery / "outputs" / j["request_id"] / "output.mp4", media_type="video/mp4")

    @api.get("/api/status")
    def status(self) -> dict:
        active = [j for j in self.jobs.values() if j["status"] in ("queued", "running")]
        return {"replicas": self.meta["replicas"], "active": len(active),
                "done": sum(j["status"] == "done" for j in self.jobs.values()),
                "median_s": self._median(), "blocked": self.blocked_reason,
                "budget": getattr(self, "budget", None), "uptime_s": round(time.time() - self.t0, 1)}

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
        free = max(0, self.meta["replicas"] - running)
        return max(0, ahead - free + 1)

    def _public(self, j: dict) -> dict:
        p = {k: j[k] for k in ("job_id", "scene_id", "prompt", "seed", "resolution", "status", "cached",
                               "seconds", "error")}
        now = time.time()
        p["elapsed_s"] = round((j["t_end"] or now) - j["t_submit"], 1)
        p["running_s"] = round((j["t_end"] or now) - j["t_start"], 1) if j["t_start"] else None
        p["queue_position"] = self._queue_position(j)
        p["expected_s"] = self._median()
        p["video_url"] = f"api/jobs/{j['job_id']}/video.mp4" if j["status"] == "done" else None
        p["host"] = j.get("host")
        return p

    async def _run(self, job: dict, scene: dict) -> None:
        from offroad_ray.storage import atomic_write_json
        req = {
            "request_id": job["request_id"], "ns_dir": str(self.gallery),
            "sample": {"prompt": job["prompt"], "video_path": scene["clip_path"], "seed": job["seed"],
                       "num_steps": 4, "resolution": job["resolution"], "max_frames": 93,
                       "num_video_frames_per_chunk": 93, "keep_input_resolution": True,
                       "edge": {"control_path": scene["control_path"], "control_weight": 1.0,
                                "preset_edge_threshold": "medium"}},
            "expect": scene["expect"],
        }
        atomic_write_json(Path(ROOT) / "runs" / "_attempt_ledger" /
                          f"{int(time.time()*1000)}_{job['request_id']}_serve.json",
                          {"ns": str(self.ns), "request_id": job["request_id"], "attempt": 0,
                           "status": "submitted", "t": time.time()})
        try:
            resp = self.painter.options(multiplexed_model_id=job["scene_id"]).paint.remote(req)
            # The router holds the request until a replica is free; mark running once assigned.
            job["status"] = "queued"
            task = asyncio.ensure_future(resp)
            while not task.done():
                await asyncio.sleep(0.5)
                if job["t_start"] is None and self._looks_assigned(job):
                    job["t_start"] = time.time()
                    job["status"] = "running"
            rec = task.result()
            job["t_end"] = time.time()
            job["host"] = rec.get("replica_host")
            if job["t_start"] is None:
                job["t_start"] = job["t_end"] - (rec.get("replica_wall_s") or 0)
            job["seconds"] = rec.get("replica_wall_s")
            if rec.get("status") == "generated_unreviewed":
                job["status"] = "done"
                self.durations.append(rec["replica_wall_s"])
            else:
                job["status"] = "failed"
                job["error"] = (rec.get("error") or rec.get("status") or "")[:500]
            rec.update({k: job[k] for k in ("prompt", "seed", "resolution", "scene_id")})
            rec["guardrails"] = self.meta["guardrails"]
            atomic_write_json(self.ns / "records" / f"{job['request_id']}.json", rec)
        except Exception as e:
            job.update(status="failed", error=f"{type(e).__name__}: {str(e)[:400]}", t_end=time.time())
        atomic_write_json(self.ns / "jobs" / f"{job['job_id']}.json", job)

    def _looks_assigned(self, job: dict) -> bool:
        # Serve does not expose per-request assignment; approximate FIFO over replicas.
        running = sum(1 for j in self.jobs.values() if j["status"] == "running")
        if running >= self.meta["replicas"]:
            return False
        earlier = [j for j in self.jobs.values() if j["status"] == "queued" and j["t_submit"] < job["t_submit"]]
        return not earlier


def build_app(settings: dict, scenes: list[dict], ns_dir: str, meta: dict, autoscale: bool = False):
    replicas = meta["replicas"]
    if autoscale:
        painter = Painter.options(autoscaling_config={
            "min_replicas": 1, "max_replicas": replicas, "target_ongoing_requests": 1,
            "upscale_delay_s": 5, "downscale_delay_s": 600}).bind(settings)
    else:
        painter = Painter.options(num_replicas=replicas).bind(settings)
    return PaintAPI.bind(painter, scenes, ns_dir, meta)
