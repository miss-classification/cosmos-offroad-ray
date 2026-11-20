"""Ray actor that owns one persistent Cosmos inference subprocess on one GPU.

The actor runs in the Anyscale Python environment. It never imports torch or
cosmos. The subprocess inherits the actor's CUDA_VISIBLE_DEVICES, so the
subprocess-local device 0 is the GPU Ray assigned; no physical ordinal is
hard-coded anywhere.
"""
from __future__ import annotations

import json
import os
import select
import shutil
import signal
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import ray

from offroad_ray.storage import atomic_write_json, make_owned_dir, unique_attempt_dir

CHILD_ENV_PASSTHROUGH = ("HOME", "USER", "LANG", "LC_ALL", "TZ", "TERM")


class ChildFailure(RuntimeError):
    """Subprocess died, hung or broke protocol: an infrastructure failure."""


class GpuSampler(threading.Thread):
    """Samples NVML for this actor's GPU and the process tree's CPU/RSS."""

    def __init__(self, interval_s: float = 1.0) -> None:
        super().__init__(daemon=True)
        self.interval_s = interval_s
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.child_pid: int | None = None
        self.error: str | None = None

    def run(self) -> None:
        try:
            import psutil
            import pynvml
            pynvml.nvmlInit()
            cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
            h = (pynvml.nvmlDeviceGetHandleByUUID(cvd) if cvd.startswith("GPU-")
                 else pynvml.nvmlDeviceGetHandleByIndex(int(cvd)))
            self.uuid = pynvml.nvmlDeviceGetUUID(h)
            me = psutil.Process()
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"
            return
        while not self._stop.is_set():
            try:
                u = pynvml.nvmlDeviceGetUtilizationRates(h)
                m = pynvml.nvmlDeviceGetMemoryInfo(h)
                p = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
                cpu = rss = 0.0
                procs = [me] + me.children(recursive=True)
                for pr in procs:
                    try:
                        cpu += pr.cpu_percent(None)
                        rss += pr.memory_info().rss
                    except Exception:
                        pass
                with self._lock:
                    self.samples.append({"t": time.time(), "gpu_util": u.gpu, "mem_util": u.memory,
                                         "mem_used_gb": round(m.used / 1e9, 2), "power_w": round(p, 1),
                                         "cpu_pct": round(cpu, 1), "rss_gb": round(rss / 1e9, 2)})
            except Exception as e:
                self.error = f"{type(e).__name__}: {e}"
            self._stop.wait(self.interval_s)

    def drain(self) -> list[dict]:
        with self._lock:
            out, self.samples = self.samples, []
        return out

    def stop(self) -> None:
        self._stop.set()


class CosmosActor:
    def __init__(self, actor_index: int, settings: dict) -> None:
        self.t_created = time.time()
        self.idx = actor_index
        self.s = settings
        self.host = socket.gethostname()
        self.node_ip = ray.util.get_node_ip_address()
        self.gpu_ids = [str(g) for g in ray.get_gpu_ids()]
        self.cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        self.served = 0
        self.busy_s = 0.0
        self.sampler = GpuSampler(settings.get("telemetry_interval_s", 1.0))
        self.sampler.start()
        self.scratch = Path(settings["scratch"]) / f"actor{actor_index}.{os.getpid()}"
        make_owned_dir(self.scratch)
        self.log_path = Path(settings["log_dir"]) / f"actor{actor_index}.{self.host}.{os.getpid()}.child.log"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._start_child()
        t0 = time.time()
        reply = self._exchange({"op": "init", "model": settings["model"], "guardrails": settings["guardrails"],
                                "scratch_dir": str(self.scratch / "cosmos"),
                                "options": settings.get("engine_options", {})}, timeout=settings["init_timeout_s"])
        if reply.get("type") != "init_ok":
            raise ChildFailure(f"init failed: {reply.get('error')}\n{reply.get('traceback', '')}\n"
                               f"--- child log tail ---\n{self._tail()}")
        self.child_info = reply["info"]
        self.init_s = time.time() - t0
        self.t_ready = time.time()

    # ------------------------------------------------------------------ child
    def _child_env(self) -> dict:
        s = self.s
        env = {k: os.environ[k] for k in CHILD_ENV_PASSTHROUGH if k in os.environ}
        modules_dir = str(Path(__import__("offroad_ray").__file__).resolve().parents[1])
        env.update({
            "PATH": f"{Path(s['cosmos_python']).parent}:/usr/local/bin:/usr/bin:/bin",
            "PYTHONPATH": modules_dir,          # only our stdlib-only worker module; nothing from Ray's env
            "PYTHONUNBUFFERED": "1", "PYTHONNOUSERSITE": "1",
            "CUDA_VISIBLE_DEVICES": self.cvd or "",
            "HF_HOME": s["hf_home"], "HF_HUB_DISABLE_TELEMETRY": "1",
            "COSMOS_EXPERIMENTAL_CHECKPOINTS": "1",
            "OMP_NUM_THREADS": str(s.get("child_omp_threads", 4)),
            "TMPDIR": str(self.scratch),
            "PYTORCH_CUDA_ALLOC_CONF": s.get("alloc_conf", "expandable_segments:True"),
        })
        if s.get("hf_offline"):
            env["HF_HUB_OFFLINE"] = "1"
        env.update({k: str(v) for k, v in (s.get("child_extra_env") or {}).items()})
        return env

    def _start_child(self) -> None:
        rfd, wfd = os.pipe()
        env = self._child_env()
        env["OFFROAD_RESULT_FD"] = str(wfd)
        self._log = open(self.log_path, "ab")
        self.proc = subprocess.Popen(
            [self.s["cosmos_python"], "-m", "offroad_ray.inference.subprocess_worker"],
            stdin=subprocess.PIPE, stdout=self._log, stderr=subprocess.STDOUT, env=env,
            cwd=self.s["cosmos_src"], pass_fds=(wfd,), start_new_session=True)
        os.close(wfd)  # parent keeps only the read end so child death gives EOF
        self._rfd = rfd
        self._buf = b""
        self.sampler.child_pid = self.proc.pid

    def _readline(self, timeout: float | None) -> bytes:
        deadline = None if timeout is None else time.time() + timeout
        while b"\n" not in self._buf:
            remaining = None if deadline is None else max(0.0, deadline - time.time())
            if remaining == 0.0:
                raise ChildFailure(f"timeout after {timeout}s waiting for subprocess reply")
            r, _, _ = select.select([self._rfd], [], [], remaining if remaining is not None else 5.0)
            if not r:
                if self.proc.poll() is not None:
                    raise ChildFailure(f"subprocess exited with code {self.proc.returncode}")
                continue
            chunk = os.read(self._rfd, 1 << 16)
            if not chunk:
                self.proc.wait(timeout=30)
                raise ChildFailure(f"subprocess closed result pipe (exit code {self.proc.returncode})")
            self._buf += chunk
        line, self._buf = self._buf.split(b"\n", 1)
        return line

    def _exchange(self, req: dict, timeout: float | None) -> dict:
        if self.proc.poll() is not None:
            raise ChildFailure(f"subprocess already exited ({self.proc.returncode})\n{self._tail()}")
        try:
            self.proc.stdin.write((json.dumps(req) + "\n").encode())
            self.proc.stdin.flush()
        except BrokenPipeError:
            raise ChildFailure(f"subprocess stdin closed\n{self._tail()}") from None
        try:
            return json.loads(self._readline(timeout))
        except ChildFailure as e:
            raise ChildFailure(f"{e}\n--- child log tail ---\n{self._tail()}") from None

    def _tail(self, n: int = 4000) -> str:
        try:
            self._log.flush()
            return self.log_path.read_text(errors="replace")[-n:]
        except Exception:
            return "(no child log)"

    # -------------------------------------------------------------- interface
    def info(self) -> dict:
        return {"actor_index": self.idx, "host": self.host, "node_ip": self.node_ip,
                "ray_gpu_ids": self.gpu_ids, "cuda_visible_devices": self.cvd,
                "nvml_uuid": getattr(self.sampler, "uuid", None), "sampler_error": self.sampler.error,
                "child_pid": self.proc.pid, "child_info": self.child_info, "init_s": round(self.init_s, 2),
                "t_created": self.t_created, "t_ready": self.t_ready, "child_log": str(self.log_path)}

    def ping(self) -> dict:
        return self._exchange({"op": "ping"}, timeout=30)

    def generate(self, req: dict) -> dict:
        """Run one request. Returns an attempt record; raises ChildFailure on infra failure."""
        from offroad_ray.video_io import probe
        t_start = time.time()
        attempt_dir = unique_attempt_dir(req["ns_dir"], req["request_id"])
        sample = dict(req["sample"])
        sample["name"] = "output"
        child_req = {"op": "generate", "request_id": req["request_id"], "out_dir": str(attempt_dir),
                     "sample": sample}
        try:
            res = self._exchange(child_req, timeout=self.s["request_timeout_s"])
        except ChildFailure:
            self._kill_child()
            raise
        t_gen = time.time()
        rec: dict[str, Any] = {
            "request_id": req["request_id"], "actor_index": self.idx, "host": self.host,
            "node_ip": self.node_ip, "cuda_visible_devices": self.cvd,
            "gpu_uuid": self.child_info.get("device_uuid"),
            "child_status": res.get("status"), "child_seconds": res.get("seconds"),
            "stage_times": res.get("stage_times", {}), "peak_mem_gb": res.get("peak_mem_gb"),
            "reserved_gb": res.get("reserved_gb"), "request_index_in_process": res.get("request_index_in_process"),
            "error": res.get("error"), "traceback": res.get("traceback"), "oom": res.get("oom", False),
            "attempt_dir": str(attempt_dir), "t_start": t_start, "t_generated": t_gen,
        }
        if res.get("type") == "error":
            rec.update(status="generation_failure", error=res.get("error"), traceback=res.get("traceback"))
        elif res["status"] == "generated":
            out = Path(res["output_path"])
            try:
                pr = probe(str(out))
                exp = req["expect"]
                problems = []
                if pr["frames"] != exp["frames"]:
                    problems.append(f"frames {pr['frames']} != {exp['frames']}")
                if (pr["width"], pr["height"]) != (exp["width"], exp["height"]):
                    problems.append(f"size {pr['width']}x{pr['height']} != {exp['width']}x{exp['height']}")
                if abs(pr["fps"] - exp["fps"]) > 0.01:
                    problems.append(f"fps {pr['fps']} != {exp['fps']}")
                if abs(pr["duration_s"] - exp["frames"] / exp["fps"]) > 1.5 / exp["fps"]:
                    problems.append(f"duration {pr['duration_s']:.3f}")
                rec["output_probe"] = pr
                if problems:
                    rec.update(status="generation_failure", error="output validation: " + "; ".join(problems))
                else:
                    final = Path(req["ns_dir"]) / "outputs" / req["request_id"]
                    final.parent.mkdir(parents=True, exist_ok=True)
                    if final.exists():
                        shutil.move(str(final), str(final) + f".superseded.{int(time.time())}")
                    os.replace(attempt_dir, final)
                    rec.update(status="generated_unreviewed", output_dir=str(final),
                               output_path=str(final / out.name),
                               control_used_path=str(final / "output_control_edge.mp4"))
            except Exception as e:
                rec.update(status="generation_failure", error=f"output validation error: {type(e).__name__}: {e}")
        elif res["status"] == "cp_follower":
            rec.update(status="cp_follower")
        elif res["status"].startswith("guardrail_blocked"):
            rec.update(status="guardrail_blocked", guardrail_stage=res["status"].rsplit("_", 1)[-1])
        else:
            rec.update(status="generation_failure")
        rec["t_end"] = time.time()
        rec["validate_s"] = round(rec["t_end"] - t_gen, 3)
        self.served += 1
        self.busy_s += rec["t_end"] - t_start
        rec["actor_served"] = self.served
        return rec

    def telemetry(self) -> dict:
        return {"actor_index": self.idx, "host": self.host, "samples": self.sampler.drain(),
                "sampler_error": self.sampler.error}

    def _kill_child(self) -> None:
        try:
            os.killpg(self.proc.pid, signal.SIGKILL)
        except Exception:
            pass

    def shutdown(self) -> dict:
        try:
            if self.proc.poll() is None:
                self._exchange({"op": "shutdown"}, timeout=60)
                self.proc.wait(timeout=60)
        except Exception:
            self._kill_child()
        finally:
            self.sampler.stop()
            try:
                self._log.close()
            except Exception:
                pass
        return {"actor_index": self.idx, "served": self.served, "busy_s": round(self.busy_s, 2),
                "exit_code": self.proc.poll()}

    def __del__(self):
        try:
            if self.proc.poll() is None:
                self._kill_child()
        except Exception:
            pass
