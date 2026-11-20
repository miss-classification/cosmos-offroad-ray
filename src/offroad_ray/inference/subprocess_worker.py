"""Persistent Cosmos inference process. Runs under the ISOLATED Cosmos interpreter.

Only the standard library is imported at module scope; torch and cosmos are
imported on the ``init`` message. This file must not import anything from the
Anyscale/Ray environment.

IPC protocol (adapted from anyscale/cosmos-transfer25-ray@466abae, child.py)
---------------------------------------------------------------------------
Requests: one JSON object per line on stdin.
Replies:  one JSON object per line on the file descriptor named by
          $OFFROAD_RESULT_FD. stdout and stderr are reserved for diagnostic logs
          (cosmos logs freely to stdout), so logs can never corrupt the protocol.

  {"op": "init", "model": ..., "guardrails": bool, "scratch_dir": ...}
        -> {"type": "init_ok", "info": {...}}  |  {"type": "error", ...}
  {"op": "generate", "request_id": ..., "out_dir": ..., "sample": {...}}
        -> {"type": "result", "request_id": ..., "status": "...", ...}
  {"op": "ping"}      -> {"type": "pong", ...}
  {"op": "shutdown"}  -> {"type": "bye"}, then exit 0

Generate statuses: "generated" (file written, not yet validated by the parent),
"guardrail_blocked_text", "guardrail_blocked_video", "generation_failure".
Exceptions are caught per request and returned with a traceback; the process
keeps serving. A crash (e.g. a CUDA fault) closes the result pipe, which the
parent reports as an infrastructure failure.
"""
from __future__ import annotations

import json
import os
import socket
import sys
import time
import traceback

_out = None


def _reply(payload: dict) -> None:
    _out.write(json.dumps(payload, default=str) + "\n")
    _out.flush()


class Engine:
    def __init__(self, model: str, guardrails: bool, scratch_dir: str, options: dict | None = None) -> None:
        options = dict(options or {})
        t0 = time.time()
        from cosmos_oss.init import init_environment, init_output_dir
        init_environment()
        import torch
        from cosmos_transfer2.config import SetupArguments
        from cosmos_transfer2.inference import Control2WorldInference

        t_import = time.time()
        from pathlib import Path
        scratch = Path(scratch_dir)  # cosmos_oss.init_output_dir and SetupArguments take a Path
        init_output_dir(scratch)  # once per process: adds loguru sinks
        self.torch = torch
        self.setup = SetupArguments(
            model=model, output_dir=scratch, disable_guardrails=not guardrails,
            keep_going=True, benchmark=True,  # benchmark=True enables per-stage timers (adds cuda syncs)
            **{k: options[k] for k in ("compile_tokenizer", "enable_parallel_tokenizer",
                                       "parallel_tokenizer_grid") if k in options})
        self.inference = Control2WorldInference(self.setup, batch_hint_keys=["edge"])
        self.rank = int(getattr(self.inference, "device_rank", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        model_obj = self.inference.inference_pipeline.model
        if options.get("compile_net"):
            model_obj.net = torch.compile(model_obj.net, dynamic=False)
        self.fine: dict[str, float] = {}
        if options.get("encode_cache"):
            self._cache_encode(model_obj, int(options.get("encode_cache_size", 64)),
                               bool(options.get("encode_cache_verify", False)))
        if options.get("fine_timers", True):
            self._wrap(model_obj, "encode", "fine_vae_encode")
            self._wrap(model_obj, "decode", "fine_vae_decode")
            self._wrap(model_obj, "generate_samples_from_batch", "fine_sample_total")
            self._wrap_net_forward(model_obj)
        torch.cuda.synchronize()
        self.load_s = time.time() - t_import
        self.import_s = t_import - t0
        self.requests = 0
        self.guardrails = guardrails
        props = torch.cuda.get_device_properties(0)
        self.info = {
            "host": socket.gethostname(), "pid": os.getpid(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "device_name": props.name, "device_uuid": str(getattr(props, "uuid", "")),
            "device_total_gb": round(props.total_memory / 1e9, 2),
            "torch": torch.__version__, "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(), "python": sys.version.split()[0],
            "import_s": round(self.import_s, 2), "load_s": round(self.load_s, 2),
            "mem_after_load_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
            "peak_after_load_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
            "guardrails": guardrails, "model": model, "options": options,
            "rank": self.rank, "world_size": self.world_size,
        }

    def _wrap(self, obj, name: str, key: str) -> None:
        """Replace obj.name with a CUDA-synchronised timer that accumulates into self.fine[key]."""
        fn = getattr(obj, name)
        torch = self.torch
        fine = self.fine

        def timed(*a, **k):
            torch.cuda.synchronize()
            t = time.time()
            try:
                return fn(*a, **k)
            finally:
                torch.cuda.synchronize()
                fine[key] = fine.get(key, 0.0) + time.time() - t
                fine[key + "_calls"] = fine.get(key + "_calls", 0) + 1
        setattr(obj, name, timed)

    def _cache_encode(self, model_obj, size: int, verify: bool) -> None:
        """Memoise VAE encode by input content.

        Per request the model encodes the scene's control video and its conditioning
        video; both depend only on the scene and resolution, not on prompt or seed.
        The key is a content fingerprint (shape, dtype, sum, strided sum of squares),
        so a different input can never reuse a stale latent unless it collides on all
        three. ``verify`` recomputes on every hit and records the max abs difference.
        """
        from collections import OrderedDict
        torch = self.torch
        self._verified_total = 0
        fn = model_obj.encode
        cache: OrderedDict = OrderedDict()
        fine = self.fine

        def key_of(x):
            xf = x.detach()
            return (tuple(xf.shape), str(xf.dtype), float(xf.float().sum()),
                    float((xf[..., ::7, ::13].float() ** 2).sum()))

        def cached(x, *a, **k):
            if a or k or not torch.is_tensor(x):
                return fn(x, *a, **k)
            key = key_of(x)
            hit = cache.get(key)
            if hit is not None:
                cache.move_to_end(key)
                fine["encode_cache_hits"] = fine.get("encode_cache_hits", 0) + 1
                if verify and fine.get("_verified", 0) + self._verified_total < 2:
                    self._verified_total += 1
                    fresh = fn(x)
                    fine["encode_cache_maxdiff"] = max(fine.get("encode_cache_maxdiff", 0.0),
                                                       float((fresh.float() - hit.float()).abs().max()))
                return hit.clone()
            out = fn(x)
            cache[key] = out.detach().clone()
            if len(cache) > size:
                cache.popitem(last=False)
            fine["encode_cache_misses"] = fine.get("encode_cache_misses", 0) + 1
            return out
        model_obj.encode = cached

    def _wrap_net_forward(self, model_obj) -> None:
        torch = self.torch
        fine = self.fine
        net = model_obj.net
        state = {}

        def pre(_m, _a):
            torch.cuda.synchronize()
            state["t"] = time.time()

        def post(_m, _a, _o):
            torch.cuda.synchronize()
            fine["fine_dit_forward"] = fine.get("fine_dit_forward", 0.0) + time.time() - state.get("t", time.time())
            fine["fine_dit_forward_calls"] = fine.get("fine_dit_forward_calls", 0) + 1
        try:
            net.register_forward_pre_hook(pre)
            net.register_forward_hook(post)
        except Exception as e:  # diagnostic only
            fine["_net_hook_error"] = repr(e)

    def _stage_times(self) -> dict:
        timer = self.inference.benchmark_timer
        out = {}
        try:
            for k, v in getattr(timer, "results", {}).items():
                vals = v if isinstance(v, list) else [v]
                if vals:
                    out[k] = round(float(vals[-1]), 3)
            timer.reset()
        except Exception as e:  # timers are diagnostic only
            out["_timer_error"] = repr(e)
        for k, v in self.fine.items():
            out[k] = round(v, 3) if isinstance(v, float) else v
        self.fine.clear()
        return out

    def generate(self, req: dict) -> dict:
        from pathlib import Path
        from cosmos_transfer2.config import InferenceArguments
        torch = self.torch
        rid = req["request_id"]
        out_dir = Path(req["out_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        sample = InferenceArguments(**req["sample"])
        torch.cuda.reset_peak_memory_stats()
        self.requests += 1
        t0 = time.time()
        try:
            path = self.inference._generate_sample(sample, out_dir, sample_id=1)
            torch.cuda.synchronize()
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            return {"type": "result", "request_id": rid, "status": "generation_failure",
                    "error": msg[:4000], "traceback": traceback.format_exc()[-6000:],
                    "oom": isinstance(e, torch.OutOfMemoryError) or "out of memory" in msg.lower(),
                    "seconds": round(time.time() - t0, 3), "stage_times": self._stage_times(),
                    "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                    "request_index_in_process": self.requests}
        secs = time.time() - t0
        name = sample.name
        if self.rank != 0:  # context-parallel follower: rank 0 writes the output
            return {"type": "result", "request_id": rid, "status": "cp_follower", "output_path": None,
                    "seconds": round(secs, 3), "stage_times": self._stage_times(),
                    "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                    "request_index_in_process": self.requests, "guardrails": self.guardrails}
        produced = path is not None and Path(path).exists()
        if produced:
            status = "generated"
        elif (out_dir / f"{name}_control_edge.mp4").exists():
            status = "guardrail_blocked_video"   # control saved before the video guardrail ran
        else:
            status = "guardrail_blocked_text"
        return {"type": "result", "request_id": rid, "status": status,
                "output_path": str(path) if produced else None,
                "seconds": round(secs, 3), "stage_times": self._stage_times(),
                "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                "reserved_gb": round(torch.cuda.memory_reserved() / 1e9, 2),
                "request_index_in_process": self.requests,
                "guardrails": self.guardrails}


def main() -> int:
    global _out
    _out = os.fdopen(int(os.environ["OFFROAD_RESULT_FD"]), "w")
    engine = None
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            op = req.get("op")
            if op == "shutdown":
                _reply({"type": "bye"})
                return 0
            if op == "ping":
                _reply({"type": "pong", "ready": engine is not None,
                        "requests": engine.requests if engine else 0})
            elif op == "init":
                if engine is not None:
                    _reply({"type": "error", "error": "already initialised"})
                    continue
                engine = Engine(req["model"], bool(req["guardrails"]), req["scratch_dir"], req.get("options"))
                _reply({"type": "init_ok", "info": engine.info})
            elif op == "generate":
                if engine is None:
                    _reply({"type": "error", "error": "generate before init"})
                    continue
                _reply(engine.generate(req))
            else:
                _reply({"type": "error", "error": f"unknown op {op!r}"})
        except Exception as e:
            _reply({"type": "error", "error": f"{type(e).__name__}: {e}",
                    "traceback": traceback.format_exc()[-6000:]})
    return 0


if __name__ == "__main__":
    sys.exit(main())
