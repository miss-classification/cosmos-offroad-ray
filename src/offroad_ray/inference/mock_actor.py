"""Mock inference actor for scheduler tests (shipped to workers via py_modules)."""
from __future__ import annotations

import time
from pathlib import Path


class MockActor:
    """Mimics CosmosActor.generate. Behaviour is driven by markers in the request."""

    def __init__(self, idx, settings):
        self.idx = idx
        self.s = settings
        self.n = 0

    def info(self):
        return {"actor_index": self.idx, "host": "mock", "cuda_visible_devices": str(self.idx), "init_s": 0.01,
                "child_info": {"device_uuid": f"mock-{self.idx}", "load_s": 0.0, "mem_after_load_gb": 0.0}}

    def generate(self, req):
        from offroad_ray.inference.actor import ChildFailure
        self.n += 1
        mode = req["sample"]["prompt"]
        flag = Path(req["ns_dir"]) / f"failed_once_{req['request_id']}"
        if mode == "infra_once" and not flag.exists():
            flag.write_text("x")
            raise ChildFailure("simulated subprocess death")
        if mode == "infra_always":
            raise ChildFailure("simulated persistent failure")
        t0 = time.time()
        time.sleep(float(req["sample"].get("sleep", 0.05)))
        if mode == "guardrail":
            return {"request_id": req["request_id"], "status": "guardrail_blocked", "guardrail_stage": "video",
                    "t_start": t0, "t_end": time.time(), "actor_index": self.idx}
        out = Path(req["ns_dir"]) / "outputs" / req["request_id"]
        out.mkdir(parents=True, exist_ok=True)
        (out / "output.mp4").write_bytes(b"fake")
        return {"request_id": req["request_id"], "status": "generated_unreviewed", "output_path": str(out / "output.mp4"),
                "t_start": t0, "t_end": time.time(), "actor_index": self.idx, "child_seconds": 0.05}

    def telemetry(self):
        return {"actor_index": self.idx, "samples": []}

    def shutdown(self):
        return {"actor_index": self.idx, "served": self.n}
