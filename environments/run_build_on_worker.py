"""Run build_cosmos_env.sh on a GPU worker node (not the CPU head) via a Ray task.

The build does not need a GPU; it reserves CPUs only, but is pinned to a worker
node by requiring the accelerator-shape custom resource fraction so it never
lands on the head.
"""
import os
import subprocess
import sys
from pathlib import Path

import ray

HERE = Path(__file__).resolve().parent


@ray.remote(num_cpus=8, resources={"anyscale/accelerator_shape:1xA100-80G": 0.001})
def build(script: str, env: dict) -> dict:
    import socket
    proc = subprocess.run(["bash", "-c", script], env={**os.environ, **env},
                          capture_output=True, text=True)
    return {"host": socket.gethostname(), "rc": proc.returncode,
            "stdout": proc.stdout[-20000:], "stderr": proc.stderr[-20000:]}


if __name__ == "__main__":
    ray.init(address="auto", logging_level="ERROR")
    script = (HERE / "build_cosmos_env.sh").read_text()
    passthrough = {k: v for k, v in os.environ.items()
                   if k in ("COSMOS_COMMIT", "COSMOS_TAG", "CUDA_EXTRA", "PYTHON_VERSION", "OFFROAD_ROOT")}
    res = ray.get(build.remote(script, passthrough))
    print(f"host={res['host']} rc={res['rc']}")
    print(res["stdout"])
    print(res["stderr"], file=sys.stderr)
    sys.exit(res["rc"])
