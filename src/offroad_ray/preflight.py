"""Cluster and environment preflight.

* ``cluster_inventory``: Ray nodes/resources plus a GPU-reserving Ray task on
  every GPU node (nvidia-smi on the CPU head is not a GPU preflight).
* ``env_check``: launches env_check_worker.py under the Cosmos interpreter on a
  worker GPU, against a real prepared clip and its cached control.
"""
from __future__ import annotations

import json
import socket
import subprocess
import time
from pathlib import Path


def _gpu_probe() -> dict:
    import os
    import ray

    def sh(c):
        r = subprocess.run(c, shell=True, capture_output=True, text=True, timeout=60)
        return r.stdout.strip()
    return {
        "host": socket.gethostname(), "node_ip": ray.util.get_node_ip_address(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "ray_gpu_ids": ray.get_gpu_ids(),
        "nvidia_smi": sh("nvidia-smi --query-gpu=name,memory.total,driver_version,uuid --format=csv,noheader"),
        "cuda_driver_api": sh("nvidia-smi | grep -o 'CUDA Version: [0-9.]*'"),
        "nvcc": sh("nvcc --version 2>/dev/null | tail -2 | head -1 || /usr/local/cuda/bin/nvcc --version | tail -2 | head -1"),
        "cpus_os": sh("nproc --all"), "mem_gb": sh("free -g | awk '/Mem/{print $2}'"),
        "local_storage": sh("df -h /mnt/local_storage | tail -1"),
        "cluster_storage": sh("df -h /mnt/cluster_storage | tail -1"),
        "ffmpeg": sh("ffmpeg -version | head -1"), "libx264": "libx264" in sh("ffmpeg -hide_banner -encoders"),
    }


def cluster_inventory() -> dict:
    import ray
    nodes = [n for n in ray.nodes() if n["Alive"]]
    gpu_nodes = [n for n in nodes if n["Resources"].get("GPU", 0) > 0]
    probe = ray.remote(num_gpus=1, num_cpus=1)(_gpu_probe)
    probes = ray.get([probe.remote() for _ in range(int(sum(n["Resources"]["GPU"] for n in gpu_nodes)))])
    uuids = [p["nvidia_smi"].split(",")[-1].strip() for p in probes]
    return {
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "ray_version": ray.__version__,
        "nodes": [{"ip": n["NodeManagerAddress"], "cpu": n["Resources"].get("CPU", 0), "gpu": n["Resources"].get("GPU", 0),
                   "memory_gb": round(n["Resources"].get("memory", 0) / 1e9, 1),
                   "object_store_gb": round(n["Resources"].get("object_store_memory", 0) / 1e9, 1),
                   "is_head": "anyscale/node-group:head" in n["Resources"]} for n in nodes],
        "num_gpu_nodes": len(gpu_nodes), "total_gpus": sum(n["Resources"]["GPU"] for n in gpu_nodes),
        "gpu_probes": probes, "distinct_gpu_uuids": len(set(uuids)),
    }


def env_check(cosmos_python: str, cosmos_src: str, clip: str, control: str, out_json: str) -> dict:
    import ray
    from offroad_ray.ray_env import SRC_DIR  # noqa: F401
    worker_src = (Path(__file__).parent / "env_check_worker.py").read_text()

    @ray.remote(num_gpus=1, num_cpus=4)
    def run(src: str) -> dict:
        import os
        import tempfile
        d = tempfile.mkdtemp(prefix="envcheck-")
        script = os.path.join(d, "env_check_worker.py")
        out = os.path.join(d, "out.json")
        open(script, "w").write(src)
        env = {"PATH": f"{os.path.dirname(cosmos_python)}:/usr/bin:/bin", "HOME": os.environ.get("HOME", "/home/ray"),
               "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""), "PYTHONNOUSERSITE": "1",
               "COSMOS_EXPERIMENTAL_CHECKPOINTS": "1"}
        t0 = time.time()
        p = subprocess.run([cosmos_python, "-I", script, clip, control, out], cwd=cosmos_src, env=env,
                           capture_output=True, text=True, timeout=1800)
        res = json.load(open(out)) if os.path.exists(out) else {}
        res.update(rc=p.returncode, seconds=round(time.time() - t0, 1), host=socket.gethostname(),
                   stdout_tail=p.stdout[-3000:], stderr_tail=p.stderr[-5000:])
        return res

    res = ray.get(run.remote(worker_src))
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(res, indent=1, default=str))
    return res


# (repo, revision or None, file to fetch). README.md / config.json can be downloadable on a
# gated repo without access (observed 2026-11-18), so each check fetches a file the
# license actually gates: a weight file or the gated repo's .gitattributes.
GATED_REPOS = [
    ("nvidia/Cosmos-Transfer2.5-2B", "bbaeedb2b57cc8b14a44653099e2551adb69dcc7", ".gitattributes"),
    ("nvidia/Cosmos-Reason1-7B", "3210bec0495fdc7a8d3dbb8d58da5711eab4b423", ".gitattributes"),
    ("nvidia/Cosmos-Predict2.5-2B", "f176dc95b4a70f53ce01c4b302851595e7322b00", ".gitattributes"),
    ("nvidia/Cosmos-Guardrail1", "d6d4bfa899a71454a700907664f3e88f503950cf", ".gitattributes"),
    # v1.5.4 text guardrail = Blocklist + Qwen3Guard (guardrail/common/presets.py). The
    # llamaGuard3 module is still in the tree but nothing imports it, so Llama Guard is not needed.
    ("Qwen/Qwen3Guard-Gen-0.6B", None, ".gitattributes"),  # open; unpinned upstream, resolved sha recorded
]


def hf_access_check(cosmos_python: str, hf_home: str, out_json: str) -> dict:
    """Real file download from every gated repo on a worker, inside the Cosmos interpreter.

    Reports per-repo ok/error and the resolved commit; never reads or prints the token.
    """
    import ray
    code = r'''
import json, sys
from huggingface_hub import hf_hub_download, HfApi
from huggingface_hub.utils import get_token
repos = json.loads(sys.argv[1]); out = {"token_present": get_token() is not None, "repos": {}}
for repo, rev, fn in repos:
    try:
        p = hf_hub_download(repo, fn, revision=rev)
        sha = HfApi().model_info(repo, revision=rev).sha
        out["repos"][repo] = {"ok": True, "requested_revision": rev, "resolved_sha": sha}
    except Exception as e:
        out["repos"][repo] = {"ok": False, "requested_revision": rev, "error": type(e).__name__ + ": " + str(e).splitlines()[0][:300]}
print(json.dumps(out))
'''

    @ray.remote(num_cpus=1, resources={"anyscale/accelerator_shape:1xA100-80G": 0.001})
    def run():
        import os
        env = {"PATH": f"{os.path.dirname(cosmos_python)}:/usr/bin:/bin", "HOME": os.environ.get("HOME", "/home/ray"),
               "HF_HOME": hf_home, "PYTHONNOUSERSITE": "1"}
        p = subprocess.run([cosmos_python, "-I", "-c", code, json.dumps(GATED_REPOS)], env=env,
                           capture_output=True, text=True, timeout=600)
        try:
            res = json.loads(p.stdout.strip().splitlines()[-1])
        except Exception:
            res = {"error": p.stderr[-2000:]}
        res["host"] = socket.gethostname()
        return res

    res = ray.get(run.remote())
    res["all_ok"] = bool(res.get("repos")) and all(v["ok"] for v in res["repos"].values())
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(res, indent=1))
    return res
