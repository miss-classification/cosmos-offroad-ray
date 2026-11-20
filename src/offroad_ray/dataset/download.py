"""Headless TartanDrive 2.0 downloader.

The official tool (castacks/tartan_drive_2.0, scripts/tartandrive_gui.py) is a
Tk GUI. It reads ``assets/files.yaml`` and fetches objects anonymously from the
AirLab S3-compatible endpoint below with boto3 (UNSIGNED). This module reuses
exactly that endpoint, bucket and key layout; it does not invent URLs or bypass
any access control.

Downloads run as Ray tasks on worker nodes and write to immutable paths:
    <root>/data/tartandrive2/raw/<run_id>/<camera>.tar      (+ .meta.json)
    <root>/data/tartandrive2/raw/<run_id>/<camera>/         (extracted, read-only use)
"""
from __future__ import annotations

import os
import socket
import tarfile
import time
from pathlib import Path

from offroad_ray.ids import sha256_file
from offroad_ray.storage import atomic_write_json, read_json

ENDPOINT = "https://airlab-cloud.andrew.cmu.edu:8080/swift/v1/AUTH_ac8533a83cff4d48bc8c608ad222d330"
BUCKET = "tartandrive2"
TOOLS_REPO = "https://github.com/castacks/tartan_drive_2.0"
TOOLS_COMMIT = "fe893b9b4132bfbd30ba84a839872b3a855e8176"
SMALL_FILES = ["data_extraction.log"]


def _client():
    import boto3
    from botocore import UNSIGNED
    from botocore.client import Config
    return boto3.client("s3", endpoint_url=ENDPOINT,
                        config=Config(signature_version=UNSIGNED, retries={"max_attempts": 5}))


def object_key(run_id: str, filename: str) -> str:
    return f"kitti/all_topics/{run_id}/{filename}"


def head_size(run_id: str, filename: str) -> int:
    return int(_client().head_object(Bucket=BUCKET, Key=object_key(run_id, filename))["ContentLength"])


def download_object(run_id: str, filename: str, dest: Path) -> dict:
    """Stream one object to ``dest`` atomically. Idempotent: verified files are skipped."""
    meta_path = dest.with_suffix(dest.suffix + ".meta.json")
    if dest.exists() and meta_path.exists():
        meta = read_json(meta_path)
        if meta.get("bytes") == dest.stat().st_size:
            return {**meta, "skipped": True, "seconds": 0.0}
    dest.parent.mkdir(parents=True, exist_ok=True)
    c = _client()
    key = object_key(run_id, filename)
    t0 = time.time()
    head = c.head_object(Bucket=BUCKET, Key=key)
    tmp = dest.with_name(f".{dest.name}.{os.getpid()}.part")
    body = c.get_object(Bucket=BUCKET, Key=key)["Body"]
    with open(tmp, "wb") as f:
        for chunk in body.iter_chunks(chunk_size=8 << 20):
            f.write(chunk)
    size = tmp.stat().st_size
    if size != int(head["ContentLength"]):
        tmp.unlink(missing_ok=True)
        raise IOError(f"size mismatch for {key}: {size} != {head['ContentLength']}")
    os.replace(tmp, dest)
    seconds = time.time() - t0
    meta = {
        "run_id": run_id, "key": key, "bucket": BUCKET, "endpoint": ENDPOINT,
        "bytes": size, "etag": head.get("ETag", "").strip('"'),
        "last_modified": str(head.get("LastModified")), "sha256": sha256_file(dest),
        "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seconds": round(seconds, 2), "host": socket.gethostname(),
    }
    atomic_write_json(meta_path, meta)
    return {**meta, "skipped": False}


def extract_tar(tar_path: Path, out_dir: Path) -> dict:
    """Extract into a temp dir then rename; refuses unsafe member paths."""
    done = out_dir / ".extracted.json"
    if done.exists():
        return read_json(done)
    tmp = out_dir.with_name(f".{out_dir.name}.extracting.{os.getpid()}")
    tmp.mkdir(parents=True, exist_ok=False)
    n = 0
    with tarfile.open(tar_path) as tf:
        tf.extractall(tmp, filter="data")  # rejects absolute paths / traversal
        n = sum(1 for m in tf.getmembers() if m.isfile())
    files = sorted(p.name for p in tmp.rglob("*") if p.is_file())
    info = {"tar": str(tar_path), "members": n, "files_sample": files[:5],
            "num_png": sum(f.endswith(".png") for f in files)}
    atomic_write_json(tmp / ".extracted.json", info)
    os.replace(tmp, out_dir)
    return info


def fetch_run(run_id: str, camera: str, raw_dir: str) -> dict:
    """Download + extract one run's camera tarball and small metadata files."""
    run_dir = Path(raw_dir) / run_id
    t0 = time.time()
    tar_meta = download_object(run_id, f"{camera}.tar", run_dir / f"{camera}.tar")
    small = {}
    for f in SMALL_FILES:
        try:
            small[f] = download_object(run_id, f, run_dir / f)["bytes"]
        except Exception as e:  # metadata is optional; record why it is missing
            small[f] = f"unavailable: {type(e).__name__}"
    t1 = time.time()
    ext = extract_tar(run_dir / f"{camera}.tar", run_dir / camera)
    t2 = time.time()
    return {"run_id": run_id, "camera": camera, "tar": tar_meta, "small_files": small,
            "extract": ext, "download_s": round(t1 - t0, 2), "extract_s": round(t2 - t1, 2),
            "host": socket.gethostname()}


def fetch_runs(runs: list[str], camera: str, raw_dir: str, max_download_gb: float) -> list[dict]:
    """Budget-check, then fetch runs in parallel as Ray tasks on worker nodes."""
    import ray
    from offroad_ray.ray_env import WORKER_ONLY

    sizes = {r: head_size(r, f"{camera}.tar") for r in runs}
    already = {r for r in runs if (Path(raw_dir) / r / f"{camera}.tar.meta.json").exists()}
    new_gb = sum(s for r, s in sizes.items() if r not in already) / 1e9
    if new_gb > max_download_gb:
        raise RuntimeError(f"planned new download {new_gb:.2f} GB exceeds max_download_gb={max_download_gb}")
    task = ray.remote(num_cpus=1, resources=WORKER_ONLY, max_retries=2)(fetch_run)
    return ray.get([task.remote(r, camera, raw_dir) for r in runs])
