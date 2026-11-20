"""Distributed clip preparation with Ray Data.

Pipeline (all heavy work on GPU-worker CPUs; the head advertises 0 CPUs):

  runs ──flat_map(plan_clips)──▶ clip candidates (metadata only)
       ──map(prepare_clip)─────▶ decode PNGs, validate, resample, encode model
                                  input, decode-back verify, Canny edge control at
                                  model resolution, lossless encode, verify
       ──write_parquet──────────▶ prepared clip catalog

Rows carry paths and metadata, never decoded frames. Each clip writes into
<prepared>/<prep_hash>/<clip_id>/ and is skipped when a validated record with the
same prep_hash and source checksum already exists.
"""
from __future__ import annotations

import hashlib
import json
import os
import socket
import time
from pathlib import Path

import numpy as np

from offroad_ray.ids import clip_id as make_clip_id, prep_config_hash
from offroad_ray.storage import atomic_write_json, make_owned_dir, read_json

PREP_CODE_VERSION = "prep-v1"
CANNY = {"very_low": (20, 50), "low": (50, 100), "medium": (100, 200),
         "high": (200, 300), "very_high": (300, 400)}  # cosmos control_input.py, use_random=False


def compute_prep_hash(cfg) -> str:
    return prep_config_hash(cfg.clips.model_dump(), cfg.control.model_dump(mode="json"), PREP_CODE_VERSION)


def plan_clips(row: dict) -> list[dict]:
    """flat_map: one run -> its candidate clip windows (metadata only)."""
    from offroad_ray.dataset.clips import select_windows
    from offroad_ray.dataset.manifest import build_run_manifest

    p = json.loads(row["params"])
    m = build_run_manifest(row["run_dir"], p["camera"])
    chosen, rejected = select_windows(
        m["timestamps"], m["valid"], num_frames=p["num_frames"], fps=p["target_fps"],
        clips_per_run=p["clips_per_run"], start_offset_s=p["start_offset_s"],
        min_gap_s=p["min_gap_s"], max_gap_s=p["max_source_gap_s"], max_reuse=p["max_frame_reuse"])
    run_stats = {k: m[k] for k in ("num_frames", "duration_s", "dt_median_s", "dt_min_s", "dt_max_s")}
    run_stats.update(missing_files=len(m["missing_files"]), exporter_filled=len(m["exporter_filled"]),
                     nonmonotonic=len(m["nonmonotonic"]), rejected_windows=len(rejected))
    out = []
    for k, (t0, rs) in enumerate(chosen):
        idx = rs.source_index
        out.append({
            "run_id": m["run_id"], "camera": p["camera"], "clip_index": k,
            "clip_id": make_clip_id(p["dataset_id"], m["run_id"], p["camera"], t0 - m["timestamps"][0],
                                    p["num_frames"], p["target_fps"]),
            "src_t0_unix": float(t0), "run_t0_unix": float(m["timestamps"][0]),
            "source_frame_paths": json.dumps([m["frame_paths"][i] for i in idx]),
            "source_frame_index": json.dumps(idx.tolist()),
            "source_timestamps": json.dumps(m["timestamps"][idx].tolist()),
            "target_timestamps": json.dumps(rs.target_times.tolist()),
            "resample_max_err_s": float(rs.abs_error_s.max()),
            "resample_duplicates": int(rs.duplicates),
            "resample_max_reuse": int(rs.max_reuse),
            "unique_source_frames": int(len(np.unique(idx))),
            "run_stats": json.dumps(run_stats),
            "rejected_windows": json.dumps(rejected[:20]),
            "params": row["params"],
        })
    if not out:
        out.append({"run_id": m["run_id"], "camera": p["camera"], "clip_index": -1, "clip_id": "",
                    "src_t0_unix": float("nan"), "run_t0_unix": float(m["timestamps"][0]) if m["num_frames"] else float("nan"),
                    "source_frame_paths": "[]", "source_frame_index": "[]", "source_timestamps": "[]",
                    "target_timestamps": "[]", "resample_max_err_s": float("nan"), "resample_duplicates": 0,
                    "resample_max_reuse": 0, "unique_source_frames": 0, "run_stats": json.dumps(run_stats),
                    "rejected_windows": json.dumps(rejected[:20]), "params": row["params"]})
    return out


def edge_control(frames_rgb: np.ndarray, model_wh: tuple[int, int], preset: str) -> np.ndarray:
    """Replicates the pinned Cosmos on-the-fly edge path: INTER_AREA resize of the
    input to model resolution, then cv2.Canny on the RGB frame with preset thresholds."""
    import cv2
    w, h = model_wh
    lo, hi = CANNY[preset]
    out = np.empty((frames_rgb.shape[0], h, w, 3), dtype=np.uint8)
    for i, f in enumerate(frames_rgb):
        r = cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA)
        e = cv2.Canny(r, lo, hi)
        out[i] = e[..., None]
    return out


def _frame_flags(frames: np.ndarray) -> dict:
    gray = frames.mean(axis=3)
    means = gray.mean(axis=(1, 2))
    stds = gray.std(axis=(1, 2))
    return {"blank_frames": int(np.sum(stds < 2.0)), "luma_mean": float(means.mean()),
            "luma_min": float(means.min()), "luma_max": float(means.max())}


_NAN = float("nan")
# Every catalog row has exactly these columns, so Parquet blocks share one schema
# whether a clip succeeded, failed or had no valid window.
CATALOG_DEFAULTS: dict = {
    "run_id": "", "camera": "", "clip_id": "", "clip_index": -1, "src_t0_unix": _NAN, "run_t0_unix": _NAN,
    "source_frame_index": "[]", "source_timestamps": "[]", "target_timestamps": "[]",
    "resample_max_err_s": _NAN, "resample_duplicates": -1, "resample_max_reuse": -1,
    "unique_source_frames": -1, "run_stats": "{}", "rejected_windows": "[]",
    "host": "", "prep_hash": "", "status": "error", "error": "", "clip_path": "", "control_path": "",
    "source_preview_path": "", "clip_sha256": "", "source_sha256": "", "control_sha256": "",
    "cache_hit": False, "source_width": -1, "source_height": -1, "source_fps_est": _NAN,
    "model_input_width": -1, "model_input_height": -1, "model_input_fps": _NAN, "model_input_frames": -1,
    "model_input_duration_s": _NAN, "control_width": -1, "control_height": -1, "edge_density": _NAN,
    "encode_psnr_db": _NAN, "resampling": "", "blank_frames": -1, "luma_mean": _NAN, "luma_min": _NAN,
    "luma_max": _NAN, "timing_decode_s": _NAN, "timing_encode_verify_s": _NAN, "timing_control_s": _NAN,
    "prep_seconds": _NAN, "license": "", "attribution": "", "dataset_id": "", "dataset_version": "",
}


def prepare_clip(row: dict) -> dict:
    """map: build one model-input clip + edge control, normalised to the catalog schema."""
    rec = _prepare_clip(row)
    out = {k: rec.get(k, d) for k, d in CATALOG_DEFAULTS.items()}
    for k, d in CATALOG_DEFAULTS.items():
        if out[k] is None:
            out[k] = d
        elif isinstance(d, float):
            out[k] = float(out[k])
        elif isinstance(d, bool):
            out[k] = bool(out[k])
        elif isinstance(d, int):
            out[k] = int(out[k])
        else:
            out[k] = str(out[k])
    return out


def _prepare_clip(row: dict) -> dict:
    """Build one model-input clip + edge control. Idempotent via cache record."""
    import cv2
    from offroad_ray.video_io import decode_rgb, encode_rgb, probe

    cv2.setNumThreads(int(os.environ.get("OPENCV_NUM_THREADS", "2")))
    p = json.loads(row["params"])
    t_start = time.time()
    base = {k: row[k] for k in ("run_id", "camera", "clip_id", "clip_index", "src_t0_unix", "run_t0_unix",
                                 "source_frame_index", "source_timestamps", "target_timestamps",
                                 "resample_max_err_s", "resample_duplicates", "resample_max_reuse",
                                 "unique_source_frames", "run_stats", "rejected_windows")}
    base.update(host=socket.gethostname(), prep_hash=p["prep_hash"], status="error", error="",
                clip_path="", control_path="", source_preview_path="", clip_sha256="", source_sha256="",
                control_sha256="", cache_hit=False)
    if row["clip_index"] < 0:
        base.update(status="no_valid_window", error="no valid window in run")
        return base
    out_dir = Path(p["prepared_root"]) / p["prep_hash"] / row["clip_id"]
    record = out_dir / "clip_record.json"
    paths = json.loads(row["source_frame_paths"])

    # Source checksum: hash of the source PNG bytes in model-frame order.
    h = hashlib.sha256()
    for fp in dict.fromkeys(paths):  # unique, order-preserving
        with open(fp, "rb") as f:
            h.update(f.read())
    src_sha = h.hexdigest()
    base["source_sha256"] = src_sha

    if record.exists():
        rec = read_json(record)
        if rec.get("prep_hash") == p["prep_hash"] and rec.get("source_sha256") == src_sha \
                and rec.get("status") == "ok" and Path(rec["clip_path"]).exists() \
                and Path(rec["control_path"]).exists():
            rec["cache_hit"] = True
            rec["host"] = socket.gethostname()
            rec["prep_seconds"] = round(time.time() - t_start, 3)
            return rec

    try:
        make_owned_dir(out_dir)
        cache: dict[str, np.ndarray] = {}
        frames = []
        corrupt = []
        for fp in paths:
            if fp not in cache:
                im = cv2.imread(fp, cv2.IMREAD_COLOR)
                if im is None:
                    corrupt.append(fp)
                    cache[fp] = None
                else:
                    cache[fp] = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            frames.append(cache[fp])
        if corrupt:
            raise ValueError(f"{len(corrupt)} corrupt/unreadable source frames, e.g. {corrupt[0]}")
        shapes = {f.shape for f in frames}
        if len(shapes) != 1:
            raise ValueError(f"inconsistent frame shapes {shapes}")
        frames = np.stack(frames)
        src_h, src_w = frames.shape[1:3]
        flags = _frame_flags(frames)
        if flags["blank_frames"]:
            raise ValueError(f"{flags['blank_frames']} blank source frames")
        t_dec = time.time()

        n, fps = p["num_frames"], p["target_fps"]
        clip_path = str(out_dir / "input.mp4")
        encode_rgb(frames, clip_path, fps, crf=p["crf"])
        # Decode the encoded clip back and validate it (frame count, size, duration).
        pr = probe(clip_path)
        dec = decode_rgb(clip_path, pr["width"], pr["height"])
        problems = []
        if pr["frames"] != n or dec.shape[0] != n:
            problems.append(f"frames {pr['frames']}/{dec.shape[0]} != {n}")
        if (pr["width"], pr["height"]) != (src_w, src_h):
            problems.append(f"size {pr['width']}x{pr['height']} != {src_w}x{src_h}")
        if abs(pr["duration_s"] - n / fps) > 1.0 / fps:
            problems.append(f"duration {pr['duration_s']:.3f}s != {n / fps:.3f}s")
        psnr = float(10 * np.log10(255.0 ** 2 / max(np.mean((dec.astype(np.float32) - frames) ** 2), 1e-9)))
        t_enc = time.time()

        # Edge control from the *decoded* model input, exactly as Cosmos would see it.
        ctrl = edge_control(dec, tuple(p["model_resolution"]), p["preset_edge_threshold"])
        control_path = str(out_dir / "control_edge.mp4")
        encode_rgb(ctrl, control_path, fps, lossless_rgb=True)
        ctrl_back = decode_rgb(control_path, ctrl.shape[2], ctrl.shape[1])
        if ctrl_back.shape != ctrl.shape or not np.array_equal(ctrl_back, ctrl):
            problems.append("lossless control round trip mismatch")
        edge_density = float((ctrl[..., 0] > 0).mean())
        t_ctrl = time.time()

        # Small source-cadence preview (unique source frames at native 10 Hz) for reports.
        uniq = list(dict.fromkeys(paths))
        prev = np.stack([cache[u] for u in uniq])
        src_fps = 1.0 / np.median(np.diff(json.loads(row["source_timestamps"]))[np.diff(json.loads(row["source_timestamps"])) > 0])
        preview_path = str(out_dir / "source_native.mp4")
        encode_rgb(prev, preview_path, float(round(src_fps, 3)), crf=18)

        rec = {**base,
               "status": "ok" if not problems else "invalid", "error": "; ".join(problems),
               "clip_path": clip_path, "control_path": control_path, "source_preview_path": preview_path,
               "clip_sha256": hashlib.sha256(Path(clip_path).read_bytes()).hexdigest(),
               "control_sha256": hashlib.sha256(Path(control_path).read_bytes()).hexdigest(),
               "source_width": int(src_w), "source_height": int(src_h), "source_fps_est": float(src_fps),
               "model_input_width": int(pr["width"]), "model_input_height": int(pr["height"]),
               "model_input_fps": float(fps), "model_input_frames": int(pr["frames"]),
               "model_input_duration_s": float(pr["duration_s"]),
               "control_width": int(ctrl.shape[2]), "control_height": int(ctrl.shape[1]),
               "edge_density": edge_density, "encode_psnr_db": psnr,
               "resampling": "nearest-timestamp frame selection (duplication), no interpolation",
               **flags,
               "timing_decode_s": round(t_dec - t_start, 3), "timing_encode_verify_s": round(t_enc - t_dec, 3),
               "timing_control_s": round(t_ctrl - t_enc, 3),
               "prep_seconds": round(time.time() - t_start, 3), "cache_hit": False,
               "license": p["license"], "attribution": p["attribution"], "dataset_id": p["dataset_id"],
               "dataset_version": p["dataset_version"]}
        atomic_write_json(out_dir / "timestamps_map.json", {
            "clip_id": row["clip_id"], "policy": "nearest",
            "target_timestamps_unix": json.loads(row["target_timestamps"]),
            "source_timestamps_unix": json.loads(row["source_timestamps"]),
            "source_frame_index": json.loads(row["source_frame_index"])})
        atomic_write_json(record, rec)
        return rec
    except Exception as e:
        base.update(status="error", error=f"{type(e).__name__}: {e}", prep_seconds=round(time.time() - t_start, 3))
        return base


def run_prepare(cfg, *, concurrency: int = 16) -> dict:
    """Execute the Ray Data preparation pipeline and write the catalog."""
    import ray
    from offroad_ray.dataset.manifest import ATTRIBUTION, LICENSE
    from offroad_ray.ray_env import WORKER_ONLY

    ph = compute_prep_hash(cfg)
    prepared_root = str(cfg.paths.prepared_dir)
    params = json.dumps({
        "camera": cfg.dataset.camera, "dataset_id": cfg.dataset.dataset_id,
        "dataset_version": cfg.dataset.dataset_version,
        "num_frames": cfg.clips.num_frames, "target_fps": cfg.clips.target_fps,
        "clips_per_run": cfg.clips.clips_per_run, "start_offset_s": cfg.clips.start_offset_s,
        "min_gap_s": cfg.clips.min_gap_s, "max_source_gap_s": cfg.clips.max_source_gap_s,
        "max_frame_reuse": cfg.clips.max_frame_reuse, "crf": cfg.clips.crf,
        "model_resolution": list(cfg.control.model_resolution),
        "preset_edge_threshold": cfg.control.preset_edge_threshold,
        "prep_hash": ph, "prepared_root": prepared_root, "license": LICENSE, "attribution": ATTRIBUTION,
    })
    runs = [{"run_id": r, "run_dir": str(cfg.paths.raw_dir / r), "params": params} for r in cfg.dataset.runs]
    t0 = time.time()
    ctx = ray.data.DataContext.get_current()
    ctx.enable_progress_bars = False
    ds = (ray.data.from_items(runs, override_num_blocks=len(runs))
          .flat_map(plan_clips, num_cpus=1, resources=WORKER_ONLY)
          .repartition(target_num_rows_per_block=1)   # one clip per block -> real parallelism
          .map(prepare_clip, num_cpus=2, resources=WORKER_ONLY,
               compute=ray.data.TaskPoolStrategy(size=concurrency)))
    out_dir = Path(prepared_root) / ph / "catalog"
    tmp_dir = out_dir.with_name(f"catalog.tmp.{int(t0)}")
    ds.write_parquet(str(tmp_dir))
    stats = ds.stats()
    if out_dir.exists():
        out_dir.rename(out_dir.with_name(f"catalog.prev.{int(t0)}"))
    tmp_dir.rename(out_dir)
    wall = time.time() - t0
    import pandas as pd
    df = pd.read_parquet(out_dir)
    summary = {
        "prep_hash": ph, "catalog": str(out_dir), "wall_s": round(wall, 2),
        "rows": int(len(df)), "ok": int((df.status == "ok").sum()),
        "status_counts": df.status.value_counts().to_dict(),
        "cache_hits": int(df.cache_hit.sum()), "hosts": df.host.value_counts().to_dict(),
        "prep_seconds_sum": float(df.get("prep_seconds", pd.Series(dtype=float)).sum()),
    }
    atomic_write_json(Path(prepared_root) / ph / f"prepare_summary.{int(t0)}.json",
                      {**summary, "ray_data_stats": stats})
    (Path(prepared_root) / ph / f"ray_data_stats.{int(t0)}.txt").write_text(stats)
    return summary
