"""Source frame manifest for a downloaded TartanDrive 2.0 KITTI-format run.

Reads ``<camera>/timestamps.txt`` (Unix seconds, one per frame, aligned with
``000000.png``...) and the exporter's ``data_extraction.log`` to flag frames the
exporter duplicated to fill gaps ("filled missing frame i <- j" for the camera
topic). Duplicated fills are treated as invalid source frames, not real data.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

CAMERA_TOPICS = {
    "image_left_color": "/multisense/left/image_rect_color",
    "image_left": "/multisense/left/image_rect",
}

LICENSE = "CC BY 4.0"
ATTRIBUTION = (
    "TartanDrive 2.0, AirLab, Carnegie Mellon University (theairlab.org/TartanDrive2). "
    "Licensed CC BY 4.0. Frames were temporally resampled, re-encoded and used as "
    "conditioning for synthetic appearance changes by this project."
)


def parse_filled_frames(log_text: str, topic: str) -> set[int]:
    pat = re.compile(rf"--\s*{re.escape(topic)}\s+filled missing frame\s+(\d+)\s*<-\s*(\d+)")
    return {int(m.group(1)) for m in pat.finditer(log_text)}


def build_run_manifest(run_dir: str | Path, camera: str) -> dict:
    """Return a dict of arrays/metadata describing all source frames of one run."""
    run_dir = Path(run_dir)
    cam_dir = run_dir / camera
    ts = np.loadtxt(cam_dir / "timestamps.txt", dtype=np.float64, ndmin=1)
    pngs = sorted(cam_dir.glob("*.png"))
    names = [p.name for p in pngs]
    expected = [f"{i:06d}.png" for i in range(len(ts))]
    missing = sorted(set(expected) - set(names))
    valid = np.ones(len(ts), dtype=bool)
    for m in missing:
        valid[int(m[:6])] = False
    filled: set[int] = set()
    log = run_dir / "data_extraction.log"
    if log.exists():
        filled = parse_filled_frames(log.read_text(errors="replace"), CAMERA_TOPICS.get(camera, camera))
        for i in filled:
            if i < len(valid):
                valid[i] = False
    d = np.diff(ts)
    nonmono = np.where(d <= 0)[0]
    valid[nonmono + 1] = False
    return {
        "run_id": run_dir.name,
        "camera": camera,
        "timestamps": ts,
        "frame_paths": [str(cam_dir / e) for e in expected],
        "valid": valid,
        "num_frames": int(len(ts)),
        "missing_files": missing,
        "exporter_filled": sorted(filled),
        "nonmonotonic": nonmono.tolist(),
        "duration_s": float(ts[-1] - ts[0]) if len(ts) else 0.0,
        "dt_median_s": float(np.median(d)) if len(d) else float("nan"),
        "dt_min_s": float(d.min()) if len(d) else float("nan"),
        "dt_max_s": float(d.max()) if len(d) else float("nan"),
    }
