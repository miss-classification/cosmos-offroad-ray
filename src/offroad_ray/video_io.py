"""FFmpeg-based encode/decode/probe used by Ray workers (Anyscale environment).

FFmpeg threads are capped explicitly (Ray CPU reservations do not cap them).
"""
from __future__ import annotations

import json
import os
import subprocess
from fractions import Fraction

import numpy as np

FFMPEG = os.environ.get("OFFROAD_FFMPEG", "ffmpeg")
FFPROBE = os.environ.get("OFFROAD_FFPROBE", "ffprobe")


def _threads() -> str:
    return os.environ.get("OFFROAD_FFMPEG_THREADS", "2")


def encode_rgb(frames: np.ndarray, path: str, fps: float, *, crf: int | None = 12,
               lossless_rgb: bool = False) -> None:
    """Encode (T,H,W,3) uint8 RGB frames to H.264 mp4 with a constant frame rate.

    lossless_rgb=True uses libx264rgb at qp 0 (bit-exact RGB), used for cached
    control videos so the model consumes exactly the control that was computed.
    """
    t, h, w, c = frames.shape
    assert c == 3 and frames.dtype == np.uint8
    rate = str(Fraction(fps).limit_denominator(1001))
    if lossless_rgb:
        codec = ["-c:v", "libx264rgb", "-qp", "0", "-preset", "veryfast", "-pix_fmt", "rgb24"]
    else:
        codec = ["-c:v", "libx264", "-crf", str(crf), "-preset", "medium", "-pix_fmt", "yuv420p"]
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-threads", _threads(),
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", rate, "-i", "-",
           *codec, "-r", rate, "-movflags", "+faststart", path]
    proc = subprocess.run(cmd, input=np.ascontiguousarray(frames).tobytes(), capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg encode failed: {proc.stderr.decode(errors='replace')[-2000:]}")


def probe(path: str) -> dict:
    cmd = [FFPROBE, "-v", "error", "-select_streams", "v:0", "-count_frames",
           "-show_entries", "stream=width,height,nb_read_frames,r_frame_rate,avg_frame_rate,duration,codec_name,pix_fmt",
           "-show_entries", "format=duration,size", "-of", "json", path]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {out.stderr[-1000:]}")
    j = json.loads(out.stdout)
    s = j["streams"][0]
    fmt = j.get("format", {})
    return {
        "width": int(s["width"]), "height": int(s["height"]),
        "frames": int(s.get("nb_read_frames", 0)),
        "fps": float(Fraction(s["r_frame_rate"])),
        "duration_s": float(s.get("duration") or fmt.get("duration") or 0.0),
        "codec": s.get("codec_name"), "pix_fmt": s.get("pix_fmt"),
        "bytes": int(fmt.get("size", 0)),
    }


def decode_rgb(path: str, width: int | None = None, height: int | None = None) -> np.ndarray:
    """Decode all frames to (T,H,W,3) uint8 RGB."""
    if width is None or height is None:
        p = probe(path)
        width, height = p["width"], p["height"]
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-threads", _threads(), "-i", path,
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    out = subprocess.run(cmd, capture_output=True)
    if out.returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed: {out.stderr.decode(errors='replace')[-1000:]}")
    buf = np.frombuffer(out.stdout, dtype=np.uint8)
    frame = width * height * 3
    if buf.size % frame:
        raise RuntimeError(f"decoded byte count {buf.size} not a multiple of frame size {frame}")
    return buf.reshape(-1, height, width, 3)
