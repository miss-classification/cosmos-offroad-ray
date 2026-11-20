"""Pure functions for clip windowing and temporal resampling (no I/O, unit tested).

Resampling policy "nearest": for target times t_k = t0 + k / fps_target,
k = 0..N-1, use the source frame whose timestamp is closest to t_k. Physical
duration is preserved (N / fps_target seconds of source time). At 10 Hz source
and 16 Hz target every source frame is shown 1 or 2 times; nothing is
interpolated or invented, and every output frame maps to exactly one source frame.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ResampleResult:
    target_times: np.ndarray       # (N,) absolute target timestamps, seconds
    source_index: np.ndarray       # (N,) index into the source frame list
    abs_error_s: np.ndarray        # (N,) |source_t - target_t|
    duplicates: int                # output frames that repeat the previous source frame
    max_reuse: int                 # most times any single source frame is used


def resample_nearest(src_t: np.ndarray, t0: float, num_frames: int, fps: float) -> ResampleResult:
    src_t = np.asarray(src_t, dtype=np.float64)
    if src_t.ndim != 1 or len(src_t) < 2:
        raise ValueError("need at least two source timestamps")
    if np.any(np.diff(src_t) <= 0):
        raise ValueError("source timestamps must be strictly increasing")
    tgt = t0 + np.arange(num_frames, dtype=np.float64) / fps
    if tgt[0] < src_t[0] - 1e-9 or tgt[-1] > src_t[-1] + 1e-9:
        raise ValueError("target window extends beyond source timestamps")
    right = np.clip(np.searchsorted(src_t, tgt), 1, len(src_t) - 1)
    left = right - 1
    choose_right = np.abs(src_t[right] - tgt) < np.abs(src_t[left] - tgt)
    idx = np.where(choose_right, right, left)
    err = np.abs(src_t[idx] - tgt)
    dups = int(np.sum(idx[1:] == idx[:-1]))
    max_reuse = int(np.bincount(idx - idx.min()).max())
    return ResampleResult(tgt, idx, err, dups, max_reuse)


def window_duration(num_frames: int, fps: float) -> float:
    """Physical duration covered by num_frames at fps (first frame start to last frame end)."""
    return num_frames / fps


def validate_window(src_t: np.ndarray, valid: np.ndarray, t0: float, num_frames: int, fps: float,
                    max_gap_s: float, max_reuse: int) -> tuple[bool, list[str], ResampleResult | None]:
    """Check a candidate window. Returns (ok, reasons, resample)."""
    reasons: list[str] = []
    t1 = t0 + (num_frames - 1) / fps
    # Work on the local neighbourhood of the window only: a bad timestamp elsewhere
    # in the run must not invalidate this window.
    local = np.where((src_t >= t0 - max_gap_s) & (src_t <= t1 + max_gap_s))[0]
    if local.size < 2:
        return False, ["fewer than two source frames in window"], None
    span = np.arange(local.min(), local.max() + 1)
    bad = span[~valid[span]]
    if bad.size:
        reasons.append(f"{bad.size} invalid source frames in window (e.g. index {int(bad[0])})")
    if np.any(np.diff(src_t[span]) <= 0):
        reasons.append("non-increasing source timestamps in window")
    if reasons:
        return False, reasons, None
    try:
        rs_local = resample_nearest(src_t[span], t0, num_frames, fps)
    except ValueError as e:
        return False, [f"resample: {e}"], None
    rs = ResampleResult(rs_local.target_times, rs_local.source_index + span[0], rs_local.abs_error_s,
                        rs_local.duplicates, rs_local.max_reuse)
    used = np.arange(rs.source_index.min(), rs.source_index.max() + 1)
    gaps = np.diff(src_t[used]) if len(used) > 1 else np.array([0.0])
    if gaps.size and gaps.max() > max_gap_s:
        reasons.append(f"source gap {gaps.max():.3f}s > {max_gap_s}s")
    if rs.max_reuse > max_reuse:
        reasons.append(f"source frame reused {rs.max_reuse}x > {max_reuse}")
    if rs.abs_error_s.max() > max_gap_s:
        reasons.append(f"nearest-frame error {rs.abs_error_s.max():.3f}s > {max_gap_s}s")
    if t1 > src_t[-1]:
        reasons.append("window exceeds run end")
    return (not reasons), reasons, rs


def select_windows(src_t: np.ndarray, valid: np.ndarray, *, num_frames: int, fps: float,
                   clips_per_run: int, start_offset_s: float, min_gap_s: float,
                   max_gap_s: float, max_reuse: int) -> tuple[list[tuple[float, ResampleResult]], list[dict]]:
    """Choose up to clips_per_run non-overlapping valid windows, spread across the run.

    Candidate starts are evenly spaced over the usable span; invalid candidates
    are recorded (not silently dropped) and the scan slides forward in 0.5 s
    steps to find a valid replacement.
    """
    src_t = np.asarray(src_t, dtype=np.float64)
    dur = window_duration(num_frames, fps)
    first = src_t[0] + start_offset_s
    last_start = src_t[-1] - (num_frames - 1) / fps
    chosen: list[tuple[float, ResampleResult]] = []
    rejected: list[dict] = []
    if last_start < first:
        rejected.append({"t0": float(first), "reasons": ["run shorter than offset + one clip"]})
        return chosen, rejected
    span = last_start - first
    max_fit = int(span // (dur + min_gap_s)) + 1
    n = min(clips_per_run, max_fit)
    starts = [first + span * (i / max(n - 1, 1)) if n > 1 else first + span / 2 for i in range(n)]
    next_allowed = first
    for s in starts:
        t0 = max(s, next_allowed)
        while t0 <= last_start + 1e-9:
            ok, reasons, rs = validate_window(src_t, valid, t0, num_frames, fps, max_gap_s, max_reuse)
            if ok:
                chosen.append((float(t0), rs))
                next_allowed = t0 + dur + min_gap_s
                break
            rejected.append({"t0": float(t0), "reasons": reasons})
            t0 += 0.5
    return chosen, rejected
