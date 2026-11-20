"""Automated output checks. Pure numpy/OpenCV functions plus one per-output entry point.

What these can and cannot establish (also in docs/methodology.md):
  * decode / frame count / size / duration: hard validity checks.
  * blank frames: near-constant frames (spatial std < 2 levels).
  * frozen frames: consecutive near-identical frames. Compared against the number
    of duplicated frames in the model INPUT, which nearest-timestamp resampling
    introduces on purpose (10 Hz source shown at 16 Hz).
  * temporal discontinuity candidates: frame-difference spikes in the output that
    the source does not have at the same index. A candidate is a pointer for human
    review, not proof of flicker.
  * edge agreement (structure proxy): Canny edges of the output vs Canny edges of
    the real source clip, with a 2 px tolerance. Fog and night legitimately remove visible edges,
    so low recall under those conditions is expected and is compared within
    condition, never as an absolute quality score.
  * luma / contrast / saturation shifts: describe the appearance change; they are
    not quality scores.
"""
from __future__ import annotations

import numpy as np

ANALYSIS_W, ANALYSIS_H = 512, 272   # downsample for speed; same aspect as 1024x544


def to_gray(frames: np.ndarray) -> np.ndarray:
    return frames.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)


def blank_frames(gray: np.ndarray, thr: float = 2.0) -> int:
    return int(np.sum(gray.std(axis=(1, 2)) < thr))


def frame_diffs(gray: np.ndarray) -> np.ndarray:
    """Mean absolute difference between consecutive frames, length T-1."""
    if len(gray) < 2:
        return np.zeros(0, np.float32)
    return np.abs(np.diff(gray, axis=0)).mean(axis=(1, 2))


def frozen_frames(diffs: np.ndarray, thr: float = 0.5) -> int:
    return int(np.sum(diffs < thr))


def discontinuity_candidates(out_diffs: np.ndarray, src_diffs: np.ndarray, k: float = 6.0,
                             min_abs: float = 8.0) -> list[int]:
    """Indices i (transition i -> i+1) where the output jumps but the source does not.

    Uses a robust z-score (median/MAD) of the output's own difference series and
    requires the source transition at the same index to be unremarkable.
    """
    if len(out_diffs) == 0:
        return []
    med = np.median(out_diffs)
    mad = np.median(np.abs(out_diffs - med)) + 1e-6
    z = (out_diffs - med) / (1.4826 * mad)
    s_med = np.median(src_diffs) if len(src_diffs) else 0.0
    s_mad = (np.median(np.abs(src_diffs - s_med)) + 1e-6) if len(src_diffs) else 1.0
    s_z = (src_diffs - s_med) / (1.4826 * s_mad) if len(src_diffs) else np.zeros_like(out_diffs)
    idx = np.where((z > k) & (out_diffs > min_abs) & (s_z[: len(out_diffs)] < k / 2))[0]
    return idx.tolist()


def temporal_luma_jitter(gray: np.ndarray) -> float:
    """Std of the frame-to-frame change in mean luma (levels). Context for flicker review."""
    m = gray.mean(axis=(1, 2))
    return float(np.diff(m).std()) if len(m) > 2 else 0.0


def edge_agreement(pred_edges: np.ndarray, ref_edges: np.ndarray, tol_px: int = 2) -> dict:
    """Precision/recall/F1 of binary edge maps with a spatial tolerance (dilation)."""
    import cv2
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol_px + 1, 2 * tol_px + 1))
    tp_p = tp_r = n_p = n_r = 0
    for p, r in zip(pred_edges, ref_edges):
        p = p > 0
        r = r > 0
        r_d = cv2.dilate(r.astype(np.uint8), k) > 0
        p_d = cv2.dilate(p.astype(np.uint8), k) > 0
        tp_p += int((p & r_d).sum())
        tp_r += int((r & p_d).sum())
        n_p += int(p.sum())
        n_r += int(r.sum())
    prec = tp_p / n_p if n_p else 0.0
    rec = tp_r / n_r if n_r else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"edge_precision": prec, "edge_recall": rec, "edge_f1": f1,
            "edge_density_out": n_p / max(pred_edges.size, 1), "edge_density_ref": n_r / max(ref_edges.size, 1)}


def appearance_stats(frames: np.ndarray) -> dict:
    import cv2
    gray = to_gray(frames)
    hsv = np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2HSV) for f in frames[:: max(1, len(frames) // 12)]])
    return {"luma_mean": float(gray.mean()), "contrast": float(gray.std(axis=(1, 2)).mean()),
            "saturation_mean": float(hsv[..., 1].mean())}


def resize_frames(frames: np.ndarray, w: int = ANALYSIS_W, h: int = ANALYSIS_H) -> np.ndarray:
    import cv2
    return np.stack([cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA) for f in frames])


def canny_frames(frames: np.ndarray, lo: int = 100, hi: int = 200) -> np.ndarray:
    import cv2
    return np.stack([cv2.Canny(f, lo, hi) for f in frames])


def compute_output_metrics(rec: dict) -> dict:
    """Ray Data map: metrics for one generation record. Decodes inside the worker."""
    import json
    import time
    from offroad_ray.video_io import decode_rgb, probe

    t0 = time.time()
    out = {"request_id": rec["request_id"], "status": rec.get("status", ""), "metrics_error": "",
           "decode_ok": False}
    if rec.get("status") != "generated_unreviewed" or not rec.get("output_path"):
        out["metrics_seconds"] = round(time.time() - t0, 3)
        return out
    try:
        expect = json.loads(rec["expect"]) if isinstance(rec.get("expect"), str) else rec.get("expect") or {}
        pr = probe(rec["output_path"])
        gen = decode_rgb(rec["output_path"], pr["width"], pr["height"])
        src = decode_rgb(rec["clip_path"])
        out.update(decode_ok=True, frames=int(gen.shape[0]), width=int(gen.shape[2]), height=int(gen.shape[1]),
                   duration_s=pr["duration_s"], fps=pr["fps"],
                   frames_ok=bool(gen.shape[0] == src.shape[0]),
                   size_ok=bool(gen.shape[1:3] == src.shape[1:3]))
        g_small, s_small = resize_frames(gen), resize_frames(src)
        g_gray, s_gray = to_gray(g_small), to_gray(s_small)
        g_d, s_d = frame_diffs(g_gray), frame_diffs(s_gray)
        cands = discontinuity_candidates(g_d, s_d)
        out.update(
            blank_frames=blank_frames(g_gray),
            frozen_frames=frozen_frames(g_d), source_frozen_frames=frozen_frames(s_d),
            discontinuity_candidates=json.dumps(cands), num_discontinuity_candidates=len(cands),
            luma_jitter=temporal_luma_jitter(g_gray), source_luma_jitter=temporal_luma_jitter(s_gray),
            mean_frame_diff=float(g_d.mean()) if len(g_d) else 0.0,
            source_mean_frame_diff=float(s_d.mean()) if len(s_d) else 0.0,
        )
        # Structure proxy: output Canny vs source Canny at the analysis resolution.
        ea = edge_agreement(canny_frames(g_small), canny_frames(s_small))
        out.update(ea)
        a_g, a_s = appearance_stats(g_small), appearance_stats(s_small)
        out.update({f"out_{k}": v for k, v in a_g.items()})
        out.update({f"src_{k}": v for k, v in a_s.items()})
        out.update(delta_luma=a_g["luma_mean"] - a_s["luma_mean"],
                   delta_contrast=a_g["contrast"] - a_s["contrast"],
                   delta_saturation=a_g["saturation_mean"] - a_s["saturation_mean"])
        flags = []
        if not out["frames_ok"]:
            flags.append("frame_count")
        if not out["size_ok"]:
            flags.append("dimensions")
        if out["blank_frames"]:
            flags.append("blank_frames")
        if out["frozen_frames"] > out["source_frozen_frames"] + 5:
            flags.append("frozen_frames")
        if cands:
            flags.append("discontinuity_candidates")
        out["auto_flags"] = ",".join(flags)
    except Exception as e:
        out["metrics_error"] = f"{type(e).__name__}: {e}"
    out["metrics_seconds"] = round(time.time() - t0, 3)
    return out
