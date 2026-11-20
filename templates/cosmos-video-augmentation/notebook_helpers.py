"""Plotting, video frames and data loading shared by the notebooks."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from offroad_ray.config import DEFAULT_ROOT

ROOT = Path(DEFAULT_ROOT)
MEDIA = ROOT / "media"
RESULTS = ROOT / "results"
LABELS = {"golden_hour_glare": "golden hour", "night_rain": "night rain", "blue_hour_dusk": "blue hour",
          "blizzard_whiteout": "blizzard", "dust_brownout": "dust brownout", "wildfire_smoke": "wildfire smoke",
          "drought_summer": "drought summer", "autumn": "autumn", "night": "night", "rain": "rain", "src": "real camera"}


def check_setup(use_gpus: bool) -> None:
    """Fail early with a clear message if the data bundle (or, with GPUs, the weights) is missing."""
    if not (MEDIA / "index.json").exists():
        raise SystemExit(f"Data not found in {ROOT}. Run: python scripts/setup_data.py")
    n = sum(1 for _ in MEDIA.rglob("*.mp4"))
    print(f"COSMOS_SHARED = {ROOT}  ({n} videos in the bundle)")
    if use_gpus and not (ROOT / "hf_home").exists():
        raise SystemExit("USE_GPUS = True needs the model weights: python scripts/setup_data.py --with-weights")
    print("Mode:", "live generation on GPUs" if use_gpus else "stored results on CPU (set USE_GPUS = True to generate)")


def scenes() -> dict:
    return json.loads((MEDIA / "index.json").read_text())["scenes"]


def read_jsonl(path: str | Path) -> pd.DataFrame:
    return pd.DataFrame([json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()])


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def frames_at(path: str | Path, times=(0.4, 2.9, 5.4), width: int = 512) -> list[np.ndarray]:
    import cv2
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 16.0
    out = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, f = cap.read()
        if ok:
            h = int(f.shape[0] * width / f.shape[1])
            out.append(cv2.cvtColor(cv2.resize(f, (width, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
    cap.release()
    return out


def show_conditions(scene_id: str, conditions: list[str], t: float = 2.9, cols: int = 3, width: int = 512):
    """One frame (same timestamp) from the real clip and each generated condition."""
    import matplotlib.pyplot as plt
    items = ["src", *conditions]
    rows = (len(items) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.6, rows * 2.6))
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, c in zip(np.ravel(axes), items):
        ax.imshow(frames_at(MEDIA / scene_id / f"{c}.mp4", (t,), width)[0])
        ax.set_title(LABELS.get(c, c), fontsize=11)
    fig.suptitle(scenes()[scene_id]["label"], fontsize=13, y=1.0)
    fig.tight_layout()
    plt.show()


def show_strip(paths: list[str | Path], labels: list[str], times=(0.4, 2.9, 5.4), width: int = 384):
    """Each row is a video, each column a timestamp: shows that motion is kept over the clip."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(paths), len(times), figsize=(len(times) * 3.6, len(paths) * 2.0), squeeze=False)
    for r, (p, lab) in enumerate(zip(paths, labels)):
        for c, f in enumerate(frames_at(p, times, width)):
            axes[r][c].imshow(f)
            axes[r][c].axis("off")
            if r == 0:
                axes[r][c].set_title(f"t = {times[c]:.1f} s", fontsize=10)
        axes[r][0].text(-0.04, 0.5, lab, transform=axes[r][0].transAxes, ha="right", va="center", fontsize=10)
    fig.tight_layout()
    plt.show()


def video(path: str | Path, width: int = 640):
    """Embed a short MP4 in the notebook output."""
    from IPython.display import Video
    return Video(str(path), embed=True, width=width, html_attributes="autoplay muted loop playsinline")


def stage_table(manifest: pd.DataFrame) -> pd.DataFrame:
    st = pd.DataFrame(list(manifest["stage_times"]))
    order = [("generate_chunk", "Denoising, 4 steps"), ("video_guardrail", "Video guardrail"),
             ("postprocessing", "Decode and postprocessing"), ("preprocessing", "Preprocessing"),
             ("text_guardrail", "Text guardrail")]
    med_total = float(manifest["child_seconds"].median())
    rows = [{"stage": name, "median_s": round(float(st[k].median()), 2),
             "share": f"{100 * st[k].median() / med_total:.1f}%"} for k, name in order if k in st]
    rows.append({"stage": "Total per request", "median_s": round(med_total, 2), "share": "100%"})
    return pd.DataFrame(rows)


def scaling_frame() -> pd.DataFrame:
    """1, 2, 4 and 8 actors on the same 24 requests (guardrails on)."""
    s = read_json(RESULTS / "benchmark" / "scaling.json")
    df = pd.DataFrame(s["scaling"])
    med = {p["num_actors"]: p["per_request_child_s_median"] for p in s["points"]}
    df["median_request_s"] = df["num_actors"].map(med)
    return df[["num_actors", "wall_s", "speedup_gen", "efficiency_gen", "speedup_e2e", "cold_start_s", "median_request_s"]]


def plot_scaling(df: pd.DataFrame):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 3.4))
    n = df["num_actors"]
    ax.plot(n, n, "--", color="#888", label="ideal")
    ax.plot(n, df["speedup_gen"], "o-", color="#2b7bd6", label="generation window")
    ax.plot(n, df["speedup_e2e"], "o-", color="#e8682f", label="end to end (incl. model load)")
    ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
    ax.set_xticks(list(n)); ax.set_xticklabels(list(n)); ax.set_yticks(list(n)); ax.set_yticklabels(list(n))
    ax.set_xlabel("actors (one A100 each)"); ax.set_ylabel("speedup over 1 actor")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.show()


def plot_timeline(points: str = "a8"):
    """Each actor loads the model once, then generates back to back."""
    import matplotlib.pyplot as plt
    ns = RESULTS / "benchmark" / points
    m = read_jsonl(ns / "manifest.jsonl")
    t0 = read_json(ns / "pool_summary.json")["t0"]
    fig, ax = plt.subplots(figsize=(10, 0.45 * m["actor_index"].nunique() + 1.2))
    for _, r in m.iterrows():
        ax.barh(r["actor_index"], r["t_end"] - r["t_dispatch"], left=r["t_dispatch"] - t0, color="#2b7bd6", edgecolor="white")
    for a, g in m.groupby("actor_index"):
        ax.barh(a, g["t_dispatch"].min() - t0, left=0, color="#c4c3b8", edgecolor="white")
    ax.set_yticks(sorted(m["actor_index"].unique()))
    ax.set_yticklabels([f"actor {i}" for i in sorted(m["actor_index"].unique())])
    ax.invert_yaxis()
    ax.set_xlabel("seconds since the pool started (grey: model load, blue: one request each)")
    ax.set_title(f"{m['actor_index'].nunique()} actors, {len(m)} requests")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    plt.tight_layout()
    plt.show()
