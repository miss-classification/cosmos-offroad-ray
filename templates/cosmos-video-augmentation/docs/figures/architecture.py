"""Draw docs/images/architecture.png (python docs/figures/architecture.py)."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "images" / "architecture.png"
C = {"data": "#e8f1fb", "data_e": "#2b7bd6", "core": "#fdf1e3", "core_e": "#e08a1e", "serve": "#eaf6ee",
     "serve_e": "#2e9a55", "store": "#f2f2f2", "store_e": "#8a8a8a", "text": "#1b1f24", "muted": "#5d6673"}

fig, ax = plt.subplots(figsize=(13, 6.6), dpi=150)
ax.set_xlim(0, 130); ax.set_ylim(0, 66); ax.axis("off")


def box(x, y, w, h, title, sub, fc, ec, fs=10.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.6", fc=fc, ec=ec, lw=1.6))
    ax.text(x + w / 2, y + h - 2.6, title, ha="center", va="top", fontsize=fs, weight="bold", color=C["text"])
    ax.text(x + w / 2, y + h - 6.3, sub, ha="center", va="top", fontsize=8.6, color=C["muted"], linespacing=1.35)


def arrow(x1, y1, x2, y2, label=None, dy=1.4):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13, lw=1.4, color="#444"))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + dy, label, ha="center", fontsize=8, color=C["muted"])


def band(y, h, label, color):
    ax.add_patch(FancyBboxPatch((1, y), 128, h, boxstyle="round,pad=0,rounding_size=2", fc="none", ec=color, lw=1, ls="--"))
    ax.text(2.5, y + h - 1.6, label, fontsize=9.5, weight="bold", color=color, va="top")


# batch band
band(33, 31, "Batch generation", C["core_e"])
box(4, 41, 20, 15, "TartanDrive 2.0", "real ATV camera\nframes, 10 Hz\n(public, CC BY 4.0)", C["store"], C["store_e"])
box(30, 41, 23, 15, "Ray Data", "decode, resample to 16 fps,\nencode, verify,\nCanny edge control\n(CPU, 1 clip per block)", C["data"], C["data_e"])
box(59, 41, 25, 15, "Ray Core scheduler", "ray.wait dispatch loop\n1 request in flight per GPU\nretries, validation, resume", C["core"], C["core_e"])
for i in range(4):
    box(88 + i * 1.2, 41.5 - i * 1.2, 21, 13.5, "" if i < 3 else "GPU actors x 8", "" if i < 3 else
        "CosmosActor + Cosmos\nTransfer 2.5 subprocess\nmodel loaded once", "white" if i < 3 else C["core"], C["core_e"])
box(115.5, 42.5, 11.5, 11, "Outputs", "MP4 +\nrecords", C["store"], C["store_e"], fs=9.5)
arrow(24.5, 48.5, 29.5, 48.5)
arrow(53.5, 48.5, 58.5, 48.5)
ax.text(56, 45.2, "clip\ncatalog", ha="center", fontsize=8, color=C["muted"])
arrow(84.5, 48.5, 87.5, 46.5)
arrow(112.6, 46, 115, 46)
box(57, 34.6, 29, 4.4, "", "", C["data"], C["data_e"])
ax.text(71.5, 36.8, "Ray Data quality pass (CPU)", ha="center", va="center", fontsize=9, weight="bold", color=C["text"])
ax.plot([121.2, 121.2], [42.2, 36.8], color="#444", lw=1.4)
arrow(121.2, 36.8, 86.5, 36.8)

# serve band
band(2, 28, "Interactive application", C["serve_e"])
box(4, 9, 20, 14, "Browser", "pick a drive,\npick a condition,\nwatch 8 GPUs", C["store"], C["store_e"])
box(32, 9, 26, 14, "PaintAPI (Ray Serve)", "FastAPI on CPU\njob API: POST, then poll\npage + stored results", C["serve"], C["serve_e"])
for i in range(4):
    box(68 + i * 1.2, 9.5 - i * 1.2, 26, 12.5, "" if i < 3 else "Painter replicas x 8", "" if i < 3 else
        "1 GPU each, 1 request each\n@serve.multiplexed by drive\n(cached video encodes)", "white" if i < 3 else C["serve"], C["serve_e"])
box(104, 8, 23, 15, "Replay mode", "ReplayPainter on CPU\nstored guardrails-on\noutputs, no GPU needed", "white", C["serve_e"])
arrow(24.5, 16, 31.5, 16, "HTTP")
arrow(58.5, 16, 67.5, 14)
ax.text(63, 8.6, "routed by\ndrive ID", ha="center", fontsize=8, color=C["muted"])
ax.text(100, 16, "or", ha="center", fontsize=9, color=C["muted"])
arrow(97, 37.3, 86, 23.6)
ax.text(93.2, 31.0, "same actor class", fontsize=8, color=C["muted"], va="center")

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, bbox_inches="tight", facecolor="white")
print(OUT)
