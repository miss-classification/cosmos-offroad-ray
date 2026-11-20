"""Synchronized, labeled comparison videos and GIFs.

All panels show the same source interval frame for frame (93 frames, 16 fps model
cadence). Generated panels are marked SYNTHETIC; every output carries the
TartanDrive 2.0 attribution and notes the 10 Hz to 16 Hz resampling.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

ATTRIB = "Source: TartanDrive 2.0 (AirLab CMU, CC BY 4.0), resampled 10 Hz to 16 Hz by frame repetition"
GEN_COLOR = (30, 132, 73)       # synthetic label (green, white text about 4.8:1 contrast)
SRC_COLOR = (40, 40, 40)
CTRL_COLOR = (60, 90, 160)


def _font(size: int):
    from PIL import ImageFont
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default(size=size)


def _bar(width: int, height: int, text: str, color, font_size: int, align: str = "left") -> np.ndarray:
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (width, height), color)
    d = ImageDraw.Draw(im)
    f = _font(font_size)
    tw = d.textlength(text, font=f)
    x = 8 if align == "left" else max(4, (width - tw) / 2)
    d.text((x, (height - font_size) / 2 - 1), text, fill=(255, 255, 255), font=f)
    return np.asarray(im)


def _resize(frames: np.ndarray, w: int, h: int) -> np.ndarray:
    import cv2
    return np.stack([cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA) for f in frames])


def load_panel(path: str, w: int, h: int, n: int | None = None) -> np.ndarray:
    from offroad_ray.video_io import decode_rgb
    fr = decode_rgb(path)
    if n is not None:
        if len(fr) < n:
            fr = np.concatenate([fr, np.repeat(fr[-1:], n - len(fr), axis=0)])
        fr = fr[:n]
    return _resize(fr, w, h)


def compose(panels: list[tuple[np.ndarray, str, tuple]], cols: int, *, title: str | None = None,
            label_h: int = 26, gap: int = 4, footer: str = ATTRIB) -> np.ndarray:
    """panels: list of (frames[T,H,W,3], label, color). Returns composed (T,H,W,3)."""
    t = min(p[0].shape[0] for p in panels)
    ph, pw = panels[0][0].shape[1:3]
    rows = int(np.ceil(len(panels) / cols))
    cell_h = ph + label_h
    title_h = 34 if title else 0
    foot_h = 22
    H = title_h + rows * cell_h + (rows - 1) * gap + foot_h
    W = cols * pw + (cols - 1) * gap
    H += H % 2
    W += W % 2
    canvas = np.full((t, H, W, 3), 255, np.uint8)
    if title:
        canvas[:, :title_h] = _bar(W, title_h, title, (20, 20, 20), 18)
    for k, (fr, label, color) in enumerate(panels):
        r, c = divmod(k, cols)
        y = title_h + r * (cell_h + gap)
        x = c * (pw + gap)
        canvas[:, y:y + label_h, x:x + pw] = _bar(pw, label_h, label, color, 15)
        canvas[:, y + label_h:y + cell_h, x:x + pw] = fr[:t]
    canvas[:, H - foot_h:H] = _bar(W, foot_h, footer, (245, 245, 245), 11)
    # footer text is white on light gray; redraw dark for legibility
    from PIL import Image, ImageDraw
    im = Image.fromarray(canvas[0, H - foot_h:H].copy())
    im.paste((245, 245, 245), (0, 0, W, foot_h))
    ImageDraw.Draw(im).text((8, 4), footer, fill=(60, 60, 60), font=_font(11))
    canvas[:, H - foot_h:H] = np.asarray(im)
    return canvas


def write_mp4(frames: np.ndarray, path: str, fps: float = 16.0, crf: int = 18) -> str:
    from offroad_ray.video_io import encode_rgb
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    encode_rgb(frames, path, fps, crf=crf)
    return path


def mp4_to_gif(mp4: str, gif: str, width: int = 960, fps: int = 10) -> str:
    Path(gif).parent.mkdir(parents=True, exist_ok=True)
    vf = f"fps={fps},scale={width}:-2:flags=lanczos,split[a][b];[a]palettegen=max_colors=192:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4"
    r = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", mp4, "-vf", vf, "-loop", "0", gif],
                       capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr[-1000:])
    return gif


def triptych(source: str, control: str, generated: str, out_mp4: str, *, condition: str, seed: int,
             w: int = 512, h: int = 272, title: str | None = None) -> str:
    """Source | edge control | generated, synchronized."""
    n = 93
    panels = [(load_panel(source, w, h, n), "SOURCE (real)", SRC_COLOR),
              (load_panel(control, w, h, n), "EDGE CONTROL (model input)", CTRL_COLOR),
              (load_panel(generated, w, h, n), f"SYNTHETIC: {condition}  seed {seed}", GEN_COLOR)]
    return write_mp4(compose(panels, 3, title=title), out_mp4)


def condition_grid(source: str, generated: dict[str, str], out_mp4: str, *, w: int = 512, h: int = 272,
                   title: str | None = None, seed: int | None = None) -> str:
    """Source plus one panel per condition (2x3 grid with source first, or 2x2 without)."""
    n = 93
    panels = [(load_panel(source, w, h, n), "SOURCE (real)", SRC_COLOR)]
    for cond, path in generated.items():
        panels.append((load_panel(path, w, h, n), f"SYNTHETIC: {cond}" + (f"  seed {seed}" if seed is not None else ""), GEN_COLOR))
    cols = 3 if len(panels) > 4 else 2
    return write_mp4(compose(panels, cols, title=title), out_mp4)


def before_after(source: str, generated: str, out_mp4: str, *, condition: str, w: int = 640, h: int = 340,
                 title: str | None = None) -> str:
    n = 93
    panels = [(load_panel(source, w, h, n), "BEFORE: real footage", SRC_COLOR),
              (load_panel(generated, w, h, n), f"AFTER: synthetic {condition}", GEN_COLOR)]
    return write_mp4(compose(panels, 2, title=title), out_mp4)


def wipe(source: str, generated: str, out_mp4: str, *, condition: str, w: int = 1024, h: int = 544) -> str:
    """Single-frame before/after with a moving vertical divider across the clip."""
    n = 93
    s = load_panel(source, w, h, n)
    g = load_panel(generated, w, h, n)
    out = s.copy()
    for i in range(n):
        x = int(w * (0.15 + 0.7 * (0.5 - 0.5 * np.cos(2 * np.pi * i / n))))
        out[i, :, x:] = g[i, :, x:]
        out[i, :, max(0, x - 2):x + 2] = 255
    canvas = compose([(out, f"REAL (left)  |  SYNTHETIC {condition} (right)", GEN_COLOR)], 1)
    return write_mp4(canvas, out_mp4)
