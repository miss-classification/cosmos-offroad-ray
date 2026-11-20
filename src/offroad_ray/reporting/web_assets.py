"""Web-sized showcase assets for the blog, slides and the static report.

    .venv/bin/python -m offroad_ray.reporting.web_assets [--skip-reencode] [--skip-heroes]

Reads only guardrails-ON namespaces (pilot, main, conditions_v3a, conditions_v3b) and the
automated VLM judge tables; never smoke, never anything named serve_demo. Originals in
results/showcase/<ns>/ are left untouched. Writes:

  results/showcase/web/<ns>/*.mp4|*.jpg   re-encoded copies of the existing showcase MP4s
  results/showcase/web/hero/*             condition grid, before/after x3, 9-scene grid, failure
  results/showcase/web/assets.json        captions, selection rationale, file sizes

Encoding: H.264, yuv420p, CRF 26, width at most 960, +faststart. Hero items also get an
animated GIF and/or WebP, each kept under 8 MB by stepping the width or quality down.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import textwrap
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[3]
SHOWCASE = PROJECT / "results" / "showcase"
WEB = SHOWCASE / "web"
RUNS = Path("/mnt/cluster_storage/cosmos-offroad/runs")
GUARDED_NS = ["pilot", "main", "conditions_v3a", "conditions_v3b"]
FORBIDDEN = ("smoke", "serve_demo")

MP4_LIMIT = 6e6
ANIM_LIMIT = 8e6
CRF = 26
MAX_W = 960
FPS = 16.0
N_FRAMES = 93
SEED = 101

REAL = (45, 45, 45)
SYNTH = (25, 110, 60)
CTRL = (50, 80, 150)
FAIL = (150, 40, 40)
ATTRIB = ("Source: TartanDrive 2.0 (AirLab CMU, CC BY 4.0), 10 Hz resampled to 16 Hz. "
          "Synthetic: Cosmos Transfer 2.5 edge/distilled, guardrails on.")

COND_NAMES = {
    "fog": "Fog", "rain": "Rain", "night": "Night", "snow": "Snow",
    "night_rain": "Night rain", "dust_brownout": "Dust brownout", "wildfire_smoke": "Wildfire smoke",
    "golden_hour_glare": "Golden hour glare", "blizzard_whiteout": "Blizzard whiteout",
    "autumn": "Autumn", "drought_summer": "Drought summer", "flooded_mud": "Flooded mud",
    "blue_hour_dusk": "Blue hour dusk", "frost_dawn": "Frost dawn",
}
VIEWED_FAILURE = "63ddf6299783bc996775"
SCORE_COLS = ["photorealism", "temporal_stability", "physical_plausibility", "condition_intensity"]


# ---------------------------------------------------------------- helpers

def cond_name(c: str) -> str:
    return COND_NAMES.get(c, c.replace("_", " ").capitalize())


def scene_name(run_or_clip_id: str) -> str:
    """'2023-11-02-16-00-43_down_meadows' -> 'down meadows'; 'sam_loop_2023-09-27-...' -> 'sam loop'."""
    s = run_or_clip_id.split("__")[0]
    s = re.sub(r"\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}", "", s).strip("_")
    return s.replace("_", " ") or run_or_clip_id[:20]


def _check_ns(ns: str) -> None:
    if any(f in ns for f in FORBIDDEN):
        raise SystemExit(f"refusing namespace {ns}: guardrails-OFF or demo namespaces are never shown")
    summ = json.loads((RUNS / ns / "main" / "pool_summary.json").read_text())
    if not summ.get("settings", {}).get("guardrails", False):
        raise SystemExit(f"refusing namespace {ns}: guardrails OFF")


def load_records() -> list[dict]:
    recs = []
    for ns in GUARDED_NS:
        _check_ns(ns)
        for line in (RUNS / ns / "main" / "manifest.jsonl").read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("status") == "generated_unreviewed":
                    r["ns"] = ns
                    recs.append(r)
    return recs


def load_judgments():
    import pandas as pd
    parts = [pd.read_parquet(RUNS / "judge" / j / "judgments.parquet") for j in ("main", "v3")]
    d = pd.concat(parts, ignore_index=True)
    d = d[(d["ok"] == True) & (d["kind"] == "synthetic")].copy()  # noqa: E712
    d["score4"] = d[SCORE_COLS].mean(axis=1)
    d["score3"] = d[SCORE_COLS[:3]].mean(axis=1)
    return d


def _font(size: int, bold: bool = False):
    from PIL import ImageFont
    p = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    try:
        return ImageFont.truetype(p, size)
    except OSError:
        return ImageFont.load_default(size=size)


def text_bar(width: int, height: int, lines: list[str], bg, fg=(255, 255, 255), size: int = 14,
             bold: bool = False, align: str = "left") -> np.ndarray:
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (width, height), bg)
    d = ImageDraw.Draw(im)
    f = _font(size, bold)
    lh = int(size * 1.3)
    y = (height - lh * len(lines)) / 2 + 1
    for ln in lines:
        tw = d.textlength(ln, font=f)
        x = 8 if align == "left" else max(4, (width - tw) / 2)
        d.text((x, y), ln, fill=fg, font=f)
        y += lh
    return np.asarray(im)


def load_panel(path: str, w: int, h: int, n: int = N_FRAMES) -> np.ndarray:
    import cv2
    from offroad_ray.video_io import decode_rgb
    fr = decode_rgb(path)
    if len(fr) < n:
        fr = np.concatenate([fr, np.repeat(fr[-1:], n - len(fr), axis=0)])
    fr = fr[:n]
    return np.stack([cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA) for f in fr])


def compose(panels: list[tuple[np.ndarray, str, tuple]], cols: int, *, title: str | None,
            label_size: int = 14, gap: int = 3, footer: str = ATTRIB,
            caption: list[str] | None = None) -> np.ndarray:
    """panels: (frames[T,H,W,3], label, label color). Label bar above each panel, all frames in sync."""
    t = min(p[0].shape[0] for p in panels)
    ph, pw = panels[0][0].shape[1:3]
    rows = int(np.ceil(len(panels) / cols))
    label_h = int(label_size * 1.6)
    cell_h = ph + label_h
    W = cols * pw + (cols - 1) * gap
    title_h = 34 if title else 0
    cap_h = (int(15 * 1.3) * len(caption) + 14) if caption else 0
    foot_h = 22
    H = title_h + rows * cell_h + (rows - 1) * gap + cap_h + foot_h
    W += W % 2
    H += H % 2
    canvas = np.full((t, H, W, 3), 255, np.uint8)
    if title:
        canvas[:, :title_h] = text_bar(W, title_h, [title], (20, 20, 20), size=17, bold=True)
    for k, (fr, label, color) in enumerate(panels):
        r, c = divmod(k, cols)
        y = title_h + r * (cell_h + gap)
        x = c * (pw + gap)
        canvas[:, y:y + label_h, x:x + pw] = text_bar(pw, label_h, [label], color, size=label_size, bold=True)
        canvas[:, y + label_h:y + cell_h, x:x + pw] = fr[:t]
    y = title_h + rows * cell_h + (rows - 1) * gap
    if caption:
        canvas[:, y:y + cap_h] = text_bar(W, cap_h, caption, (250, 245, 235), fg=(30, 30, 30), size=15)
        y += cap_h
    canvas[:, H - foot_h:H] = text_bar(W, foot_h, [footer], (238, 238, 238), fg=(50, 50, 50), size=11)
    return canvas


def encode_mp4(frames: np.ndarray, path: Path, crf: int = CRF) -> Path:
    from offroad_ray.video_io import encode_rgb
    path.parent.mkdir(parents=True, exist_ok=True)
    encode_rgb(frames, str(path), FPS, crf=crf)
    if path.stat().st_size > MP4_LIMIT and crf < 34:
        return encode_mp4(frames, path, crf + 2)
    return path


def _ff(args: list[str]) -> None:
    r = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr[-1500:])


def reencode(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    for crf in (CRF, 28, 30, 32):
        _ff(["-i", str(src), "-vf", f"scale='min({MAX_W},iw)':-2:flags=lanczos", "-c:v", "libx264", "-preset", "slow",
             "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(dst)])
        if dst.stat().st_size <= MP4_LIMIT:
            break
    return dst


def still(mp4: Path, jpg: Path, frame: int = 46) -> Path:
    _ff(["-i", str(mp4), "-vf", f"select=eq(n\\,{frame})", "-vframes", "1", "-q:v", "3", str(jpg)])
    return jpg


def animate(mp4: Path, stem: Path) -> dict:
    """GIF and animated WebP under ANIM_LIMIT; returns {'gif': path|None, 'webp': path|None}."""
    out = {"gif": None, "webp": None}
    gif = stem.with_suffix(".gif")
    for w, fps in ((800, 10), (640, 10), (560, 8), (480, 8)):
        vf = (f"fps={fps},scale={w}:-2:flags=lanczos,split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];"
              f"[b][p]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle")
        _ff(["-i", str(mp4), "-vf", vf, "-loop", "0", str(gif)])
        if gif.stat().st_size <= ANIM_LIMIT:
            out["gif"] = gif
            break
    else:
        gif.unlink(missing_ok=True)
    webp = stem.with_suffix(".webp")
    for q in (70, 55, 40):
        _ff(["-i", str(mp4), "-vf", f"fps=12,scale='min({MAX_W},iw)':-2:flags=lanczos", "-c:v", "libwebp_anim",
             "-quality", str(q), "-loop", "0", "-an", str(webp)])
        if webp.stat().st_size <= ANIM_LIMIT:
            out["webp"] = webp
            break
    else:
        webp.unlink(missing_ok=True)
    return out


# ---------------------------------------------------------------- task 1: re-encode existing showcase

def _reencode_one(src: str, dst: str) -> tuple[str, int, str, int]:
    d = reencode(Path(src), Path(dst))
    j = still(d, d.with_suffix(".jpg"))
    return str(d), d.stat().st_size, str(j), j.stat().st_size


def reencode_showcase(workers: int = 8) -> list[dict]:
    jobs = []
    for ns in ("pilot", "conditions_v3a", "conditions_v3b"):
        for mp4 in sorted((SHOWCASE / ns).glob("*.mp4")):
            jobs.append((str(mp4), str(WEB / ns / mp4.name)))
    out = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for (src, _), res in zip(jobs, ex.map(_reencode_one, *zip(*jobs))):
            out.append({"source": str(Path(src).relative_to(PROJECT)), "mp4": str(Path(res[0]).relative_to(PROJECT)),
                        "mp4_bytes": res[1], "jpg": str(Path(res[2]).relative_to(PROJECT)), "jpg_bytes": res[3]})
    return out


# ---------------------------------------------------------------- task 2: hero assets

def pick_before_after(j, recs_by_rid: dict, k: int = 3) -> list[dict]:
    """Greedy: highest judge score4 (tie: score3, then condition-level mean), distinct scenes and conditions."""
    cond_mean = j.groupby("requested_condition")["score4"].mean()
    j = j[j["request_id"].isin(recs_by_rid)].copy()
    j["cond_mean"] = j["requested_condition"].map(cond_mean)
    j = j.sort_values(["score4", "score3", "cond_mean", "request_id"], ascending=[False, False, False, True])
    picked, scenes, conds = [], set(), set()
    for _, r in j.iterrows():
        sc = scene_name(r["run_id"])
        if sc in scenes or r["requested_condition"] in conds:
            continue
        picked.append({"request_id": r["request_id"], "scene": sc, "condition": r["requested_condition"],
                       "seed": int(r["seed"]), "set": r["set"],
                       "scores": {c: float(r[c]) for c in SCORE_COLS},
                       "judge_observed_condition": r["observed_condition"]})
        scenes.add(sc)
        conds.add(r["requested_condition"])
        if len(picked) == k:
            break
    return picked


def pick_scene_condition(j) -> str:
    g = j[j["seed"] == SEED].groupby("requested_condition")[["score4", "score3"]].mean()
    return g.sort_values(["score4", "score3"], ascending=False).index[0]


def pick_failure(j, quality_flags: list[dict]) -> dict:
    w = j.sort_values(["score3", "score4", "request_id"]).iloc[0]
    return {"request_id": w["request_id"], "scene": scene_name(w["run_id"]), "condition": w["requested_condition"],
            "seed": int(w["seed"]), "set": w["set"], "scores": {c: float(w[c]) for c in SCORE_COLS},
            "judge_summary": w["summary"], "judge_observed_condition": w["observed_condition"],
            "auto_flags_main": quality_flags}


def hero_condition_grid(recs: list[dict], clip_id: str, out: Path) -> dict:
    pw, ph = 236, 126
    sel = [r for r in recs if r["clip_id"] == clip_id and r["seed"] == SEED]
    order = ["fog", "rain", "snow", "night", "night_rain", "blue_hour_dusk", "golden_hour_glare", "autumn",
             "drought_summer", "dust_brownout", "wildfire_smoke", "blizzard_whiteout", "frost_dawn", "flooded_mud"]
    by_c = {r["condition"]: r for r in sel}
    conds = [c for c in order if c in by_c] + [c for c in by_c if c not in order]
    src = sel[0]
    panels = [(load_panel(src["clip_path"], pw, ph), "Real source", REAL),
              (load_panel(src["control_path"], pw, ph), "Edge control (input)", CTRL)]
    panels += [(load_panel(by_c[c]["output_path"], pw, ph), cond_name(c), SYNTH) for c in conds]
    sc = scene_name(clip_id)
    title = f"{sc}: real source, edge control and {len(conds)} synthetic conditions"
    frames = compose(panels, 4, title=title, label_size=12, gap=2)
    mp4 = encode_mp4(frames, out)
    # hi-res still for slides (panel 480 wide)
    big = compose([(load_panel(p, 480, 256)[46:47].copy(), lab, col) for p, lab, col in
                   [(src["clip_path"], "Real source", REAL), (src["control_path"], "Edge control (model input)", CTRL)]
                   + [(by_c[c]["output_path"], cond_name(c), SYNTH) for c in conds]],
                  4, title=title, label_size=20, gap=4)
    from PIL import Image
    Image.fromarray(big[0]).save(out.with_name(out.stem + "_still_hires.jpg"), quality=88)
    return {"mp4": mp4, "scene": sc, "clip_id": clip_id, "conditions": conds,
            "namespaces": sorted({by_c[c]["ns"] for c in conds})}


def hero_before_after(rec: dict, src_rec: dict, out: Path, scores: dict) -> Path:
    pw, ph = 476, 253
    sc = scene_name(rec["clip_id"])
    panels = [(load_panel(src_rec["clip_path"], pw, ph), "Before: real footage", REAL),
              (load_panel(rec["output_path"], pw, ph), f"After: synthetic {cond_name(rec['condition']).lower()}", SYNTH)]
    short = {"photorealism": "realism", "temporal_stability": "stability",
             "physical_plausibility": "plausibility", "condition_intensity": "intensity"}
    sc_txt = ", ".join(f"{short.get(k, k)} {v:.0f}" for k, v in scores.items())
    title = f"{sc}: {cond_name(rec['condition']).lower()}, seed {rec['seed']}"
    cap = [f"Automated blind judge (Cosmos-Reason1-7B, not human), 1 to 5: {sc_txt}."]
    return encode_mp4(compose(panels, 2, title=title, label_size=15, gap=4, caption=cap), out)


def hero_multi_scene(recs: list[dict], condition: str, out: Path) -> dict:
    pw, ph = 316, 168
    sel = sorted([r for r in recs if r["condition"] == condition and r["seed"] == SEED], key=lambda r: r["clip_id"])
    by_scene = {}
    for r in sel:
        by_scene.setdefault(scene_name(r["clip_id"]), r)
    items = list(by_scene.items())[:9]
    panels = [(load_panel(r["output_path"], pw, ph), sc, SYNTH) for sc, r in items]
    title = f"Synthetic {cond_name(condition).lower()} across {len(items)} scenes (seed {SEED})"
    return {"mp4": encode_mp4(compose(panels, 3, title=title, label_size=13, gap=3), out),
            "scenes": [sc for sc, _ in items], "condition": condition}


def hero_failure(rec: dict, src_rec: dict, out: Path, caption: list[str]) -> Path:
    pw, ph = 476, 253
    panels = [(load_panel(src_rec["clip_path"], pw, ph), "Real source", REAL),
              (load_panel(rec["output_path"], pw, ph), f"Synthetic {cond_name(rec['condition']).lower()}: failure example", FAIL)]
    title = f"Failure example: {scene_name(rec['clip_id'])}, {cond_name(rec['condition']).lower()}, seed {rec['seed']}"
    return encode_mp4(compose(panels, 2, title=title, label_size=15, gap=4, caption=caption), out)


def _rel(p) -> str | None:
    return str(Path(p).relative_to(PROJECT)) if p else None


def _size(p) -> int | None:
    return Path(p).stat().st_size if p else None


def build_heroes() -> dict:
    import pandas as pd
    recs = load_records()
    by_rid = {r["request_id"]: r for r in recs}
    src_by_clip = {r["clip_id"]: r for r in recs}
    j = load_judgments()
    hero = WEB / "hero"
    hero.mkdir(parents=True, exist_ok=True)
    meta: dict = {}

    # (a) condition grid: the t3.000 clip with the highest mean judge score4 over all 14 conditions at seed 101
    js = j[(j["seed"] == SEED) & j["clip_id"].str.contains("__t3.000__")]
    per_clip = js.groupby("clip_id")["score4"].agg(["mean", "size"]).sort_values("mean", ascending=False)
    grid_clip = per_clip.index[0]
    g = hero_condition_grid(recs, grid_clip, hero / "condition_grid.mp4")
    meta["condition_grid"] = {**{k: v for k, v in g.items() if k != "mp4"}, "mp4": g["mp4"],
                              "why": (f"Clip with the highest mean automated judge score (mean of 4 criteria) across "
                                      f"its seed-{SEED} conditions: {per_clip.iloc[0]['mean']:.2f} over "
                                      f"{int(per_clip.iloc[0]['size'])} judged outputs.")}

    # (b) before/after x3
    ba = pick_before_after(j, by_rid)
    meta["before_after"] = []
    for i, p in enumerate(ba, 1):
        rec = by_rid[p["request_id"]]
        slug = f"before_after_{i}_{p['scene'].replace(' ', '_')}_{p['condition']}"
        mp4 = hero_before_after(rec, src_by_clip[rec["clip_id"]], hero / f"{slug}.mp4", p["scores"])
        meta["before_after"].append({**p, "mp4": mp4})

    # (c) 9 scenes under one condition
    cond = pick_scene_condition(j)
    ms = hero_multi_scene(recs, cond, hero / f"multi_scene_{cond}.mp4")
    cm = j[(j["seed"] == SEED) & (j["requested_condition"] == cond)]["score4"].mean()
    meta["multi_scene"] = {**ms, "why": f"Condition with the highest mean automated judge score at seed {SEED} ({cm:.2f})."}

    # (d) failure
    q = pd.read_parquet(RUNS / "main" / "main" / "quality" / "quality_table.parquet")
    flags = q[q["auto_flags"] != ""][["request_id", "clip_id", "condition", "seed", "auto_flags",
                                       "frozen_frames", "source_frozen_frames"]].to_dict("records")
    f = pick_failure(j, flags)
    frec = by_rid[f["request_id"]]
    sc = f["scores"]
    caption = [
        f"Lowest automated judge score of {len(j)} judged outputs: temporal stability {sc['temporal_stability']:.0f} of 5,",
        f"physical plausibility {sc['physical_plausibility']:.0f} of 5, photorealism {sc['photorealism']:.0f} of 5.",
    ]
    if f["request_id"] == VIEWED_FAILURE:  # description written after looking at this clip frame by frame
        caption += ["The low sun disc is visible early and then dissolves into the haze, and smoke puffs appear and",
                    "vanish between frames with no source in the scene. Trail layout and camera motion still follow",
                    "the real clip, because the edge control fixes them."]
    f["mp4"] = hero_failure(frec, src_by_clip[frec["clip_id"]], hero / f"failure_{f['scene'].replace(' ', '_')}_{f['condition']}.mp4", caption)
    f["caption"] = " ".join(caption)
    meta["failure"] = f

    # stills and animations
    def finish(entry: dict, anim: bool = True) -> None:
        mp4 = Path(entry["mp4"])
        entry["jpg"] = still(mp4, mp4.with_suffix(".jpg"))
        if anim:
            a = animate(mp4, mp4.with_suffix(""))
            entry["gif"], entry["webp"] = a["gif"], a["webp"]
        for k in ("mp4", "jpg", "gif", "webp"):
            if k in entry:
                entry[k + "_bytes"] = _size(entry[k])
                entry[k] = _rel(entry[k])
    finish(meta["condition_grid"])
    hi = hero / "condition_grid_still_hires.jpg"
    meta["condition_grid"]["jpg_hires"] = _rel(hi)
    meta["condition_grid"]["jpg_hires_bytes"] = _size(hi)
    for e in meta["before_after"]:
        finish(e)
    finish(meta["multi_scene"])
    finish(meta["failure"])
    return meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-reencode", action="store_true")
    ap.add_argument("--skip-heroes", action="store_true")
    a = ap.parse_args(argv)
    WEB.mkdir(parents=True, exist_ok=True)
    manifest_path = WEB / "assets.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if not a.skip_reencode:
        manifest["reencoded"] = reencode_showcase()
    if not a.skip_heroes:
        manifest["hero"] = build_heroes()
    manifest["encoding"] = {"codec": "H.264 libx264", "pix_fmt": "yuv420p", "crf": CRF, "max_width": MAX_W,
                            "faststart": True, "mp4_limit_bytes": MP4_LIMIT, "anim_limit_bytes": ANIM_LIMIT}
    manifest["note"] = ("Guardrails-ON namespaces only (pilot, main, conditions_v3a, conditions_v3b). Judge scores are "
                        "automated and blind (Cosmos-Reason1-7B), not human review.")
    manifest_path.write_text(json.dumps(manifest, indent=1, default=str))
    for p in sorted(WEB.rglob("*")):
        if p.is_file():
            print(f"{p.stat().st_size / 1e6:6.2f} MB  {p.relative_to(PROJECT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
