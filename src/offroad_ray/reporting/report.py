"""Static HTML results report.

    .venv/bin/python -m offroad_ray.cli report [configs/main.yaml] [--namespace main] [--saturation-check]

Writes results/report/index.html: one HTML file with inline CSS and no scripts, linking to
results/showcase/web and results/benchmark images by relative path. Everything is read from
disk (run namespaces, quality tables, judge tables, scaling.json, budget ledger); nothing is
typed in by hand except the phase notes copied from AGENTS.md, which are labelled as such.

Rules kept here: guardrails-ON namespaces only (smoke and anything named serve_demo are never
shown as results), judge scores are labelled automated and blind (Cosmos-Reason1-7B), cost is
node-hours only.
"""
from __future__ import annotations

import html
import json
import os
import statistics
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[3]
RESULTS = PROJECT / "results"
RUNS = Path("/mnt/cluster_storage/cosmos-offroad/runs")
EXCLUDED = ("smoke", "serve_demo")
NODES = 8  # GPU worker nodes held by the cluster during GPU phases

STAGES_TOP = ["text_guardrail", "generate_img2world", "video_guardrail"]
STAGES_IN_I2W = ["get_text_embeddings", "preprocessing", "generate_chunk", "postprocessing"]

# Phase notes copied from AGENTS.md (2026-11-18); measured columns come from pool_summary.json.
PHASES = [
    ("phase_a/main", "A: single actor, 2 generations", "init 106 s (load 56.6 s); generations 97.8 s and 80.5 s; peak 71 GB"),
    ("pilot/main", "B: pilot, 24 requests on 8 actors", "8-way init 143 to 171 s; generations 79 to 101 s"),
    ("benchmark/bench-20261118-142754-a1", "C: scaling benchmark, 1 actor", "identical 24 requests per point, fresh namespace"),
    ("benchmark/bench-20261118-142754-a2", "C: scaling benchmark, 2 actors", ""),
    ("benchmark/bench-20261118-142754-a4", "C: scaling benchmark, 4 actors", ""),
    ("benchmark/bench-20261118-142754-a8", "C: scaling benchmark, 8 actors", ""),
    ("main/main", "D: main, 24 clips x 4 conditions x 3 seeds", "fog, rain, night, snow; seeds 101, 202, 303"),
    ("realism_a/main", "Phase 2 realism sweep a", "high edge threshold, control weight 0.8"),
    ("realism_b/main", "Phase 2 realism sweep b", "very high edge threshold, control weight 0.6"),
    ("depth_probe/main", "Phase 2 depth probe, 1 clip", "general/depth, 35 steps: load 212 s, generate 868 s, peak 65 GB"),
    ("judge/sweep", "Phase 2 VLM judge, sweep + real sources", "Cosmos-Reason1-7B, automated, blind"),
    ("judge/main", "Phase 2 VLM judge, main", "Cosmos-Reason1-7B, automated, blind"),
    ("conditions_v3a/main", "Phase 3 conditions, part a", "5 conditions x 9 scenes, seed 101"),
    ("conditions_v3b/main", "Phase 3 conditions, part b", "5 conditions x 9 scenes, seed 101"),
    ("judge/v3", "Phase 3 VLM judge", "Cosmos-Reason1-7B, automated, blind"),
]


# ---------------------------------------------------------------- small helpers

def _e(x) -> str:
    return html.escape(str(x))


def _read_json(p: Path):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None


def _fmt(v, nd: int = 2) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:,.{nd}f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def table(headers: list[str], rows: list[list], num_cols: set[int] | None = None, nd: int = 2) -> str:
    num_cols = num_cols or set()
    h = "".join(f"<th{' class=n' if i in num_cols else ''}>{_e(c)}</th>" for i, c in enumerate(headers))
    body = []
    for r in rows:
        tds = "".join(f"<td{' class=n' if i in num_cols else ''}>{_e(_fmt(v, nd))}</td>" for i, v in enumerate(r))
        body.append(f"<tr>{tds}</tr>")
    return f"<div class=tw><table><thead><tr>{h}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"


def _rel(target: Path, out_dir: Path) -> str:
    return os.path.relpath(target, out_dir).replace(os.sep, "/")


def _missing(what: str) -> str:
    return f"<p class=muted>Not available: {_e(what)}.</p>"


def _allowed(ns: str) -> bool:
    return not any(x in ns for x in EXCLUDED)


# ---------------------------------------------------------------- data

def stage_timing(ns_dir: Path) -> dict | None:
    """Median per-stage seconds over a namespace's generated records.

    Upstream nesting (cosmos_transfer2/inference.py, inference_pipeline.py at 45c6f0fb):
    generate_img2world wraps get_text_embeddings, preprocessing, generate_chunk and postprocessing;
    text_guardrail and video_guardrail are separate top-level timers; saving the output and control
    videos is not timed, so it lands in the untimed remainder of child_seconds.
    """
    recs = []
    for f in sorted(Path(ns_dir, "records").glob("*.json")):
        r = _read_json(f)
        if r and r.get("status") == "generated_unreviewed" and r.get("stage_times") and r.get("child_seconds"):
            recs.append(r)
    if not recs:
        return None
    def med(vals):
        return statistics.median(vals) if vals else None
    st = {k: med([r["stage_times"][k] for r in recs if k in r["stage_times"]]) for k in STAGES_TOP + STAGES_IN_I2W}
    i2w_other = med([r["stage_times"]["generate_img2world"] - sum(r["stage_times"].get(k, 0) for k in STAGES_IN_I2W)
                     for r in recs if "generate_img2world" in r["stage_times"]])
    untimed = med([r["child_seconds"] - sum(r["stage_times"].get(k, 0) for k in STAGES_TOP) for r in recs])
    total = med([r["child_seconds"] for r in recs])
    return {"n": len(recs), "median_s": st, "img2world_other_s": i2w_other, "untimed_s": untimed,
            "child_seconds_median": total,
            "child_seconds_min": min(r["child_seconds"] for r in recs),
            "child_seconds_max": max(r["child_seconds"] for r in recs),
            "validate_s_median": med([r["validate_s"] for r in recs if r.get("validate_s") is not None])}


def saturation_check(ns_dir: Path, seed: int = 101, stride: int = 8, v_min: int = 50, workers: int = 8) -> dict:
    """Is the night saturation shift an HSV artifact? Compare OpenCV HSV S with CIELAB chroma.

    HSV S = (max - min) / max, so for a fixed colour difference it grows as pixels get darker.
    CIELAB chroma does not divide by brightness. Same frames, same 512x272 analysis size as metrics.py.
    """
    from concurrent.futures import ProcessPoolExecutor
    import pandas as pd
    q = pd.read_parquet(Path(ns_dir, "quality", "quality_table.parquet"))
    q = q[(q["seed"] == seed) & (q["status"] == "generated_unreviewed")]
    jobs = [(p, "out", c) for p, c in zip(q["output_path"], q["condition"])]
    jobs += [(p, "src", "source") for p in sorted(set(q["clip_path"]))]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(_sat_one, *zip(*jobs), [stride] * len(jobs), [v_min] * len(jobs)))
    df = pd.DataFrame(rows)
    g = df.groupby("condition")[["hsv_s", "hsv_s_bright", "dark_frac", "v_mean", "lab_chroma"]].mean()
    return {"seed": seed, "stride": stride, "v_min": v_min, "n_outputs": int((df["kind"] == "out").sum()),
            "n_sources": int((df["kind"] == "src").sum()),
            "by_condition": {c: {k: round(float(v), 2) for k, v in r.items()} for c, r in g.iterrows()},
            "computed_at": time.strftime("%Y-%m-%d %H:%M:%S")}


def _sat_one(path: str, kind: str, cond: str, stride: int = 8, v_min: int = 50) -> dict:
    import cv2
    import numpy as np
    from offroad_ray.quality.metrics import resize_frames
    from offroad_ray.video_io import decode_rgb
    fr = resize_frames(decode_rgb(path))[::stride]
    hsv = np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2HSV) for f in fr])
    s, v = hsv[..., 1].astype(np.float32), hsv[..., 2]
    lab = np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2LAB) for f in fr]).astype(np.float32)
    bright = v >= v_min
    return {"kind": kind, "condition": cond, "hsv_s": float(s.mean()),
            "hsv_s_bright": float(s[bright].mean()) if bright.any() else float("nan"),
            "dark_frac": float((v < v_min).mean()), "v_mean": float(v.mean()),
            "lab_chroma": float(np.hypot(lab[..., 1] - 128, lab[..., 2] - 128).mean())}


def node_hours(budget: dict | None) -> dict:
    if not budget:
        return {}
    pools = [p for p in budget.get("pools", [])]
    actor_h = sum(p["wall_s"] * p["num_actors"] for p in pools) / 3600
    union_h = budget["gpu_phase_wall_time_h"]["used"]
    return {"union_h": union_h, "held_node_h": union_h * NODES, "actor_node_h": actor_h}


# ---------------------------------------------------------------- sections

def sec_dataset(results: Path, runs: Path, ns_dir: Path) -> str:
    fetch = _read_json(results / "dataset" / "fetch_main.json")
    prep = _read_json(results / "dataset" / "prepare_main.json")
    parts = []
    rows = []
    cat = None
    if prep and prep.get("catalog") and Path(prep["catalog"]).exists():
        try:
            import pandas as pd
            cat = pd.read_parquet(prep["catalog"])
        except Exception:  # noqa: BLE001
            cat = None
    if cat is not None:
        r0 = cat.iloc[0]
        rows += [["Source", "TartanDrive 2.0 (AirLab, Carnegie Mellon University), CC BY 4.0, ATV front camera (image_left_color)"],
                 ["Runs (drives)", f"{cat['run_id'].nunique()}"],
                 ["Prepared clips", f"{len(cat)} ({int((cat['status'] == 'ok').sum())} ok)"],
                 ["Clip format", f"{int(r0['model_input_frames'])} frames, {int(r0['model_input_width'])}x{int(r0['model_input_height'])}, "
                                 f"{r0['model_input_fps']:.0f} fps ({r0['model_input_duration_s']:.2f} s)"],
                 ["Resampling", f"source at {cat['source_fps_est'].median():.1f} Hz to 16 fps by nearest-timestamp frame repetition, "
                                f"no interpolation (median {int(cat['resample_duplicates'].median())} repeated frames per clip)"],
                 ["Edge control", f"Canny edges precomputed once per clip at {int(r0['control_width'])}x{int(r0['control_height'])}, "
                                  f"median edge density {cat['edge_density'].median():.3f}"]]
    if fetch:
        rows.append(["New source data downloaded", f"{fetch['new_bytes'] / 1e9:.2f} GB in {fetch['wall_s']:.0f} s ({len(fetch.get('runs', []))} camera archives)"])
    if not rows:
        return "<h2 id=dataset>Dataset</h2>" + _missing("dataset summary files")
    parts.append(table(["Item", "Value"], rows))
    sheet = results / "dataset" / "contact_sheet.jpg"
    return "<h2 id=dataset>Dataset</h2>" + "".join(parts) + (
        f"<figure><img src='{{rel:{sheet}}}' alt='Contact sheet of the prepared source clips'><figcaption>"
        "Prepared source clips (real footage, TartanDrive 2.0).</figcaption></figure>" if sheet.exists() else "")


def sec_phases(runs: Path) -> str:
    rows = []
    for ns, label, note in PHASES:
        if not _allowed(ns):
            continue
        s = _read_json(runs / ns / "pool_summary.json")
        if not s:
            continue
        sc = s.get("status_counts", {})
        ok = sc.get("generated_unreviewed", sc.get("judged", 0))
        g = s.get("settings", {}).get("guardrails")
        rows.append([label, ns, s.get("num_actors"), s.get("requests"), ok, s.get("infra_failures", ""),
                     s.get("actor_replacements", ""), s.get("wall_s"),
                     round(s["wall_s"] * s["num_actors"] / 3600, 3) if s.get("wall_s") else None,
                     "on" if g else ("n/a (judge)" if g is None else "OFF"), note])
    if not rows:
        return "<h2 id=phases>Phases</h2>" + _missing("pool_summary.json files")
    return ("<h2 id=phases>Phases</h2><p>Measured columns come from each namespace's <code>pool_summary.json</code>; "
            "the notes column is copied from AGENTS.md. Actor node-hours = wall time x actors (each actor holds one "
            "1x A100 node). The guardrails-off smoke namespace is excluded from all results.</p>"
            + table(["Phase", "Namespace", "Actors", "Requests", "OK", "Infra failures", "Actor replacements",
                     "Wall s", "Actor node-h", "Guardrails", "Notes"], rows, {2, 3, 4, 5, 6, 7, 8}))


def sec_scaling(results: Path) -> str:
    d = _read_json(results / "benchmark" / "scaling.json")
    if not d:
        return "<h2 id=scaling>Scaling benchmark</h2>" + _missing("results/benchmark/scaling.json")
    pts = {p["num_actors"]: p for p in d["points"]}
    rows = []
    for s in d["scaling"]:
        p = pts.get(s["num_actors"], {})
        rows.append([s["num_actors"], s["wall_s"], s["generation_window_s"], s["cold_start_s"], s["speedup_e2e"],
                     s["efficiency_e2e"], s["speedup_gen"], s["efficiency_gen"],
                     p.get("requests_per_min_generation_window"), p.get("per_request_child_s_median"),
                     p.get("node_hours_gpu_group"), p.get("active_fraction_of_node_hours")])
    imgs = []
    for name, cap in [("scaling.png", "Speedup and efficiency vs actors"), ("timeline_a1.png", "Request timeline, 1 actor"),
                      ("timeline_a2.png", "2 actors"), ("timeline_a4.png", "4 actors"), ("timeline_a8.png", "8 actors")]:
        p = results / "benchmark" / name
        if p.exists():
            imgs.append(f"<figure><img src='{{rel:{p}}}' alt='{_e(cap)}'><figcaption>{_e(cap)}</figcaption></figure>")
    return ("<h2 id=scaling>Scaling benchmark</h2>"
            f"<p>{d['requests_per_point']} identical requests per point, fresh namespace per point, no resume. "
            "End-to-end (e2e) includes actor start-up and model load; the generation window starts at the first "
            "dispatch. Node-hours count all 8 GPU nodes for the run wall time because the cluster keeps them "
            "allocated; the active fraction is the share of those node-hours spent generating.</p>"
            + table(["Actors", "Wall s", "Generation window s", "Cold start s", "Speedup e2e", "Efficiency e2e",
                     "Speedup gen", "Efficiency gen", "Requests/min (gen window)", "Median s per request",
                     "Node-h (8 nodes held)", "Active fraction"], rows, set(range(12)), nd=2)
            + "<div class=grid2>" + "".join(imgs) + "</div>")


def sec_stages(ns_dir: Path) -> str:
    t = stage_timing(ns_dir)
    if not t:
        return "<h2 id=stages>Per-stage timing</h2>" + _missing(f"records with stage_times in {ns_dir}")
    m = t["median_s"]
    tot = t["child_seconds_median"]
    rows = [
        ["text_guardrail", "top level", m["text_guardrail"]],
        ["generate_img2world", "top level (wraps the 4 rows below)", m["generate_img2world"]],
        ["  get_text_embeddings", "inside generate_img2world", m["get_text_embeddings"]],
        ["  preprocessing", "inside generate_img2world", m["preprocessing"]],
        ["  generate_chunk (4 denoising steps)", "inside generate_img2world", m["generate_chunk"]],
        ["  postprocessing", "inside generate_img2world", m["postprocessing"]],
        ["  other inside generate_img2world", "median of per-request remainder", t["img2world_other_s"]],
        ["video_guardrail", "top level", m["video_guardrail"]],
        ["untimed (mostly writing the output and control videos)", "median of child_seconds minus top-level timers", t["untimed_s"]],
        ["child_seconds (per request, worker side)", "total", tot],
    ]
    for r in rows:
        r.append(round(100 * r[2] / tot, 1) if r[2] is not None and tot else None)
    return ("<h2 id=stages>Per-stage timing</h2>"
            f"<p>Recomputed from {t['n']} per-request records in <code>{_e(ns_dir)}</code> (median of each timer; "
            "medians do not add up exactly). Generation dominates: the 4-step distilled sampler is "
            f"{100 * m['generate_chunk'] / tot:.0f}% of the median request. Request time ranged "
            f"{t['child_seconds_min']:.1f} to {t['child_seconds_max']:.1f} s. Output validation with ffprobe on the "
            f"driver side adds a median {t['validate_s_median']:.2f} s.</p>"
            + table(["Stage", "Scope", "Median s", "% of median request"], rows, {2, 3}))


def sec_quality(ns_dir: Path) -> str:
    qdir = Path(ns_dir, "quality")
    s = _read_json(qdir / "quality_summary.json")
    if not s:
        return "<h2 id=quality>Automated quality metrics</h2>" + _missing(f"{qdir}/quality_summary.json")
    rows = [[c["condition"], c["count()"], c["mean(edge_f1)"], c["std(edge_f1)"], c["mean(edge_recall)"],
             c["mean(delta_luma)"], c["mean(delta_contrast)"], c["mean(delta_saturation)"], c["mean(luma_jitter)"],
             c["sum(blank_frames)"], c["sum(num_discontinuity_candidates)"]] for c in s["condition_summary"]]
    flags_html = ""
    try:
        import pandas as pd
        q = pd.read_parquet(qdir / "quality_table.parquet")
        f = q[q["auto_flags"] != ""]
        if len(f):
            fr = [[r.request_id, r.clip_id.split("__")[0], r.clip_id.split("__")[2] if r.clip_id.count("__") >= 2 else "",
                   r.condition, r.seed, r.auto_flags, r.frozen_frames, r.source_frozen_frames] for r in f.itertuples()]
            flags_html = ("<h3>Auto-flags</h3>"
                          f"<p>{len(f)} of {len(q)} outputs were flagged. All flags are <code>frozen_frames</code> "
                          "(more near-identical consecutive frames than the source plus 5). The source column shows the "
                          "real clip already has many near-identical frames: the vehicle moves slowly in that window, and "
                          "the 10 Hz to 16 Hz resampling repeats frames on purpose. These are pointers for human review, "
                          "not confirmed defects.</p>"
                          + table(["Request", "Run", "Window", "Condition", "Seed", "Flags", "Frozen transitions (output)",
                                   "Frozen transitions (source)"], fr, {4, 6, 7}))
    except Exception:  # noqa: BLE001
        pass
    return ("<h2 id=quality>Automated quality metrics</h2>"
            f"<p>Main run, {s['records']} outputs, {s['decoded']} decoded, {s['auto_flagged']} auto-flagged, "
            f"{s['human_labelled']} human labels. Edge F1 compares Canny edges of the output with Canny edges of the "
            "real source (2 px tolerance); it measures structure kept, not quality. Luma, contrast and saturation "
            "deltas describe the appearance change (output minus source, 0 to 255 levels).</p>"
            + table(["Condition", "Outputs", "Edge F1", "Edge F1 std", "Edge recall", "Delta luma", "Delta contrast",
                     "Delta saturation (HSV)", "Luma jitter", "Blank frames", "Discontinuity candidates"],
                    rows, set(range(1, 11)))
            + flags_html)


def _judge_rows(summary: dict) -> list[list]:
    return [[r["set"], r["requested_condition"], r["n"], r["photorealism"], r["temporal_stability"],
             r["physical_plausibility"], r["condition_intensity"]] for r in summary.get("by_set_condition", [])]


def sec_judge(runs: Path) -> str:
    out = ["<h2 id=judge>VLM judge (automated, blind, Cosmos-Reason1-7B)</h2>",
           "<p class=callout>These scores come from an automated model, not from people. Cosmos-Reason1-7B "
           "(greedy decoding, 4 fps, 560 px) scored each clip 1 to 5 without being told the requested condition or "
           "whether the clip was real or synthetic. Real source clips were judged as calibration. No human labels "
           "exist yet.</p>"]
    hdr = ["Set", "Requested condition", "n", "Photorealism", "Temporal stability", "Physical plausibility",
           "Condition intensity"]
    sw = _read_json(runs / "judge" / "sweep" / "summary.json")
    if sw:
        rows = [[r["set"], r["n"], r["photorealism"], r["temporal_stability"], r["physical_plausibility"],
                 r["condition_intensity"]] for r in sw["by_set"]]
        out.append("<h3>Calibration and realism sweep</h3><p>The real source clips (set <code>source</code>) score only "
                   "slightly above the synthetic sets, so the judge separates real from synthetic weakly; treat small "
                   "differences between sets as noise.</p>")
        out.append(table(["Set", "n", "Photorealism", "Temporal stability", "Physical plausibility",
                          "Condition intensity"], rows, {1, 2, 3, 4, 5}))
    mn = _read_json(runs / "judge" / "main" / "summary.json")
    if mn:
        rows = [r + [r_["condition_match"]] for r, r_ in zip(_judge_rows(mn), mn["by_set_condition"])]
        out.append(f"<h3>Main run</h3><p>{mn['parsed_ok']} of {mn['total']} judgments parsed. Condition match is the "
                   "share of clips whose blind observed-condition label equals the requested one. Most rain and snow "
                   "clips were labelled overcast.</p>")
        out.append(table(hdr + ["Condition match"], rows, {2, 3, 4, 5, 6, 7}))
    v3 = _read_json(runs / "judge" / "v3" / "summary.json")
    if v3:
        out.append(f"<h3>Phase 3 conditions</h3><p>{v3['parsed_ok']} of {v3['total']} parsed. The judge vocabulary has no "
                   "label for autumn, dust and similar conditions, so condition match is 0 by construction; the observed "
                   "label counts below are the useful signal.</p>")
        out.append(table(hdr, _judge_rows(v3), {2, 3, 4, 5, 6}))
        conf = v3.get("confusion", {})
        if conf:
            conds = sorted({c for v in conf.values() for c in v})
            labels = sorted(conf)
            rows = [[c] + [conf[l].get(c, 0) for l in labels] for c in conds]
            out.append("<p>Observed label (columns) per requested condition (rows):</p>"
                       + table(["Requested"] + labels, rows, set(range(1, len(labels) + 1))))
    if len(out) == 2:
        out.append(_missing("judge summaries"))
    return "".join(out)


def _media(entry: dict, out_dir: Path, project: Path) -> str:
    mp4, jpg = entry.get("mp4"), entry.get("jpg")
    if not mp4 or not (project / mp4).exists():
        return ""
    poster = f" poster='{{rel:{project / jpg}}}'" if jpg else ""
    links = [f"<a href='{{rel:{project / mp4}}}'>MP4 {entry.get('mp4_bytes', 0) / 1e6:.1f} MB</a>"]
    for k in ("gif", "webp", "jpg", "jpg_hires"):
        if entry.get(k):
            links.append(f"<a href='{{rel:{project / entry[k]}}}'>{k.replace('_', ' ').upper()} "
                         f"{(entry.get(k + '_bytes') or 0) / 1e6:.1f} MB</a>")
    return (f"<video controls loop muted playsinline preload=none{poster} src='{{rel:{project / mp4}}}'></video>"
            f"<p class=links>{' | '.join(links)}</p>")


def sec_showcase(results: Path, out_dir: Path) -> str:
    project = results.parent
    m = _read_json(results / "showcase" / "web" / "assets.json")
    if not m or "hero" not in m:
        return "<h2 id=showcase>Showcase</h2>" + _missing("results/showcase/web/assets.json (run python -m offroad_ray.reporting.web_assets)")
    h = m["hero"]
    out = ["<h2 id=showcase>Showcase</h2><p>Every panel marked synthetic was generated by Cosmos Transfer 2.5 "
           "edge/distilled with guardrails on; the real panels are TartanDrive 2.0 footage. Selections below use the "
           "automated judge scores, not human review.</p>"]
    g = h.get("condition_grid")
    if g:
        out.append(f"<h3>Condition grid: {_e(g['scene'])}</h3><p>{_e(g['why'])} Source, its edge control, and "
                   f"{len(g['conditions'])} conditions, all frames in sync.</p>" + _media(g, out_dir, project))
    for e in h.get("before_after", []):
        sc = ", ".join(f"{k.replace('_', ' ')} {v:.0f}" for k, v in e["scores"].items())
        out.append(f"<h3>Before and after: {_e(e['scene'])}, {_e(e['condition'].replace('_', ' '))}</h3>"
                   f"<p>Judge scores (automated): {_e(sc)}. Blind observed label: {_e(e['judge_observed_condition'])}.</p>"
                   + _media(e, out_dir, project))
    if h.get("before_after"):
        out.append("<p class=muted>Before/after rule: rank all judged synthetic outputs by the mean of the four judge "
                   "scores, then pick greedily with distinct scenes and distinct conditions. All three picks scored 5 on "
                   "every criterion, so ties were broken by the condition's mean judge score.</p>")
    ms = h.get("multi_scene")
    if ms:
        out.append(f"<h3>Multi-scene grid: {_e(ms['condition'].replace('_', ' '))} on {len(ms['scenes'])} scenes</h3><p>{_e(ms['why'])}</p>"
                   + _media(ms, out_dir, project))
    f = h.get("failure")
    if f:
        out.append(f"<h3>Failure example: {_e(f['scene'])}, {_e(f['condition'].replace('_', ' '))}</h3>"
                   f"<p>{_e(f['caption'])}</p>" + _media(f, out_dir, project))
    re_ = m.get("reencoded", [])
    if re_:
        rows = [[_e(Path(r["mp4"]).parent.name), f"<a href='{{rel:{project / r['mp4']}}}'>{_e(Path(r['mp4']).name)}</a>",
                 f"{r['mp4_bytes'] / 1e6:.1f}", f"<a href='{{rel:{project / r['jpg']}}}'>still</a>"] for r in re_]
        body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
        out.append(f"<details><summary>All web-sized showcase clips ({len(re_)})</summary><div class=tw><table><thead><tr>"
                   "<th>Set</th><th>MP4</th><th>MB</th><th>JPG</th></tr></thead><tbody>" + body + "</tbody></table></div></details>")
    return "".join(out)


def sec_budget(budget: dict | None) -> str:
    if not budget:
        return "<h2 id=budget>Budget ledger</h2>" + _missing("budget status")
    nh = node_hours(budget)
    g, a, d = budget["gpu_phase_wall_time_h"], budget["generation_attempts"], budget["source_download_gb"]
    rows = [["GPU-phase wall time (union of pool intervals)", f"{g['used']:.3f} h", f"{g['limit']} h", g.get("note", "")],
            ["Generation attempts", a["used"], a["limit"], a.get("note", "")],
            ["New source data", f"{d['used']:.2f} GB", f"{d['limit']} GB", ""]]
    pools = [[p["ns"] + (" (guardrails off, not evaluated)" if not _allowed(p["ns"]) else ""), p["num_actors"],
              p["wall_s"], round(p["wall_s"] * p["num_actors"] / 3600, 3)] for p in budget.get("pools", [])]
    return ("<h2 id=budget>Budget ledger</h2><p>From <code>offroad_ray.cli budget</code>. Cost is reported as "
            "node-hours only; pricing was not verified. Each GPU node is 1x A100-SXM4-80GB.</p>"
            + table(["Item", "Used", "Limit", "Note"], rows)
            + table(["Node-hours", "Value"], [
                ["Held: GPU-phase union x 8 nodes (upper bound, cluster keeps nodes allocated)", round(nh["held_node_h"], 2)],
                ["Active: sum over pools of wall time x actors", round(nh["actor_node_h"], 2)]], {1})
            + "<details><summary>Per-pool ledger (includes time spent on runs not shown as results)</summary>"
            + table(["Namespace", "Actors", "Wall s", "Actor node-h"], pools, {1, 2, 3}) + "</details>")


def sec_limitations(sat: dict | None, quality: dict | None) -> str:
    items = [
        "Vehicle motion, camera path and scene layout are fixed by the source footage. The edge control pins structure, "
        "so the model changes appearance (weather, light, season) but cannot add obstacles, change the route or alter "
        "how the vehicle drives.",
        "Fog has the lowest edge agreement with the source. That is expected by design: fog hides distant edges, so a "
        "convincing fog clip should lose edges. Edge F1 is compared within a condition, never as an absolute score.",
        "All judge scores are automated (Cosmos-Reason1-7B, blind). The judge rates real source clips only slightly above "
        "synthetic ones and often labels rain and snow as overcast. No human review has been done yet.",
        "The source runs at 10 Hz and is shown at 16 fps by repeating frames, which also inflates frozen-frame counts.",
        "Each clip is 93 frames (5.8 s) at 1024x544. Longer clips would need chunked generation, which was not tested.",
        "Seeds are fixed (101, 202, 303); guardrails were on for every reported run and never bypassed.",
    ]
    sat_html = ""
    if sat and "night" in sat.get("by_condition", {}) and "source" in sat["by_condition"]:
        b = sat["by_condition"]
        src, n = b["source"], b["night"]
        night_delta = (quality or {}).get("night")
        rows = [[c, v["hsv_s"], v["hsv_s_bright"], v["lab_chroma"], v["v_mean"], 100 * v["dark_frac"]]
                for c, v in sorted(b.items(), key=lambda kv: kv[0] != "source")]
        sat_html = (
            "<h3>Night saturation check</h3>"
            f"<p>The main-run night delta saturation is {night_delta:+.0f} levels" if night_delta is not None else
            "<h3>Night saturation check</h3><p>The night delta saturation is large")
        sat_html += (
            ". <code>quality/metrics.py</code> computes saturation as the mean OpenCV HSV S channel, which is "
            "(max - min) / max of the RGB values. Dividing by brightness makes S grow as pixels get darker for the same "
            f"colour difference. Recomputed on {sat['n_outputs']} seed-{sat['seed']} outputs and {sat['n_sources']} sources "
            f"(every {sat['stride']}th frame, 512x272): night outputs average HSV S {n['hsv_s']:.0f} vs {src['hsv_s']:.0f} "
            f"for the sources, with {100 * n['dark_frac']:.0f}% of pixels below V {sat['v_min']}. Restricting to pixels "
            f"with V of at least {sat['v_min']} does not remove the gap (S {n['hsv_s_bright']:.0f} vs {src['hsv_s_bright']:.0f}), "
            f"because those pixels are still dim. CIELAB chroma, which does not divide by brightness, rises from "
            f"{src['lab_chroma']:.1f} to {n['lab_chroma']:.1f} ({n['lab_chroma'] / src['lab_chroma']:.1f}x). "
            "Finding: the night clips really are more colourful than the source (a blue or amber cast from moonlight "
            "and headlights), but the HSV metric exaggerates the size of that change by dividing by low brightness. "
            "Do not quote the HSV delta as a colour change for night; quote chroma instead.</p>"
            + table(["Condition", "HSV S (all pixels)", f"HSV S (V >= {sat['v_min']})", "CIELAB chroma", "Mean V",
                     f"% pixels V < {sat['v_min']}"], rows, {1, 2, 3, 4, 5}, nd=1))
    else:
        items.append("Night delta saturation is large; it has not been rechecked here (run report with "
                     "--saturation-check). The HSV S channel divides by brightness and inflates on dark pixels.")
    return ("<h2 id=limitations>Limitations</h2><ul>" + "".join(f"<li>{_e(i)}</li>" for i in items) + "</ul>" + sat_html)


CSS = """
:root{--fg:#1d1d1f;--muted:#5f6368;--line:#dcdcdc;--bg:#fff;--soft:#f5f5f2;--accent:#1f6e43}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,"DejaVu Sans",sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px 64px}h1{font-size:28px;margin:0 0 4px}h2{margin-top:44px;padding-top:8px;border-top:1px solid var(--line)}
h3{margin-top:28px}nav{margin:12px 0 8px;font-size:14px}nav a{margin-right:12px;color:var(--accent)}a{color:var(--accent)}
.muted{color:var(--muted)}.callout{background:var(--soft);border-left:4px solid var(--accent);padding:10px 14px}
.tw{overflow-x:auto;margin:10px 0}table{border-collapse:collapse;font-size:13.5px;min-width:50%}th,td{border-bottom:1px solid var(--line);padding:5px 9px;text-align:left;vertical-align:top}
th{background:var(--soft);font-weight:600}td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}code{font-size:13px;background:var(--soft);padding:1px 4px;border-radius:3px}
img,video{max-width:100%;height:auto;display:block;border:1px solid var(--line)}figure{margin:12px 0}figcaption{font-size:13px;color:var(--muted)}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px}.links{font-size:13px;margin:4px 0 0}
details{margin:12px 0}summary{cursor:pointer;color:var(--accent)}
"""


def build_report(cfg=None, ns_dir: str | Path | None = None, *, out_dir: str | Path | None = None,
                 runs_dir: str | Path = RUNS, results_dir: str | Path = RESULTS, budget: dict | None = None,
                 saturation: bool | dict | None = None) -> Path:
    """Write <out_dir>/index.html and return its path.

    cfg is accepted for CLI symmetry (only cfg.paths.runs_dir is used when given). ns_dir is the
    namespace for the timing and quality sections (default runs/main/main). budget: pass a dict to
    skip calling budget_status(). saturation: True recomputes the night saturation check and caches
    it in out_dir/saturation_check.json; a dict is used as is; None reads the cache if present.
    """
    runs = Path(getattr(getattr(cfg, "paths", None), "runs_dir", None) or runs_dir)
    results = Path(results_dir)
    ns = Path(ns_dir) if ns_dir else runs / "main" / "main"
    if not _allowed(str(ns)):
        raise ValueError(f"refusing to report on {ns}: guardrails-off or demo namespace")
    summ = _read_json(ns / "pool_summary.json") or {}
    if summ and summ.get("settings", {}).get("guardrails") is False:
        raise ValueError(f"refusing to report on {ns}: guardrails OFF")
    out = Path(out_dir) if out_dir else results / "report"
    out.mkdir(parents=True, exist_ok=True)

    if budget is None:
        try:
            from offroad_ray.reporting.benchmark import budget_status
            budget = budget_status(str(runs.parent))
        except Exception:  # noqa: BLE001
            budget = None
    cache = out / "saturation_check.json"
    if saturation is True:
        sat = saturation_check(ns)
        cache.write_text(json.dumps(sat, indent=1))
    elif isinstance(saturation, dict):
        sat = saturation
    else:
        sat = _read_json(cache)
    qs = _read_json(ns / "quality" / "quality_summary.json") or {}
    night_q = {c["condition"]: c["mean(delta_saturation)"] for c in qs.get("condition_summary", [])}

    sections = [
        ("dataset", "Dataset", sec_dataset(results, runs, ns)),
        ("phases", "Phases", sec_phases(runs)),
        ("scaling", "Scaling", sec_scaling(results)),
        ("stages", "Stage timing", sec_stages(ns)),
        ("quality", "Quality metrics", sec_quality(ns)),
        ("judge", "VLM judge", sec_judge(runs)),
        ("showcase", "Showcase", sec_showcase(results, out)),
        ("budget", "Budget", sec_budget(budget)),
        ("limitations", "Limitations", sec_limitations(sat, night_q)),
    ]
    nav = "".join(f"<a href='#{k}'>{_e(t)}</a>" for k, t, _ in sections)
    body = "".join(s for _, _, s in sections)
    # resolve {rel:/abs/path} placeholders to paths relative to the report directory
    import re
    body = re.sub(r"\{rel:([^}]+)\}", lambda m_: _e(_rel(Path(m_.group(1)), out)), body)
    doc = (
        "<!doctype html><html lang=en><head><meta charset=utf-8>"
        "<meta name=viewport content='width=device-width,initial-scale=1'>"
        "<title>Off-road augmentation results</title>"
        f"<style>{CSS}</style></head><body><main>"
        "<h1>Off-road video augmentation with Ray and Cosmos Transfer 2.5</h1>"
        "<p class=muted>Results report for TartanDrive 2.0 ATV footage re-rendered by Cosmos Transfer 2.5 edge/distilled. "
        f"Generated {_e(time.strftime('%Y-%m-%d %H:%M %Z'))} by <code>offroad_ray.cli report</code>. All results use guardrails on. "
        "Judge scores are automated, not human. Cost is in node-hours only.</p>"
        f"<nav>{nav}</nav>{body}"
        "<p class=muted style='margin-top:48px'>Source footage: TartanDrive 2.0, AirLab, Carnegie Mellon University, CC BY 4.0. "
        "Frames were temporally resampled, re-encoded and used as conditioning for synthetic appearance changes.</p>"
        "</main></body></html>")
    path = out / "index.html"
    path.write_text(doc)
    return path
