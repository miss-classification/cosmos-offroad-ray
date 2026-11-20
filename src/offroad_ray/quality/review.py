"""Quality processing (Ray Data) and the static human-review interface.

Pipeline:
  manifest.jsonl ──Ray Data map(compute_output_metrics)──▶ metrics parquet
               ──map(make_triptych)──▶ per-request synchronized review video
               ──groupby(condition).aggregate──▶ evaluation table
Human labels are never inferred. They come only from a labels JSON exported by
the review page (or written by a reviewer) and merged by request_id.
"""
from __future__ import annotations

import html
import json
import os
import time
from pathlib import Path

RUBRIC = [
    ("trail_layout", "Trail layout preserved"),
    ("obstacles", "Obstacle placement preserved"),
    ("persistence", "Objects persist (no popping)"),
    ("shape", "Shapes plausible"),
    ("camera_motion", "Camera motion continuous"),
    ("temporal", "Temporally consistent (no flicker)"),
    ("appearance", "Requested appearance achieved"),
    ("hallucination", "No severe hallucination"),
]
DECISIONS = ("keep", "reject", "uncertain")
DECISION_STATE = {"keep": "quality_accepted", "reject": "quality_rejected", "uncertain": "review_uncertain"}


def load_manifest(ns_dir: str | Path) -> list[dict]:
    p = Path(ns_dir) / "manifest.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def make_triptych(rec: dict) -> dict:
    """Ray Data map: render source | control | generated review video for one record."""
    from offroad_ray.reporting.montage import triptych
    out = {"request_id": rec["request_id"], "triptych_path": "", "triptych_error": ""}
    if rec.get("status") != "generated_unreviewed":
        return out
    try:
        dst = str(Path(rec["output_dir"]) / "review_triptych.mp4")
        if not Path(dst).exists():
            triptych(rec["clip_path"], rec["control_path"], rec["output_path"], dst,
                     condition=rec["condition"], seed=int(rec["seed"]),
                     title=f"{rec['clip_id'][:60]}   request {rec['request_id'][:10]}")
        out["triptych_path"] = dst
    except Exception as e:
        out["triptych_error"] = f"{type(e).__name__}: {e}"
    return out


def merge_labels(recs: list[dict], labels_path: Path) -> tuple[list[dict], int]:
    """Attach human labels; returns (records, number labelled). Missing labels stay unreviewed."""
    if not labels_path.exists():
        return recs, 0
    labels = json.loads(labels_path.read_text())
    by_id = {l["request_id"]: l for l in labels.get("labels", [])}
    n = 0
    for r in recs:
        lab = by_id.get(r["request_id"])
        if lab and lab.get("decision") in DECISIONS and r.get("status") == "generated_unreviewed":
            r["review_state"] = DECISION_STATE[lab["decision"]]
            r["review"] = lab
            n += 1
        else:
            r.setdefault("review_state", r.get("status"))
    return recs, n


def run_quality(cfg, ns_dir: str, *, render_triptychs: bool = True) -> dict:
    import pandas as pd
    import ray
    from offroad_ray.quality.metrics import compute_output_metrics
    from offroad_ray.ray_env import WORKER_ONLY
    from offroad_ray.storage import atomic_write_json

    ns = Path(ns_dir)
    recs = load_manifest(ns)
    rows = [{"request_id": r["request_id"], "status": r.get("status", ""), "output_path": r.get("output_path") or "",
             "output_dir": r.get("output_dir") or "", "clip_path": r.get("clip_path") or "",
             "control_path": r.get("control_path") or "", "condition": r.get("condition", ""),
             "seed": int(r.get("seed", -1)), "clip_id": r.get("clip_id", ""), "run_id": r.get("run_id", "")}
            for r in recs]
    t0 = time.time()
    ctx = ray.data.DataContext.get_current()
    ctx.enable_progress_bars = False
    ds = ray.data.from_items(rows, override_num_blocks=max(1, len(rows)))
    metrics = ds.map(compute_output_metrics, num_cpus=2, resources=WORKER_ONLY,
                     compute=ray.data.TaskPoolStrategy(size=32))
    mdir = ns / "quality" / "metrics_parquet"
    if mdir.exists():
        mdir.rename(mdir.with_name(f"metrics_parquet.prev.{int(t0)}"))
    metrics.write_parquet(str(mdir))
    t_metrics = time.time()
    trip = {}
    if render_triptychs:
        tds = ds.map(make_triptych, num_cpus=2, resources=WORKER_ONLY, compute=ray.data.TaskPoolStrategy(size=32))
        trip = {r["request_id"]: r for r in tds.take_all()}
    t_trip = time.time()

    # Aggregate with Ray Data (per condition), then pandas for the small final table.
    mds = ray.data.read_parquet(str(mdir))
    ok = mds.filter(lambda r: bool(r.get("decode_ok")))
    joined = ok.map(lambda r: {**r, "condition": next((x["condition"] for x in rows if x["request_id"] == r["request_id"]), "")})
    agg = joined.groupby("condition").aggregate(
        ray.data.aggregate.Count(), ray.data.aggregate.Mean("edge_f1"), ray.data.aggregate.Std("edge_f1"),
        ray.data.aggregate.Mean("edge_recall"), ray.data.aggregate.Mean("delta_luma"),
        ray.data.aggregate.Mean("delta_contrast"), ray.data.aggregate.Mean("delta_saturation"),
        ray.data.aggregate.Sum("num_discontinuity_candidates"), ray.data.aggregate.Sum("blank_frames"),
        ray.data.aggregate.Mean("luma_jitter")).to_pandas()
    t_agg = time.time()

    mdf = pd.read_parquet(mdir)
    recs, n_labelled = merge_labels(recs, ns / "review" / "human_labels.json")
    rdf = pd.DataFrame(recs)
    full = rdf.merge(mdf.drop(columns=["status"], errors="ignore"), on="request_id", how="left")
    if trip:
        full["triptych_path"] = full.request_id.map(lambda i: trip.get(i, {}).get("triptych_path", ""))
    qdir = ns / "quality"
    qdir.mkdir(parents=True, exist_ok=True)
    full.drop(columns=[c for c in ("traceback", "identity", "stage_times", "output_probe", "review") if c in full],
              errors="ignore").to_parquet(qdir / "quality_table.parquet")
    agg.to_csv(qdir / "condition_summary.csv", index=False)
    page = write_review_page(cfg, ns, full.to_dict("records"))
    summary = {"records": len(recs), "decoded": int(mdf.decode_ok.sum()), "human_labelled": n_labelled,
               "auto_flagged": int((mdf.get("auto_flags", pd.Series(dtype=str)).fillna("") != "").sum()),
               "metrics_s": round(t_metrics - t0, 2), "triptych_s": round(t_trip - t_metrics, 2),
               "aggregate_s": round(t_agg - t_trip, 2), "review_page": str(page),
               "condition_summary": agg.round(4).to_dict("records")}
    atomic_write_json(qdir / "quality_summary.json", summary)
    return summary


def _rel(target: str, start: Path) -> str:
    return os.path.relpath(target, start) if target else ""


def write_review_page(cfg, ns: Path, rows: list[dict]) -> Path:
    """Static HTML review page with rubric, flags and label export. No server needed."""
    out_dir = ns / "review"
    out_dir.mkdir(parents=True, exist_ok=True)
    cards = []
    for r in sorted(rows, key=lambda x: (x.get("clip_id", ""), x.get("condition", ""), x.get("seed", 0))):
        rid = r["request_id"]
        st = r.get("status", "")
        vid = r.get("triptych_path") or ""
        media = (f'<video src="{html.escape(_rel(vid, out_dir))}" controls loop muted preload="none" '
                 f'width="100%"></video>' if vid else f'<div class="missing">no video ({html.escape(st)})</div>')
        flags = r.get("auto_flags") or ""
        metrics = (f"edge F1 {r.get('edge_f1', float('nan')):.3f} | dLuma {r.get('delta_luma', float('nan')):+.1f} | "
                   f"dContrast {r.get('delta_contrast', float('nan')):+.1f} | jumps {r.get('num_discontinuity_candidates', '')}"
                   if r.get("decode_ok") else "")
        rubric = "".join(
            f'<label><input type="checkbox" data-k="{k}"> {html.escape(lbl)}</label>' for k, lbl in RUBRIC)
        dec = "".join(f'<label><input type="radio" name="d_{rid}" value="{d}"> {d}</label>' for d in DECISIONS)
        cards.append(f"""
<section class="card" data-rid="{rid}" data-status="{html.escape(st)}">
  <header><b>{html.escape(r.get('condition', ''))}</b> seed {r.get('seed')} | {html.escape(r.get('clip_id', '')[:70])}
  <span class="st st-{html.escape(st)}">{html.escape(r.get('review_state', st))}</span></header>
  {media}
  <div class="meta">request {rid} | model {html.escape(cfg.generation.model)}, {cfg.generation.num_steps} steps,
  control weight {r.get('control_weight')} | guardrails {'on' if cfg.generation.guardrails else 'off'} |
  gen {r.get('child_seconds', '')} s on actor {r.get('actor_index', '')}</div>
  <div class="meta">{html.escape(metrics)} {('| <span class="flag">flags: ' + html.escape(flags) + '</span>') if flags else ''}</div>
  <details><summary>prompt</summary><p>{html.escape(r.get('prompt', '') or '')}</p></details>
  <div class="rubric">{rubric}</div>
  <div class="decision">{dec}
    <select class="cat"><option value="">failure category</option>
      <option>structure_drift</option><option>object_hallucination</option><option>object_removal</option>
      <option>temporal_flicker</option><option>weak_appearance</option><option>artifact</option><option>other</option></select>
    <input class="ts" placeholder="time (s) of issue"> <input class="note" placeholder="note"></div>
</section>""")
    page = out_dir / "index.html"
    page.write_text(REVIEW_TEMPLATE.replace("{{TITLE}}", html.escape(f"{cfg.name} / {ns.name} review"))
                    .replace("{{COUNT}}", str(len(rows))).replace("{{CARDS}}", "\n".join(cards)))
    return page


REVIEW_TEMPLATE = """<!doctype html><meta charset="utf-8"><title>{{TITLE}}</title>
<style>
body{font:14px/1.4 system-ui,sans-serif;margin:16px;background:#fafafa;color:#222}
.card{background:#fff;border:1px solid #ddd;border-radius:6px;padding:10px;margin:12px 0}
header{margin-bottom:6px}.meta{color:#555;font-size:12px;margin:4px 0}.flag{color:#b3261e}
.st{float:right;font-size:12px;padding:1px 6px;border-radius:3px;background:#eee}
.st-guardrail_blocked,.st-generation_failure,.st-infra_failure{background:#fde7e7}
.rubric label{display:inline-block;margin-right:12px;font-size:12px}.decision{margin-top:6px}
.missing{padding:30px;text-align:center;background:#f3f3f3;color:#777}
#bar{position:sticky;top:0;background:#fff;padding:8px;border-bottom:1px solid #ddd;z-index:2}
textarea{width:100%;height:120px}
</style>
<div id="bar"><b>{{TITLE}}</b> ({{COUNT}} requests). Synthetic outputs are generated by Cosmos Transfer 2.5;
source footage TartanDrive 2.0 (CC BY 4.0). Labels are saved in this browser until exported.
<button onclick="exportLabels()">Export labels JSON</button>
<label><input type="checkbox" id="blind" onchange="toggleBlind()"> hide flags/metrics while reviewing</label></div>
{{CARDS}}
<h3>Exported labels</h3><p>Save as <code>review/human_labels.json</code> in this run directory, then rerun
<code>offroad-ray review</code>.</p><textarea id="out"></textarea>
<script>
const KEY='labels:'+document.title;
function cardState(c){const d=c.querySelector('input[type=radio]:checked');
 const rub={};c.querySelectorAll('.rubric input').forEach(i=>rub[i.dataset.k]=i.checked);
 return {request_id:c.dataset.rid,decision:d?d.value:null,rubric:rub,
  failure_category:c.querySelector('.cat').value,issue_time_s:c.querySelector('.ts').value,
  note:c.querySelector('.note').value,reviewed_at:new Date().toISOString()};}
function save(){const all={};document.querySelectorAll('.card').forEach(c=>{const s=cardState(c);if(s.decision)all[s.request_id]=s});
 try{localStorage.setItem(KEY,JSON.stringify(all))}catch(e){}}
function load(){let all={};try{all=JSON.parse(localStorage.getItem(KEY)||'{}')}catch(e){}
 document.querySelectorAll('.card').forEach(c=>{const s=all[c.dataset.rid];if(!s)return;
  const r=c.querySelector(`input[value=${s.decision}]`);if(r)r.checked=true;
  Object.entries(s.rubric||{}).forEach(([k,v])=>{const i=c.querySelector(`[data-k=${k}]`);if(i)i.checked=v});
  c.querySelector('.cat').value=s.failure_category||'';c.querySelector('.ts').value=s.issue_time_s||'';
  c.querySelector('.note').value=s.note||'';});}
function exportLabels(){save();let all={};try{all=JSON.parse(localStorage.getItem(KEY)||'{}')}catch(e){}
 const labels=[];document.querySelectorAll('.card').forEach(c=>{const s=all[c.dataset.rid]||cardState(c);if(s.decision)labels.push(s)});
 document.getElementById('out').value=JSON.stringify({reviewer:'',labels:labels},null,1);}
function toggleBlind(){const b=document.getElementById('blind').checked;
 document.querySelectorAll('.meta').forEach(m=>m.style.visibility=b?'hidden':'visible');}
document.addEventListener('change',save);document.addEventListener('input',save);load();
</script>"""
