"""Maintainers only: build the template's data bundle from a finished set of runs.

The bundle is what ``scripts/setup_data.py`` downloads. It holds everything the notebooks
show without GPUs:

  media/<scene>/src.mp4, <condition>.mp4, thumb.jpg   9 showcase drives, 10 conditions (H.264 CRF 20, 1024x544)
  media/index.json                                    scene names, request IDs and prompts
  results/...                                         manifests, timings, quality, judge and scaling results

Only guardrails-on runs are included. Host names, IP addresses, GPU IDs and absolute
paths are removed from every record.

  python scripts/make_bundle.py --runs /path/to/runs --media docs/blog/replay --results results --out /path/to/bundle
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

DROP_KEYS = re.compile(r"(host|node_ip|gpu_uuid|cuda_visible|attempt_dir|hostname|^ip$|node_id|pid$)", re.I)
NAMES = {
    "dd5a2ee56e": "Grassy two-track", "790e27532e": "Dirt road, blue sky", "833f53f9d2": "Narrow gravel path",
    "5d20270fcc": "Straight dirt lane", "242efb7e7e": "Rocky meadow trail", "6e36465c49": "Wet gravel road",
    "fb9d3bafac": "Rocks under trees", "1b210bceff": "Power-line road", "9ab6805b7c": "Open gravel flat",
}


def clean(x, root: str):
    if isinstance(x, dict):
        return {k: clean(v, root) for k, v in x.items() if not DROP_KEYS.search(k)}
    if isinstance(x, list):
        return [clean(v, root) for v in x]
    if isinstance(x, str):
        x = x.replace(root, "${COSMOS_SHARED}")
        return re.sub(r"[a-z0-9-]+\.[a-z0-9-]+\.c\.[a-z0-9-]+\.internal", "<host>", x)
    return x


def copy_json(src: Path, dst: Path, root: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(clean(json.loads(src.read_text()), root), indent=1))


def copy_jsonl(src: Path, dst: Path, root: str) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    rows = [clean(json.loads(l), root) for l in src.read_text().splitlines() if l.strip()]
    assert all(r.get("identity", {}).get("guardrails", True) for r in rows), f"guardrails-off rows in {src}"
    dst.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="the runs/ directory of the original study")
    ap.add_argument("--media", required=True, help="folder with v/<scene>/*.mp4 and index.json (replay export)")
    ap.add_argument("--results", required=True, help="the study's results/ folder (benchmark, dataset, report)")
    ap.add_argument("--serve-session", default=None, help="a guardrails-on Serve session folder, for its timings")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    runs, out = Path(a.runs), Path(a.out)
    root = str(runs.parent)
    if out.exists():
        shutil.rmtree(out)
    res = out / "results"

    # media
    idx = json.loads((Path(a.media) / "index.json").read_text())
    shutil.copytree(Path(a.media) / "v", out / "media")
    v3 = {json.loads(l)["identity"]["clip_sha256"][:10]: json.loads(l) for ns in ("conditions_v3a", "conditions_v3b")
          for l in (runs / ns / "main" / "manifest.jsonl").read_text().splitlines() if l.strip()}
    scenes = {sid: {"label": NAMES.get(sid, s["run_id"]), "run_id": s["run_id"], "clip_id": s["clip_id"],
                    "clip_sha256": v3[sid]["identity"]["clip_sha256"], "conditions": s["conds"]}
              for sid, s in idx["scenes"].items()}
    (out / "media" / "index.json").write_text(json.dumps({"scenes": scenes, "prompts": idx["prompts"]}, indent=1))

    # results
    for ns, name in (("main/main", "main"), ("conditions_v3a/main", "conditions_v3a"),
                     ("conditions_v3b/main", "conditions_v3b"), ("pilot/main", "pilot"),
                     ("depth_probe/main", "depth_probe")):
        copy_jsonl(runs / ns / "manifest.jsonl", res / name / "manifest.jsonl", root) if (runs / ns / "manifest.jsonl").exists() else None
        copy_json(runs / ns / "pool_summary.json", res / name / "pool_summary.json", root)
    copy_json(runs / "depth_probe/main/probe_result.json", res / "depth_probe" / "probe_result.json", root)
    q = runs / "main/main/quality"
    copy_json(q / "quality_summary.json", res / "main" / "quality" / "quality_summary.json", root)
    shutil.copy(q / "condition_summary.csv", res / "main" / "quality" / "condition_summary.csv")
    import pandas as pd
    qt = pd.read_parquet(q / "quality_table.parquet")
    qt = qt.drop(columns=[c for c in qt.columns if DROP_KEYS.search(c) or c == "triptych_path"])
    for c in qt.columns:
        if qt[c].dtype == object:
            qt[c] = qt[c].map(lambda v: clean(v, root) if isinstance(v, str) else v)
    qt.to_parquet(res / "main" / "quality" / "quality_table.parquet")
    for p in sorted((runs / "benchmark").iterdir()):
        n = p.name.rsplit("-", 1)[-1]          # a1, a2, a4, a8
        copy_jsonl(p / "manifest.jsonl", res / "benchmark" / n / "manifest.jsonl", root)
        copy_json(p / "pool_summary.json", res / "benchmark" / n / "pool_summary.json", root)
    copy_json(Path(a.results) / "benchmark" / "scaling.json", res / "benchmark" / "scaling.json", root)
    for j in ("main", "v3", "sweep"):
        copy_json(runs / "judge" / j / "summary.json", res / "judge" / j / "summary.json", root)
    copy_json(Path(a.results) / "report" / "saturation_check.json", res / "main" / "quality" / "saturation_check.json", root)
    for f in ("fetch_main.json", "prepare_main.json", "prepare_conditions_v3a.json", "prepare_realism_b.json"):
        copy_json(Path(a.results) / "dataset" / f, res / "dataset" / f, root)
    shutil.copy(Path(a.results) / "dataset" / "contact_sheet.jpg", res / "dataset" / "contact_sheet.jpg")
    if a.serve_session:
        s = Path(a.serve_session)
        for f in sorted((s / "jobs").glob("*.json")):
            copy_json(f, res / "serve" / "jobs" / f.name, root)
        for f in sorted((s / "records").glob("*.json")):
            rec = json.loads(f.read_text())
            assert rec.get("guardrails") is True, f"{f} is not a guardrails-on record"
            copy_json(f, res / "serve" / "records" / f.name, root)
    total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    leaks = [str(f) for f in out.rglob("*.json*") if re.search(r"\.internal|10\.\d+\.\d+\.\d+", f.read_text())]
    qt_text = pd.read_parquet(res / "main" / "quality" / "quality_table.parquet").astype(str).to_string()
    if re.search(r"\.internal|10\.\d+\.\d+\.\d+|/mnt/cluster_storage", qt_text):
        leaks.append("quality_table.parquet")
    assert not leaks, f"host names or IPs left in {leaks[:3]}"
    print(f"bundle: {out} ({total / 1e6:.0f} MB, {sum(1 for _ in out.rglob('*') if _.is_file())} files)")


if __name__ == "__main__":
    main()
