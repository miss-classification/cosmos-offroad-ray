"""CPU checks: the downloaded results reproduce the figures quoted in README.md, and the
replay-mode pieces work without GPUs.

  python tests/test_template.py          (or: pytest tests/test_template.py)
"""
import json
import os
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from offroad_ray.config import DEFAULT_ROOT, load_config  # noqa: E402

ROOT = Path(DEFAULT_ROOT)
RES = ROOT / "results"


def _jsonl(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def test_bundle_present():
    idx = json.loads((ROOT / "media" / "index.json").read_text())
    assert len(idx["scenes"]) == 9
    for sid, s in idx["scenes"].items():
        assert (ROOT / "media" / sid / "src.mp4").exists()
        assert len(s["conditions"]) == 10
        for c in s["conditions"]:
            assert (ROOT / "media" / sid / f"{c}.mp4").exists(), (sid, c)


def test_main_run():
    m = _jsonl(RES / "main" / "manifest.jsonl")
    assert len(m) == 288
    assert all(r["status"] == "generated_unreviewed" and r["attempts"] == 1 for r in m)
    assert all(r["identity"]["guardrails"] for r in m)
    assert round(statistics.median(r["child_seconds"] for r in m), 2) == 80.32
    gen = statistics.median(r["stage_times"]["generate_chunk"] for r in m)
    assert round(gen, 2) == 63.57
    assert round(json.loads((RES / "main" / "pool_summary.json").read_text())["wall_s"], 2) == 3089.64


def test_scaling():
    s = json.loads((RES / "benchmark" / "scaling.json").read_text())
    by_n = {r["num_actors"]: r for r in s["scaling"]}
    assert by_n[8]["speedup_gen"] == 6.976 and by_n[8]["efficiency_gen"] == 0.872
    assert by_n[8]["speedup_e2e"] == 5.19
    med = {p["num_actors"]: p["per_request_child_s_median"] for p in s["points"]}
    assert med[1] == 79.79 and med[8] == 80.27


def test_quality():
    import pandas as pd
    cs = pd.read_csv(RES / "main" / "quality" / "condition_summary.csv").set_index("condition")
    f1 = cs["mean(edge_f1)"].round(2).to_dict()
    assert f1 == {"snow": 0.82, "rain": 0.78, "night": 0.64, "fog": 0.53}
    qt = pd.read_parquet(RES / "main" / "quality" / "quality_table.parquet")
    assert int((qt["auto_flags"].fillna("") != "").sum()) == 6


def test_configs():
    for name in ("quickstart", "main", "benchmark"):
        cfg = load_config(HERE / "configs" / f"{name}.yaml")
        assert cfg.generation.guardrails is True
    assert [c.name for c in load_config(HERE / "configs" / "quickstart.yaml").generation.conditions] == \
        ["golden_hour_glare", "night_rain"]


def test_replay_presets_match_bundle():
    from offroad_ray.serve.app import bundle_scenes, load_presets
    presets = {p["name"] for p in load_presets(HERE / "configs")}
    idx = json.loads((ROOT / "media" / "index.json").read_text())
    stored = {c for s in idx["scenes"].values() for c in s["conditions"]}
    assert stored <= presets
    assert len(bundle_scenes(ROOT)) == 9


if __name__ == "__main__":
    fns = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok", fn.__name__)
    print(f"{len(fns)} checks passed against {ROOT}")
