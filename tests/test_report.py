"""CPU tests for the static HTML report (no Ray, no video decoding)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from offroad_ray.reporting.report import build_report, stage_timing
from offroad_ray.reporting.web_assets import scene_name


def _rec(i: int, chunk: float) -> dict:
    st = {"text_guardrail": 0.6, "get_text_embeddings": 0.1, "preprocessing": 1.0, "generate_chunk": chunk,
          "postprocessing": 4.0, "video_guardrail": 5.0}
    st["generate_img2world"] = st["get_text_embeddings"] + st["preprocessing"] + chunk + st["postprocessing"] + 0.5
    child = st["text_guardrail"] + st["generate_img2world"] + st["video_guardrail"] + 3.0
    return {"request_id": f"r{i}", "status": "generated_unreviewed", "stage_times": st,
            "child_seconds": child, "validate_s": 0.9}


def _fake_runs(root: Path) -> Path:
    ns = root / "runs" / "main" / "main"
    (ns / "records").mkdir(parents=True)
    for i, chunk in enumerate([60.0, 62.0, 64.0]):
        (ns / "records" / f"r{i}.json").write_text(json.dumps(_rec(i, chunk)))
    (ns / "pool_summary.json").write_text(json.dumps({
        "num_actors": 2, "requests": 3, "wall_s": 200.0, "infra_failures": 0, "actor_replacements": 0,
        "settings": {"guardrails": True}, "status_counts": {"generated_unreviewed": 3}}))
    smoke = root / "runs" / "smoke" / "main"
    smoke.mkdir(parents=True)
    (smoke / "pool_summary.json").write_text(json.dumps({"num_actors": 1, "requests": 2, "wall_s": 10.0,
                                                         "settings": {"guardrails": False}, "status_counts": {}}))
    return ns


def test_stage_timing_medians_and_nesting(tmp_path):
    ns = _fake_runs(tmp_path)
    t = stage_timing(ns)
    assert t["n"] == 3
    assert t["median_s"]["generate_chunk"] == pytest.approx(62.0)
    assert t["img2world_other_s"] == pytest.approx(0.5)
    assert t["untimed_s"] == pytest.approx(3.0)


def test_build_report_writes_html_with_all_sections(tmp_path):
    ns = _fake_runs(tmp_path)
    budget = {"gpu_phase_wall_time_h": {"used": 1.0, "limit": 4.0}, "generation_attempts": {"used": 3, "limit": 600},
              "source_download_gb": {"used": 1.0, "limit": 100},
              "pools": [{"ns": "main/main", "num_actors": 2, "wall_s": 200.0},
                        {"ns": "smoke/main", "num_actors": 1, "wall_s": 10.0}]}
    sat = {"seed": 101, "stride": 8, "v_min": 50, "n_outputs": 1, "n_sources": 1,
           "by_condition": {"source": {"hsv_s": 40, "hsv_s_bright": 36, "lab_chroma": 8.0, "v_mean": 120, "dark_frac": 0.1},
                            "night": {"hsv_s": 170, "hsv_s_bright": 185, "lab_chroma": 15.0, "v_mean": 48, "dark_frac": 0.6}}}
    out = build_report(None, ns, out_dir=tmp_path / "report", runs_dir=tmp_path / "runs",
                       results_dir=tmp_path / "results", budget=budget, saturation=sat)
    doc = out.read_text()
    for sid in ("dataset", "phases", "scaling", "stages", "quality", "judge", "showcase", "budget", "limitations"):
        assert f"id={sid}" in doc
    assert "62.00" in doc                      # generate_chunk median
    assert "not from people" in doc            # judge labelled as automated
    assert "1.9x" in doc                       # chroma ratio 15/8
    assert "smoke/main (guardrails off, not evaluated)" in doc
    assert "—" not in doc and "~" not in doc and "$" not in doc


def test_build_report_refuses_guardrails_off(tmp_path):
    _fake_runs(tmp_path)
    with pytest.raises(ValueError):
        build_report(None, tmp_path / "runs" / "smoke" / "main", out_dir=tmp_path / "r", budget={}, saturation={})


def test_scene_name():
    assert scene_name("2023-11-02-16-00-43_down_meadows__image_left_color__t3.000__715cd636") == "down meadows"
    assert scene_name("sam_loop_2023-09-27-13-04-54") == "sam loop"
