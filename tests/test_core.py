"""CPU tests: config, ids, resampling, windowing, manifest parsing, storage, resume."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from offroad_ray.config import ExperimentConfig, load_config
from offroad_ray.dataset.clips import resample_nearest, select_windows, validate_window
from offroad_ray.dataset.manifest import build_run_manifest, parse_filled_frames
from offroad_ray.ids import clip_id, request_id, stable_hash
from offroad_ray.storage import atomic_write_json, make_owned_dir, safe_rmtree

PROJECT = Path(__file__).resolve().parents[1]


def _cfg(**kw):
    base = {"name": "t", "dataset": {"runs": ["r1"]},
            "generation": {"conditions": [{"name": "fog", "prompt": "fog"}], "seeds": [1]}}
    base.update(kw)
    return ExperimentConfig.model_validate(base)


# ---------------------------------------------------------------- config
def test_shipped_configs_validate():
    for name in ("pilot", "benchmark", "main"):
        p = PROJECT / "configs" / f"{name}.yaml"
        if p.exists():
            cfg = load_config(p)
            assert cfg.clips.num_frames == 93
            assert cfg.generation.num_steps == 4


def test_config_rejects_wrong_steps_and_unknown_fields():
    with pytest.raises(Exception):
        _cfg(generation={"conditions": [{"name": "a", "prompt": "a"}], "num_steps": 35})
    with pytest.raises(Exception):
        _cfg(bogus=1)
    with pytest.raises(Exception):
        _cfg(clips={"num_frames": 121})
    with pytest.raises(Exception):
        _cfg(generation={"conditions": [{"name": "a", "prompt": "x"}, {"name": "a", "prompt": "y"}]})


# ------------------------------------------------------------------- ids
def _rid(**over):
    kw = dict(clip_checksum="c" * 64, prep_hash="p", model="edge/distilled", cosmos_commit="x",
              checkpoint_revision="y", prompt="fog", seed=1, control_weight=1.0, num_steps=4,
              resolution="720", guardrails=True, code_version="gen-v1")
    kw.update(over)
    return request_id(**kw)


def test_request_id_stable_and_sensitive():
    assert _rid() == _rid()
    for k, v in [("seed", 2), ("prompt", "rain"), ("control_weight", 0.5), ("guardrails", False),
                 ("prep_hash", "q"), ("clip_checksum", "d" * 64), ("checkpoint_revision", "z")]:
        assert _rid(**{k: v}) != _rid(), k
    assert _rid(control_weight=1) == _rid(control_weight=1.0)


def test_stable_hash_key_order_invariant():
    assert stable_hash({"a": 1, "b": 2}) == stable_hash({"b": 2, "a": 1})


def test_clip_id_stable():
    a = clip_id("td2", "run", "cam", 7.925, 93, 16.0)
    assert a == clip_id("td2", "run", "cam", 7.925, 93, 16.0)
    assert a != clip_id("td2", "run", "cam", 8.0, 93, 16.0)
    assert a.startswith("run__cam__t7.925__")


# ------------------------------------------------------------ resampling
def test_resample_preserves_duration_and_maps_each_frame():
    src = 1000.0 + np.arange(200) * 0.1  # 10 Hz
    rs = resample_nearest(src, src[0] + 1.0, 93, 16.0)
    assert len(rs.source_index) == 93
    # physical span of targets equals (N-1)/fps
    assert rs.target_times[-1] - rs.target_times[0] == pytest.approx(92 / 16.0)
    assert rs.abs_error_s.max() <= 0.05 + 1e-9
    assert rs.max_reuse <= 2
    # 93 targets over ~5.75 s of 10 Hz source -> 58 or 59 unique frames, rest duplicates
    assert 57 <= len(np.unique(rs.source_index)) <= 59
    assert rs.duplicates == 93 - len(np.unique(rs.source_index))
    assert np.all(np.diff(rs.source_index) >= 0)


def test_resample_identity_when_rates_match():
    src = np.arange(100) / 16.0
    rs = resample_nearest(src, 0.0, 93, 16.0)
    assert np.array_equal(rs.source_index, np.arange(93))
    assert rs.duplicates == 0


def test_resample_rejects_bad_timestamps():
    with pytest.raises(ValueError):
        resample_nearest(np.array([0.0, 0.1, 0.1, 0.2]), 0.0, 2, 16.0)
    with pytest.raises(ValueError):
        resample_nearest(np.arange(10) * 0.1, 0.5, 93, 16.0)  # beyond end


def test_validate_window_flags_gap_and_invalid_frames():
    src = np.arange(300) * 0.1
    valid = np.ones(300, bool)
    ok, reasons, _ = validate_window(src, valid, 1.0, 93, 16.0, 0.25, 3)
    assert ok, reasons
    gap = src.copy()
    gap[30:] += 1.0  # 1.1 s gap after frame 29
    ok, reasons, _ = validate_window(gap, valid, 1.0, 93, 16.0, 0.25, 3)
    assert not ok and any("gap" in r for r in reasons)
    bad = valid.copy()
    bad[40] = False
    ok, reasons, _ = validate_window(src, bad, 1.0, 93, 16.0, 0.25, 3)
    assert not ok and any("invalid" in r for r in reasons)


def test_select_windows_non_overlapping_and_skips_corrupt():
    src = 100.0 + np.arange(600) * 0.1  # 60 s
    valid = np.ones(600, bool)
    valid[60:70] = False  # corrupt frames near the start
    chosen, rejected = select_windows(src, valid, num_frames=93, fps=16.0, clips_per_run=3,
                                      start_offset_s=3.0, min_gap_s=2.0, max_gap_s=0.25, max_reuse=3)
    assert len(chosen) == 3
    starts = [c[0] for c in chosen]
    assert all(b - a >= 93 / 16 + 2.0 - 1e-6 for a, b in zip(starts, starts[1:]))
    for t0, rs in chosen:
        assert valid[rs.source_index].all()
    assert rejected, "corrupt-frame windows must be recorded, not silently dropped"


def test_bad_timestamp_elsewhere_does_not_poison_run():
    src = 100.0 + np.arange(600) * 0.1
    src[500] = src[499]  # duplicate stamp late in the run
    valid = np.ones(600, bool)
    valid[500] = False
    chosen, rejected = select_windows(src, valid, num_frames=93, fps=16.0, clips_per_run=3,
                                      start_offset_s=3.0, min_gap_s=2.0, max_gap_s=0.25, max_reuse=3)
    assert len(chosen) == 3
    for _, rs in chosen:
        assert 500 not in rs.source_index
        assert np.all(np.diff(src[rs.source_index]) >= 0)


def test_select_windows_short_run():
    src = np.arange(30) * 0.1
    chosen, rejected = select_windows(src, np.ones(30, bool), num_frames=93, fps=16.0, clips_per_run=3,
                                      start_offset_s=0.0, min_gap_s=2.0, max_gap_s=0.25, max_reuse=3)
    assert chosen == [] and rejected


# -------------------------------------------------------------- manifest
LOG = """
   -- /multisense/left/image_rect_color filled missing frame 12 <- 11: 1694706339.2
   -- /multisense/left/image_rect filled missing frame 13 <- 12: 1694706339.3
   -- /wheel_rpm filled missing frame 57 <- 56: 1694706339.2
"""


def test_parse_filled_frames_topic_specific():
    assert parse_filled_frames(LOG, "/multisense/left/image_rect_color") == {12}
    assert parse_filled_frames(LOG, "/multisense/left/image_rect") == {13}


def test_build_run_manifest_flags_missing_filled_and_nonmonotonic(tmp_path):
    run = tmp_path / "runA"
    cam = run / "image_left_color"
    cam.mkdir(parents=True)
    ts = 100 + np.arange(20) * 0.1
    ts[15] = ts[14]  # non-monotonic duplicate stamp
    np.savetxt(cam / "timestamps.txt", ts)
    for i in range(20):
        if i != 5:  # frame 5 missing
            (cam / f"{i:06d}.png").write_bytes(b"x")
    (run / "data_extraction.log").write_text(LOG)
    m = build_run_manifest(run, "image_left_color")
    assert m["num_frames"] == 20
    assert m["missing_files"] == ["000005.png"]
    assert m["exporter_filled"] == [12]
    assert not m["valid"][5] and not m["valid"][12] and not m["valid"][15]
    assert m["valid"].sum() == 17


# --------------------------------------------------------------- storage
def test_atomic_write_and_guarded_delete(tmp_path):
    p = tmp_path / "a" / "b.json"
    atomic_write_json(p, {"x": 1})
    assert json.loads(p.read_text()) == {"x": 1}
    assert not list(p.parent.glob(".*.tmp"))
    owned = make_owned_dir(tmp_path / "root" / "scratch1")
    assert safe_rmtree(owned, [tmp_path / "root"])
    unowned = tmp_path / "root" / "other"
    unowned.mkdir()
    with pytest.raises(ValueError):
        safe_rmtree(unowned, [tmp_path / "root"])
    with pytest.raises(ValueError):
        safe_rmtree(tmp_path / "root", [tmp_path / "root"])  # never the root itself
    with pytest.raises(ValueError):
        safe_rmtree(owned.parent.parent, [tmp_path / "root"])
