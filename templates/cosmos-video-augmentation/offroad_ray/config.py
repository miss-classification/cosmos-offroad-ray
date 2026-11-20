"""Typed experiment configuration loaded from YAML.

Every field that changes generated pixels participates in a stable hash
(see ``ids.py``), so changing it creates new request identities instead of
silently reusing stale outputs.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import pydantic
import yaml

# Shared storage that every node of a Workspace or Job can read. Override with COSMOS_SHARED.
DEFAULT_ROOT = os.environ.get("COSMOS_SHARED", "/mnt/user_storage/cosmos-video-augmentation")


class _Strict(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)


class PathsConfig(_Strict):
    root: str = DEFAULT_ROOT
    """Durable shared root (NFS). Raw data, prepared clips, envs, checkpoints, runs."""
    scratch: str = "/mnt/local_storage/cosmos-video-augmentation-scratch"
    """Node-local scratch. Disposable; bounded by scratch_max_gb."""
    scratch_max_gb: float = 40.0

    @property
    def raw_dir(self) -> Path:
        return Path(self.root) / "data" / "tartandrive2" / "raw"

    @property
    def prepared_dir(self) -> Path:
        return Path(self.root) / "data" / "prepared"

    @property
    def runs_dir(self) -> Path:
        return Path(self.root) / "runs"

    @property
    def hf_home(self) -> Path:
        return Path(self.root) / "hf_home"


class DatasetConfig(_Strict):
    dataset_id: str = "tartandrive2"
    dataset_version: str = "kitti/all_topics export (files.yaml @ castacks/tartan_drive_2.0 fe893b9)"
    camera: str = "image_left_color"
    """KITTI-export tarball stem. Multisense left rectified color camera."""
    runs: list[str]
    max_download_gb: float = 30.0


class ClipConfig(_Strict):
    num_frames: int = 93
    """Model-input frames per clip. Cosmos Transfer 2.5 distilled requires exactly 93."""
    target_fps: float = 16.0
    """Model-input cadence; the edge checkpoints are registered with fps=16."""
    resampling: Literal["nearest"] = "nearest"
    """Temporal resampling policy from source timestamps to the target grid."""
    clips_per_run: int = 3
    min_gap_s: float = 2.0
    """Minimum gap between consecutive clips from the same run."""
    start_offset_s: float = 3.0
    """Skip the first seconds of a run (vehicle often stationary at start)."""
    max_source_gap_s: float = 0.25
    """Reject a window whose source frames have a timestamp gap above this."""
    max_frame_reuse: int = 3
    """Reject if any single source frame is reused more than this many times."""
    crf: int = 12
    """libx264 CRF for the prepared model-input clips (visually near-lossless)."""


class ControlConfig(_Strict):
    kind: Literal["edge"] = "edge"
    preset_edge_threshold: Literal["very_low", "low", "medium", "high", "very_high"] = "medium"
    model_resolution: tuple[int, int] = (1280, 704)
    """(W, H) the pinned Cosmos revision resizes a 1024x544 input to ('720' / '16,9')."""


class Condition(_Strict):
    name: str
    prompt: str


class GenerationConfig(_Strict):
    model: Literal["edge/distilled"] = "edge/distilled"
    cosmos_commit: str = "45c6f0fb4c21d4bed8907a1e153326ad4b6686c0"
    checkpoint_revision: str = "bbaeedb2b57cc8b14a44653099e2551adb69dcc7"
    num_steps: int = 4
    resolution: str = "720"
    control_weights: list[float] = [1.0]
    seeds: list[int] = [1]
    conditions: list[Condition]
    guardrails: bool = True
    keep_input_resolution: bool = True

    @pydantic.field_validator("num_steps")
    @classmethod
    def _distilled_steps(cls, v: int) -> int:
        if v != 4:
            raise ValueError("edge/distilled is trained for exactly 4 sampling steps")
        return v


class SchedulerConfig(_Strict):
    num_actors: int = 8
    cpus_per_actor: float = 4.0
    max_attempts_per_request: int = 2
    """Total attempts for transient infrastructure failures (never for guardrail blocks)."""
    request_timeout_s: float = 900.0
    init_timeout_s: float = 1800.0
    max_actor_restarts: int = 2


class BudgetConfig(_Strict):
    max_generation_attempts: int = 500
    max_gpu_worker_hours: float = 2.0


class ExperimentConfig(_Strict):
    name: str
    paths: PathsConfig = PathsConfig()
    dataset: DatasetConfig
    clips: ClipConfig = ClipConfig()
    control: ControlConfig = ControlConfig()
    generation: GenerationConfig
    scheduler: SchedulerConfig = SchedulerConfig()
    budget: BudgetConfig = BudgetConfig()
    max_clips: int | None = None
    """Cap on prepared clips used for generation (pilot uses a small subset)."""

    @pydantic.model_validator(mode="after")
    def _check(self) -> "ExperimentConfig":
        if self.clips.num_frames != 93 and self.generation.model == "edge/distilled":
            raise ValueError("edge/distilled is intended for exactly 93 frames")
        names = [c.name for c in self.generation.conditions]
        if len(set(names)) != len(names):
            raise ValueError("condition names must be unique")
        if len(set(self.generation.seeds)) != len(self.generation.seeds):
            raise ValueError("seeds must be unique")
        return self


def load_config(path: str | Path, **overrides) -> ExperimentConfig:
    """Load YAML. ``generation.conditions_file`` (relative to the YAML) supplies shared prompts."""
    path = Path(path)
    data = yaml.safe_load(path.read_text())
    data.update(overrides)
    gen = data.get("generation", {})
    if "conditions_file" in gen:
        cf = path.parent / gen.pop("conditions_file")
        gen["conditions"] = yaml.safe_load(cf.read_text())["conditions"]
    only = gen.pop("only_conditions", None)   # optional subset of the conditions file, by name
    if only:
        gen["conditions"] = [c for c in gen["conditions"] if c["name"] in set(only)]
        if len(gen["conditions"]) != len(set(only)):
            raise ValueError(f"unknown condition in only_conditions: {only}")
    return ExperimentConfig.model_validate(data)
