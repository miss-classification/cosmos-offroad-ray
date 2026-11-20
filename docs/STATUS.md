# Project status

Last updated: 2026-11-18 (session 1). Times are PST unless marked UTC.

## Verified environment (discovery, 2026-11-18)

| item | observed |
|---|---|
| Ray | 2.60.0, Python 3.13.18 (`/home/ray/anaconda3`), connected via `address="auto"` |
| head | 16 CPU / 62 GB RAM host; advertises **0 Ray CPUs** (no work can schedule there) |
| GPU workers | **8 separate nodes**, each 1x NVIDIA A100-SXM4-80GB, 12 Ray CPUs, 167 GB RAM, 369 GB local NVMe |
| driver | 580.178.04 (CUDA 13.0 capable); nvcc 13.0 present on workers |
| storage | `/mnt/cluster_storage` NFS (2.5 TB, shared by all nodes, workspace scoped); `/mnt/local_storage` node local NVMe |
| tools | ffmpeg 4.4.2 with libx264, ffprobe, git-lfs, rsync, gcc, uv 0.13.5 |
| torch | not installed in the Anyscale environment (expected; Cosmos runs isolated) |
| cloud | GCP us-central1; pricing not verified, so cost is reported as node-hours |

The brief described one 8-GPU node. The cluster is eight 1-GPU nodes. Consequence: a
1-actor benchmark leaves 7 GPU nodes idle but still allocated; node-hours are counted
for all 8 GPU nodes for the whole experiment window.

## Decisions

- Cosmos Transfer 2.5 pinned to tag v1.5.4, commit `45c6f0fb4c21d4bed8907a1e153326ad4b6686c0`,
  installed with `uv sync --locked --extra cu130` from its committed `uv.lock` into an
  isolated env on shared storage (Option A: actor plus persistent subprocess).
- Model `edge/distilled`, checkpoint `nvidia/Cosmos-Transfer2.5-2B@bbaeedb2b57cc8b14a44653099e2551adb69dcc7`
  (`distilled/general/edge/41f07f13-...`), 4 steps, 93 frames, requires `COSMOS_EXPERIMENTAL_CHECKPOINTS=1`.
- Source: TartanDrive 2.0 KITTI-format export, `image_left_color` (1024x544 PNG, 10 Hz).
- Clips resampled 10 Hz to 16 Hz by nearest-timestamp frame selection (duplication only).
- Edge controls cached at the model resolution (1280x704), losslessly (libx264rgb qp0).
- Internal repo `anyscale/cosmos-transfer25-ray@466abae` became readable mid-session;
  its subprocess protocol, guardrail-status handling and findings were reused.

## Done

- Inventory, GPU probe on all 8 workers via Ray GPU tasks.
- Project skeleton, config, ids, storage, downloader, clip planner, Ray Data prepare
  pipeline (validated on one run: 1 clip, 93 frames, 5.8125 s, lossless control round trip).
- Inference subprocess worker, actor, dynamic scheduler (not yet run on GPU).
- 14 CPU tests passing.

## Validated so far

- Cosmos env READY (`envs/ct25-v1.5.4-45c6f0fb-cu130-py3.13`, uv sync 884 s). Intentional
  override: PyPI `decord==0.6.0` (locked cu130 wheel needs libavformat.so.60, absent).
- `env-check` on an A100: cuDNN Conv3D, flash-attn (sm_80), TransformerEngine, natten, decord
  decode of a real clip, and edge-control parity with the upstream path (100.000% pixel agreement).
- 9 runs fetched (13.39 GB, 83 s). Ray Data prepare: 27 clips, all 8 worker nodes, 26.6 s with
  18 cache hits; contact sheet at `results/dataset/contact_sheet.jpg`.
- 19 tests pass (15 CPU, 4 mocked-actor scheduler tests on the live cluster).
- `nvidia/Cosmos-Reason1-7B@3210bec` staged to the shared HF cache (16.6 GB).

## Blocked

**Hugging Face token.** Real weight-file downloads return 401 for three gated repos:
`nvidia/Cosmos-Transfer2.5-2B`, `nvidia/Cosmos-Predict2.5-2B`, `nvidia/Cosmos-Guardrail1`.
Correction: `meta-llama/Llama-Guard-3-8B` is NOT needed. v1.5.4's text guardrail is Blocklist +
`Qwen/Qwen3Guard-Gen-0.6B` (open); the llamaGuard3 module is unreferenced. Correction recorded: an earlier check that fetched README.md
reported false passes; `hf-check` now fetches gated files.
Agreed plan once the token works: 2-request guardrails-OFF smoke test (labelled, not evaluated),
then Phase A with guardrails ON, then pilot, benchmark and main with guardrails ON only.

## Phase results so far

- Smoke (guardrails OFF, labelled, not evaluated): 2/2 generated, 1024x544, 93 frames, 16 fps.
  Two earlier init failures were fixed: Path type for output_dir, and an import-time download of
  `Predict2.5-2B/robot/action-cond` (now staged; upstream `main` = 85f8ae7).
- Phase A (guardrails ON, 1 actor, offline): init 106 s (load 56.6 s), then 2 generations in the
  same actor: 97.8 s (first, includes guardrail warm-up) and 80.5 s; peak 71.0 / 47.2 GB.

## Budget decision (2026-11-18, user)

Estimated pilot + full benchmark + main 288 = about 125 min of elapsed GPU-phase time, over
the 2 h limit. The user approved raising the GPU wall-time limit to **2.5 h** (elapsed time of
GPU phases, as previously reported), keeping the full 1/2/4/8 benchmark and main 288.

## Budget decision 2 (2026-11-18 about 16:00, user)

GPU-phase limit raised to **3.0 h** for: realism sweep (24 gens), Cosmos-Reason1 VLM judge,
showcase regeneration with the winning setting (36 gens), and a single-clip depth timing probe.
Depth study beyond one clip needs a new decision. See AGENTS.md "Phase 2".

## Resource ledger

| resource | used | limit |
|---|---|---|
| generation attempts | 4 | 500 |
| new source download | 13.63 GB | 100 GB |
| checkpoints staged | 16.6 GB | n/a |
| GPU-phase wall time | see `offroad_ray.cli budget` (elapsed union of pool intervals) | 3.0 h (raised twice by user) |

## Next actions (in order)

1. `python -m offroad_ray.cli hf-check configs/pilot.yaml` must PASS all five repos.
2. `python jobs/stage_checkpoints.py` (one worker stages the rest; actors then run offline).
3. Smoke test, then Phase A, pilot, benchmark, main, review, report, showcase GIFs.
