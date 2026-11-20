# AGENTS.md: handoff for agents working on cosmos-offroad-ray

Read this first, then `docs/STATUS.md` (ledger and phase results). Last updated 2026-11-20.
**If you are running on the B200 cloud, jump to "Phase 5" at the bottom first.**

## What this project is

"Scaling Off-Road Video Augmentation with Ray and NVIDIA Cosmos Transfer 2.5". Real ATV footage
from **TartanDrive 2.0** (CC BY 4.0; never call it footage we collected, never call it an RC car,
never invent locations) is turned into 93-frame clips; **Cosmos Transfer 2.5 edge/distilled**
re-renders each clip under fog / rain / night / snow prompts. Ray Data does preprocessing and
quality metrics; Ray Core persistent GPU actors do generation; runs on Anyscale.

Deliverables still owed: blog draft `docs/blog.md`, 6 to 8 slide deck in `docs/presentation/`,
`docs/demo_script.md` plus a recorded demo, `docs/{architecture,methodology,limitations}.md`,
static HTML results report, `results/showcase/` (GIFs, before/after, condition grid, multi-scene
grid, a failure example: the user explicitly wants this to look impressive), README,
`environments/Dockerfile`, `.gitignore`, Anyscale Job config.

## Hard rules from the user (do not break)

- Ask before: provisioning resources, changing machine pools/permissions/network, touching host
  NVIDIA drivers, paid external APIs, exceeding limits, publishing or pushing to a remote,
  terminating the shared cluster. Do not publish externally.
- Limits: GPU-phase wall time **3.5 h**, attempts **600** (raised 2 h -> 2.5 h -> 3.0 h -> 3.5 h, 500 -> 600, by the user on 2026-11-18; counted as
  elapsed time while any GPU pool runs, union of intervals), **500** generation attempts, **100 GB** new source data.
  If a phase would exceed a limit, stop before it and ask.
- Never print credentials (filter env vars by NAME). Never bake HF/cloud creds into images.
- No checkpoints, model init or GPU generation on the CPU head (head has 0 Ray CPUs anyway).
- Never delete a working env; never overwrite source used by running workers.
- Never change seeds to dodge guardrails; never silently lower resolution/steps after OOM.
- All reported results use **guardrails ON**. The `smoke` namespace (guardrails OFF) is never
  evaluated or shown.
- Never invent human reviews; never fabricate accepted-clip throughput.
- Do not migrate the main pipeline to Cosmos 3. Exception (user, 2026-11-20): a side-by-side Cosmos 3
  transfer COMPARISON on the B200 cloud is approved (Phase 5); results stay a separate namespace. Do not commit large video into git. Cost: report node-hours only
  (pricing not verified).
- Blog style: no em dashes, no tilde approximations, no hype words, no "One X, Four Y" titles,
  no "This is not X. It is Y." framing.

## Cluster and environment (verified)

- Ray 2.60.0, Python 3.13.18 at `/home/ray/anaconda3`; `ray.init(address="auto")`.
- **8 separate GPU nodes, each 1x A100-SXM4-80GB**, 12 Ray CPUs, 167 GB RAM, local NVMe.
  Head: 0 Ray CPUs. Worker-only resource `anyscale/accelerator_shape:1xA100-80G`.
- `/mnt/cluster_storage` = shared NFS (all paths below). The head sees new NFS paths late
  (negative lookup cache): retry a listing before concluding a file is missing.
- Driver venv: `.venv` (system site packages over Ray). Always run `.venv/bin/python -m offroad_ray.cli ...`.
- Cosmos env (isolated): `/mnt/cluster_storage/cosmos-offroad/envs/ct25-v1.5.4-45c6f0fb-cu130-py3.13`
  (READY.json). Cosmos Transfer 2.5 tag v1.5.4 commit `45c6f0fb4c21`, torch 2.9.1+cu130.
  Only intentional override: PyPI `decord==0.6.0`.
- Source checkout: `/mnt/cluster_storage/cosmos-offroad/src/cosmos-transfer2.5-45c6f0fb4c21`.
- HF cache: `/mnt/cluster_storage/cosmos-offroad/hf_home` (token file there, mode 600; never cat it).
  Staged (see `logs/stage_checkpoints.json`): Transfer2.5-2B edge distilled @bbaeedb,
  Predict2.5-2B tokenizer @f176dc9, Predict2.5-2B robot/action-cond @main=85f8ae7 (see gotchas),
  Reason1-7B @3210bec, Guardrail1 @d6d4bfa, Qwen3Guard-Gen-0.6B @fada3b2, siglip-so400m @9fdffc5.
  Actors run with `HF_HUB_OFFLINE=1`.
- **Llama Guard is NOT needed**: v1.5.4 text guardrail = Blocklist + Qwen3Guard.

## Layout

```
configs/      smoke.yaml (guardrails OFF), phase_a.yaml, pilot.yaml, benchmark.yaml, main.yaml, conditions.yaml
environments/ build_cosmos_env.sh, run_build_on_worker.py
jobs/         stage_checkpoints.py, run_remaining.py (unattended driver, see below)
src/offroad_ray/
  config.py ids.py storage.py ray_env.py video_io.py preprocessing.py preflight.py cli.py
  dataset/{download,clips,manifest}.py
  inference/{subprocess_worker,actor,scheduler,mock_actor}.py
  quality/{metrics,review}.py           # review.run_quality NOT yet exercised on real data
  reporting/{montage,benchmark}.py      # report.py (build_report) NOT written yet
tests/        test_core.py, test_scheduler.py (19 pass; `.venv/bin/python -m pytest -q`)
results/      preflight/, dataset/ (contact_sheet.jpg), benchmark/ (filled by Phase C)
docs/         STATUS.md (+ the docs still to write)
```

Run data: `/mnt/cluster_storage/cosmos-offroad/runs/<config name>/<namespace>/`
(`manifest.jsonl`, `records/`, `attempts/`, `outputs/<request_id>/output.mp4`, `events.json`,
`pool_summary.json`, `telemetry/`, `logs/`). Attempt ledger: `runs/_attempt_ledger/`.
Prepared clips: `data/prepared/f55bfa53d072457b/catalog` (27 clips, 9 runs).

## Design in one paragraph

Each `CosmosActor` (Ray actor, 1 GPU, Anyscale env) owns one persistent child process running
`subprocess_worker.py` under the Cosmos interpreter: JSON requests on stdin, replies on a
dedicated fd. The model loads once per actor; the scheduler (`run_pool`) keeps one request in
flight per actor via `ray.wait`, retries infra failures on a replacement actor (never guardrail
blocks), validates every output with ffprobe, and writes per-request records for resume.
Edge controls are precomputed once per clip, losslessly (100% pixel parity with upstream).

## Phase status

| phase | state | key numbers |
|---|---|---|
| smoke (guardrails OFF) | done, labelled, not evaluated | 2/2 ok |
| A (1 GPU, 2 gens, 1 load) | **done** | init 106 s (load 56.6 s); gens 97.8 s and 80.5 s; peak 71 GB |
| B pilot 24 on 8 actors | **done** 24/24, wall 469.5 s, 0 infra failures | 8-way init 143 to 171 s; gens 79 to 101 s |
| C benchmark 1/2/4/8 | queued in driver | identical 24 requests, fresh namespaces, no resume |
| D main 288 | queued in driver | budget-checked first, see below |
| pilot review (metrics) | **done**, rc=0, 0 auto-flags, 0 human labels | edge F1 vs source: snow 0.84, rain 0.81, night 0.66, fog 0.54 |
| report / showcase / docs | not started | |

## Phase 2 (approved by the user 2026-11-18 about 16:00)

User asked for more realistic, "wow" conditions, a VLM judge, and asked about depth. Approved:
limit 3.0 h; realism sweep; VLM judge; regenerate showcase scenes with the winning setting;
depth: **measure ONE clip first**, then ask before any depth study.

- Realism levers on the distilled edge model: sparser edges (`control.preset_edge_threshold`
  high / very_high; the medium map is almost solid grass texture, see
  `results/realism/edge_density_medium_high_veryhigh.jpg`), lower `control_weight`, stronger
  prompts (`configs/conditions_v2.yaml`). Configs: `realism_a` (high, 0.8), `realism_b`
  (very_high, 0.6); both on the 3 pilot clips, seed 101, guardrails ON. Catalogs prepared.
- VLM judge: `quality/vlm_judge.py` + `vlm_judge_worker.py`, CLI `judge`. Cosmos-Reason1-7B in
  the Cosmos env (transformers, flash-attn, greedy, 4 fps, 560 px). BLIND: not told the requested
  condition or real vs synthetic; real source clips judged as calibration. Scores are automated,
  never mixed with human labels. Outputs: `runs/judge/<name>/{judgments.parquet,summary.json}`.
- Depth: only the non-distilled `general/depth` model (35 steps, guidance 3) exists at v1.5.4;
  depth estimator Video-Depth-Anything-Small (Apache-2.0; Large is non-commercial, not used).
  `jobs/depth_probe.py` times one clip (meadows, v2 snow, seed 101).
- Driver `jobs/run_phase2.py` (detached) waits for run_remaining, then realism a/b -> judge
  (7 GPUs) concurrent with depth probe (1 GPU) -> winner rule -> `configs/showcase_v2.yaml`
  (9 scenes x 4 conditions) -> judge main + showcase_v2. Skips any step that would pass 3.0 h.
  Log: `logs/run_phase2.log`, state `logs/run_phase2_state.json`, winner `results/realism/sweep_scores.json`.
- After it: rebuild showcase GIFs with `jobs/build_showcase.py <ns>` (pilot GIFs already in
  `results/showcase/pilot/`), and report the depth timing to the user before any depth study.

## Phase 3 (approved by the user 2026-11-18 about 16:30)

User asked for more impressive, real conditions, submitted as jobs. Approved: all 10 conditions in
`configs/conditions_v3.yaml` (dust_brownout, golden_hour_glare, blue_hour_dusk, blizzard_whiteout,
wildfire_smoke, autumn, flooded_mud, frost_dawn, night_rain, drought_summer) on the 9 showcase
scenes, seed 101, with the phase-2 winning setting. Limits raised to **3.5 h** GPU-phase wall time
and **600** attempts. Submission: **Ray Jobs on this workspace cluster** (no new cloud resources;
Anyscale Jobs were explicitly not chosen). Driver `jobs/run_phase3.py` waits for phase 2, writes
`configs/conditions_v3{a,b}_run.yaml`, submits Ray Jobs `offroad-v3-v3a-*`, `offroad-v3-v3b-*`,
then `offroad-v3-judge-*`, then builds GIFs with `jobs/build_showcase.py` into
`results/showcase/conditions_v3{a,b}/`. Log `logs/run_phase3.log`; `ray job list` shows the jobs.
`ray job status` prints lowercase "succeeded"; match case-insensitively.

## What is running unattended right now

`jobs/run_remaining.py` (detached with nohup/setsid; survives the session). It:
1. waits for the pilot; if the pilot is incomplete it resumes it once;
2. runs `review configs/pilot.yaml` (CPU; failure does not block);
3. runs `benchmark configs/benchmark.yaml --actors 1 2 4 8` -> `results/benchmark/scaling.{json,png}`, `timeline_a*.png`;
4. budget check: used GPU-phase hours (+0.12 h uncounted) vs 2.5 h, projected main from the
   measured 8-actor per-request time. If 288 does not fit, it runs only whole clips that fit
   (`--limit` in multiples of 12) and leaves the rest pending for a later `generate configs/main.yaml`
   (resume reuses validated outputs);
5. runs main, then `review configs/main.yaml`.

Check progress:
```bash
tail -f /mnt/cluster_storage/cosmos-offroad/logs/run_remaining.log
cat /mnt/cluster_storage/cosmos-offroad/logs/run_remaining_state.json
grep '^[0-9][0-9]:' /mnt/cluster_storage/cosmos-offroad/logs/{pilot,benchmark,main}.log | tail
.venv/bin/python -m offroad_ray.cli budget
ps -eo pid,args | grep '[r]un_remaining.py'
```
To stop it: `kill <pid of run_remaining.py>` then kill any `offroad_ray.cli generate|benchmark`
child; the scheduler's actors die with the driver. Do not kill with a `pkill -f` pattern that
matches your own shell command line (that happened once).

## Next steps after the driver finishes

1. Pilot review worked on real data. If the main review crashed (`logs/review_main.log`): fix `quality/review.py` (never run on real data before), rerun `review`.
   Also fix the docstring in `quality/metrics.py` ("vs control edges" should say "vs source Canny").
2. Ask the user for human labels via the static review page (`<ns>/review/index.html`); never invent them.
3. Write `reporting/report.py` (`build_report`) -> static HTML report from manifest, quality
   summary, `results/benchmark/scaling.json` and charts.
4. Build `results/showcase/` with `reporting/montage.py` (small GIFs and JPGs only; full MP4s stay
   on cluster storage). Guardrails-ON namespaces only.
5. Write docs, blog, deck, demo script; update STATUS.md and this file.

## Open questions for analysis

- Night `mean(delta_saturation)` is +132 (other conditions within about 40). Likely HSV saturation
  inflating on very dark pixels; check `quality/metrics.py` before quoting it anywhere.
- Fog has the lowest edge agreement, which is expected (fog hides distant edges); do not report it
  as a failure without looking at the clips.

## Gotchas already hit (do not rediscover)

- `cosmos_oss.init.init_output_dir` and `SetupArguments.output_dir` need a `pathlib.Path`, not str.
- Importing the v1.5.4 experiment registry downloads `nvidia/Cosmos-Predict2.5-2B`
  `robot/action-cond/38c6...pt` at revision `main` via `uvx hf download` (unused by edge/distilled).
  It must be in the cache or offline actors fail at init. Now staged.
- `_control_edge.mp4` (`output_control_edge.mp4`) is written on every run, not only on guardrail blocks.
- HF access checks must fetch a gated file (`.gitattributes`); README/config download even without access.
- Ray Data function `map`: use `compute=ray.data.TaskPoolStrategy(size=N)`, not a concurrency tuple.
- pytest `tmp_path` is head-local: scheduler tests use `/mnt/cluster_storage/cosmos-offroad/tmp/pytest`.
- distilled edge model: exactly 4 steps; guidance ignored; `sigma_max` crashes. Prompt and
  `control_weight` are the levers. 1024x544 input runs at 1280x704 and is resized back.
- 8 actors loading at once from NFS: init about 150 to 170 s vs 106 s alone.

## Phase 2 + 3 outcome (all finished 17:48, 2026-11-18)
- Budget at end: GPU-phase union 3.32 h of 4.0 h (plus about 0.12 h uncounted), attempts 527 of 600, downloads 13.6 GB.
- Realism sweep (Cosmos-Reason1 blind judge, seed-101 subset, about 10-12 clips each): pilot quality 4.38, realism_a 4.30, realism_b 4.30 -> winner pilot (medium edges, weight 1.0), so no showcase_v2. realism_b (very_high edges, weight 0.6) had condition_match 0.92 vs pilot 0.42: worth a human look.
- Sweep judge started 40 s before the lenient-parser fix and was re-merged on CPU at 18:45 (72/75 parsed; old file kept as judgments.pre_reparse.parquet). Winner unchanged.
- Orphaned judge-main from the old phase2 (16:43) overlapped the new one; the new run resumed from its shards. main: 284/288 parsed.
- Depth probe: 1 clip, 35 steps, guidance 3: load 212 s, generate 868 s, peak 65 GB. Edge distilled median is 80 s per request (benchmark), so depth is about 11x slower per clip. Ask the user before any depth study.
- v3 conditions: Ray Jobs offroad-v3-{v3a,v3b,judge}-171832 all SUCCEEDED; 90/90 generated, 87 judged. condition_match is 0 by construction (the judge vocabulary has no names like autumn or dust); use per-condition observed labels instead.
- Showcase GIFs: results/showcase/conditions_v3{a,b} (about 400 MB each) still need compressing before blog/slides.

## Phase 4: live Ray Serve demo + latency work (2026-11-19, session 2)
User approved: Serve demo with guardrails OFF (demo only, "not going to land in anyone's hands"),
try every latency lever except H100s. Namespaces `runs/latency_profile/*` and `runs/serve_demo/*` are
guardrails OFF: never evaluated, never shown as results (same status as smoke). If a recorded demo goes
into the blog, re-record with `--guardrails`.
- Workspace fix: `/home/ray/default/.anyscaleignore` now excludes the big showcase dirs (Ray's working_dir
  upload was 994 MB > 512 MB limit). Copies: `/mnt/cluster_storage/cosmos-offroad/showcase_originals/`.
  `.venv` was missing after the workspace restore and was rebuilt (`uv venv --system-site-packages` + `uv pip install -e '.[test]'`).
- Latency profile (`jobs/profile_latency.py`, results `runs/latency_profile/p20261119_152847/results.json`),
  warm, 1 A100, guardrails off: 720p 74.7 s = DiT 4 steps 44.3 (11 s/step) + VAE encode 9.7 (2 calls) +
  decode 8.1 + other 12.6. 480p 24.1 s but output is a blurry wash (unusable; distilled model is 720p only).
  torch.compile (net + tokenizer aggressive) 69.7 s after warm-ups of 183 s and 89 s. Context parallel across
  nodes is SLOWER (cp2 103.7 s, cp4 97.7 s): 1-GPU nodes, NCCL over the network. Guardrails cost about 5.7 s.
- Encode cache (`engine_options.encode_cache`, subprocess_worker): both VAE encodes depend only on scene +
  resolution. Content-keyed GPU LRU. Verified exact: max abs diff 0.0 on 16 hits. Saves about 10 s (to about 65 s).
- Serve app: `src/offroad_ray/serve/{app.py,ui.html}`, launcher `jobs/serve_demo.py up|down|status|ui`.
  Painter (1 GPU, max_ongoing_requests=1, reuses CosmosActor) + PaintAPI (FastAPI ingress, job API, UI).
  Scene affinity via `@serve.multiplexed` (scene_id as model id) so the per-replica latent cache hits.
  Outputs shared in `runs/serve_demo/gallery/outputs/<rid>` (repeat prompt = instant). GPU time booked by a
  30 s heartbeat into `runs/serve_demo/<session>/pool_summary.json` (undercounts up to 30 s + teardown);
  API refuses work at the GPU/attempt limit. Code ships as an immutable snapshot `runs/serve_demo/code/<session>`
  (Serve rejects local py_modules). UI html is passed via meta (module __file__ in replicas points at the driver path).
- Measured on 8 replicas (session s20261119_153944): cold-up 112 s; 8 scenes in parallel 78 to 82 s wall;
  first request per replica 80 s; with cache hits 70 to 75 s while verify was on.
- Quality note: thunderstorm on medium edges turns grass into white foam; blowing snow reads well.
  Sparser edges (realism_b: very_high, weight 0.6) are the obvious next lever for the live demo.
- Budget after phase 4: GPU-phase 3.57 of 4.0 h, attempts 555 of 600. Serve is DOWN.
- Report and web assets (CPU): `results/report/index.html` (`cli report`), `results/showcase/web/` (122 MB,
  hero/ grids, before/after, failure example) via `python -m offroad_ray.reporting.web_assets`.
  Night saturation: HSV S exaggerates (171 vs 42); quote CIELAB chroma (8.5 -> 15.4). Judge is weak evidence
  (labels rain/snow "overcast", dust "clear_day"); report says so.

## Limits lifted (2026-11-19 20:15)
User: "there is no quota, u can keep going". GPU-phase and attempt limits in `reporting/benchmark.py` set to
1000 h / 1,000,000 (effectively off, still tracked and reported). At that moment: 4.05 h GPU-phase, 574 attempts.
All other hard rules unchanged (ask before provisioning, publishing, etc.).
- Gotcha: Serve replicas import `offroad_ray` from the py_modules shipped by the job that FIRST started the Serve
  controller (it comes before the PYTHONPATH snapshot), so runtime imports in replicas can be hours stale.
  Limits therefore reach the app through `meta` (`gpu_limit_h`, `attempt_limit`). Deployment classes themselves are
  pickled from the driver, so app.py changes do take effect.
- New guided UI (2026-11-19 evening): autoplaying before/after preview at the top (`runs/serve_demo/peek/*.mp4`,
  served at `/api/peek/`), 3 steps that unlock in order, a "what's happening now" panel (phase is estimated
  from elapsed vs typical time), and plain scene names (`NAMES` in ui.html) with the dataset label as small print.

## Blog site (2026-11-19 evening)
- `docs/blog/index.html` blog preview (Anyscale template format, metadata block at top), `docs/blog/blog.md` CMS copy,
  `docs/blog/replay/` static interactive replay (9 scenes x 10 conditions, guardrails ON, native 1024x544, CRF 20),
  `docs/blog/assets/` hero/four_worlds/nine_drives/wipes (MP4 + `gif/`), click-through recording, charts.
- Click-through recorded with Playwright (scratchpad venv; needs `LD_LIBRARY_PATH` to a conda prefix with atk/at-spi2)
  against the live app with `--guardrails` (session s20261119_205340: 8 drives, 100.9 s click to last clip).
- Facts verified in scratchpad fact_sheet (main run is 24 clips x 4 x 3 = 288; node-hours 26.55/17.40 = through phase 3).
- Site zip for Cloudflare Pages: `/home/ray/default/paint-the-trail-site.zip` (excluded in .anyscaleignore, as are blog media dirs).

## Anyscale template (2026-11-19 late)
- `templates/cosmos-video-augmentation/`: README (+ipynb), notebooks 01/02/03 (CPU path executed, outputs saved),
  vendored `offroad_ray/` (COSMOS_SHARED / COSMOS_ENV_ROOT / OFFROAD_WORKER_RESOURCE env-configurable), Serve app
  with GPU `Painter` and CPU `ReplayPainter`, Dockerfile, jobs/{job,service}.yaml, tests/test_template.py (6 pass).
- Notebooks are generated by `scripts/build_notebooks.py`; data bundle by `scripts/make_bundle.py`
  (sanitized; tar at /mnt/cluster_storage/cosmos-offroad/cosmos-video-augmentation-bundle-v1.tar, needs public hosting).
- Test data installed at /mnt/user_storage/cosmos-video-augmentation. `templates/PUBLISHING.md` lists what is
  verified, what is not (GPU path in template form, Docker build) and the /template and /register-template steps.

## Phase 5: B200 cloud, more depth gens + Cosmos 3 transfer comparison (asked 2026-11-20)
User has B200s on a different cloud (not this Anyscale workspace) and wants (a) more depth-controlled
generations to compare against edge, and (b) the newest Cosmos transfer model for comparison. The Phase 2
rule "ask before any depth study" is lifted for this B200 work; all other hard rules still apply
(guardrails ON for anything reported, ask before provisioning/publishing, never copy the HF token).
Still confirm the exact plan (clips x conditions x seeds per model) with the user before a large batch.

### Getting there (bundle made on the Anyscale side)
`bash jobs/make_b200_bundle.sh` (use `--dry-run` for sizes: code 147 MB, assets 52 GB) writes to
`/mnt/cluster_storage/cosmos-offroad/transfer/`: `offroad-code.tar.gz`, `offroad-assets.tar`, `SHA256SUMS`.
Excluded on purpose: HF token, `envs/` (venv not relocatable), raw TartanDrive bags, guardrails-OFF runs.
**Chosen: `--slim` (user, 2026-11-20): 3.6 GB** = `data/prepared` + guardrails-ON comparison runs (v3a/v3b, pilot,
depth_probe, judge, realism, phase_a, attempt ledger, logs). NO weights: on the B200 cluster run `jobs/stage_checkpoints.py`
(pinned revisions; edit its hard-coded ROOT/PY if needed). Gaps in that list: the depth checkpoint
(`nvidia/Cosmos-Transfer2.5-2B` general/depth @dea7737) and Video-Depth-Anything-Small are fetched on first use by
`depth_probe.py` (not offline); `Cosmos-Predict2.5-2B` action-cond is pinned to `main` upstream and may have moved;
Qwen3Guard and siglip are unpinned upstream (compare resolved shas with `logs/stage_checkpoints.json` from the bundle).
`runs/main` (4.5 GB) and `runs/benchmark` are not in the slim bundle.
On the B200 box:
```bash
sha256sum -c SHA256SUMS
sudo mkdir -p /mnt/cluster_storage/cosmos-offroad && sudo chown $USER /mnt/cluster_storage/cosmos-offroad
# (or put it on a big local NVMe and symlink /mnt/cluster_storage/cosmos-offroad -> there)
tar -C /mnt/cluster_storage/cosmos-offroad -xf offroad-assets.tar
tar -C ~ -xzf offroad-code.tar.gz && cd ~/cosmos-offroad-ray
```
Keep the root at `/mnt/cluster_storage/cosmos-offroad`: `config.DEFAULT_ROOT` and every catalog parquet
(`clip_path`, `control_path`) hold absolute paths under it. Then:
- `hf auth login` (new token, never copied from the old box). Gated repos used: Cosmos-Transfer2.5-2B,
  Guardrail1, Cosmos 3 checkpoints.
- Docker image `environments/Dockerfile` (base `anyscale/ray:2.60.0-py313-cu130`, verified on Docker Hub): OS deps,
  orchestration deps from pyproject, build-time checks. NO Cosmos envs inside: build them onto the mounted root once
  (same as the A100 cluster; they persist across containers). Not built yet (no Docker in the Anyscale workspace).
  ```bash
  docker build -f environments/Dockerfile -t offroad-cosmos:b200 environments/   # no COPY: also valid for the Anyscale image builder
  docker run --gpus all --ipc=host --shm-size=64g -it \
    -v /mnt/cluster_storage/cosmos-offroad:/mnt/cluster_storage/cosmos-offroad \
    -v ~/cosmos-offroad-ray:/home/ray/project -w /home/ray/project offroad-cosmos:b200 bash
  # inside, once:
  UV_CACHE_DIR=/tmp/uv-cache bash environments/build_cosmos_env.sh   # Cosmos Transfer 2.5 (edge + depth), READY.json
  bash environments/build_cosmos3_env.sh      # Cosmos 3 (diffusers pinned via DIFFUSERS_REF), READY.json
  uv venv --system-site-packages .venv && uv pip install -e '.[test]'
  hf auth login
  ray start --head --resources='{"anyscale/accelerator_shape:1xA100-80G": 8}'   # see below
  ```
  Container user is `ray` (uid 1000): the mounted root must be writable by it. Host driver >= 580 (CUDA 13).

### Things that assume the Anyscale A100 cluster (fix before running)
- `ray_env.WORKER_ONLY` and `preflight.py` request `anyscale/accelerator_shape:1xA100-80G`. On a plain
  multi-GPU B200 node with a local `ray start --head`, that resource does not exist and tasks hang pending.
  Make it configurable (env var) or empty there; do not break the A100 default.
  Zero-code workaround: start Ray with that custom resource declared (`ray start --head --resources=...` above);
  it is only a label, the real limit is still `num_gpus`. Fine for a first run; still label results as B200.
- Workspace was 8 nodes x 1 GPU over NFS; a B200 box is likely 8 GPUs in one node (NVLink). Context parallel
  was slower here only because of cross-node NCCL; on one 8xB200 node it may help. Measure, don't assume.
- Report text and benchmark axis labels say "one A100 each": label B200 numbers as B200, never mix them into
  the A100 scaling chart. Keep node-hours per hardware type separate. Cost: node-hours only.
- `jobs/depth_probe.py` is the working depth recipe (model="depth", 35 steps, guidance 3, Video-Depth-Anything-Small
  computed on the fly, staged in `hf_home/cosmos_depth_models`; checkpoint `Cosmos-Transfer2.5-2B` general @dea7737).
  A100 baseline: load 212 s, generate 868 s, peak 65 GB (`runs/depth_probe/main/probe_result.json`). Re-run that
  one clip first on B200 to get a like-for-like timing before scaling out.

### (a) More depth gens
Write a depth namespace (e.g. `runs/depth_b200/<ns>`) using the same clip/prompt/seed keys as the edge runs
so pairs line up: start with the 9 showcase scenes x `conditions_v3` (seed 101, matches `runs/conditions_v3{a,b}`),
then pilot clips x conditions_v2. Precompute depth controls once per clip (like edge) instead of on the fly.
Compare with: the blind Cosmos-Reason1 judge (`cli judge`, same rules: blind, automated, never mixed with human
labels), edge F1 vs source Canny, CIELAB chroma (not HSV S), and per-request time. Depth will not match source edges
by construction, so do not read lower edge F1 as worse quality.

### (b) "Cosmos Transfer 3" = Cosmos 3 (verify before coding)
There is no standalone "Cosmos Transfer 3". NVIDIA released Cosmos 3 (HF collection `nvidia/cosmos3`, repo
github.com/nvidia/cosmos, June 2026); `cosmos-transfer2.5` is now in maintenance mode. Transfer is part of the
Cosmos 3 family. Per the repo README as read on 2026-11-20 (unverified in code): `Cosmos3-Nano-Sim-Transfer`
("auto-regressive transfer - depth, segmentation, edge, blur") and `Cosmos3-Nano-Transfer-Auto` (multi-view
driving); base sizes Edge 4B / Nano 16B / Super 64B; Nano lists B200 as supported; license OpenMDW 1.1; Python 3.13 + uv,
diffusers from git main. Generation docs mention 256p/480p, 50 to 150 frames: our clips are 93 frames at
1024x544 run at 720p, so check the transfer checkpoint's real resolution/frame limits in `docs/reference/models.md`
and `COOKBOOK.md` before committing to a comparison, and report any resolution difference next to every result.
Rules: separate isolated env (never touch the 2.5 env), pin an exact commit + checkpoint revision (record both in a
READY.json like the 2.5 env), guardrails ON, same clips/prompts/seeds as the 2.5 runs, edge control from our
precomputed `control_edge.mp4` where the model accepts it (and depth where it does). Results go in their own
namespace (`runs/cosmos3_transfer/<ns>`), never overwrite 2.5 results. Note the different licenses in the blog.
Pick the checkpoint with the user if it is ambiguous (Nano-Sim-Transfer is the closest match to our use).
