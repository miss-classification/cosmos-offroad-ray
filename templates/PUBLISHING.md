# Publishing `cosmos-video-augmentation`

This folder is the output of Step 1 of the template workflow: a complete template whose CPU path is
execution-verified. Steps 2 and 3 need Buildkite access and checkouts of `anyscale/templates` and
`anyscale/product`, so they run from your machine.

## What was verified (2026-11-19, this workspace)

| Check | Result |
|---|---|
| `python scripts/setup_data.py --from <bundle>` | Installs the 353 MB bundle; reruns skip it |
| `python tests/test_template.py` | 6 of 6 checks pass (bundle, main run, scaling, quality, configs, replay presets) |
| 01, 02, 03 notebooks with `USE_GPUS = False` | Execute end to end with no errors (outputs saved in the notebooks) |
| Replay-mode Serve application in a browser | Presets load, 4 drives return, no page errors |
| Bundle and notebook outputs | No host names, IP addresses or internal paths |

## Not verified yet

* **GPU path in template form.** The generation, scheduling and Serve code is the code used for the study
  (288 + 90 clips, live app), but the template copy changes where it finds the Cosmos environment
  (`COSMOS_ENV_ROOT`, default `/opt/cosmos`) and shared storage (`COSMOS_SHARED`). Run notebook 01 and 03
  once with `USE_GPUS = True` on the template image before publishing.
* **Docker image.** `Dockerfile` has not been built here (no Docker in this workspace). Confirm the
  `BASE_IMAGE` tag for Ray 2.60 with Python 3.13 and CUDA 13, and that `build_cosmos_env.sh` finishes
  inside the build (it clones Cosmos Transfer 2.5 at commit 45c6f0fb and runs `uv sync --locked`).
* **Compute configs.** `jobs/*.yaml` use GCP `a2-ultragpu-1g` (1x A100 80 GB) with AWS `p4de.24xlarge`
  noted in comments. The model needs 80 GB GPUs (71 GB peak on the first request of each actor).

## Before Step 2

1. **Host the data bundle** where the notebooks and CI can download it, then set `BUNDLE_URL` in
   `scripts/setup_data.py`:
   * file: `/mnt/cluster_storage/cosmos-offroad/cosmos-video-augmentation-bundle-v1.tar` (353 MB)
   * sha256 prefix: `f9e23e4bae65bf22`
   * contents: `media/` (9 drives x 10 conditions, guardrails on) and `results/` (sanitized run records)
   * rebuild with `scripts/make_bundle.py` if the results change.
2. Model weights are gated on Hugging Face (NVIDIA Open Model License). The GPU path needs `HF_TOKEN`;
   the CPU path does not.

## Step 2: publish with `/template`

From your `anyscale/templates` checkout:

```
/template publish a new template, content is at <path>/cosmos-video-augmentation
```

Notes for the skill:
* Test command: `python scripts/setup_data.py && python tests/test_template.py`, then execute the three
  notebooks with the default `USE_GPUS = False` (CPU only, about 5 minutes).
* Notebook 03 deploys a Serve application named `paint` at `/` and deletes it in its last cell.
* Get a review stamp from the Ray Data and Ray Serve owners for a new template.

## Step 3: register with `/register-template`

From your `anyscale/product` checkout:

```
/register-template register the cosmos-video-augmentation template in the console
```

Suggested gallery card:
* **Title:** Generate weather variations of real driving video
* **Description:** Use Ray Data, Ray Core and Ray Serve with NVIDIA Cosmos Transfer 2.5 to turn real
  off-road footage into new weather, lighting and seasons, then serve it as an interactive app.
* **Labels:** Ray Data, Ray Serve, Ray Core, video generation, physical AI, synthetic data, GPU
* **Image:** `docs/images/four_worlds.jpg`
