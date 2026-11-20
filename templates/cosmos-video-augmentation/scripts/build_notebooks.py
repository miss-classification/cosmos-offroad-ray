"""Maintainers only: write the three notebooks from the cell sources below.

  python scripts/build_notebooks.py && jupyter nbconvert --to notebook --execute --inplace 0*.ipynb
"""
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parents[1]

SETUP = '''# Setup
USE_GPUS = False   # False: stored results on CPU. True: generate live on GPUs (template image + weights).

import os, sys
sys.path.insert(0, os.getcwd())
import pandas as pd
import notebook_helpers as nh

pd.set_option("display.max_colwidth", 80)
nh.check_setup(USE_GPUS)'''


def nb(cells):
    n = nbf.v4.new_notebook()
    n.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    n.cells = [nbf.v4.new_markdown_cell(c[3:]) if c.startswith("md:") else nbf.v4.new_code_cell(c) for c in cells]
    return n


NB01 = [
'''md:# 01: Prepare clips and generate weather variations

This notebook covers the batch side of the workflow:

1. **Ray Core**: tasks and actors, the building blocks of the generation pool.
2. **Ray Data**: turning raw TartanDrive 2.0 camera frames into model-ready clips and edge controls.
3. **Generation**: a pool of persistent GPU actors running NVIDIA Cosmos Transfer 2.5.
4. **Scaling**: the same 24 requests on 1, 2, 4 and 8 GPUs.

With `USE_GPUS = False` (the default) every cell runs on CPU and reads stored results from the data bundle. Set `USE_GPUS = True` to generate live.''',
SETUP,
'''md:## 1. Ray Core in brief

A **task** is a function that runs somewhere on the cluster. An **actor** is a class instance that lives on one node and keeps its state between calls. The generation pool in this template is built from actors, because loading Cosmos Transfer 2.5 takes about a minute and should happen once per GPU, not once per clip.''',
'''import ray
from offroad_ray.ray_env import init_ray

init_ray()
print({k: v for k, v in ray.cluster_resources().items() if k in ("CPU", "GPU", "memory")})''',
'''from pathlib import Path

@ray.remote(num_cpus=1)
def probe_clip(path: str) -> dict:
    """A task: read the video header of one clip."""
    from offroad_ray.video_io import probe
    p = probe(path)
    return {"clip": Path(path).parent.name, "width": p["width"], "height": p["height"],
            "frames": p["frames"], "fps": round(p["fps"], 2)}

refs = [probe_clip.remote(str(nh.MEDIA / sid / "src.mp4")) for sid in nh.scenes()]   # runs in parallel
pd.DataFrame(ray.get(refs))''',
'''@ray.remote(num_cpus=0.5)
class FrameReader:
    """An actor: state (here a call counter) persists between calls, as a loaded model does."""
    def __init__(self):
        self.calls = 0

    def first_frame_shape(self, path: str):
        import cv2
        self.calls += 1
        ok, frame = cv2.VideoCapture(path).read()
        return frame.shape, self.calls

reader = FrameReader.remote()
ray.get([reader.first_frame_shape.remote(str(nh.MEDIA / sid / "src.mp4")) for sid in list(nh.scenes())[:3]])''',
'''md:## 2. Prepare clips with Ray Data

TartanDrive 2.0 records the ATV's front camera at 10 Hz. Cosmos Transfer 2.5 expects 93 frames at 16 fps, plus a control video: here, the Canny edge map of every frame. The preparation pipeline decodes frames, checks timestamps, resamples by repeating the nearest frame, encodes the model input, verifies it by decoding it back and computes the edge control. Ray Data spreads this across CPUs with one clip per block.

```python
ds = (ray.data.from_items(runs)
      .flat_map(plan_clips)                          # choose clip windows per drive
      .repartition(target_num_rows_per_block=1)      # one clip per block
      .map(prepare_clip, num_cpus=2,                 # decode, resample, edges, verify
           compute=ray.data.TaskPoolStrategy(size=concurrency)))
ds.write_parquet(catalog_dir)
```

The cell below runs the pipeline live when the raw footage is present (`python scripts/setup_data.py --with-raw`, about 1.5 GB). Otherwise it shows the stored statistics from the original 27-clip run.''',
'''from offroad_ray.config import load_config

cfg = load_config("configs/quickstart.yaml")
if (cfg.paths.raw_dir / cfg.dataset.runs[0]).exists():
    from offroad_ray.preprocessing import run_prepare
    stats = run_prepare(cfg, concurrency=4)
else:
    print("Raw footage not downloaded: showing the stored preparation run.")
    stats = nh.read_json(nh.RESULTS / "dataset" / "prepare_main.json")
{k: stats[k] for k in ("ok", "rows", "wall_s", "cache_hits") if k in stats}''',
'''from IPython.display import Image
Image(str(nh.RESULTS / "dataset" / "contact_sheet.jpg"), width=900)''',
'''md:## 3. Generate with persistent GPU actors

Each actor owns one GPU and one long-lived child process running Cosmos Transfer 2.5 in its own Python environment, so its pinned PyTorch and CUDA stack never conflicts with the driver. The model loads once per actor and serves every request sent to that GPU.

A scheduler built on `ray.wait` keeps one request in flight per actor and dispatches the next one as soon as an actor is free. It retries infrastructure failures on a replacement actor, never retries a guardrail block, validates each output and writes a record per request, so an interrupted run resumes from the last completed request.

```python
while pending or in_flight:
    done, _ = ray.wait(list(in_flight), num_returns=1, timeout=5.0)
    for ref in done:
        i, req, t0 = in_flight.pop(ref)
        record(req, ray.get(ref))
        free.append(i)
    while free and pending:
        i, req = free.popleft(), pending.popleft()
        in_flight[actors[i].generate.remote(req)] = (i, req, time.time())
```''',
'''if USE_GPUS:
    from offroad_ray.cli import _catalog
    from offroad_ray.inference.scheduler import build_requests, run_pool
    ns = str(cfg.paths.runs_dir / cfg.name / "main")
    requests = build_requests(cfg, _catalog(cfg), ns)
    summary = run_pool(cfg, requests, ns, num_actors=cfg.scheduler.num_actors, resume=True)
    manifest = nh.read_jsonl(Path(ns) / "manifest.jsonl")
else:
    manifest = nh.read_jsonl(nh.RESULTS / "main" / "manifest.jsonl")   # 24 clips x 4 conditions x 3 seeds
    summary = nh.read_json(nh.RESULTS / "main" / "pool_summary.json")

print(f"{len(manifest)} requests, {summary['wall_s']:.0f} s wall on {summary['num_actors']} GPUs")
manifest["status"].value_counts()''',
'''md:The 4 denoising steps account for most of each request, so throughput scales with the number of GPUs working in parallel:''',
'''nh.stage_table(manifest)''',
'''md:One generated frame per condition, from the same real clip and edge map:''',
'''nh.show_conditions("5d20270fcc", ["golden_hour_glare", "night_rain", "autumn", "blue_hour_dusk", "blizzard_whiteout"])''',
'''md:Across the clip, the generated video follows the real vehicle motion frame by frame:''',
'''sid = "5d20270fcc"
nh.show_strip([nh.MEDIA / sid / "src.mp4", nh.MEDIA / sid / "autumn.mp4", nh.MEDIA / sid / "night_rain.mp4"],
              ["real", "autumn", "night rain"])''',
'''md:## 4. Scaling from 1 to 8 GPUs

The same 24 requests ran on 1, 2, 4 and 8 actors. The median request time stays flat, so actors do not slow each other down. End-to-end speedup is lower because every actor first loads the model (about 106 seconds), which only 3 requests per actor amortize in the 8-actor case.''',
'''scaling = nh.scaling_frame()
scaling''',
'''nh.plot_scaling(scaling)''',
'''nh.plot_timeline("a8")''',
'''md:## Next

* [02_quality_and_review.ipynb](02_quality_and_review.ipynb) scores the outputs with a Ray Data pass on CPU.
* [03_serve.ipynb](03_serve.ipynb) puts the same actors behind an interactive Ray Serve application.''',
]

NB02 = [
'''md:# 02: Score outputs with Ray Data

Generated video needs checking at scale. This notebook runs the quality pass live on CPU over the bundled outputs, then compares it with the stored results of the full 288-clip run.

* **Edge F1** compares the edges of each output with those of the source clip within 2 pixels and measures how much of the original structure is preserved.
* **Temporal checks** look for blank frames, frozen frames and sudden jumps.
* **Color** compares brightness and chroma with the source.''',
SETUP,
'''import logging, ray
from offroad_ray.ray_env import init_ray
init_ray()
ray.data.DataContext.get_current().enable_progress_bars = False
logging.getLogger("ray.data").setLevel(logging.WARNING)

rows = []
for sid, s in nh.scenes().items():
    for cond in s["conditions"]:
        rows.append({"request_id": f"{sid}-{cond}", "scene": s["label"], "condition": cond,
                     "status": "generated_unreviewed",
                     "output_path": str(nh.MEDIA / sid / f"{cond}.mp4"), "clip_path": str(nh.MEDIA / sid / "src.mp4")})
print(len(rows), "outputs to score")''',
'''md:The same function the batch pipeline uses, mapped over every output with Ray Data. Each worker decodes its own videos, so only paths travel through the pipeline:''',
'''from offroad_ray.quality.metrics import compute_output_metrics

ds = ray.data.from_items(rows, override_num_blocks=len(rows))
metrics = ds.map(compute_output_metrics, num_cpus=1).to_pandas()
scored = pd.DataFrame(rows).merge(metrics, on="request_id")
scored.groupby("condition")[["edge_f1", "frozen_frames", "blank_frames"]].mean().round(3).sort_values("edge_f1")''',
'''md:The bundled videos are re-encoded copies, so values differ slightly from the originals. The full run scored the original outputs, 72 per condition:''',
'''full = pd.read_csv(nh.RESULTS / "main" / "quality" / "condition_summary.csv")
full = full.rename(columns={"count()": "outputs", "mean(edge_f1)": "edge_f1", "mean(delta_luma)": "change_in_brightness",
                            "sum(blank_frames)": "blank_frames"})
full[["condition", "outputs", "edge_f1", "change_in_brightness", "blank_frames"]].round(3).sort_values("edge_f1")''',
'''md:## Measuring color on dark frames

HSV saturation is normalized by brightness, so it rises sharply on dark night frames even when the frame holds no more color. CIELAB chroma does not depend on brightness. On this drive, the night version measures far higher in HSV saturation while its CIELAB chroma stays close to the real clip; golden hour adds real color and rises in both:''',
'''import cv2, numpy as np

def color_stats(path):
    cap, hsv_s, chroma = cv2.VideoCapture(str(path)), [], []
    for i in range(0, 93, 8):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, f = cap.read()
        if not ok:
            break
        hsv_s.append(cv2.cvtColor(f, cv2.COLOR_BGR2HSV)[..., 1].mean())
        lab = cv2.cvtColor(f, cv2.COLOR_BGR2LAB).astype(np.float32)
        chroma.append(np.hypot(lab[..., 1] - 128, lab[..., 2] - 128).mean())
    return {"HSV saturation": round(float(np.mean(hsv_s)), 1), "CIELAB chroma": round(float(np.mean(chroma)), 2)}

sid = "5d20270fcc"
pd.DataFrame({c: color_stats(nh.MEDIA / sid / f"{c}.mp4") for c in ("src", "night", "golden_hour_glare")}).T''',
'''md:Over the full run (24 source clips and 96 outputs, every 8th frame), the same comparison:''',
'''sat = nh.read_json(nh.RESULTS / "main" / "quality" / "saturation_check.json")["by_condition"]
pd.DataFrame(sat).T.rename(columns={"hsv_s": "HSV saturation", "lab_chroma": "CIELAB chroma",
                                    "v_mean": "brightness"})[["brightness", "HSV saturation", "CIELAB chroma"]]''',
'''md:## An automated judge, and its limits

The study also asked a vision-language model, Cosmos-Reason1-7B, to rate every clip without telling it the requested condition. Its scores were useful to rank showcase candidates, but it is weak evidence: it labeled most rain clips "overcast". Use human review for any claim about realism.''',
'''judge = nh.read_json(nh.RESULTS / "judge" / "main" / "summary.json")
pd.DataFrame(judge["by_set_condition"]).query("set == 'main'")[
    ["requested_condition", "n", "photorealism", "temporal_stability", "physical_plausibility", "condition_match"]]''',
'''md:## Failure cases

Automatic flags point to clips for manual review. In the full run, all 6 flags were frozen frames from one slow-moving source clip, which already repeats frames because 10 Hz footage is stretched to 16 fps.''',
'''qt = pd.read_parquet(nh.RESULTS / "main" / "quality" / "quality_table.parquet")
flagged = qt[qt["auto_flags"].fillna("") != ""]
print(f"{len(flagged)} of {len(qt)} outputs flagged")
flagged[["clip_id", "condition", "seed", "auto_flags", "frozen_frames", "source_frozen_frames"]]''',
'''md:The lowest-scoring output of the study: wildfire smoke, where the sun disc dissolves and smoke appears and vanishes between frames.''',
'''sid = "833f53f9d2"
nh.show_strip([nh.MEDIA / sid / "src.mp4", nh.MEDIA / sid / "wildfire_smoke.mp4"], ["real", "wildfire smoke"])''',
]

NB03 = [
'''md:# 03: Serve the generator with Ray Serve

This notebook deploys the interactive application: pick a real drive, choose a condition, and Cosmos Transfer 2.5 generates it on a GPU.

* **`Painter`**: 1 GPU per replica, `max_ongoing_requests=1`. Wraps the same actor class as the batch pipeline, so the model loads once per replica.
* **`PaintAPI`**: a FastAPI front end on CPU. It serves the page and a job API: `POST /api/paint` returns job IDs at once and the browser polls for progress.
* **Scene affinity**: scene IDs are passed as multiplexed model IDs, so Serve sends repeat requests for a drive to a replica that has already encoded it.

With `USE_GPUS = False` the application runs in **replay mode**: a CPU `ReplayPainter` returns stored guardrails-on outputs for the preset conditions, so you can try the whole application, including routing and the UI, without GPUs.''',
SETUP,
'''md:```python
@serve.deployment(max_ongoing_requests=1, ray_actor_options={"num_gpus": 1, "num_cpus": 4})
class Painter:
    def __init__(self, settings):
        self.engine = CosmosActor(idx, settings)          # same engine as the batch pipeline

    @serve.multiplexed(max_num_models_per_replica=12)
    async def _scene(self, scene_id): return scene_id

    async def paint(self, req):
        await self._scene(serve.get_multiplexed_model_id())
        return await asyncio.to_thread(self.engine.generate, req)

# In PaintAPI: route each job to a replica that already holds its drive
self.painter.options(multiplexed_model_id=job["scene_id"]).paint.remote(req)
```''',
'''import logging
from ray import serve
from offroad_ray.ray_env import init_ray
from offroad_ray.serve.app import build_app

logging.getLogger("ray.serve").setLevel(logging.WARNING)

ROUTE = os.environ.get("PAINT_ROUTE", "/")
APP_NAME = os.environ.get("PAINT_APP", "paint")
init_ray(log_to_driver=False)   # keep replica request logs out of the notebook
serve.run(build_app(use_gpus=USE_GPUS, replicas=8 if USE_GPUS else 4), name=APP_NAME, route_prefix=ROUTE)
print(serve.status().applications[APP_NAME].status)''',
'''md:## Call the API

Send one preset condition to three drives, then poll the jobs until they finish. In GPU mode each clip takes about 80 seconds and all three run in parallel; in replay mode they return after a few seconds.''',
'''import time, requests

BASE = "http://localhost:8000" + ROUTE.rstrip("/")
info = requests.get(f"{BASE}/api/scenes").json()
print("mode:", info["mode"], "| replicas:", info["replicas"], "| scenes:", len(info["scenes"]))
preset = next(p for p in info["presets"] if p[0].lower().startswith("golden"))
scene_ids = [s["scene_id"] for s in info["scenes"][:3]]

jobs = requests.post(f"{BASE}/api/paint", json={"prompt": preset[2], "scenes": scene_ids}).json()["jobs"]
t0 = time.time()
while True:
    states = [requests.get(f"{BASE}/api/jobs/{j['job_id']}").json() for j in jobs]
    if all(s["status"] in ("done", "failed") for s in states):
        break
    time.sleep(2)
print(f"finished in {time.time() - t0:.1f} s")
pd.DataFrame(states)[["scene_id", "status", "cached", "seconds", "error"]]''',
'''md:The same request again returns at once: outputs are stored by a hash of the drive, prompt and seed.''',
'''again = requests.post(f"{BASE}/api/paint", json={"prompt": preset[2], "scenes": scene_ids}).json()["jobs"]
pd.DataFrame(again)[["scene_id", "status", "cached"]]''',
'''md:## Open the page

In a Workspace, open the Workspace URL with `?port=8000` added. If your browser reports an invalid referer, paste the URL into a new tab instead of clicking a link to it.

The page walks a first-time visitor through three steps (pick a drive, pick a condition, generate), shows each GPU's progress and lays out the results with a slider that compares the real and generated clips.

![The live application](docs/images/app.gif)''',
'''md:## Share it as an Anyscale Service

```bash
anyscale service deploy -f jobs/service.yaml     # prints a URL to share
anyscale service terminate -n cosmos-video-augmentation
```

The Service runs on its own cluster, with 8 A100 replicas in GPU mode. It keeps its GPUs until you terminate it. Anyone with the URL can submit generation requests, so keep the guardrails on and share the URL only with your audience.''',
'''md:## Clean up''',
'''serve.delete(APP_NAME)
print("application deleted")''',
]

for name, cells in (("01_prepare_and_generate.ipynb", NB01), ("02_quality_and_review.ipynb", NB02),
                    ("03_serve.ipynb", NB03)):
    nbf.write(nb(cells), HERE / name)
    print("wrote", name)
