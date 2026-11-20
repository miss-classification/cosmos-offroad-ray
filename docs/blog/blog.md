Blog metadata (draft, for the CMS)

Blog image
    `assets/four_worlds_poster.jpg` (alt: `assets/hero_poster.jpg`)
Publish date & time
    TBD
Author(s)
    Alicia Chua, Alexandr Plashchinsky, Nickhil Gupta, Abhishek Sriram, Abhishek Monangi, Omar Shorbaji
Blog type
    Product
Eng reviewer
    TBD
Target keywords
    Ray, Anyscale, Ray Serve, Ray Data, multi GPU, video generation, world foundation models, NVIDIA Cosmos, Cosmos Transfer 2.5, synthetic data, off-road autonomy, data augmentation, batch inference, model multiplexing
Related resources
    https://docs.ray.io/en/latest/serve/model-multiplexing.html  
https://docs.ray.io/en/latest/data/data.html  
https://docs.ray.io/en/latest/ray-core/actors.html
SEO description
    Generate weather variations of real off-road video with NVIDIA Cosmos Transfer 2.5 on Ray: 288 clips on 8 A100s and a live Ray Serve application. (145 characters)
Blog URL
    /every-season-every-trail-ray-cosmos-transfer

Ray Data · Ray Core · Ray Serve · NVIDIA Cosmos

# Every Season on Every Trail: Scaling Off-Road Video Augmentation with Ray and NVIDIA Cosmos Transfer 2.5

By Alicia Chua, Alexandr Plashchinsky, Nickhil Gupta, Abhishek Sriram, Abhishek Monangi and Omar Shorbaji

Off-road perception models need examples of night rain, snow and smoke, yet these conditions are scarce in recorded driving data. We used [NVIDIA Cosmos Transfer 2.5](https://github.com/nvidia-cosmos/cosmos-transfer2.5) [2] to generate new weather and lighting for real ATV footage from [TartanDrive 2.0](https://theairlab.org/TartanDrive2) [1], a public off-road driving dataset from the AirLab at Carnegie Mellon University, preserving the terrain, vehicle motion and camera path of each recording. On 8 NVIDIA A100 GPUs, [Ray](https://www.ray.io) [6] generated 288 variants in 51.5 minutes, and every request succeeded on its first attempt.

[VIDEO: assets/hero_one_drive.mp4 (autoplay, muted, loop); GIF fallback: assets/gif/hero_one_drive.gif]

A real 5.8-second recording followed by Cosmos Transfer 2.5 outputs for golden hour, night rain, autumn, blue hour, blizzard and rain. Every output is generated from the same camera clip and its edge map.

The workflow runs on a single Ray cluster. [Ray Data](https://docs.ray.io/en/latest/data/data.html) prepares clips and scores outputs, persistent [Ray actors](https://docs.ray.io/en/latest/ray-core/actors.html) keep the 2B-parameter model loaded on each GPU, a scheduler built on `ray.wait` handles retries and resumption, and [Ray Serve](https://docs.ray.io/en/latest/serve/index.html) exposes the same actors through an interactive application in which users select a drive, describe a condition and watch 8 GPUs generate results in parallel. The template includes a replay of stored results that runs without GPUs; generating new clips requires GPU resources.

**288 / 288** clips succeeded on the first attempt, 0 out-of-memory errors

**51.5 min** for the full run on 8 A100s (6.45 GPU-hours of work)

**6.98x** generation speedup on 8 GPUs vs 1 (87% efficiency)

**100.9 s** to generate 8 drives in the live Ray Serve application

## Interactive demo

Select a drive and a condition. The player replays outputs from the batch pipeline at the model's native 1024×544 resolution, with the original footage to the left of the divider and the generated version to the right. Select "All 9 drives" to view the same prompt applied to every scene.

Interactive replay: stored outputs, no GPU needed[open full screen](replay/)

[EMBED: iframe of https://paint-the-trail.pages.dev/replay/, height 640 px]

### The live application

The replay reproduces a live Ray Serve application. In the recording below, a new user selects a drive, chooses golden hour and submits the request to all 8 GPUs. The page reports the progress of each GPU during generation and then displays the results with a comparison slider.

[VIDEO: assets/clickthrough.mp4 (autoplay, muted, loop); GIF fallback: assets/gif/clickthrough.gif]

Screen recording of the live application on Ray Serve with safety guardrails enabled. The generation period is shown at 12x speed. All 8 clips completed 100.9 seconds after submission.

## Motivation

Autonomous ground vehicles operating off-road encounter conditions that their training data rarely covers. Collecting footage of a night rainstorm on a specific trail requires waiting for the weather, deploying a crew in it and labeling the new data. Simulators provide control, but they rarely reproduce the appearance of real terrain such as wet grass and loose gravel.

World foundation models provide an alternative [4]. Cosmos Transfer 2.5 [2] takes a real video, a control signal derived from it and a text prompt, and generates a new video that follows the control. We use the edge map of every frame as the control. Because the edges come from the real recording, the generated clip keeps the trail's layout and the vehicle's motion, while the prompt sets the light, weather, season and surface.

[VIDEO: assets/four_worlds.mp4 (autoplay, muted, loop); GIF fallback: assets/gif/four_worlds.gif]

Golden hour, night rain, autumn and blizzard, generated from the same recording and edge map.

## Generation pipeline

  * **Input.** 93 frames (5.8 seconds) from the ATV's front camera, resampled from 10 Hz to the model's 16 fps by repeating the nearest frame without interpolation.
  * **Control.** Canny edges computed once per clip at the model's working resolution and stored losslessly, so every condition and seed reuses the same control video.
  * **Generation.** The distilled edge model runs 4 denoising steps and writes a 1024×544 video.
  * **Guardrails.** The Cosmos safety guardrails [4]: a text check before generation (keyword blocklist and Qwen3Guard [5]) and a video check after it (Cosmos-Guardrail1). All results in this post were generated with both checks enabled.
  * **Validation.** Each output is probed for frame count, size and frame rate before it is recorded as done.

### Dataset: TartanDrive 2.0

Item| Value  
---|---  
Source| [TartanDrive 2.0](https://theairlab.org/TartanDrive2) [1], AirLab, Carnegie Mellon University; CC BY 4.0; ATV front camera (`image_left_color`)  
Drives| 9 recorded runs  
Clips| 27 prepared (3 per drive); 24 used in the main run, 9 showcase scenes for the extra conditions  
Download| 13.39 GB in 82.75 s  
Clip format| 93 frames, 1024×544, 16 fps  
  
## Results

### Scaling

We ran the same 24 requests on 1, 2, 4 and 8 actors, each with a dedicated A100, in fresh namespaces with no resume. The median request took 79.79 seconds on a single actor and 80.27 seconds with eight running concurrently, so parallel actors do not slow individual requests.

Actors| Wall time| Speedup, generation| Speedup, end to end| Median per request  
---|---|---|---|---  
1| 2067.75 s| 1.00x| 1.00x| 79.79 s  
2| 1107.90 s| 1.96x| 1.87x| 80.19 s  
4| 624.94 s| 3.75x| 3.31x| 80.26 s  
8| 398.39 s| 6.98x| 5.19x| 80.27 s  
  
End-to-end speedup includes a fixed cold start of about 106 seconds per actor (loading the model), spread over only 3 requests per actor in the 8-actor case. On longer runs this cost becomes small: the main run below spent 88 seconds before its first dispatch and 51.5 minutes generating.

![Speedup and parallel efficiency for 1, 2, 4 and 8 actors](assets/scaling.png) Speedup relative to a single actor (left) and parallel efficiency (right). The generation window excludes the model cold start. ![Timeline of 8 actors each loading the model then generating 3 requests back to back](assets/timeline_a8.png) Every actor loads the model once, then processes its requests back to back with no idle gaps.

### The main run

The main run combined 24 clips with 4 conditions (fog, rain, night, snow) and 3 seeds: 288 requests on 8 A100s. All 288 succeeded on the first attempt in 3089.64 seconds, with no out-of-memory errors, no infrastructure failures and no replacement actors. Executed sequentially, the same work would require 23,219 seconds of GPU time.

### Time per request

The denoising steps account for 79% of each request, so throughput scales primarily with the number of GPUs working in parallel.

Stage (median of 288)| Seconds| Share  
---|---|---  
Denoising, 4 steps| 63.57| 79.1%  
Video guardrail| 5.03| 6.3%  
Writing output and control videos| 4.85| 6.0%  
Decode and postprocessing| 4.03| 5.0%  
Preprocessing| 1.18| 1.5%  
Text guardrail| 0.64| 0.8%  
**Total per request**| **80.32**|  100%  
  
### Consistency across drives

Because the scene structure comes from each recording, the same prompt produces results that follow the layout of every drive. Night rain received the highest automated review score among the additional conditions.

[VIDEO: assets/nine_drives_night_rain.mp4 (autoplay, muted, loop); GIF fallback: assets/gif/nine_drives_night_rain.gif]

The night rain prompt applied to all nine showcase drives. Headlight falloff, wet ground and rain streaks follow each trail's own layout.

[VIDEO: assets/wipe_fb9d3bafac_autumn.mp4 (autoplay, muted, loop); GIF fallback: assets/gif/wipe_fb9d3bafac_autumn.gif]

Autumn on a rocky trail under trees.

[VIDEO: assets/wipe_6e36465c49_blue_hour_dusk.mp4 (autoplay, muted, loop); GIF fallback: assets/gif/wipe_6e36465c49_blue_hour_dusk.gif]

Blue hour on a wet gravel road.

### Automated quality metrics

We scored every output with a Ray Data pass on CPU. Edge F1 compares the edges of each output with those of the source clip within 2 pixels and measures how much of the original structure is preserved. Fog scores lowest, as expected, because fog obscures distant edges.

Condition (72 outputs each)| Edge F1 vs source| Change in brightness  
---|---|---  
Snow| 0.820| +17.3  
Rain| 0.779| −33.9  
Night| 0.640| −82.4  
Fog| 0.529| +7.8  
  
HSV saturation overstates color on dark frames because it is normalized by brightness: night outputs measure 170.7 against 41.8 for the source footage. CIELAB chroma, which is independent of brightness, shows a smaller increase from 8.47 to 15.44. The automated vision-language judge ([Cosmos-Reason1-7B](https://huggingface.co/nvidia/Cosmos-Reason1-7B) [3], blind to the requested condition) proved to be weak evidence, labeling 64 of 71 rain clips as "overcast", so its scores were used only to rank showcase candidates. Human review of the outputs is planned.

### Failure cases

[VIDEO: assets/failure_wildfire_smoke.mp4 (autoplay, muted, loop); GIF fallback: assets/gif/failure_wildfire_smoke.gif]

The lowest-scoring of 371 judged outputs: wildfire smoke on a winding gravel path. The sun disc dissolves and smoke puffs appear and vanish between frames.

Our automatic checks flagged 6 of the 288 main-run outputs, all for frozen frames and all from the same source clip, where the vehicle barely moves. That clip already repeats frames, because the camera's 10 Hz recording is stretched to 16 fps, so the flags identify clips for manual review.

## System design on Ray

Clip preparation, model hosting, GPU scheduling, quality scoring and interactive serving each present a distinct scaling problem. Each maps to a Ray abstraction, and all of them run on the same cluster of 8 nodes with an A100 GPU on each.

### Ray Data: clip preparation

Clip preparation decodes the raw camera frames, checks timestamps, resamples to 16 fps, encodes the model input, verifies it by decoding it back and computes the edge control. A Ray Data pipeline distributes the work with a single clip per block across the GPU nodes' CPUs, so the GPUs stay free for generation:

**Ray Data · Clip preparation** Python · excerpt
    
    
    ds = (ray.data.from_items(runs, override_num_blocks=len(runs))
          .flat_map(plan_clips, num_cpus=1, resources=WORKER_ONLY)       # choose clip windows per drive
          .repartition(target_num_rows_per_block=1)                      # one clip per block
          .map(prepare_clip, num_cpus=2, resources=WORKER_ONLY,          # decode, resample, edges, verify
               compute=ray.data.TaskPoolStrategy(size=concurrency)))
    ds.write_parquet(catalog_dir)                                         # the prepared clip catalog

**The Ray advantage:** preparing 9 clips from scratch took 24.06 seconds of wall time for 129.49 seconds of per-clip work. Each clip writes into a directory keyed by a hash of its settings, so a rerun over all 27 clips with the same settings finished in 7.28 seconds.

### Ray Core: persistent model actors

Cosmos Transfer 2.5 pins its own PyTorch and CUDA stack, which differs from the driver's environment. Each generation worker is a Ray actor that starts a long-lived child process under the Cosmos interpreter and talks to it over a pipe. The model loads once per actor, in 52 to 57 seconds, and then serves every request sent to that GPU:

**Ray Core · Persistent GPU actor** Python · excerpt
    
    
    class CosmosActor:                               # created with ray.remote(CosmosActor), one GPU each
        def _start_child(self):
            rfd, wfd = os.pipe()                      # replies come back on a dedicated pipe
            self.proc = subprocess.Popen(
                [self.s["cosmos_python"], "-m", "offroad_ray.inference.subprocess_worker"],
                stdin=subprocess.PIPE, env=self._child_env(), pass_fds=(wfd,))
    
        def generate(self, req: dict) -> dict:     # model stays loaded between calls
            ...

**The Ray advantage:** the actor owns the GPU, the child process and its logs. If the child dies, the actor reports it and the scheduler replaces the actor on a healthy node, and the Cosmos dependencies remain isolated from the driver.

### Scheduling with `ray.wait`

Generation requests vary little in cost, so the scheduler keeps a single request in flight per actor and dispatches the next request as soon as an actor becomes free. It retries infrastructure failures on a replacement actor, never retries a guardrail block, validates every output and writes a record per request, so an interrupted run resumes from the last completed request:

**Ray Core · Dispatch loop** Python · simplified
    
    
    while pending or in_flight:
        done, _ = ray.wait(list(in_flight), num_returns=1, timeout=5.0)
        for ref in done:
            i, req, t0 = in_flight.pop(ref)
            record(req, ray.get(ref))                 # validate with ffprobe, write the record
            free.append(i)
        while free and pending:                    # refill idle GPUs immediately
            i, req = free.popleft(), pending.popleft()
            in_flight[actors[i].generate.remote(req)] = (i, req, time.time())

**The Ray advantage:** because requests are dispatched individually from a shared queue, a GPU never waits on another. In the main run each of the 8 actors served exactly 36 requests and was busy for 2905 to 2957 of the 3090 seconds.

### Ray Data: quality scoring

The quality pass reads every output, computes edge agreement, brightness, contrast, color and temporal metrics, and aggregates them per condition. It ran on all 288 outputs in 60.12 seconds and also builds side-by-side review images for a static review page:

**Ray Data · Quality metrics** Python · simplified
    
    
    ds = ray.data.from_items(rows, override_num_blocks=len(rows))
    metrics = ds.map(compute_output_metrics, num_cpus=2, resources=WORKER_ONLY,
                     compute=ray.data.TaskPoolStrategy(size=32))
    metrics.write_parquet(metrics_dir)
    summary = ray.data.read_parquet(metrics_dir).groupby("condition").aggregate(
        Count(), Mean("edge_f1"), Std("edge_f1"), Mean("delta_luma"), ...).to_pandas()

### Ray Serve: the interactive application

The live application reuses the same actor class inside a Serve deployment. Each GPU replica processes a single request at a time. A CPU front end built on FastAPI serves the page, queues jobs and lets the browser poll for progress, so a request that takes 80 seconds never holds a connection open. Scene IDs act as [multiplexed model IDs](https://docs.ray.io/en/latest/serve/model-multiplexing.html), which sends repeat requests for a drive to a replica that has already encoded it:

**Ray Serve · GPU tier and scene affinity** Python · excerpt
    
    
    @serve.deployment(max_ongoing_requests=1, ray_actor_options={"num_gpus": 1, "num_cpus": 4})
    class Painter:
        def __init__(self, settings):
            self.engine = CosmosActor(idx, settings)      # same engine as the batch pipeline
    
        @serve.multiplexed(max_num_models_per_replica=12)
        async def _scene(self, scene_id): return scene_id
    
        async def paint(self, req):
            await self._scene(serve.get_multiplexed_model_id())
            return await asyncio.to_thread(self.engine.generate, req)
    
    # In the FastAPI front end: route each job to a replica that already holds its scene
    self.painter.options(multiplexed_model_id=job["scene_id"]).paint.remote(req)

**The Ray advantage:** Serve's model multiplexing also provides cache-affinity routing. The video encodes that depend only on the drive are cached on the GPU and reused bit for bit, and identical requests replay instantly from shared storage.

With guardrails on, 8 replicas were ready 121 seconds after deployment. In the recorded session, 8 drives finished 100.9 seconds after the click, with each clip taking 97.9 to 100.5 seconds. Those were the first requests on each replica, and the guardrail models were still warming up: the video check took 13.56 seconds against a median of 5.03 seconds in the batch run.

### Single-clip latency

We also evaluated ways to reduce the latency of an individual clip. Splitting a clip across GPUs with context parallelism was slower than a single GPU in both 2-node and 4-node configurations, because each node in this cluster has a single GPU and the split adds network traffic to every step. Generating at 480p was about 3 times faster, but this distilled model is trained for 720p and the output turned to blur. `torch.compile` shaved a few seconds per clip, after warm-up runs of up to three minutes. Depth control, an alternative conditioning method, needs the non-distilled model with 35 steps: 867.5 seconds per clip, 10.8 times the edge model. On this hardware, the more effective strategy is to process more clips in parallel, which Ray supports directly.

## Limitations

  * **Vehicle motion is fixed.** Motion and camera path come from the real drive, so a blizzard clip drives at the speed of a summer day. Choosing slow source segments for harsh weather helps.
  * **Edges carry texture.** Dense grass produces dense edges, and under heavy rain or snow prompts that texture can turn into white foam. Sparser edge maps followed the prompt better in a small sweep (condition match 0.92 against 0.42 in our automated review) and are the next setting to test.
  * **Human review is pending.** All quality figures in this post are automated, and the automated judge is weak evidence.
  * **Frame duplication.** The camera records at 10 Hz and the model expects 16 fps, so slow clips already contain repeated frames.

## Cost

The batch work in this post, from the first smoke test through the ten extra conditions, used 3.32 hours of cluster time on 8 GPU nodes: 26.55 node-hours held and 17.40 node-hours of active generation. The main run alone held 6.87 node-hours.

## Key takeaways

A world foundation model can turn a few hours of real off-road driving into the same trails at night, in rain, in smoke or in autumn, while the geometry of every frame stays true to the recording. The work is dominated by a fixed 4-step sampler on each clip, so throughput comes from running many clips at once: per-request time stayed flat from 1 to 8 GPUs, and 288 clips finished in under an hour without a failed request.

Ray supplies the execution layer around the model: Ray Data for preparation and scoring, persistent actors that isolate the model's dependencies, a dispatch loop that keeps every GPU busy and resumes cleanly, and Ray Serve to put the same engine behind an interactive app.

## Get started

Open the template on Anyscale (DRAFT: replace with the template link before publishing). It includes the batch pipeline, the quality pass, the Ray Serve application and the replay page shown above, which runs without GPUs. Enable GPU execution when you are ready to generate new clips.

**Acknowledgements.** We thank the TartanDrive 2.0 team at the AirLab, Carnegie Mellon University, for releasing their off-road driving dataset openly, and the NVIDIA Cosmos team for the Cosmos Transfer 2.5, Cosmos-Reason1 and Cosmos-Guardrail1 models and code.

## References

  1. M. Sivaprakasam, P. Maheshwari, M. Guaman Castro, S. Triest, M. Nye, S. Willits, A. Saba, W. Wang and S. Scherer. _TartanDrive 2.0: More Modalities and Better Infrastructure to Further Self-Supervised Learning Research in Off-Road Driving Tasks._ arXiv:2402.01913, 2024. [arxiv.org/abs/2402.01913](https://arxiv.org/abs/2402.01913). Dataset: [theairlab.org/TartanDrive2](https://theairlab.org/TartanDrive2).
  2. NVIDIA. _World Simulation with Video Foundation Models for Physical AI._ arXiv:2511.00062, 2025. [arxiv.org/abs/2511.00062](https://arxiv.org/abs/2511.00062). Code: [github.com/nvidia-cosmos/cosmos-transfer2.5](https://github.com/nvidia-cosmos/cosmos-transfer2.5). Model: [huggingface.co/nvidia/Cosmos-Transfer2.5-2B](https://huggingface.co/nvidia/Cosmos-Transfer2.5-2B). Project page: [research.nvidia.com/labs/dir/cosmos-transfer2.5](https://research.nvidia.com/labs/dir/cosmos-transfer2.5/).
  3. NVIDIA. _Cosmos-Reason1: From Physical Common Sense To Embodied Reasoning._ arXiv:2503.15558, 2025. [arxiv.org/abs/2503.15558](https://arxiv.org/abs/2503.15558). Model: [huggingface.co/nvidia/Cosmos-Reason1-7B](https://huggingface.co/nvidia/Cosmos-Reason1-7B).
  4. NVIDIA. _Cosmos World Foundation Model Platform for Physical AI._ arXiv:2501.03575, 2025. [arxiv.org/abs/2501.03575](https://arxiv.org/abs/2501.03575). Guardrail model: [huggingface.co/nvidia/Cosmos-Guardrail1](https://huggingface.co/nvidia/Cosmos-Guardrail1). Platform: [nvidia.com/en-us/ai/cosmos](https://www.nvidia.com/en-us/ai/cosmos/).
  5. H. Zhao, C. Yuan, F. Huang, X. Hu, Y. Zhang, A. Yang, B. Yu, D. Liu, J. Zhou, J. Lin and others. _Qwen3Guard Technical Report._ arXiv:2510.14276, 2025. Model: [huggingface.co/Qwen/Qwen3Guard-Gen-0.6B](https://huggingface.co/Qwen/Qwen3Guard-Gen-0.6B).
  6. P. Moritz, R. Nishihara, S. Wang, A. Tumanov, R. Liaw, E. Liang, M. Elibol, Z. Yang, W. Paul, M. I. Jordan and I. Stoica. _Ray: A Distributed Framework for Emerging AI Applications._ 13th USENIX Symposium on Operating Systems Design and Implementation (OSDI), 2018. [arxiv.org/abs/1712.05889](https://arxiv.org/abs/1712.05889). Project: [ray.io](https://www.ray.io), documentation: [docs.ray.io](https://docs.ray.io), code: [github.com/ray-project/ray](https://github.com/ray-project/ray).
  7. X. Zhai, B. Mustafa, A. Kolesnikov and L. Beyer. _Sigmoid Loss for Language Image Pre-Training._ arXiv:2303.15343, 2023. Model: [huggingface.co/google/siglip-so400m-patch14-384](https://huggingface.co/google/siglip-so400m-patch14-384) (loaded by the Cosmos guardrail stack).

## Licenses and attribution

  * **Driving footage:** TartanDrive 2.0, AirLab, Carnegie Mellon University, licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Changes: frames were temporally resampled from 10 Hz to 16 fps, re-encoded, and used as conditioning for generated changes in appearance. All generated videos in this post are derived from this footage.
  * **Models:** Cosmos-Transfer2.5-2B, Cosmos-Reason1-7B and Cosmos-Guardrail1 are governed by the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license); Cosmos-Guardrail1 also lists the Llama 2 Community License as additional information. Qwen3Guard-Gen-0.6B and SigLIP are licensed under Apache 2.0.
  * **Code:** Cosmos Transfer 2.5 and Ray are released under the Apache License 2.0.
  * **Trademarks:** NVIDIA, Cosmos and A100 are trademarks of NVIDIA Corporation.

## Get involved with Ray

To connect with the Ray community, join the Ray Slack or ask questions on the Discuss forum. To run this workload on managed Ray, get started with [Anyscale](https://www.anyscale.com). To learn more about Ray, see [ray.io](https://www.ray.io) and [docs.ray.io](https://docs.ray.io).
