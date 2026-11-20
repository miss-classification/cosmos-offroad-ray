"""Stage pinned checkpoints into the shared HF cache once, before any actor starts.

Runs in the isolated Cosmos interpreter on ONE worker (never on the head), so
eight actors later load from cache with HF_HUB_OFFLINE=1 instead of racing to
download the same files. File list mirrors cosmos-transfer2.5 v1.5.4 for
model=edge/distilled with guardrails enabled.

Usage: python stage_checkpoints.py <out.json>
"""
import json
import os
import sys
import time

FILES = [  # (repo, revision, filename)
    ("nvidia/Cosmos-Transfer2.5-2B", "bbaeedb2b57cc8b14a44653099e2551adb69dcc7",
     "distilled/general/edge/41f07f13-f2e4-4e34-ba4c-86f595acbc20_ema_bf16.pt"),
    ("nvidia/Cosmos-Predict2.5-2B", "f176dc95b4a70f53ce01c4b302851595e7322b00", "tokenizer.pth"),
    # Not used by edge/distilled, but v1.5.4 downloads it as an import side effect of
    # interactive/configs/registry_experiment/experiments_dmd2_ac_predict2p5.py (_build_no_s3_run),
    # at revision "main" (upstream unpinned). Must be cached or offline actors fail to initialise.
    ("nvidia/Cosmos-Predict2.5-2B", "main", "robot/action-cond/38c6c645-7d41-4560-8eeb-6f4ddc0e6574_ema_bf16.pt"),
]
SNAPSHOTS = [  # (repo, revision) full directories
    ("nvidia/Cosmos-Reason1-7B", "3210bec0495fdc7a8d3dbb8d58da5711eab4b423"),
    ("nvidia/Cosmos-Guardrail1", "d6d4bfa899a71454a700907664f3e88f503950cf"),
    # Guardrail models actually loaded at v1.5.4 (guardrail/common/presets.py): text = Blocklist +
    # Qwen3Guard; video = VideoContentSafetyFilter (SigLIP) + RetinaFace. Llama Guard is unused.
    ("Qwen/Qwen3Guard-Gen-0.6B", None),             # open; unpinned upstream, resolved sha recorded
    ("google/siglip-so400m-patch14-384", None),     # open; unpinned upstream, resolved sha recorded
]


def size_of(p):
    if os.path.isfile(p):
        return os.path.getsize(os.path.realpath(p))
    return sum(os.path.getsize(os.path.realpath(os.path.join(r, f))) for r, _, fs in os.walk(p) for f in fs)


def main(out):
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    res = {"hf_home": os.environ.get("HF_HOME"), "items": []}
    for repo, rev, fn in FILES:
        t0 = time.time()
        try:
            p = hf_hub_download(repo, fn, revision=rev)
            res["items"].append({"repo": repo, "revision": rev, "file": fn, "ok": True, "bytes": size_of(p),
                                 "resolved_sha": p.split("/snapshots/")[1].split("/")[0] if "/snapshots/" in p else None,
                                 "seconds": round(time.time() - t0, 1)})
        except Exception as e:
            res["items"].append({"repo": repo, "revision": rev, "file": fn, "ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}"})
    for repo, rev in SNAPSHOTS:
        t0 = time.time()
        try:
            p = snapshot_download(repo, revision=rev)
            sha = HfApi().model_info(repo, revision=rev).sha
            res["items"].append({"repo": repo, "revision": rev, "resolved_sha": sha, "ok": True,
                                 "bytes": size_of(p), "seconds": round(time.time() - t0, 1)})
        except Exception as e:
            res["items"].append({"repo": repo, "revision": rev, "ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}"})
    res["total_gb"] = round(sum(i.get("bytes", 0) for i in res["items"]) / 1e9, 2)
    res["all_ok"] = all(i["ok"] for i in res["items"])
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main(sys.argv[1])
