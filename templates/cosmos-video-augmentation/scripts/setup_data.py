"""One-time setup: download the data bundle, and optionally model weights and raw footage.

  python scripts/setup_data.py                     # data bundle, about 350 MB (enough for every notebook on CPU)
  python scripts/setup_data.py --with-weights      # + Cosmos Transfer 2.5 and guardrail weights, about 40 GB (needs HF_TOKEN)
  python scripts/setup_data.py --with-raw          # + TartanDrive 2.0 camera archive for the quickstart drive, about 1.5 GB

Everything goes under $COSMOS_SHARED (default /mnt/user_storage/cosmos-video-augmentation),
which every node of a Workspace or Job can read. The script skips anything already present.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from offroad_ray.config import DEFAULT_ROOT, load_config  # noqa: E402

# Public location of the bundle built by scripts/make_bundle.py. Override with COSMOS_BUNDLE_URL.
BUNDLE_URL = os.environ.get("COSMOS_BUNDLE_URL", "https://REPLACE-WITH-PUBLIC-BUCKET/cosmos-video-augmentation/bundle-v1.tar")


def have_bundle(root: Path) -> bool:
    return (root / "media" / "index.json").exists() and (root / "results" / "main" / "manifest.jsonl").exists()


def get_bundle(root: Path, src: str | None) -> None:
    if have_bundle(root):
        print(f"bundle: already in {root}")
        return
    root.mkdir(parents=True, exist_ok=True)
    if src and Path(src).is_dir():                       # a local or shared copy of the bundle
        for sub in ("media", "results"):
            shutil.copytree(Path(src) / sub, root / sub, dirs_exist_ok=True)
        print(f"bundle: copied from {src}")
    else:
        url = src or BUNDLE_URL
        if "REPLACE-WITH" in url:
            sys.exit("No bundle location configured: pass --from <folder or URL> or set COSMOS_BUNDLE_URL.")
        with tempfile.TemporaryDirectory() as td:
            tar = Path(td) / "bundle.tar"
            print(f"bundle: downloading {url}")
            urllib.request.urlretrieve(url, tar)
            with tarfile.open(tar) as t:
                t.extractall(root, filter="data")
    assert have_bundle(root), f"bundle incomplete in {root}"
    n = sum(1 for _ in (root / "media").rglob("*.mp4"))
    print(f"bundle: ready in {root} ({n} videos)")


def get_weights(root: Path) -> None:
    """Stage the pinned checkpoints into $COSMOS_SHARED/hf_home with the Cosmos interpreter."""
    from offroad_ray.inference.scheduler import actor_settings
    if not os.environ.get("HF_TOKEN"):
        sys.exit("Set HF_TOKEN and accept the NVIDIA Open Model License for nvidia/Cosmos-Transfer2.5-2B, "
                 "nvidia/Cosmos-Predict2.5-2B, nvidia/Cosmos-Guardrail1 and nvidia/Cosmos-Reason1-7B on Hugging Face.")
    cfg = load_config(HERE / "configs" / "quickstart.yaml")
    s = actor_settings(cfg, str(root / "runs" / "_setup"), hf_offline=False)
    env = {**os.environ, "HF_HOME": str(cfg.paths.hf_home), "HF_HUB_ENABLE_HF_TRANSFER": "1"}
    out = root / "weights_manifest.json"
    print(f"weights: staging into {cfg.paths.hf_home} (about 40 GB, 10 to 20 minutes)")
    subprocess.run([s["cosmos_python"], str(HERE / "offroad_ray" / "stage_checkpoints.py"), str(out)], env=env, check=True)
    print(f"weights: done, manifest {out}")


def get_raw(config: str) -> None:
    subprocess.run([sys.executable, "-m", "offroad_ray.cli", "fetch", str(HERE / config)], cwd=HERE, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="src", default=None, help="bundle folder or URL (default: COSMOS_BUNDLE_URL)")
    ap.add_argument("--with-weights", action="store_true")
    ap.add_argument("--with-raw", action="store_true")
    ap.add_argument("--raw-config", default="configs/quickstart.yaml")
    a = ap.parse_args()
    root = Path(DEFAULT_ROOT)
    get_bundle(root, a.src)
    if a.with_weights:
        get_weights(root)
    if a.with_raw:
        get_raw(a.raw_config)


if __name__ == "__main__":
    main()
