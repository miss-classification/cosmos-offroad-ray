#!/usr/bin/env bash
# Build the isolated Cosmos Transfer 2.5 environment on shared storage.
#
# Design (see environments/dependency_notes.md):
#   * Source checkout pinned to an exact upstream commit, never `main`.
#   * Python interpreter installed by uv INTO shared storage so every node can
#     execute it (a venv whose base interpreter lives in ~/.local on the head
#     node is not portable to workers).
#   * `uv sync --locked` against upstream's committed uv.lock. No upgrades.
#   * A flock serializes concurrent builds; READY.json is written only after
#     every check in this script passes. GPU checks happen separately at runtime
#     (python -m offroad_ray.cli env-check), never during the build.
#
# Usage: build_cosmos_env.sh  (all settings via env vars below)
set -euo pipefail

COSMOS_REPO="${COSMOS_REPO:-https://github.com/nvidia-cosmos/cosmos-transfer2.5}"
COSMOS_COMMIT="${COSMOS_COMMIT:-45c6f0fb4c21d4bed8907a1e153326ad4b6686c0}"   # tag v1.5.4
COSMOS_TAG="${COSMOS_TAG:-v1.5.4}"
CUDA_EXTRA="${CUDA_EXTRA:-cu130}"
PYTHON_VERSION="${PYTHON_VERSION:-3.13}"
ROOT="${OFFROAD_ROOT:-/mnt/cluster_storage/cosmos-offroad}"
ENV_ID="ct25-${COSMOS_TAG}-${COSMOS_COMMIT:0:8}-${CUDA_EXTRA}-py${PYTHON_VERSION}"

SRC_DIR="$ROOT/src/cosmos-transfer2.5-${COSMOS_COMMIT:0:12}"
ENV_DIR="$ROOT/envs/$ENV_ID"
LOCK="$ROOT/envs/.build.lock"
export UV_PYTHON_INSTALL_DIR="$ENV_DIR/python"
export UV_PROJECT_ENVIRONMENT="$ENV_DIR/venv"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/mnt/local_storage/uv-cache}"   # node-local, disposable
export UV_LINK_MODE=copy
export UV_PYTHON_PREFERENCE=only-managed
export GIT_LFS_SKIP_SMUDGE=1

mkdir -p "$ROOT/src" "$ROOT/envs"
exec 9>"$LOCK"
echo "[build] waiting for build lock $LOCK"
flock 9
echo "[build] lock acquired on $(hostname)"

if [[ -f "$ENV_DIR/READY.json" ]]; then
  echo "[build] $ENV_DIR already READY; verifying identity"
  grep -q "\"cosmos_commit\": \"$COSMOS_COMMIT\"" "$ENV_DIR/READY.json" \
    && grep -q "\"cuda_extra\": \"$CUDA_EXTRA\"" "$ENV_DIR/READY.json" \
    && { echo "[build] identity matches; nothing to do"; exit 0; }
  echo "[build] READY.json identity mismatch; refusing to overwrite $ENV_DIR" >&2
  exit 2
fi
if [[ -d "$ENV_DIR/venv" ]]; then
  # Not READY, so nothing can be using it. `uv sync --locked` below is idempotent and
  # repairs or verifies it in place instead of re-downloading 9 GB.
  echo "[build] found unverified env at $ENV_DIR (no READY.json); re-syncing and re-verifying in place"
fi

# 1. Immutable source checkout.
if [[ ! -d "$SRC_DIR/.git" ]]; then
  tmp="${SRC_DIR}.tmp.$$"
  rm -rf "$tmp"
  git clone --quiet "$COSMOS_REPO" "$tmp"
  git -C "$tmp" checkout --quiet "$COSMOS_COMMIT"
  mv "$tmp" "$SRC_DIR"
fi
actual=$(git -C "$SRC_DIR" rev-parse HEAD)
[[ "$actual" == "$COSMOS_COMMIT" ]] || { echo "source commit mismatch: $actual" >&2; exit 3; }
[[ -z "$(git -C "$SRC_DIR" status --porcelain --untracked-files=no)" ]] || { echo "source tree modified" >&2; exit 3; }

# 2. Portable interpreter + locked environment.
mkdir -p "$ENV_DIR"
cd "$SRC_DIR"
uv --version
uv python install "$PYTHON_VERSION"
t0=$(date +%s)
uv sync --locked --extra="$CUDA_EXTRA" --python "$PYTHON_VERSION" --no-dev
t1=$(date +%s)
echo "[build] uv sync took $((t1-t0)) s"

PY="$ENV_DIR/venv/bin/python"

# Intentional override 1 (recorded in READY.json and dependency_notes.md):
# the locked decord==0.6.0+cu130.torch29 (cosmos index) links libavformat.so.60
# (FFmpeg 6), which this image does not provide (system FFmpeg 4.4 -> .so.58).
# The PyPI decord 0.6.0 manylinux wheel bundles its own FFmpeg libraries. Cosmos
# uses decord only with CPU contexts. --reinstall-package is required: a local
# version label (+cu130...) satisfies a plain "decord==0.6.0" pin under PEP 440.
DECORD_OVERRIDE="decord==0.6.0"
uv pip install --python "$PY" --index-url https://pypi.org/simple --reinstall-package decord "$DECORD_OVERRIDE"
"$PY" -c "import decord, importlib.metadata as m; print('[build] decord', m.version('decord'))"
# 3. Portability checks: the venv's base interpreter must live on shared storage.
base_prefix=$("$PY" -c 'import sys; print(sys.base_prefix)')
case "$base_prefix" in
  "$ENV_DIR"/*) echo "[build] base_prefix on shared storage: $base_prefix";;
  *) echo "base_prefix $base_prefix is not under $ENV_DIR; env not portable" >&2; exit 4;;
esac
readlink -f "$PY"

# 4. Dependency consistency (CPU-only; no GPU required).
# Upstream v1.5.4 intentionally overrides three constraints (see pyproject.toml
# [tool.uv] override-dependencies and its workspace layout). Those, and only those,
# are accepted; any other incompatibility fails the build.
uv pip check --python "$PY" > "$ENV_DIR/pip_check.txt" 2>&1 || true
cat "$ENV_DIR/pip_check.txt"
grep '^The package' "$ENV_DIR/pip_check.txt" \
  | grep -v -F -e 'The package `megatron-core` requires `numpy<2.0.0`' \
               -e 'The package `torch` requires `nvidia-cudnn-cu13==' \
               -e 'The package `cosmos-transfer2` requires `cosmos-oss==0.1.0`' \
               -e 'The package `decord` was built for a different platform' \
  > "$ENV_DIR/pip_check_unexpected.txt" || true
if [[ -s "$ENV_DIR/pip_check_unexpected.txt" ]]; then
  echo "[build] unexpected dependency incompatibilities:" >&2
  cat "$ENV_DIR/pip_check_unexpected.txt" >&2
  exit 5
fi
echo "[build] pip check: only the three documented upstream overrides present"
uv pip freeze --python "$PY" > "$ENV_DIR/freeze.txt"
cp "$SRC_DIR/uv.lock" "$ENV_DIR/uv.lock"

# 5. Import smoke test (not a substitute for the runtime GPU check).
"$PY" - <<'EOF' > "$ENV_DIR/import_check.json"
import json, importlib, sys
mods = ["torch", "torchvision", "flash_attn", "transformer_engine", "natten", "decord",
        "cosmos_transfer2", "cosmos_oss", "transformers", "av", "cv2", "numpy"]
out = {"python": sys.version}
for m in mods:
    try:
        mod = importlib.import_module(m)
        out[m] = getattr(mod, "__version__", "ok")
    except Exception as e:  # record, do not hide
        out[m] = f"IMPORT_ERROR: {type(e).__name__}: {e}"
import torch
out["torch_cuda_runtime"] = torch.version.cuda
out["cudnn"] = torch.backends.cudnn.version()
print(json.dumps(out, indent=1))
EOF
cat "$ENV_DIR/import_check.json"
if grep -q IMPORT_ERROR "$ENV_DIR/import_check.json"; then
  echo "[build] import errors; not marking READY" >&2; exit 6
fi

cat > "$ENV_DIR/READY.json" <<EOF
{
 "env_id": "$ENV_ID",
 "cosmos_repo": "$COSMOS_REPO",
 "cosmos_tag": "$COSMOS_TAG",
 "cosmos_commit": "$COSMOS_COMMIT",
 "cuda_extra": "$CUDA_EXTRA",
 "python_request": "$PYTHON_VERSION",
 "uv_version": "$(uv --version | awk '{print $2}')",
 "uv_lock_sha256": "$(sha256sum "$SRC_DIR/uv.lock" | awk '{print $1}')",
 "intentional_overrides": ["$DECORD_OVERRIDE from PyPI replaces locked decord==0.6.0+cu130.torch29 (needs libavformat.so.60, absent)"],
 "accepted_pip_check_overrides_from_upstream": ["numpy>=2 over megatron-core<2", "nvidia-cudnn-cu13 9.21.1.3 over torch pin 9.13.0.50", "cosmos-oss workspace 1.5.0 vs 0.1.0 label"],
 "built_on_host": "$(hostname)",
 "built_at_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
 "sync_seconds": $((t1-t0))
}
EOF
echo "[build] READY: $ENV_DIR"
