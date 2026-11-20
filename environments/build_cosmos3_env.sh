#!/usr/bin/env bash
# Build the isolated Cosmos 3 environment (comparison only; never shares packages with the 2.5 env).
# Upstream installs diffusers from git main; we pin an exact commit. --torch-backend=cu130 is explicit
# so the wheel choice does not depend on driver probing. Writes READY.json only after the import check.
#
# Usage: DIFFUSERS_REF=<commit sha> bash environments/build_cosmos3_env.sh
set -euo pipefail
DIFFUSERS_REF="${DIFFUSERS_REF:-d3d8af89e5b08a352cf47bba3a451c5fed9a49b1}"   # diffusers main, 2026-11-05
PYTHON_VERSION="${PYTHON_VERSION:-3.13}"
ROOT="${OFFROAD_ROOT:-/mnt/cluster_storage/cosmos-offroad}"
ENV_DIR="$ROOT/envs/cosmos3-${DIFFUSERS_REF:0:12}-cu130-py${PYTHON_VERSION}"
export UV_PYTHON_INSTALL_DIR="$ENV_DIR/python" UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}" UV_LINK_MODE=copy

[[ -f "$ENV_DIR/READY.json" ]] && { echo "[cosmos3] $ENV_DIR already READY"; exit 0; }
mkdir -p "$ENV_DIR"
uv venv "$ENV_DIR/venv" --python "$PYTHON_VERSION" --seed --managed-python --allow-existing
PY="$ENV_DIR/venv/bin/python"
uv pip install --python "$PY" --torch-backend=cu130 \
  "diffusers @ git+https://github.com/huggingface/diffusers.git@${DIFFUSERS_REF}" \
  accelerate av cosmos_guardrail huggingface_hub imageio imageio-ffmpeg torch torchvision transformers
"$PY" - <<'EOF' | tee "$ENV_DIR/import_check.json"
import json, torch, diffusers
from diffusers import Cosmos3OmniPipeline  # noqa: F401  (fails the build if this diffusers lacks Cosmos 3)
print(json.dumps({"torch": torch.__version__, "cuda": torch.version.cuda, "diffusers": diffusers.__version__,
                  "cuda_available": torch.cuda.is_available(),
                  "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}))
EOF
uv pip freeze --python "$PY" > "$ENV_DIR/freeze.txt"
printf '{"diffusers_ref": "%s", "torch_backend": "cu130", "python": "%s", "built_on_host": "%s", "built_at_utc": "%s"}\n' \
  "$DIFFUSERS_REF" "$PYTHON_VERSION" "$(hostname)" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$ENV_DIR/READY.json"
echo "[cosmos3] READY: $ENV_DIR (python: $PY)"
