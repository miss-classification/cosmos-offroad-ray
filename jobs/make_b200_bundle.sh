#!/usr/bin/env bash
# Pack this project for the B200 cloud (see AGENTS.md "Phase 5").
#
# Writes two tars to $OUT (default /mnt/cluster_storage/cosmos-offroad/transfer):
#   offroad-code.tar.gz    the repo (~150 MB): code, configs, AGENTS.md, results/ (no .venv, no caches)
#   offroad-assets.tar     shared-storage subset (~45 GB, already-compressed media/weights, so no gzip):
#                          hf_home/hub (weights, minus the unused Llama-Guard stub), hf_home/cosmos_depth_models,
#                          src/ (Cosmos Transfer 2.5 checkout), data/prepared/ (catalogs + clips + edge controls),
#                          runs/ (guardrails-ON namespaces only, for side-by-side comparison)
# NOT included on purpose:
#   hf_home/token  (credential: log in again on the new box)
#   envs/          (venv is not relocatable; rebuild with environments/build_cosmos_env.sh)
#   data/tartandrive2 raw bags (26 GB; only needed to re-run preprocessing)
#   runs/{smoke,serve_demo,latency_profile}  (guardrails OFF, never evaluated)
#
# Usage: bash jobs/make_b200_bundle.sh [--dry-run] [--slim]
#   --slim (default choice, user 2026-11-20): assets = data/prepared + guardrails-ON comparison runs only (~3.6 GB).
#          No weights: the B200 cluster downloads them from Hugging Face (jobs/stage_checkpoints.py; depth
#          checkpoints download on first use). runs/main (4.5 GB) and runs/benchmark (1.1 GB) are left out.
set -euo pipefail
PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="${OFFROAD_ROOT:-/mnt/cluster_storage/cosmos-offroad}"
OUT="${OUT:-$ROOT/transfer}"
DRY=0; SLIM=0
for a in "$@"; do case "$a" in --dry-run) DRY=1;; --slim) SLIM=1;; *) echo "unknown arg $a" >&2; exit 2;; esac; done

ASSETS=(hf_home/hub hf_home/cosmos_depth_models src data/prepared runs)
ASSET_EXCLUDES=(
  --exclude=hf_home/token --exclude='hf_home/hub/models--meta-llama--Llama-Guard-3-8B'
  --exclude=runs/smoke --exclude=runs/serve_demo --exclude=runs/latency_profile
  --exclude='*/.git/objects/pack/*.pack.tmp*'
)
if (( SLIM )); then
  ASSETS=(data/prepared runs/conditions_v3a runs/conditions_v3b runs/pilot runs/depth_probe runs/judge
          runs/realism_a runs/realism_b runs/phase_a runs/_attempt_ledger logs)
fi
CODE_EXCLUDES=(--exclude='.venv' --exclude='__pycache__' --exclude='.pytest_cache' --exclude='.wrangler')

if (( DRY )); then
  echo "code:";   du -sh --exclude=.venv "$PROJECT"
  echo "assets:"; (cd "$ROOT" && du -sch "${ASSETS[@]}" --exclude=token --exclude='models--meta-llama--*' \
                     --exclude=smoke --exclude=serve_demo --exclude=latency_profile)
  exit 0
fi

mkdir -p "$OUT"
echo "[bundle] code -> $OUT/offroad-code.tar.gz"
tar -C "$(dirname "$PROJECT")" "${CODE_EXCLUDES[@]}" -czf "$OUT/offroad-code.tar.gz" "$(basename "$PROJECT")"

echo "[bundle] assets -> $OUT/offroad-assets.tar (tens of GB, takes a while over NFS)"
tar -C "$ROOT" "${ASSET_EXCLUDES[@]}" -cf "$OUT/offroad-assets.tar" "${ASSETS[@]}"

# Safety: the HF token must never be in the bundle.
if tar -tf "$OUT/offroad-assets.tar" | grep -qx 'hf_home/token'; then
  echo "[bundle] ERROR: hf_home/token ended up in the tar; deleting it" >&2; rm -f "$OUT/offroad-assets.tar"; exit 3
fi
(cd "$OUT" && sha256sum offroad-code.tar.gz offroad-assets.tar > SHA256SUMS)
ls -lh "$OUT"
