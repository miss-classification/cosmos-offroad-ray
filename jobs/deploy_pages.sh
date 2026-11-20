#!/usr/bin/env bash
# Deploy docs/blog (blog preview + replay app) to Cloudflare Pages without downloading anything.
# Needs: ~/.cloudflare_token (API token with "Cloudflare Pages: Edit"), optional ~/.cloudflare_account_id.
set -euo pipefail
S=/tmp/claude-1000/-home-ray-default/9decc653-6f87-468b-8cdf-e1f41228434b/scratchpad
PROJECT="${1:-paint-the-trail}"
export PATH="$S/node/bin:$PATH"
export CLOUDFLARE_API_TOKEN="$(cat ~/.cloudflare_token)"
[ -f ~/.cloudflare_account_id ] && export CLOUDFLARE_ACCOUNT_ID="$(cat ~/.cloudflare_account_id)"
W="$S/wr/node_modules/.bin/wrangler"
SITE=/home/ray/default/cosmos-offroad-ray/docs/blog
"$W" pages project create "$PROJECT" --production-branch main 2>/dev/null || true
"$W" pages deploy "$SITE" --project-name "$PROJECT" --branch main --commit-dirty=true
