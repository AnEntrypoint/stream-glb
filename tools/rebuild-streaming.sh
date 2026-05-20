#!/usr/bin/env bash
# Rebuild the entire streaming-bake set, squash into one commit on top of
# the streaming-base tag, then force-push to the assets repo. Each run
# produces exactly one commit between streaming-base and master — the
# repo never accumulates stale bake history.
#
# Usage:
#   tools/rebuild-streaming.sh
#
# Env:
#   PARALLEL=N      (default 8) parallel bake workers
#   MAX_TEX_SIZE=N  (default 2048) per-bake texture pyramid cap
#   ASSETS_REPO=    (default $HOME/dev/assets) path to AnEntrypoint/assets clone
#   SOURCE_DIR=     (default $ASSETS_REPO) where to scan for glb/vrm sources

set -euo pipefail

STREAMGLB_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ASSETS_REPO="${ASSETS_REPO:-${HOME}/dev/assets}"
[ -d "$ASSETS_REPO/.git" ] || ASSETS_REPO="C:/dev/assets"
SOURCE_DIR="${SOURCE_DIR:-$ASSETS_REPO}"
PARALLEL="${PARALLEL:-8}"
MAX_TEX_SIZE="${MAX_TEX_SIZE:-2048}"

cd "$ASSETS_REPO"
if ! git rev-parse streaming-base >/dev/null 2>&1; then
  echo "[rebuild] FATAL: tag 'streaming-base' missing in $ASSETS_REPO" >&2
  echo "[rebuild] create it at the desired pre-bake commit, e.g.:" >&2
  echo "[rebuild]   git -C $ASSETS_REPO tag streaming-base <sha>" >&2
  exit 1
fi

echo "[rebuild] assets repo : $ASSETS_REPO"
echo "[rebuild] source dir  : $SOURCE_DIR"
echo "[rebuild] parallel    : $PARALLEL"
echo "[rebuild] max tex size: $MAX_TEX_SIZE"

# 1. Clear old streaming output + regenerate.
echo "[rebuild] clearing old streaming/"
rm -rf "$ASSETS_REPO/streaming"

echo "[rebuild] baking via tools/bake-all.mjs ..."
PARALLEL="$PARALLEL" \
  MAX_TEX_SIZE="$MAX_TEX_SIZE" \
  BAKER=bake-streaming.mjs \
  OUTPUT_BASE="$ASSETS_REPO/streaming" \
  SKIP_EXISTING=0 \
  node "$STREAMGLB_ROOT/tools/bake-all.mjs" "$SOURCE_DIR"

# 2. Regenerate manifest.baked.json.
if [ -f "$ASSETS_REPO/scripts/build-baked-manifest.mjs" ]; then
  echo "[rebuild] regenerating manifest.baked.json"
  node "$ASSETS_REPO/scripts/build-baked-manifest.mjs"
fi

# 3. Squash: soft-reset back to streaming-base, recommit as one, force-push.
echo "[rebuild] squashing history to streaming-base"
git -C "$ASSETS_REPO" reset --soft streaming-base

git -C "$ASSETS_REPO" add streaming/ manifest.baked.json manifest.json scripts/ .github/

if git -C "$ASSETS_REPO" diff --cached --quiet; then
  echo "[rebuild] no changes — nothing to commit"
  exit 0
fi

git -C "$ASSETS_REPO" commit -m "Streaming GLB bakes (rebuild $(date -u +%Y-%m-%dT%H:%M:%SZ))

Rebuilt by stream-glb/tools/rebuild-streaming.sh. Single commit on top of
streaming-base tag; previous bake commits are squashed out of history.

- $(ls "$ASSETS_REPO/streaming" | wc -l) entries
- max tex size: $MAX_TEX_SIZE
- baker: stream-glb/tools/bake-streaming.mjs (single-file streaming GLB
  format, extras.LOCAL_progressive v2)"

echo "[rebuild] force-pushing master"
git -C "$ASSETS_REPO" push --force origin master

echo "[rebuild] DONE"
