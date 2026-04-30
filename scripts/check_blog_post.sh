#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ $# -lt 1 ]]; then
  echo "Usage: scripts/check_blog_post.sh <post.md> [--rendered-live] [--url https://eightmm.github.io]"
  exit 1
fi

POST="$1"
shift || true

echo "== Structural validation =="
python3 scripts/validate_blog_post.py "$POST"

echo

echo "== Quick protocol checks =="
python3 scripts/post_preflight.py "$POST"

if [[ ${1:-} == "--rendered-live" ]]; then
  shift
  URL="${1:-https://eightmm.github.io}"
  if [[ ${1:-} != https://* && ${1:-} != http://* ]]; then
    URL="https://eightmm.github.io"
  else
    shift || true
  fi
  SLUG="$(basename "$POST" .md | sed -E 's/^[0-9]{4}-[0-9]{2}-[0-9]{2}-//')"
  echo
  echo "== Rendered validation =="
  python3 scripts/validate_blog_rendered.py --url "$URL" --posts "$SLUG"
fi

echo

echo "All requested checks completed."
