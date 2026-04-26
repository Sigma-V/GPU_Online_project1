#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

SMALL_COUNT="${SMALL_COUNT:-240}"
SMALL_LENGTH="${SMALL_LENGTH:-4096}"
LARGE_COUNT="${LARGE_COUNT:-24}"
LARGE_LENGTH="${LARGE_LENGTH:-262144}"

python3 "$ROOT_DIR/scripts/generate_signals.py" \
  --output_dir "$ROOT_DIR/data/small_signals" \
  --count "$SMALL_COUNT" \
  --length "$SMALL_LENGTH" \
  --seed 2026

python3 "$ROOT_DIR/scripts/generate_signals.py" \
  --output_dir "$ROOT_DIR/data/large_signals" \
  --count "$LARGE_COUNT" \
  --length "$LARGE_LENGTH" \
  --seed 4000

echo "Created dataset variants:"
echo "  small_signals: $SMALL_COUNT files x $SMALL_LENGTH samples"
echo "  large_signals: $LARGE_COUNT files x $LARGE_LENGTH samples"
