#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

./scripts/create_demo_data.sh

make clean
make

mkdir -p artifacts output/small output/large

./bin/cuda_signal_batch \
  --input_dir data/small_signals \
  --output_dir output/small \
  --window_radius 4 \
  --threads_per_block 256 \
  --artifact_path artifacts/proof_small_signals.txt

./bin/cuda_signal_batch \
  --input_dir data/large_signals \
  --output_dir output/large \
  --window_radius 8 \
  --threads_per_block 256 \
  --artifact_path artifacts/proof_large_signals.txt

echo "Demo complete."
echo "Artifacts:"
echo "  artifacts/proof_small_signals.txt"
echo "  artifacts/proof_large_signals.txt"
