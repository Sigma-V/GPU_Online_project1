#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

echo "Kaggle GPU demo runner"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
else
  echo "nvidia-smi not found (this may be CPU runtime)."
fi

python3 - <<'PY'
import torch
print('torch version:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu:', torch.cuda.get_device_name(0))
PY

./scripts/create_demo_data.sh

mkdir -p artifacts output/small output/large

python3 scripts/torch_signal_batch.py \
  --input_dir data/small_signals \
  --output_dir output/small \
  --window_radius 4 \
  --device auto \
  --artifact_path artifacts/proof_small_signals.txt

python3 scripts/torch_signal_batch.py \
  --input_dir data/large_signals \
  --output_dir output/large \
  --window_radius 8 \
  --device auto \
  --artifact_path artifacts/proof_large_signals.txt

echo "Kaggle demo complete."
echo "Artifacts:"
echo "  artifacts/proof_small_signals.txt"
echo "  artifacts/proof_large_signals.txt"
