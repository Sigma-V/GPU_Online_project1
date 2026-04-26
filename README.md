# CUDA at Scale: Batch Signal Processing

This repository now supports **two GPU execution paths**:
- Native CUDA C++ kernels (`src/main.cu`) for CUDA lab machines with `nvcc`
- **Kaggle-friendly PyTorch GPU pipeline** (`scripts/torch_signal_batch.py`) that does not require compiling CUDA code

If you are on a MacBook M1 without NVIDIA GPU, use the Kaggle path.

## Kaggle Quick Start (Recommended)

1. Create a Kaggle Notebook and set **Accelerator = GPU**.
2. Upload this project folder (or zip) as a Kaggle Dataset, or upload directly in the notebook session.
3. In a notebook cell, run:

```bash
# Example if your project zip is in a Kaggle dataset/input:
# !unzip -q /kaggle/input/<your-dataset>/Project2.zip -d /kaggle/working/

!cd "/kaggle/working/Project 2" && bash scripts/run_kaggle_demo.sh
```

4. Submission-ready proof files will be created:
- `artifacts/proof_small_signals.txt`
- `artifacts/proof_large_signals.txt`

## What the Pipeline Does

For each signal CSV file, the pipeline performs GPU stages:
1. Min-max normalization
2. Sliding-window smoothing
3. Gradient magnitude extraction

Outputs:
- per input: `*_smooth.csv`
- per input: `*_gradient.csv`
- summary: `signal_metrics.csv`
- run artifact with timing + throughput

## Repository Structure

- `src/main.cu` - native CUDA C++ CLI and kernels
- `scripts/torch_signal_batch.py` - PyTorch GPU CLI (Kaggle path)
- `scripts/generate_signals.py` - synthetic dataset generator
- `scripts/create_demo_data.sh` - create small + large datasets
- `scripts/run_kaggle_demo.sh` - Kaggle end-to-end runner
- `scripts/run_demo.sh` - native CUDA (`nvcc`) end-to-end runner
- `Makefile` - native CUDA build helpers
- `PROJECT_DESCRIPTION.md` - rubric-aligned project narrative

## Kaggle CLI Usage

```bash
python3 scripts/torch_signal_batch.py \
  --input_dir <path> \
  --output_dir <path> \
  --window_radius 4 \
  --device auto \
  --artifact_path artifacts/last_run_summary.txt
```

Arguments:
- `--input_dir` required
- `--output_dir` required
- `--window_radius` default `4`
- `--max_files` default `-1`
- `--artifact_path` default `artifacts/last_run_summary.txt`
- `--device` one of `auto`, `cuda`, `cpu` (default `auto`)

## Dataset Generation

Creates both rubric-friendly scales:
- small: `240` files x `4096` samples
- large: `24` files x `262144` samples
- each folder includes `manifest.csv` (ignored by both pipelines)

```bash
./scripts/create_demo_data.sh
```

Override sizes if needed:

```bash
SMALL_COUNT=300 SMALL_LENGTH=4096 LARGE_COUNT=20 LARGE_LENGTH=300000 ./scripts/create_demo_data.sh
```

## Native CUDA Path (Optional)

If you are on a machine with CUDA Toolkit + `nvcc`:

```bash
make
./scripts/run_demo.sh
```

## Rubric Mapping

- Code repository quality:
  - CLI tools, README, scripts, and structured outputs
- Proof of execution artifacts:
  - `artifacts/proof_small_signals.txt`
  - `artifacts/proof_large_signals.txt`
- Project description quality:
  - `PROJECT_DESCRIPTION.md` explains goals, algorithms, scaling, and lessons
