# Project Description: GPU Batch Signal Processing

## Goal

Build a GPU-based program that performs signal processing over a large amount of data in one execution run.
This project targets both rubric scenarios:
- hundreds of small signal files
- tens of large signal files

The repository provides two equivalent GPU implementations:
- Native CUDA C++ kernels (`src/main.cu`)
- PyTorch CUDA implementation for Kaggle (`scripts/torch_signal_batch.py`)

## Input and Output

Input:
- directory of `.csv` files containing 1D signal samples

Output:
- one smoothed output file per input: `*_smooth.csv`
- one gradient-magnitude output file per input: `*_gradient.csv`
- aggregate metrics: `signal_metrics.csv`
- run artifact with timing + throughput: `artifacts/proof_*.txt`

## GPU Algorithms / Stages

### 1) NormalizeKernel

Per-sample min-max normalization using per-signal statistics:

```
normalized = (value - min_signal) / (max_signal - min_signal)
```

This stage keeps each signal in a comparable scale and stabilizes downstream processing.

### 2) MovingAverageKernel

Sliding-window denoising filter with configurable radius `r`:

```
out[i] = average(in[i-r ... i+r])
```

This smooths local noise while preserving broad structure.

### 3) GradientMagnitudeKernel

Approximates local derivative magnitude:

```
grad[i] = abs((in[i+1] - in[i-1]) / 2)
```

This highlights transitions/change points in each signal.

## Scaling Strategy

- Load many input files into one padded batch tensor (`num_signals x max_length`)
- Copy batch once to GPU
- Run kernels over the full flattened tensor
- Copy final outputs once back to host

This keeps the execution model simple while still demonstrating large-batch GPU computation.

## Why This Is Beyond Hello World

- Multi-stage GPU pipeline (not a single toy kernel)
- Handles variable input lengths safely with per-signal length bounds
- Processes large collections of files in one run
- Emits measurable artifact files with timing breakdown and throughput

## Lessons Learned

- Batch-oriented memory layout dramatically simplifies large-run execution.
- Explicit artifact logging is crucial for proving scale and performance claims.
- For mixed-length datasets, padded layouts plus length masks are practical and robust.

## Suggested Experiment Matrix for Submission

Run these two commands and include both artifacts:

1. Small-scale workload: 240-300 files x ~4096 samples
2. Large-scale workload: 20-30 files x 200k+ samples

This clearly demonstrates the "hundreds of small" and "tens of large" rubric expectations.
