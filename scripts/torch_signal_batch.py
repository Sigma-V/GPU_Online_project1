#!/usr/bin/env python3
"""Batch signal processing with GPU acceleration via PyTorch."""

import argparse
import csv
import datetime as dt
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process CSV signals in batch using PyTorch GPU operations"
    )
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--window_radius", type=int, default=4)
    parser.add_argument("--max_files", type=int, default=-1)
    parser.add_argument(
        "--artifact_path",
        type=Path,
        default=Path("artifacts/last_run_summary.txt"),
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Execution device. auto picks CUDA if available.",
    )
    return parser.parse_args()


def detect_device(mode: str) -> torch.device:
    if mode == "cpu":
        return torch.device("cpu")
    if mode == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested but CUDA is unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def trim(text: str) -> str:
    return text.strip(" \t\r\n")


def read_signal_csv(path: Path) -> np.ndarray:
    values: List[float] = []
    with path.open("r", newline="") as f:
        for raw_line in f:
            line = trim(raw_line)
            if not line:
                continue
            tokens = line.split(",")
            for token in tokens:
                token = trim(token)
                if not token:
                    continue
                try:
                    values.append(float(token))
                except ValueError:
                    continue
    if not values:
        raise RuntimeError(f"Input file has no numeric samples: {path}")
    return np.asarray(values, dtype=np.float32)


def load_dataset(input_dir: Path, max_files: int) -> Tuple[List[Path], List[np.ndarray]]:
    if not input_dir.exists():
        raise RuntimeError(f"Input directory does not exist: {input_dir}")

    files = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix == ".csv" and path.name != "manifest.csv"
    )

    if max_files > 0:
        files = files[:max_files]

    if not files:
        raise RuntimeError(f"No CSV input files found in {input_dir}")

    signals = [read_signal_csv(path) for path in files]
    return files, signals


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_signal_csv(path: Path, values: np.ndarray) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "value"])
        for idx, value in enumerate(values.tolist()):
            writer.writerow([idx, f"{value:.7f}"])


def elapsed_ms(start_s: float, end_s: float) -> float:
    return (end_s - start_s) * 1000.0


def run_pipeline(
    signals: List[np.ndarray], window_radius: int, device: torch.device
) -> Tuple[np.ndarray, np.ndarray, dict]:
    lengths = np.asarray([len(signal) for signal in signals], dtype=np.int64)
    mins = np.asarray([float(signal.min()) for signal in signals], dtype=np.float32)
    maxs = np.asarray([float(signal.max()) for signal in signals], dtype=np.float32)
    ranges = np.maximum(maxs - mins, 1e-6).astype(np.float32)

    num_signals = len(signals)
    max_length = int(lengths.max())

    host_input = np.zeros((num_signals, max_length), dtype=np.float32)
    for i, signal in enumerate(signals):
        host_input[i, : signal.shape[0]] = signal

    stats = {
        "h2d_ms": 0.0,
        "normalize_ms": 0.0,
        "smooth_ms": 0.0,
        "gradient_ms": 0.0,
        "d2h_ms": 0.0,
        "total_ms": 0.0,
        "total_samples": int(lengths.sum()),
    }

    if device.type == "cuda":
        total_start = torch.cuda.Event(enable_timing=True)
        total_stop = torch.cuda.Event(enable_timing=True)
        h2d_start = torch.cuda.Event(enable_timing=True)
        h2d_stop = torch.cuda.Event(enable_timing=True)
        norm_start = torch.cuda.Event(enable_timing=True)
        norm_stop = torch.cuda.Event(enable_timing=True)
        smooth_start = torch.cuda.Event(enable_timing=True)
        smooth_stop = torch.cuda.Event(enable_timing=True)
        grad_start = torch.cuda.Event(enable_timing=True)
        grad_stop = torch.cuda.Event(enable_timing=True)
        d2h_start = torch.cuda.Event(enable_timing=True)
        d2h_stop = torch.cuda.Event(enable_timing=True)

        total_start.record()

        h2d_start.record()
        x = torch.from_numpy(host_input).to(device)
        length_t = torch.from_numpy(lengths).to(device)
        min_t = torch.from_numpy(mins).to(device)
        range_t = torch.from_numpy(ranges).to(device)
        h2d_stop.record()

        idx = torch.arange(max_length, device=device).unsqueeze(0)
        mask = idx < length_t.unsqueeze(1)

        norm_start.record()
        normalized = (x - min_t.unsqueeze(1)) / range_t.unsqueeze(1)
        normalized = torch.where(mask, normalized, torch.zeros_like(normalized))
        norm_stop.record()

        smooth_start.record()
        kernel_size = 2 * window_radius + 1
        kernel = torch.ones((1, 1, kernel_size), dtype=torch.float32, device=device)
        sum_vals = F.conv1d(normalized.unsqueeze(1), kernel, padding=window_radius)
        counts = F.conv1d(mask.float().unsqueeze(1), kernel, padding=window_radius)
        smoothed = sum_vals / torch.clamp(counts, min=1.0)
        smoothed = torch.where(mask.unsqueeze(1), smoothed, torch.zeros_like(smoothed))
        smoothed = smoothed.squeeze(1)
        smooth_stop.record()

        grad_start.record()
        base_idx = torch.arange(max_length, device=device).unsqueeze(0).expand(
            num_signals, -1
        )
        last_valid = (length_t - 1).unsqueeze(1)
        left_idx = torch.clamp(base_idx - 1, min=0)
        left_idx = torch.minimum(left_idx, last_valid)
        right_idx = torch.minimum(base_idx + 1, last_valid)
        left_vals = torch.gather(smoothed, 1, left_idx)
        right_vals = torch.gather(smoothed, 1, right_idx)
        gradient = torch.abs(0.5 * (right_vals - left_vals))
        gradient = torch.where(mask, gradient, torch.zeros_like(gradient))
        grad_stop.record()

        d2h_start.record()
        smoothed_host = smoothed.detach().cpu().numpy()
        gradient_host = gradient.detach().cpu().numpy()
        d2h_stop.record()

        total_stop.record()
        torch.cuda.synchronize()

        stats["h2d_ms"] = h2d_start.elapsed_time(h2d_stop)
        stats["normalize_ms"] = norm_start.elapsed_time(norm_stop)
        stats["smooth_ms"] = smooth_start.elapsed_time(smooth_stop)
        stats["gradient_ms"] = grad_start.elapsed_time(grad_stop)
        stats["d2h_ms"] = d2h_start.elapsed_time(d2h_stop)
        stats["total_ms"] = total_start.elapsed_time(total_stop)
    else:
        total_s = time.perf_counter()

        h2d_s = time.perf_counter()
        x = torch.from_numpy(host_input)
        length_t = torch.from_numpy(lengths)
        min_t = torch.from_numpy(mins)
        range_t = torch.from_numpy(ranges)
        h2d_e = time.perf_counter()

        idx = torch.arange(max_length).unsqueeze(0)
        mask = idx < length_t.unsqueeze(1)

        norm_s = time.perf_counter()
        normalized = (x - min_t.unsqueeze(1)) / range_t.unsqueeze(1)
        normalized = torch.where(mask, normalized, torch.zeros_like(normalized))
        norm_e = time.perf_counter()

        smooth_s = time.perf_counter()
        kernel_size = 2 * window_radius + 1
        kernel = torch.ones((1, 1, kernel_size), dtype=torch.float32)
        sum_vals = F.conv1d(normalized.unsqueeze(1), kernel, padding=window_radius)
        counts = F.conv1d(mask.float().unsqueeze(1), kernel, padding=window_radius)
        smoothed = sum_vals / torch.clamp(counts, min=1.0)
        smoothed = torch.where(mask.unsqueeze(1), smoothed, torch.zeros_like(smoothed))
        smoothed = smoothed.squeeze(1)
        smooth_e = time.perf_counter()

        grad_s = time.perf_counter()
        base_idx = torch.arange(max_length).unsqueeze(0).expand(num_signals, -1)
        last_valid = (length_t - 1).unsqueeze(1)
        left_idx = torch.clamp(base_idx - 1, min=0)
        left_idx = torch.minimum(left_idx, last_valid)
        right_idx = torch.minimum(base_idx + 1, last_valid)
        left_vals = torch.gather(smoothed, 1, left_idx)
        right_vals = torch.gather(smoothed, 1, right_idx)
        gradient = torch.abs(0.5 * (right_vals - left_vals))
        gradient = torch.where(mask, gradient, torch.zeros_like(gradient))
        grad_e = time.perf_counter()

        d2h_s = time.perf_counter()
        smoothed_host = smoothed.numpy()
        gradient_host = gradient.numpy()
        d2h_e = time.perf_counter()

        total_e = time.perf_counter()

        stats["h2d_ms"] = elapsed_ms(h2d_s, h2d_e)
        stats["normalize_ms"] = elapsed_ms(norm_s, norm_e)
        stats["smooth_ms"] = elapsed_ms(smooth_s, smooth_e)
        stats["gradient_ms"] = elapsed_ms(grad_s, grad_e)
        stats["d2h_ms"] = elapsed_ms(d2h_s, d2h_e)
        stats["total_ms"] = elapsed_ms(total_s, total_e)

    total_ms = max(stats["total_ms"], 1e-6)
    stats["throughput_million_samples_per_sec"] = (
        stats["total_samples"] / (total_ms * 1000.0)
    )

    return smoothed_host, gradient_host, stats


def write_outputs(
    files: List[Path],
    signals: List[np.ndarray],
    smoothed: np.ndarray,
    gradient: np.ndarray,
    output_dir: Path,
) -> None:
    ensure_dir(output_dir)

    metrics_path = output_dir / "signal_metrics.csv"
    with metrics_path.open("w", newline="") as mf:
        writer = csv.writer(mf)
        writer.writerow(
            [
                "input_file",
                "num_samples",
                "smooth_min",
                "smooth_max",
                "smooth_mean",
                "gradient_mean",
                "gradient_max",
            ]
        )

        for i, path in enumerate(files):
            length = signals[i].shape[0]
            smooth_valid = smoothed[i, :length]
            grad_valid = gradient[i, :length]

            write_signal_csv(output_dir / f"{path.stem}_smooth.csv", smooth_valid)
            write_signal_csv(output_dir / f"{path.stem}_gradient.csv", grad_valid)

            writer.writerow(
                [
                    path.name,
                    length,
                    f"{float(np.min(smooth_valid)):.7f}",
                    f"{float(np.max(smooth_valid)):.7f}",
                    f"{float(np.mean(smooth_valid)):.7f}",
                    f"{float(np.mean(grad_valid)):.7f}",
                    f"{float(np.max(grad_valid)):.7f}",
                ]
            )


def write_artifact(
    artifact_path: Path,
    input_dir: Path,
    output_dir: Path,
    files: List[Path],
    signals: List[np.ndarray],
    window_radius: int,
    device: torch.device,
    stats: dict,
) -> None:
    ensure_dir(artifact_path.parent)

    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
    else:
        gpu_name = "CPU"

    lengths = [len(signal) for signal in signals]

    with artifact_path.open("w") as f:
        f.write("PyTorch Signal Processing Run Summary\n")
        f.write("===================================\n")
        f.write(
            f"Timestamp: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        f.write(f"Device: {device.type}\n")
        f.write(f"GPU: {gpu_name}\n")
        f.write(f"Input directory: {input_dir}\n")
        f.write(f"Output directory: {output_dir}\n")
        f.write(f"Input files processed: {len(files)}\n")
        f.write(f"Min length: {min(lengths)}\n")
        f.write(f"Max length: {max(lengths)}\n")
        f.write(f"Total samples: {stats['total_samples']}\n")
        f.write(f"Window radius: {window_radius}\n")
        f.write("\nTiming (ms):\n")
        f.write(f"  H2D copy: {stats['h2d_ms']:.3f}\n")
        f.write(f"  Normalize stage: {stats['normalize_ms']:.3f}\n")
        f.write(f"  Smooth stage: {stats['smooth_ms']:.3f}\n")
        f.write(f"  Gradient stage: {stats['gradient_ms']:.3f}\n")
        f.write(f"  D2H copy: {stats['d2h_ms']:.3f}\n")
        f.write(f"  Total: {stats['total_ms']:.3f}\n")
        f.write(
            "Throughput (million samples/second): "
            f"{stats['throughput_million_samples_per_sec']:.3f}\n"
        )
        f.write("\nGenerated files:\n")
        f.write("  - Per-input *_smooth.csv\n")
        f.write("  - Per-input *_gradient.csv\n")
        f.write("  - signal_metrics.csv\n")


def main() -> None:
    args = parse_args()

    if args.window_radius < 0:
        raise RuntimeError("--window_radius must be >= 0")

    device = detect_device(args.device)

    files, signals = load_dataset(args.input_dir, args.max_files)

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    print(
        f"Loaded {len(files)} files "
        f"(min length={min(len(s) for s in signals)}, max length={max(len(s) for s in signals)})"
    )

    smoothed, gradient, stats = run_pipeline(signals, args.window_radius, device)

    write_outputs(files, signals, smoothed, gradient, args.output_dir)
    write_artifact(
        args.artifact_path,
        args.input_dir,
        args.output_dir,
        files,
        signals,
        args.window_radius,
        device,
        stats,
    )

    print("Run complete")
    print(f"  Files processed: {len(files)}")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Total time (ms): {stats['total_ms']:.3f}")
    print(
        "  Throughput (million samples/s): "
        f"{stats['throughput_million_samples_per_sec']:.3f}"
    )
    print(f"  Artifact: {args.artifact_path}")


if __name__ == "__main__":
    main()
