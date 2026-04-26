#!/usr/bin/env python3
"""Generate synthetic CSV signal datasets for CUDA batch processing."""

import argparse
import csv
import math
import random
from pathlib import Path
from typing import List


def build_signal(length: int, seed: int) -> List[float]:
    rng = random.Random(seed)

    phase_1 = rng.uniform(0.0, 2.0 * math.pi)
    phase_2 = rng.uniform(0.0, 2.0 * math.pi)
    phase_3 = rng.uniform(0.0, 2.0 * math.pi)

    freq_1 = rng.uniform(1.0, 5.0)
    freq_2 = rng.uniform(6.0, 18.0)
    freq_3 = rng.uniform(19.0, 45.0)

    trend = rng.uniform(-0.25, 0.25)
    noise_scale = rng.uniform(0.03, 0.12)

    spike_count = max(2, length // 4000)
    spike_positions = sorted(rng.randrange(0, length) for _ in range(spike_count))
    spike_lookup = {pos: rng.uniform(0.8, 1.8) for pos in spike_positions}

    values: List[float] = []
    for index in range(length):
        t = index / max(1, length - 1)

        # Multi-frequency baseline with slight trend and Gaussian noise.
        value = 0.8 * math.sin(2.0 * math.pi * freq_1 * t + phase_1)
        value += 0.3 * math.sin(2.0 * math.pi * freq_2 * t + phase_2)
        value += 0.15 * math.sin(2.0 * math.pi * freq_3 * t + phase_3)
        value += trend * t
        value += rng.gauss(0.0, noise_scale)

        if index in spike_lookup:
            value += spike_lookup[index]

        values.append(value)

    return values


def write_signal_csv(path: Path, values: List[float]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["value"])
        for value in values:
            writer.writerow([f"{value:.8f}"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic CSV signals for GPU batch processing"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory where signal_XXXX.csv files will be created",
    )
    parser.add_argument(
        "--count",
        type=int,
        required=True,
        help="Number of signal files to create",
    )
    parser.add_argument(
        "--length",
        type=int,
        required=True,
        help="Samples per signal",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Base random seed for reproducibility",
    )

    args = parser.parse_args()

    if args.count <= 0:
        raise ValueError("--count must be > 0")
    if args.length <= 0:
        raise ValueError("--length must be > 0")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = args.output_dir / "manifest.csv"
    with manifest_path.open("w", newline="") as manifest_file:
        manifest = csv.writer(manifest_file)
        manifest.writerow(["file", "length", "seed"])

        for idx in range(args.count):
            filename = f"signal_{idx:04d}.csv"
            signal_path = args.output_dir / filename
            signal_seed = args.seed + idx
            values = build_signal(args.length, signal_seed)
            write_signal_csv(signal_path, values)
            manifest.writerow([filename, args.length, signal_seed])

    print(f"Generated {args.count} files in {args.output_dir}")
    print(f"Signal length: {args.length}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
