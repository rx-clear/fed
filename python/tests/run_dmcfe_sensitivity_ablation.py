"""Reproducible fixed-point aggregation sensitivity ablation.

This benchmark intentionally measures the non-secure ``ChunkedReferenceBackend``.
Its timing is a software reference for vector quantization and integer weighted
aggregation, not a measurement of production functional-encryption cost.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import sys
import time
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "python/fedml/simulation/sp/fedavg/optimized_dmcfe_ip.py"
_spec = importlib.util.spec_from_file_location("optimized_dmcfe_ip", MODULE_PATH)
if _spec is None or _spec.loader is None:  # pragma: no cover - environment guard
    raise ImportError(f"cannot load {MODULE_PATH}")
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)
OptimizedDMCFEIPAggregator = _module.OptimizedDMCFEIPAggregator


SCALES = (1 << 10, 1 << 14, 1 << 18)
CLIP_BOUNDS = (0.5, 1.0, 2.0, 8.0)
CHUNK_SIZES = (256, 1024, 4096, 16384, 65536)
SEEDS = (0, 1, 2, 3, 4)


def make_vectors(seed: int, clients: int = 8, coordinates: int = 32768):
    """Create identical-shape client updates with deterministic heavy tails."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    vectors = []
    for client in range(clients):
        values = torch.randn(coordinates, generator=generator, dtype=torch.float64) * 0.75
        # Deterministic sparse outliers exercise the clipping axis.
        stride = 97 + client
        indices = torch.arange(client, coordinates, stride, dtype=torch.long)
        signs = torch.where((indices + seed + client) % 2 == 0, 1.0, -1.0)
        values[indices] += signs * (3.0 + 0.25 * client)
        vectors.append(values)
    weights = [32 + 7 * client for client in range(clients)]
    return vectors, weights


def median(values):
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def run(output: Path, timing_repeats: int = 7) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in SEEDS:
        vectors, weights = make_vectors(seed)
        for scale in SCALES:
            for clip_bound in CLIP_BOUNDS:
                for chunk_size in CHUNK_SIZES:
                    aggregator = OptimizedDMCFEIPAggregator(
                        scale=scale,
                        clip_bound=clip_bound,
                        chunk_size=chunk_size,
                    )
                    # Warm up once so import/cache effects are not included.
                    aggregator.aggregate_vectors(vectors, weights, round_idx=seed)
                    timings = []
                    diagnostics = None
                    for _ in range(timing_repeats):
                        start = time.perf_counter()
                        _, diagnostics = aggregator.aggregate_vectors(
                            vectors, weights, round_idx=seed
                        )
                        timings.append((time.perf_counter() - start) * 1000.0)
                    assert diagnostics is not None
                    rows.append(
                        {
                            "seed": seed,
                            "clients": diagnostics.clients,
                            "coordinates": diagnostics.coordinates,
                            "scale": scale,
                            "clip_bound": clip_bound,
                            "chunk_size": chunk_size,
                            "chunks": diagnostics.chunks,
                            "clipped_fraction": diagnostics.clipped_fraction,
                            "max_abs_error": diagnostics.max_abs_error,
                            "mean_abs_error": diagnostics.mean_abs_error,
                            "elapsed_ms_median": median(timings),
                            "elapsed_ms_min": min(timings),
                            "elapsed_ms_max": max(timings),
                            "timing_repeats": timing_repeats,
                            "backend": diagnostics.backend,
                        }
                    )

    fieldnames = list(rows[0].keys())
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT
        / "python/examples/federate/simulation/sp_fedavg_mnist_lr_example/results/dmcfe_sensitivity_ablation.csv",
    )
    parser.add_argument("--timing-repeats", type=int, default=7)
    args = parser.parse_args()
    if args.timing_repeats < 3:
        raise ValueError("timing-repeats must be >= 3")
    run(args.output, timing_repeats=args.timing_repeats)


if __name__ == "__main__":
    main()
