"""Measure coded aggregation latency as clients and vector size grow."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch

from fedml.simulation.sp.fedavg.coded_ip_fedmf import CodedIPFedMFAggregator


def benchmark(client_counts=(8, 16, 32), dimensions=(128, 512), repeats=3, seed=19):
    client_counts = tuple(client_counts)
    dimensions = tuple(dimensions)
    if not client_counts or not dimensions:
        raise ValueError("client_counts and dimensions must not be empty")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be positive")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if any(
        isinstance(client_num, bool)
        or not isinstance(client_num, int)
        or client_num < 1
        for client_num in client_counts
    ):
        raise ValueError("client counts must be positive")
    if any(
        isinstance(dimension, bool)
        or not isinstance(dimension, int)
        or dimension < 1
        for dimension in dimensions
    ):
        raise ValueError("dimensions must be positive")
    rows = []
    for client_num in client_counts:
        threshold = max(1, int(client_num * 0.7))
        for dimension in dimensions:
            generator = torch.Generator().manual_seed(seed + client_num + dimension)
            updates = {i: torch.randn(dimension, generator=generator) for i in range(client_num)}
            aggregator = CodedIPFedMFAggregator(client_num, threshold, seed=seed)
            online = tuple(sorted(
                int(client_id)
                for client_id in torch.randperm(client_num, generator=generator)[:threshold]
            ))
            encode_times, aggregate_times, plaintext_times = [], [], []
            for repeat in range(repeats):
                started = time.perf_counter()
                coded = aggregator.encode_client_updates(updates, label=f"bench-{repeat}")
                encode_times.append((time.perf_counter() - started) * 1000.0)
                started = time.perf_counter()
                aggregator.aggregate(coded, online)
                aggregate_times.append((time.perf_counter() - started) * 1000.0)
                started = time.perf_counter()
                torch.stack([updates[client_id] for client_id in online]).sum(dim=0)
                plaintext_times.append((time.perf_counter() - started) * 1000.0)
            rows.append({
                "client_num": client_num,
                "dimension": dimension,
                "recovery_threshold": threshold,
                "dropout_clients": client_num - threshold,
                "dropout_rate": 1.0 - threshold / float(client_num),
                "online_clients": threshold,
                "online_client_ids": ",".join(str(client_id) for client_id in online),
                "redundancy_ratio": client_num / float(threshold),
                "encode_ms_mean": sum(encode_times) / len(encode_times),
                "aggregate_ms_mean": sum(aggregate_times) / len(aggregate_times),
                "plaintext_aggregate_ms_mean": sum(plaintext_times) / len(plaintext_times),
                "coded_to_plain_aggregate_ratio": (
                    sum(aggregate_times) / sum(plaintext_times)
                    if sum(plaintext_times) > 0
                    else float("inf")
                ),
                "coded_payload_values": client_num * dimension,
                "online_payload_values": threshold * dimension,
                "coded_to_online_payload_ratio": client_num / float(threshold),
                "coded_value_payload_bytes": client_num * dimension * 8,
                "online_value_payload_bytes": threshold * dimension * 8,
                "coded_communication_rounds": 1,
            })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/communication_benchmark.csv"))
    parser.add_argument("--seed", type=int, default=19)
    args = parser.parse_args()
    rows = benchmark(seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} benchmark rows to {args.output}")


if __name__ == "__main__":
    main()
