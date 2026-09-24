"""Run the Coded-IP-FedMF experiment sweep used by the paper tables."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Iterable, List

import torch

from fedml.simulation.sp.fedavg.coded_ip_fedmf import CodedIPFedMFAggregator


def run_sweep(*, client_num: int = 20, recovery_threshold: int = 14,
              dimension: int = 128,
              dropout_rates: Iterable[float] = (0.0, 0.1, 0.2, 0.3),
              repeats: int = 3, seed: int = 7) -> List[dict]:
    """Return one row per dropout/repeat point.

    Updates are reused across methods. The coded method recovers the sum of
    all client updates, while the plaintext baseline only sums online ones.
    """
    if client_num < 1 or recovery_threshold < 1 or recovery_threshold > client_num:
        raise ValueError("require 1 <= recovery_threshold <= client_num")
    if repeats < 1:
        raise ValueError("repeats must be positive")
    if not isinstance(dimension, int) or dimension < 1:
        raise ValueError("dimension must be a positive integer")
    dropout_rates = tuple(float(rate) for rate in dropout_rates)
    if not dropout_rates:
        raise ValueError("dropout_rates must not be empty")
    rows: List[dict] = []
    for repeat in range(repeats):
        generator = torch.Generator().manual_seed(seed + repeat)
        updates = {i: torch.randn(dimension, generator=generator) * 0.1
                   for i in range(client_num)}
        expected = torch.stack(list(updates.values())).sum(dim=0)
        aggregator = CodedIPFedMFAggregator(client_num, recovery_threshold,
                                            seed=seed + repeat)
        coded_round = aggregator.encode_client_updates(updates,
                                                        label=f"sweep-{repeat}")
        for dropout_rate in dropout_rates:
            rate = float(dropout_rate)
            if not 0.0 <= rate < 1.0:
                raise ValueError("dropout rates must be in [0, 1)")
            dropout = min(client_num - recovery_threshold,
                          int(round(client_num * rate)))
            online_count = client_num - dropout
            # Sample an arbitrary online subset rather than always taking the
            # lowest client IDs.  This exercises the MDS guarantee for varied
            # dropout patterns while remaining deterministic per repeat.
            online = tuple(sorted(
                int(client_id)
                for client_id in torch.randperm(client_num, generator=generator)[:online_count]
            ))
            started = time.perf_counter()
            result = aggregator.aggregate(coded_round, online)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            online_sum = torch.stack([updates[i] for i in online]).sum(dim=0)
            rows.append({
                "repeat": repeat,
                "client_num": client_num,
                "recovery_threshold": recovery_threshold,
                "dropout_rate": rate,
                "dropout_clients": dropout,
                "online_clients": len(online),
                "online_client_ids": ",".join(str(client_id) for client_id in online),
                "coded_max_error": float((result.aggregate - expected).abs().max()),
                "plaintext_online_max_error": float((online_sum - expected).abs().max()),
                "coded_communication_rounds": 1,
                "secagg_baseline_rounds": 1 if dropout == 0 else 3,
                "redundancy_ratio": client_num / float(recovery_threshold),
                "coded_aggregate_ms": elapsed_ms,
            })
    return rows


def write_csv(rows: List[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("cannot write an empty experiment")
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--client-num", type=int, default=20)
    parser.add_argument("--recovery-threshold", type=int, default=14)
    parser.add_argument("--dimension", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path,
                        default=Path("results/coded_ip_fedmf_sweep.csv"))
    args = parser.parse_args()
    rows = run_sweep(client_num=args.client_num,
                     recovery_threshold=args.recovery_threshold,
                     dimension=args.dimension, repeats=args.repeats,
                     seed=args.seed)
    write_csv(rows, args.output)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
