"""Synthetic federated matrix-factorisation experiment with coded aggregation."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import torch

from fedml.simulation.sp.fedavg.coded_ip_fedmf import CodedIPFedMFAggregator


def _make_ratings(clients: int, items: int, rank: int, ratings_per_client: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    true_users = torch.randn(clients, rank, generator=generator) * 0.7
    true_items = torch.randn(items, rank, generator=generator) * 0.7
    result = []
    for client_id in range(clients):
        item_ids = torch.randperm(items, generator=generator)[:ratings_per_client]
        values = (true_users[client_id] @ true_items[item_ids].T +
                  0.05 * torch.randn(len(item_ids), generator=generator)).clamp(-2, 2)
        result.append((item_ids, values))
    return result


def _client_item_gradient(item_factors, user_factor, item_ids, ratings, regularization):
    selected = item_factors[item_ids]
    residual = selected @ user_factor - ratings
    gradient = torch.zeros_like(item_factors)
    for row, item_id in enumerate(item_ids):
        gradient[item_id] += residual[row] * user_factor + regularization * selected[row]
    return gradient / max(1, len(item_ids))


def run_experiment(clients=12, items=30, rank=4, rounds=8, dropout_rate=0.2,
                   ratings_per_client=12, learning_rate=0.15, seed=11):
    if clients < 2 or items < 1 or rank < 1 or ratings_per_client < 1:
        raise ValueError("clients >= 2, items/rank/ratings_per_client must be positive")
    if not isinstance(rounds, int) or rounds < 1:
        raise ValueError("rounds must be a positive integer")
    if not math.isfinite(float(dropout_rate)) or not 0 <= dropout_rate < 1:
        raise ValueError("dropout_rate must be finite and in [0, 1)")
    ratings = _make_ratings(clients, items, rank, ratings_per_client, seed)
    generator = torch.Generator().manual_seed(seed + 1)
    item_factors = torch.randn(items, rank, generator=generator) * 0.1
    user_factors = torch.randn(clients, rank, generator=generator) * 0.1
    threshold = max(1, clients - int(round(clients * dropout_rate)))
    aggregator = CodedIPFedMFAggregator(clients, threshold, seed=seed)
    dropout_generator = torch.Generator().manual_seed(seed + 100_003)
    rows = []
    for round_idx in range(rounds):
        updates = {}
        for client_id, (item_ids, values) in enumerate(ratings):
            updates[client_id] = _client_item_gradient(
                item_factors, user_factors[client_id], item_ids, values, 1e-3
            ).reshape(-1)
        expected_update = torch.stack(list(updates.values())).sum(dim=0)
        online = tuple(sorted(
            int(client_id)
            for client_id in torch.randperm(clients, generator=dropout_generator)[:threshold]
        ))
        result = aggregator.aggregate_updates(updates, online, label=f"mf-{round_idx}")
        item_factors = item_factors - learning_rate * result.aggregate.reshape(items, rank) / clients
        squared_error = 0.0
        count = 0
        for client_id, (item_ids, values) in enumerate(ratings):
            prediction = (item_factors[item_ids] * user_factors[client_id]).sum(dim=1)
            squared_error += float(((prediction - values) ** 2).sum())
            count += len(values)
        rows.append({
            "round": round_idx,
            "rmse": (squared_error / count) ** 0.5,
            "dropout_rate": dropout_rate,
            "online_clients": len(online),
            "recovery_threshold": threshold,
            "max_recovery_error": float((result.aggregate - expected_update).abs().max()),
            "communication_rounds": 1,
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/coded_ip_fedmf_rmse.csv"))
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--dropout-rate", type=float, default=0.2)
    args = parser.parse_args()
    rows = run_experiment(rounds=args.rounds, dropout_rate=args.dropout_rate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rounds to {args.output}")


if __name__ == "__main__":
    main()
