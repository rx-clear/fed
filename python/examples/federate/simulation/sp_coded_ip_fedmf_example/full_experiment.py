"""End-to-end MF accuracy and dropout comparison for Coded-IP-FedMF."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import torch

from fedml.simulation.sp.fedavg.coded_ip_fedmf import CodedIPFedMFAggregator
from movielens import load_ratings, make_synthetic_ratings


def _gradient(items, user, item_ids, ratings, regularization=1e-3):
    selected = items[item_ids]
    residual = selected @ user - ratings
    gradient = torch.zeros_like(items)
    for row, item_id in enumerate(item_ids):
        gradient[item_id] += residual[row] * user + regularization * selected[row]
    return gradient / max(1, len(item_ids))


def _evaluate(items, users, ratings):
    errors = []
    for client_id, (item_ids, values) in enumerate(ratings):
        prediction = (items[item_ids] * users[client_id]).sum(dim=1)
        errors.append(prediction - values)
    error = torch.cat(errors)
    return float(error.square().mean().sqrt()), float(error.abs().mean())


def run_comparison(ratings, rank=8, rounds=20, learning_rate=0.05,
                   dropout_rate=0.2, seed=17):
    clients = len(ratings)
    if clients < 1:
        raise ValueError("ratings must contain at least one client")
    if not isinstance(rounds, int) or rounds < 1:
        raise ValueError("rounds must be a positive integer")
    if not isinstance(rank, int) or rank < 1:
        raise ValueError("rank must be a positive integer")
    if not math.isfinite(float(dropout_rate)) or not 0.0 <= dropout_rate < 1.0:
        raise ValueError("dropout_rate must be finite and in [0, 1)")
    non_empty_ids = [ids for ids, _ in ratings if len(ids)]
    if not non_empty_ids:
        raise ValueError("ratings must contain at least one rating")
    item_num = max(int(ids.max()) for ids in non_empty_ids) + 1
    threshold = max(1, clients - int(round(clients * dropout_rate)))
    generator = torch.Generator().manual_seed(seed)
    initial_items = torch.randn(item_num, rank, generator=generator) * 0.1
    initial_users = torch.randn(clients, rank, generator=generator) * 0.1
    coded_items, plain_items = initial_items.clone(), initial_items.clone()
    coded_users, plain_users = initial_users.clone(), initial_users.clone()
    aggregator = CodedIPFedMFAggregator(clients, threshold, seed=seed)
    dropout_generator = torch.Generator().manual_seed(seed + 100_003)
    rows = []
    for round_idx in range(rounds):
        online = tuple(sorted(
            int(client_id)
            for client_id in torch.randperm(clients, generator=dropout_generator)[:threshold]
        ))
        coded_updates, plain_updates = {}, {}
        for client_id, (ids, values) in enumerate(ratings):
            coded_updates[client_id] = _gradient(coded_items, coded_users[client_id], ids, values).reshape(-1)
            plain_updates[client_id] = _gradient(plain_items, plain_users[client_id], ids, values).reshape(-1)
        coded_result = aggregator.aggregate_updates(coded_updates, online, label=f"full-{round_idx}")
        plain_sum = torch.stack([plain_updates[i] for i in online]).sum(dim=0) / len(online)
        coded_items -= learning_rate * coded_result.aggregate.reshape_as(coded_items) / clients
        plain_items -= learning_rate * plain_sum.reshape_as(plain_items)
        coded_rmse, coded_mae = _evaluate(coded_items, coded_users, ratings)
        plain_rmse, plain_mae = _evaluate(plain_items, plain_users, ratings)
        rows.append({"round": round_idx, "dropout_rate": dropout_rate,
                     "online_clients": len(online),
                     "online_client_ids": ",".join(str(client_id) for client_id in online),
                     "coded_rmse": coded_rmse,
                     "coded_mae": coded_mae, "plaintext_rmse": plain_rmse,
                     "plaintext_mae": plain_mae, "coded_communication_rounds": 1,
                     "secagg_baseline_rounds": 1 if not dropout_rate else 3,
                     "recovery_error": float((coded_result.aggregate - torch.stack(list(coded_updates.values())).sum(0)).abs().max())})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratings", help="MovieLens ratings.dat or ratings.csv")
    parser.add_argument("--client-num", type=int, default=12)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--dropout-rate", type=float, default=0.2)
    parser.add_argument("--output", type=Path, default=Path("results/full_fedmf_comparison.csv"))
    args = parser.parse_args()
    ratings = (load_ratings(args.ratings, args.client_num) if args.ratings else
               make_synthetic_ratings(args.client_num, 40, 8, 15))
    rows = run_comparison(ratings, rounds=args.rounds, dropout_rate=args.dropout_rate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rounds to {args.output}")


if __name__ == "__main__":
    main()
