"""Personalized federated matrix factorisation with DMCFE-IP aggregation.

Each client keeps a private user embedding and sends only a fixed-point item
gradient to the server.  The example uses the optimized reference backend so
that convergence, dropout, and numerical error can be measured without
pretending that the local backend provides cryptographic security.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable, Sequence

import torch

from fedml.simulation.sp.fedavg.optimized_dmcfe_ip import (
    OptimizedDMCFEIPAggregator,
)
from movielens import RatingClient, load_ratings, make_synthetic_ratings


def _split_train_test(ratings: Sequence[RatingClient]) -> tuple[list[RatingClient], list[RatingClient]]:
    train, test = [], []
    for item_ids, values in ratings:
        if len(item_ids) < 2:
            train.append((item_ids, values))
            test.append((item_ids[:0], values[:0]))
            continue
        train.append((item_ids[:-1], values[:-1]))
        test.append((item_ids[-1:], values[-1:]))
    return train, test


def _item_gradient(
    items: torch.Tensor,
    user: torch.Tensor,
    item_ids: torch.Tensor,
    ratings: torch.Tensor,
    regularization: float,
) -> torch.Tensor:
    gradient = torch.zeros_like(items)
    if len(item_ids) == 0:
        return gradient
    selected = items[item_ids]
    residual = selected @ user - ratings
    gradient.index_add_(0, item_ids, residual[:, None] * user + regularization * selected)
    return gradient / float(len(item_ids))


def _update_private_user(
    items: torch.Tensor,
    user: torch.Tensor,
    item_ids: torch.Tensor,
    ratings: torch.Tensor,
    learning_rate: float,
    regularization: float,
) -> torch.Tensor:
    if len(item_ids) == 0:
        return user
    selected = items[item_ids]
    residual = selected @ user - ratings
    gradient = residual[:, None] * selected
    return user - learning_rate * (gradient.mean(dim=0) + regularization * user)


def _ranking_metrics(
    items: torch.Tensor,
    users: torch.Tensor,
    train: Sequence[RatingClient],
    held_out: Sequence[RatingClient],
    k: int,
) -> tuple[float, float]:
    recalls, ndcgs = [], []
    for user, (train_client, test_client) in zip(users, zip(train, held_out)):
        item_ids, _ = test_client
        if len(item_ids) == 0:
            continue
        scores = items @ user
        seen = set(int(value) for value in train_client[0].tolist())
        if seen:
            scores[list(seen)] = -torch.inf
        topk = torch.topk(scores, min(k, len(scores))).indices.tolist()
        target = int(item_ids[0])
        if target in topk:
            rank = topk.index(target) + 1
            recalls.append(1.0)
            ndcgs.append(1.0 / math.log2(rank + 1.0))
        else:
            recalls.append(0.0)
            ndcgs.append(0.0)
    return (
        float(sum(recalls) / max(1, len(recalls))),
        float(sum(ndcgs) / max(1, len(ndcgs))),
    )


def run_experiment(
    ratings: Sequence[RatingClient],
    *,
    rank: int = 8,
    rounds: int = 20,
    learning_rate: float = 0.05,
    dropout_rate: float = 0.2,
    top_k: int = 10,
    seed: int = 17,
) -> list[dict[str, object]]:
    if not ratings:
        raise ValueError("ratings must contain at least one client")
    if rank < 1 or rounds < 1 or top_k < 1:
        raise ValueError("rank, rounds and top_k must be positive")
    if not math.isfinite(float(dropout_rate)) or not 0.0 <= dropout_rate < 1.0:
        raise ValueError("dropout_rate must be in [0, 1)")
    train, test = _split_train_test(ratings)
    item_num = max((int(ids.max()) for ids, _ in ratings if len(ids)), default=-1) + 1
    if item_num < 1:
        raise ValueError("ratings must contain at least one item")
    clients = len(train)
    generator = torch.Generator().manual_seed(seed)
    items = torch.randn(item_num, rank, generator=generator, dtype=torch.float64) * 0.1
    users = torch.randn(clients, rank, generator=generator, dtype=torch.float64) * 0.1
    aggregator = OptimizedDMCFEIPAggregator(
        scale=1 << 14,
        clip_bound=8.0,
        chunk_size=1 << 15,
    )
    dropout_generator = torch.Generator().manual_seed(seed + 100003)
    rows: list[dict[str, object]] = []
    for round_idx in range(rounds):
        online_count = max(1, clients - int(round(clients * dropout_rate)))
        online = sorted(int(value) for value in torch.randperm(clients, generator=dropout_generator)[:online_count])
        updates, weights = [], []
        for client_id in online:
            ids, values = train[client_id]
            users[client_id] = _update_private_user(
                items, users[client_id], ids, values, learning_rate, 1e-3
            )
            updates.append(
                _item_gradient(items, users[client_id], ids, values, 1e-3).reshape(-1)
            )
            weights.append(max(1, len(ids)))
        aggregate, diagnostics = aggregator.aggregate_vectors(
            updates, weights, round_idx=round_idx, label="pfedmf-item-gradient"
        )
        items = items - learning_rate * aggregate.reshape_as(items)
        recall, ndcg = _ranking_metrics(items, users, train, test, top_k)
        rows.append({
            "round": round_idx,
            "dropout_rate": dropout_rate,
            "online_clients": len(online),
            "online_client_ids": ",".join(str(value) for value in online),
            "recall_at_k": recall,
            "ndcg_at_k": ndcg,
            "dmcfe_max_error": diagnostics.max_abs_error,
            "dmcfe_mean_error": diagnostics.mean_abs_error,
            "clipped_fraction": diagnostics.clipped_fraction,
            "aggregation_chunks": diagnostics.chunks,
            "backend_secure": bool(aggregator.backend.is_secure),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratings", help="MovieLens ratings.dat or ratings.csv")
    parser.add_argument("--client-num", type=int, default=12)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--dropout-rate", type=float, default=0.2)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("results/dmcfe_pfedmf.csv"))
    args = parser.parse_args()
    ratings = (
        load_ratings(args.ratings, args.client_num)
        if args.ratings
        else make_synthetic_ratings(args.client_num, 40, args.rank, 15)
    )
    rows = run_experiment(
        ratings,
        rank=args.rank,
        rounds=args.rounds,
        dropout_rate=args.dropout_rate,
        top_k=args.top_k,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rounds to {args.output}")


if __name__ == "__main__":
    main()
