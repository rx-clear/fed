"""MovieLens rating reader and deterministic client partitioning utilities."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Tuple

import torch


RatingClient = Tuple[torch.Tensor, torch.Tensor]


def load_ratings(path: str, client_num: int, max_ratings_per_client: int = 0) -> List[RatingClient]:
    """Read MovieLens ``ratings.dat`` or ``ratings.csv`` into client shards.

    Users are assigned to clients by ``user_id % client_num``. This preserves
    the user boundary and avoids sharing a user's ratings between clients.
    """
    if client_num < 1:
        raise ValueError("client_num must be positive")
    if max_ratings_per_client < 0:
        raise ValueError("max_ratings_per_client must be non-negative")
    rows = []
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    with source.open("r", encoding="utf-8", errors="replace", newline="") as stream:
        first = stream.readline()
        stream.seek(0)
        delimiter = "::" if "::" in first else ","
        has_header = False
        if delimiter == "::":
            reader = (line.rstrip("\n").split("::") for line in stream)
            # MovieLens ratings.dat has no header.  If a header is present,
            # the numeric conversion below naturally skips it.
        else:
            has_header = any(
                token.strip().lower() in {"userid", "user_id", "movieid", "item_id", "rating"}
                for token in first.split(",")
            )
            reader = csv.DictReader(stream) if has_header else csv.reader(stream)
        for row in reader:
            if delimiter == "::":
                if len(row) < 3:
                    continue
                user_id, item_id, rating = row[:3]
            elif has_header:
                user_id = row.get("userId", row.get("user_id", ""))
                item_id = row.get("movieId", row.get("item_id", ""))
                rating = row.get("rating", "")
            else:
                if len(row) < 3:
                    continue
                user_id, item_id, rating = row[:3]
            try:
                rows.append((int(user_id), int(item_id), float(rating)))
            except (TypeError, ValueError):
                continue
    if not rows:
        raise ValueError(f"no valid ratings found in {source}")
    item_ids = {item_id: index for index, item_id in enumerate(sorted({r[1] for r in rows}))}
    clients: List[List[Tuple[int, float]]] = [[] for _ in range(client_num)]
    for user_id, item_id, rating in rows:
        clients[user_id % client_num].append((item_ids[item_id], rating))
    result = []
    for entries in clients:
        if max_ratings_per_client > 0:
            entries = entries[:max_ratings_per_client]
        result.append((torch.tensor([e[0] for e in entries], dtype=torch.long),
                       torch.tensor([e[1] for e in entries], dtype=torch.float32)))
    return result


def make_synthetic_ratings(client_num: int, item_num: int, rank: int,
                           ratings_per_client: int, seed: int = 11) -> List[RatingClient]:
    """Create a MovieLens-shaped fallback dataset for CI and smoke runs."""
    if client_num < 1 or item_num < 1 or rank < 1 or ratings_per_client < 1:
        raise ValueError("client_num, item_num, rank and ratings_per_client must be positive")
    generator = torch.Generator().manual_seed(seed)
    users = torch.randn(client_num, rank, generator=generator) * 0.7
    items = torch.randn(item_num, rank, generator=generator) * 0.7
    output = []
    for client_id in range(client_num):
        ids = torch.randperm(item_num, generator=generator)[:ratings_per_client]
        values = (users[client_id] @ items[ids].T +
                  0.05 * torch.randn(len(ids), generator=generator)).clamp(1, 5)
        output.append((ids, values))
    return output
