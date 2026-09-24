"""Reproducible accuracy baselines for the Coded-IP-FedMF experiment.

The implementations here share the same data, initial factors and optimizer
as the coded reference.  This makes the comparisons useful for checking the
paper's numerical claims while keeping the security scope explicit:

* ``centralized`` averages every client update;
* ``coded`` recovers that same all-client sum from an online subset;
* ``plaintext_online`` averages only updates from online clients; and
* ``ldp_online`` adds calibrated Gaussian noise locally before the online
  plaintext average.

There is deliberately no fake Paillier implementation.  A real Paillier
baseline must be supplied by a reviewed library before cryptographic overhead
claims are made.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch

from fedml.simulation.sp.fedavg.coded_ip_fedmf import CodedIPFedMFAggregator
from paper_components import GaussianDPAccountant, clip_and_add_gaussian_noise


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
        if len(item_ids):
            prediction = (items[item_ids] * users[client_id]).sum(dim=1)
            errors.append(prediction - values)
    if not errors:
        raise ValueError("ratings must contain at least one rating")
    error = torch.cat(errors)
    return float(error.square().mean().sqrt()), float(error.abs().mean())


def _validate_inputs(ratings, rank, rounds, dropout_rate, ldp_noise_multiplier):
    clients = len(ratings)
    if clients < 2:
        raise ValueError("ratings must contain at least two clients")
    if not isinstance(rank, int) or rank < 1:
        raise ValueError("rank must be a positive integer")
    if not isinstance(rounds, int) or rounds < 1:
        raise ValueError("rounds must be a positive integer")
    if not math.isfinite(float(dropout_rate)) or not 0 <= dropout_rate < 1:
        raise ValueError("dropout_rate must be finite and in [0, 1)")
    if not math.isfinite(float(ldp_noise_multiplier)) or ldp_noise_multiplier < 0:
        raise ValueError("ldp_noise_multiplier must be finite and non-negative")
    non_empty = [ids for ids, _ in ratings if len(ids)]
    if not non_empty:
        raise ValueError("ratings must contain at least one rating")
    item_num = max(int(ids.max()) for ids in non_empty) + 1
    return clients, item_num


def run_baseline_comparison(
    ratings,
    rank: int = 8,
    rounds: int = 20,
    learning_rate: float = 0.05,
    dropout_rate: float = 0.2,
    ldp_noise_multiplier: float = 1.0,
    dp_clip_norm: float = 1.0,
    seed: int = 17,
) -> List[Dict[str, object]]:
    """Run one paired baseline experiment and return one row per round."""
    clients, item_num = _validate_inputs(
        ratings, rank, rounds, dropout_rate, ldp_noise_multiplier
    )
    if not math.isfinite(float(learning_rate)) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if not math.isfinite(float(dp_clip_norm)) or dp_clip_norm <= 0:
        raise ValueError("dp_clip_norm must be finite and positive")

    dropout = min(clients - 1, int(round(clients * dropout_rate)))
    threshold = clients - dropout
    online = tuple(range(threshold))
    generator = torch.Generator().manual_seed(seed)
    initial_items = torch.randn(item_num, rank, generator=generator, dtype=torch.float64) * 0.1
    initial_users = torch.randn(clients, rank, generator=generator, dtype=torch.float64) * 0.1
    models = {
        "centralized": initial_items.clone(),
        "coded": initial_items.clone(),
        "plaintext_online": initial_items.clone(),
        "ldp_online": initial_items.clone(),
    }
    users = initial_users.clone()
    aggregator = CodedIPFedMFAggregator(clients, threshold, seed=seed)
    dropout_generator = torch.Generator().manual_seed(seed + 100_003)
    accountant: Optional[GaussianDPAccountant] = None
    if ldp_noise_multiplier > 0:
        accountant = GaussianDPAccountant(
            noise_multiplier=ldp_noise_multiplier,
            sampling_rate=threshold / float(clients),
            delta=1e-5,
        )

    rows: List[Dict[str, object]] = []
    for round_idx in range(rounds):
        # Use arbitrary online sets rather than a fixed prefix.  This checks
        # that recovery depends on the client IDs supplied to the decoder,
        # not on an accidental ordering convention.
        online = tuple(sorted(
            int(client_id)
            for client_id in torch.randperm(clients, generator=dropout_generator)[:threshold]
        ))
        updates = {
            method: {
                client_id: _gradient(
                    models[method], users[client_id], item_ids, values
                ).reshape(-1)
                for client_id, (item_ids, values) in enumerate(ratings)
            }
            for method in models
        }
        all_sum = sum(
            (updates["centralized"][i] for i in range(clients)),
            torch.zeros_like(next(iter(updates["centralized"].values()))),
        )
        central_average = all_sum / clients
        coded_result = aggregator.aggregate_updates(updates["coded"], online, label=f"baseline-{round_idx}")
        online_average = sum(
            (updates["plaintext_online"][i] for i in online),
            torch.zeros_like(next(iter(updates["plaintext_online"].values()))),
        ) / len(online)

        ldp_updates = {}
        for client_id in online:
            if ldp_noise_multiplier == 0:
                ldp_updates[client_id] = updates["ldp_online"][client_id]
            else:
                ldp_updates[client_id] = clip_and_add_gaussian_noise(
                    updates["ldp_online"][client_id],
                    clip_norm=dp_clip_norm,
                    noise_multiplier=ldp_noise_multiplier,
                    generator=torch.Generator().manual_seed(
                        seed + round_idx * clients + client_id
                    ),
                )
        ldp_average = sum(
            ldp_updates.values(), torch.zeros_like(next(iter(ldp_updates.values())))
        ) / len(online)
        if accountant is not None:
            accountant.step()

        models["centralized"] -= learning_rate * central_average.reshape_as(models["centralized"])
        models["coded"] -= learning_rate * coded_result.aggregate.reshape_as(models["coded"]) / clients
        models["plaintext_online"] -= learning_rate * online_average.reshape_as(models["plaintext_online"])
        models["ldp_online"] -= learning_rate * ldp_average.reshape_as(models["ldp_online"])

        metrics = {}
        for method, item_factors in models.items():
            metrics[f"{method}_rmse"], metrics[f"{method}_mae"] = _evaluate(
                item_factors, users, ratings
            )
        metrics.update(
            {
                "round": round_idx,
                "dropout_rate": dropout_rate,
                "online_clients": len(online),
                "online_client_ids": ",".join(str(client_id) for client_id in online),
                "recovery_threshold": threshold,
                "coded_max_recovery_error": float(
                    (coded_result.aggregate - all_sum).abs().max()
                ),
                "coded_vs_central_max_error": float(
                    (models["coded"] - models["centralized"]).abs().max()
                ),
                "coded_communication_rounds": 1,
                "secagg_baseline_rounds": 1 if dropout == 0 else 3,
                "ldp_epsilon": accountant.epsilon if accountant is not None else 0.0,
                "ldp_delta": accountant.delta if accountant is not None else 0.0,
                "ldp_noise_multiplier": ldp_noise_multiplier,
            }
        )
        rows.append(metrics)
    return rows
