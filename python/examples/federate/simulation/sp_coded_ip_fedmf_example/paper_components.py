"""Reference-only components used to test the non-MDS parts of the paper.

These implementations are intentionally transparent test doubles:

* :class:`PairwiseMaskSecAgg` demonstrates why a conventional SecAgg round
  needs mask-recovery material when clients drop out; and
* :class:`GaussianDPAccountant` provides a conservative Gaussian-mechanism
  composition calculation for smoke experiments.

Neither class is a production cryptographic primitive.  In particular, the
server in ``PairwiseMaskSecAgg`` can derive every mask, so its output must not
be used as evidence of SecAgg or DP security.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import torch


TensorLike = Union[torch.Tensor, Sequence[float]]


@dataclass(frozen=True)
class SecAggResult:
    aggregate: torch.Tensor
    online_client_ids: Tuple[int, ...]
    dropped_client_ids: Tuple[int, ...]
    communication_rounds: int
    recovery_messages: int


class PairwiseMaskSecAgg:
    """Transparent pairwise-mask SecAgg model with dropout reconstruction."""

    is_secure = False

    def __init__(self, client_num: int, seed: int = 0):
        if not isinstance(client_num, int) or client_num < 1:
            raise ValueError("client_num must be a positive integer")
        self.client_num = client_num
        self._seed = int(seed)

    @staticmethod
    def _normalise_ids(client_ids: Iterable[int], client_num: int) -> Tuple[int, ...]:
        ids = tuple(int(client_id) for client_id in client_ids)
        if len(set(ids)) != len(ids):
            raise ValueError("client_ids must be unique")
        if any(client_id < 0 or client_id >= client_num for client_id in ids):
            raise IndexError("client id is outside the client set")
        return ids

    def _pair_mask(self, left: int, right: int, shape: torch.Size) -> torch.Tensor:
        if left == right:
            raise ValueError("pairwise masks require two distinct clients")
        low, high = sorted((int(left), int(right)))
        digest = hashlib.sha256(f"{self._seed}:{low}:{high}".encode()).digest()
        pair_seed = int.from_bytes(digest[:8], "big") % (2**63 - 1)
        generator = torch.Generator().manual_seed(pair_seed)
        return torch.randn(shape, generator=generator, dtype=torch.float64)

    def mask_updates(
        self, updates: Mapping[int, TensorLike]
    ) -> Dict[int, torch.Tensor]:
        """Return masked updates; every client masks against every peer."""
        expected_ids = set(range(self.client_num))
        if set(updates) != expected_ids:
            raise ValueError(
                f"updates must contain exactly client ids 0..{self.client_num - 1}"
            )
        values = {
            client_id: torch.as_tensor(update, dtype=torch.float64).detach().cpu()
            for client_id, update in updates.items()
        }
        if not all(torch.isfinite(value).all() for value in values.values()):
            raise ValueError("updates contain NaN or infinity")
        shapes = {value.shape for value in values.values()}
        if len(shapes) != 1:
            raise ValueError("all updates must have the same shape")
        masked = {client_id: value.clone() for client_id, value in values.items()}
        for client_id in range(self.client_num):
            for peer_id in range(client_id + 1, self.client_num):
                mask = self._pair_mask(client_id, peer_id, values[client_id].shape)
                masked[client_id] += mask
                masked[peer_id] -= mask
        return masked

    def aggregate(
        self,
        masked_updates: Mapping[int, TensorLike],
        online_client_ids: Iterable[int],
    ) -> SecAggResult:
        """Recover the online plaintext sum and account for dropout recovery."""
        online = self._normalise_ids(online_client_ids, self.client_num)
        if not online:
            raise ValueError("at least one online client is required")
        if not set(online).issubset(masked_updates):
            raise ValueError("masked_updates is missing an online client")
        dropped = tuple(client_id for client_id in range(self.client_num) if client_id not in online)
        values = {
            client_id: torch.as_tensor(masked_updates[client_id], dtype=torch.float64)
            for client_id in online
        }
        shapes = {value.shape for value in values.values()}
        if len(shapes) != 1:
            raise ValueError("all masked updates must have the same shape")
        aggregate = sum(values.values(), torch.zeros_like(next(iter(values.values()))))

        # Masks between two online clients cancel.  Edges to a dropped client
        # remain and require one recovery contribution per online/dropout pair.
        residual = torch.zeros_like(aggregate)
        for client_id in online:
            for dropped_id in dropped:
                mask = self._pair_mask(client_id, dropped_id, aggregate.shape)
                residual += mask if client_id < dropped_id else -mask
        aggregate = aggregate - residual
        recovery_messages = len(online) * len(dropped)
        return SecAggResult(
            aggregate=aggregate,
            online_client_ids=online,
            dropped_client_ids=dropped,
            communication_rounds=1 if not dropped else 3,
            recovery_messages=recovery_messages,
        )


@dataclass
class GaussianDPAccountant:
    """Simple Gaussian-mechanism composition accountant for experiments."""

    noise_multiplier: float
    sampling_rate: float = 1.0
    delta: float = 1e-5
    steps: int = 0

    def __post_init__(self) -> None:
        if not math.isfinite(self.noise_multiplier) or self.noise_multiplier <= 0:
            raise ValueError("noise_multiplier must be finite and positive")
        if not math.isfinite(self.sampling_rate) or not 0 < self.sampling_rate <= 1:
            raise ValueError("sampling_rate must be in (0, 1]")
        if not math.isfinite(self.delta) or not 0 < self.delta < 1:
            raise ValueError("delta must be in (0, 1)")
        if self.steps < 0:
            raise ValueError("steps cannot be negative")

    def step(self, count: int = 1) -> None:
        if not isinstance(count, int) or count < 1:
            raise ValueError("count must be a positive integer")
        self.steps += count

    @property
    def epsilon(self) -> float:
        """Return a conservative basic-composition epsilon upper bound."""
        per_step = (
            self.sampling_rate
            * math.sqrt(2.0 * math.log(1.25 / self.delta))
            / self.noise_multiplier
        )
        return self.steps * per_step

    def report(self) -> Dict[str, float]:
        return {
            "epsilon": self.epsilon,
            "delta": self.delta,
            "noise_multiplier": self.noise_multiplier,
            "sampling_rate": self.sampling_rate,
            "steps": float(self.steps),
        }


def clip_and_add_gaussian_noise(
    vector: TensorLike,
    clip_norm: float,
    noise_multiplier: float,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Clip a vector and add Gaussian noise for a DP smoke experiment."""
    if not math.isfinite(clip_norm) or clip_norm <= 0:
        raise ValueError("clip_norm must be finite and positive")
    if not math.isfinite(noise_multiplier) or noise_multiplier <= 0:
        raise ValueError("noise_multiplier must be finite and positive")
    value = torch.as_tensor(vector, dtype=torch.float64).detach().cpu()
    if not torch.isfinite(value).all():
        raise ValueError("vector contains NaN or infinity")
    norm = torch.linalg.vector_norm(value)
    if float(norm) > clip_norm:
        value = value * (clip_norm / norm)
    noise = torch.randn(value.shape, generator=generator, dtype=torch.float64)
    return value + noise * (clip_norm * noise_multiplier)
