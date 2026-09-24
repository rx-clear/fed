"""Coded-IP-FedMF reference implementation.

This module contains the linear-coding part of the protocol described in the
paper supplied with the project.  It is deliberately independent from the
FedAvg trainer so it can be tested with tensors, model deltas, or a matrix
factorisation trainer.

The ``ReferenceInnerProduct`` backend models an inner-product functional
encryption API.  It masks each vector with a key-derived pad and only exposes
the requested weighted sum during decryption.  It is useful for protocol and
dropout experiments, but it is *not* a production cryptographic primitive.
Replace it with the project's DMCFE backend before making security claims.
"""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass, replace
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import torch


TensorLike = Union[torch.Tensor, Sequence[float]]


def normalize_and_quantize(
    vector: TensorLike,
    scale: int = 1 << 14,
    clip_bound: Optional[float] = None,
) -> torch.Tensor:
    """Return a deterministic signed int64 representation of a vector."""

    if isinstance(scale, bool) or not isinstance(scale, int) or scale <= 0:
        raise ValueError("scale must be positive")
    result = torch.as_tensor(vector, dtype=torch.float64).detach().cpu()
    if not torch.isfinite(result).all():
        raise ValueError("vector contains NaN or infinity")
    if clip_bound is not None:
        if not torch.isfinite(torch.tensor(float(clip_bound))) or clip_bound <= 0:
            raise ValueError("clip_bound must be positive")
        result = result.clamp(-float(clip_bound), float(clip_bound))
    quantized = torch.round(result * int(scale))
    # max-int64 converts to the same float as 2**63, so use the exclusive
    # upper bound directly before converting to int64.
    if quantized.numel() and float(quantized.abs().max()) >= float(1 << 63):
        raise OverflowError("quantized vector does not fit in int64")
    return quantized.to(torch.int64)


def dequantize(vector: torch.Tensor, scale: int = 1 << 14) -> torch.Tensor:
    """Convert a fixed-point vector back to float64."""

    if isinstance(scale, bool) or not isinstance(scale, int) or scale <= 0:
        raise ValueError("scale must be positive")
    if vector.dtype != torch.int64:
        raise TypeError("dequantize expects an int64 tensor")
    return vector.to(torch.float64) / float(scale)


class MDSCode:
    """A small systematic-independent Vandermonde MDS code.

    ``k`` source shards are encoded into ``n`` shares.  Any ``k`` distinct
    shares recover the source shards, which is the dropout guarantee used by
    Coded-IP-FedMF.
    """

    def __init__(self, n: int, k: int, *, dtype: torch.dtype = torch.float64):
        if not isinstance(n, int) or not isinstance(k, int):
            raise TypeError("n and k must be integers")
        if n < 1 or k < 1 or k > n:
            raise ValueError("require 1 <= k <= n")
        self.n = n
        self.k = k
        self.dtype = dtype
        points = torch.arange(1, n + 1, dtype=dtype)
        self.generator_matrix = torch.stack([points.pow(power) for power in range(k)], dim=1)

    def _validate_shards(self, shards: torch.Tensor) -> torch.Tensor:
        value = torch.as_tensor(shards, dtype=self.dtype).detach().cpu()
        if value.ndim < 1 or value.shape[0] != self.k:
            raise ValueError(f"expected {self.k} source shards, got shape {tuple(value.shape)}")
        return value

    def encode(self, shards: TensorLike) -> torch.Tensor:
        """Encode ``[k, ...]`` source shards into ``[n, ...]`` shares."""
        source = self._validate_shards(torch.as_tensor(shards))
        flat = source.reshape(self.k, -1)
        encoded = self.generator_matrix.matmul(flat)
        return encoded.reshape((self.n,) + tuple(source.shape[1:]))

    @staticmethod
    def _normalise_ids(client_ids: Iterable[int], n: int) -> Tuple[int, ...]:
        ids = tuple(int(client_id) for client_id in client_ids)
        if len(set(ids)) != len(ids):
            raise ValueError("client_ids must be unique")
        if any(client_id < 0 or client_id >= n for client_id in ids):
            raise IndexError("client id is outside the codeword")
        return ids

    def recovery_weights(self, client_ids: Iterable[int]) -> torch.Tensor:
        """Return weights whose inner product with shares equals source sum."""
        ids = self._normalise_ids(client_ids, self.n)
        if len(ids) < self.k:
            raise ValueError(f"at least {self.k} online clients are required")
        selected = self.generator_matrix[list(ids)]
        target = torch.ones(self.k, 1, dtype=self.dtype)
        # With more than k responses this is the minimum-norm exact solution.
        weights = torch.linalg.lstsq(selected.transpose(0, 1), target).solution
        residual = selected.transpose(0, 1).matmul(weights) - target
        # Float32 Vandermonde solves can have residuals around 1e-5 even for
        # full-rank rows. Scale the tolerance with the dtype and matrix size.
        eps = torch.finfo(self.dtype).eps
        conditioning_scale = max(
            1.0,
            float(selected.abs().max()) * max(1.0, float(weights.abs().max())),
        )
        tolerance = 100.0 * float(eps) * conditioning_scale
        if float(residual.abs().max()) > tolerance:
            raise ValueError("online client rows are not full rank")
        return weights.reshape(-1)

    def decode(
        self,
        shares: Union[Mapping[int, TensorLike], torch.Tensor],
        client_ids: Optional[Iterable[int]] = None,
    ) -> torch.Tensor:
        """Recover source shards from a share mapping or a share tensor."""
        if isinstance(shares, Mapping):
            ids = self._normalise_ids(shares.keys(), self.n)
            if client_ids is not None:
                raise ValueError("client_ids must be omitted for a mapping")
            share_values = torch.stack([torch.as_tensor(shares[i], dtype=self.dtype) for i in ids])
        else:
            if client_ids is None:
                raise ValueError("client_ids is required for a share tensor")
            ids = self._normalise_ids(client_ids, self.n)
            share_values = torch.as_tensor(shares, dtype=self.dtype).detach().cpu()
            if share_values.ndim < 1 or share_values.shape[0] != len(ids):
                raise ValueError("share tensor and client_ids have different lengths")
        if len(ids) < self.k:
            raise ValueError(f"at least {self.k} online clients are required")
        selected_ids = ids[: self.k]
        selected_values = share_values[: self.k]
        matrix = self.generator_matrix[list(selected_ids)]
        flat = selected_values.reshape(self.k, -1)
        decoded = torch.linalg.solve(matrix, flat)
        return decoded.reshape((self.k,) + tuple(selected_values.shape[1:]))

    def decode_sum(
        self,
        shares: Union[Mapping[int, TensorLike], torch.Tensor],
        client_ids: Optional[Iterable[int]] = None,
    ) -> torch.Tensor:
        """Recover and sum source shards without exposing them to callers."""
        return self.decode(shares, client_ids).sum(dim=0)


@dataclass(frozen=True)
class EncryptedVector:
    client_id: int
    values: torch.Tensor
    label: str


class ReferenceInnerProduct:
    """Protocol-compatible masked inner-product backend for local experiments."""

    is_secure = False

    def __init__(self, seed: Optional[int] = None):
        self._seed = seed
        self._keys: Dict[int, bytes] = {}

    def setup(self, client_ids: Iterable[int]) -> Dict[int, bytes]:
        self._keys = {
            int(client_id): secrets.token_bytes(32)
            if self._seed is None
            else hashlib.sha256(f"{self._seed}:{int(client_id)}".encode()).digest()
            for client_id in client_ids
        }
        return dict(self._keys)

    def _pad(self, client_id: int, label: str, shape: torch.Size) -> torch.Tensor:
        if client_id not in self._keys:
            raise KeyError(f"unknown client id {client_id}")
        count = 1
        for size in shape:
            count *= int(size)
        output = []
        key = self._keys[client_id]
        for index in range(count):
            digest = hashlib.sha256(key + label.encode() + index.to_bytes(8, "big")).digest()
            raw = int.from_bytes(digest[:8], "big") / float(1 << 64)
            output.append(raw - 0.5)
        return torch.tensor(output, dtype=torch.float64).reshape(shape)

    def encrypt(self, client_id: int, vector: TensorLike, label: str) -> EncryptedVector:
        value = torch.as_tensor(vector, dtype=torch.float64).detach().cpu()
        if not torch.isfinite(value).all():
            raise ValueError("cannot encrypt NaN or infinity")
        return EncryptedVector(int(client_id), value + self._pad(int(client_id), label, value.shape), label)

    def derive_key(self, client_ids: Iterable[int], weights: TensorLike, label: str, shape: torch.Size) -> torch.Tensor:
        ids = tuple(int(client_id) for client_id in client_ids)
        weight_tensor = torch.as_tensor(weights, dtype=torch.float64).reshape(-1)
        if len(ids) != len(weight_tensor):
            raise ValueError("client_ids and weights have different lengths")
        result = torch.zeros(shape, dtype=torch.float64)
        for client_id, weight in zip(ids, weight_tensor):
            result += float(weight) * self._pad(client_id, label, shape)
        return result

    def decrypt(
        self,
        ciphertexts: Sequence[EncryptedVector],
        weights: TensorLike,
        functional_key: torch.Tensor,
    ) -> torch.Tensor:
        if not ciphertexts:
            raise ValueError("ciphertexts cannot be empty")
        weight_tensor = torch.as_tensor(weights, dtype=torch.float64).reshape(-1)
        if len(ciphertexts) != len(weight_tensor):
            raise ValueError("ciphertexts and weights have different lengths")
        shape = ciphertexts[0].values.shape
        label = ciphertexts[0].label
        if any(ciphertext.values.shape != shape or ciphertext.label != label for ciphertext in ciphertexts):
            raise ValueError("ciphertexts must have matching labels and shapes")
        result = torch.zeros_like(ciphertexts[0].values, dtype=torch.float64)
        for ciphertext, weight in zip(ciphertexts, weight_tensor):
            result += float(weight) * ciphertext.values
        return result - functional_key


@dataclass(frozen=True)
class CodedRound:
    ciphertexts: Mapping[int, EncryptedVector]
    source_shape: Tuple[int, ...]
    label: str
    value_dtype: torch.dtype = torch.float64


@dataclass(frozen=True)
class AggregationResult:
    aggregate: torch.Tensor
    online_client_ids: Tuple[int, ...]
    used_client_ids: Tuple[int, ...]
    recovery_weights: torch.Tensor
    scores: Optional[Mapping[int, float]] = None
    accepted_client_ids: Optional[Tuple[int, ...]] = None


class CodedIPFedMFAggregator:
    """Encode source shards, encrypt shares, and aggregate after dropouts."""

    def __init__(
        self,
        client_num: int,
        recovery_threshold: int,
        *,
        quantization_scale: Optional[int] = None,
        seed: Optional[int] = 0,
    ):
        self.code = MDSCode(client_num, recovery_threshold)
        self.ipfe = ReferenceInnerProduct(seed=seed)
        self.client_num = client_num
        self.recovery_threshold = recovery_threshold
        self.quantization_scale = quantization_scale
        self.ipfe.setup(range(client_num))

    def encode_shards(self, source_shards: TensorLike, *, label: str = "coded-ip-fedmf") -> CodedRound:
        raw_source = torch.as_tensor(source_shards)
        output_dtype = raw_source.dtype if torch.is_floating_point(raw_source) else torch.float64
        source = self.code._validate_shards(raw_source)
        shares = self.code.encode(source)
        ciphertexts = {
            client_id: self.ipfe.encrypt(client_id, shares[client_id], label)
            for client_id in range(self.client_num)
        }
        return CodedRound(ciphertexts, tuple(source.shape), label, output_dtype)

    def encode_client_updates(
        self,
        updates: Union[Mapping[int, TensorLike], Sequence[TensorLike]],
        *,
        label: str = "coded-ip-fedmf",
    ) -> CodedRound:
        """Group client updates into source shards while preserving their sum.

        Grouping makes the reference simulator useful when ``client_num`` is
        larger than the recovery threshold.  A production coded worker setup
        should construct the source shards at the data-placement layer.
        """
        if isinstance(updates, Mapping):
            expected_ids = set(range(self.client_num))
            if set(updates) != expected_ids:
                raise ValueError(
                    f"update mapping must contain exactly client ids 0..{self.client_num - 1}"
                )
            values = [updates[i] for i in range(self.client_num)]
        else:
            values = list(updates)
        if len(values) != self.client_num:
            raise ValueError(f"expected {self.client_num} client updates")
        raw_tensors = [torch.as_tensor(value).detach().cpu() for value in values]
        output_dtype = raw_tensors[0].dtype if torch.is_floating_point(raw_tensors[0]) else torch.float64
        if any(value.shape != raw_tensors[0].shape for value in raw_tensors):
            raise ValueError("all client updates must have the same shape")
        tensors = [value.to(torch.float64) for value in raw_tensors]
        if self.quantization_scale is not None:
            tensors = [
                dequantize(normalize_and_quantize(value, self.quantization_scale))
                for value in tensors
            ]
        groups = torch.tensor_split(torch.stack(tensors), self.recovery_threshold, dim=0)
        source = torch.stack([group.sum(dim=0) for group in groups])
        coded_round = self.encode_shards(source, label=label)
        return CodedRound(
            coded_round.ciphertexts,
            coded_round.source_shape,
            coded_round.label,
            output_dtype,
        )

    def aggregate_updates(
        self,
        updates: Union[Mapping[int, TensorLike], Sequence[TensorLike]],
        online_client_ids: Iterable[int],
        *,
        label: str = "coded-ip-fedmf",
        scores: Optional[Mapping[int, float]] = None,
    ) -> AggregationResult:
        """Convenience API for one simulated federated round."""
        return self.aggregate(
            self.encode_client_updates(updates, label=label),
            online_client_ids,
            scores=scores,
        )

    def collect_uploaded_shares(
        self,
        coded_round: CodedRound,
        online_client_ids: Iterable[int],
    ) -> CodedRound:
        """Return the server view after only selected clients upload shares.

        ``encode_shards`` is still a local reference operation, but this
        helper makes the network boundary explicit: the returned round only
        contains ciphertexts that arrived.  It is therefore suitable for
        testing share loss during transport, while a real deployment must
        construct the source shards on clients before this step.
        """
        online = self.code._normalise_ids(online_client_ids, self.client_num)
        missing = [client_id for client_id in online if client_id not in coded_round.ciphertexts]
        if missing:
            raise ValueError(f"coded round is missing uploads for clients {missing}")
        return replace(
            coded_round,
            ciphertexts={client_id: coded_round.ciphertexts[client_id] for client_id in online},
        )

    def aggregate_weighted_updates(
        self,
        updates: Union[Mapping[int, TensorLike], Sequence[TensorLike]],
        client_weights: Union[Mapping[int, float], Sequence[float]],
        online_client_ids: Iterable[int],
        *,
        label: str = "coded-ip-fedmf",
    ) -> AggregationResult:
        """Aggregate an arbitrary per-client weighted sum.

        This is the dynamic-weighting operation described in the paper.  The
        weights are applied before coding, so the MDS recovery still works for
        any online subset of at least ``recovery_threshold`` clients.
        """
        if isinstance(updates, Mapping):
            expected_ids = set(range(self.client_num))
            if set(updates) != expected_ids:
                raise ValueError(
                    f"update mapping must contain exactly client ids 0..{self.client_num - 1}"
                )
            values = [updates[i] for i in range(self.client_num)]
        else:
            values = list(updates)
        if len(values) != self.client_num:
            raise ValueError(f"expected {self.client_num} client updates")

        if isinstance(client_weights, Mapping):
            expected_ids = set(range(self.client_num))
            if set(client_weights) != expected_ids:
                raise ValueError(
                    f"weight mapping must contain exactly client ids 0..{self.client_num - 1}"
                )
            weights = torch.tensor(
                [client_weights[i] for i in range(self.client_num)], dtype=torch.float64
            )
        else:
            weights = torch.as_tensor(client_weights, dtype=torch.float64).reshape(-1)
        if len(weights) != self.client_num:
            raise ValueError("client_weights and updates have different lengths")
        if not torch.isfinite(weights).all():
            raise ValueError("client_weights contain NaN or infinity")
        weighted = {
            client_id: torch.as_tensor(values[client_id]) * float(weights[client_id])
            for client_id in range(self.client_num)
        }
        return self.aggregate_updates(weighted, online_client_ids, label=label)

    def aggregate_with_byzantine_filter(
        self,
        updates: Union[Mapping[int, TensorLike], Sequence[TensorLike]],
        online_client_ids: Iterable[int],
        reference: TensorLike,
        score_threshold: float,
        *,
        label: str = "coded-ip-fedmf",
    ) -> AggregationResult:
        """Filter low-cosine-score updates before coded aggregation.

        ``score_updates`` is intentionally a plaintext reference filter.  It
        exercises the paper's acceptance rule, but does not claim that the
        insecure reference backend provides encrypted per-client scores.
        """
        if isinstance(updates, Mapping):
            expected_ids = set(range(self.client_num))
            if set(updates) != expected_ids:
                raise ValueError(
                    f"update mapping must contain exactly client ids 0..{self.client_num - 1}"
                )
            materialized = {i: updates[i] for i in range(self.client_num)}
        else:
            values = list(updates)
            if len(values) != self.client_num:
                raise ValueError(f"expected {self.client_num} client updates")
            materialized = {i: values[i] for i in range(self.client_num)}
        scores = self.score_updates(materialized, reference)
        accepted = self.valid_clients(scores, score_threshold)
        accepted_set = set(accepted)
        first = torch.as_tensor(materialized[0])
        filtered = {
            client_id: value if client_id in accepted_set else torch.zeros_like(first)
            for client_id, value in materialized.items()
        }
        result = self.aggregate_updates(
            filtered, online_client_ids, label=label, scores=scores
        )
        return replace(result, accepted_client_ids=accepted)

    def aggregate(
        self,
        coded_round: CodedRound,
        online_client_ids: Iterable[int],
        *,
        scores: Optional[Mapping[int, float]] = None,
    ) -> AggregationResult:
        online = self.code._normalise_ids(online_client_ids, self.client_num)
        if len(online) < self.recovery_threshold:
            raise ValueError(
                f"{len(online)} online clients are insufficient; "
                f"need {self.recovery_threshold}"
            )
        missing = [client_id for client_id in online if client_id not in coded_round.ciphertexts]
        if missing:
            raise ValueError(f"coded round is missing uploads for clients {missing}")
        used = online
        weights = self.code.recovery_weights(used)
        ciphertexts = [coded_round.ciphertexts[client_id] for client_id in used]
        functional_key = self.ipfe.derive_key(used, weights, coded_round.label, ciphertexts[0].values.shape)
        aggregate = self.ipfe.decrypt(ciphertexts, weights, functional_key).to(coded_round.value_dtype)
        return AggregationResult(aggregate, online, used, weights, scores)

    @staticmethod
    def score_updates(updates: Mapping[int, TensorLike], reference: TensorLike) -> Dict[int, float]:
        """Compute cosine scores used by the paper's Byzantine filter."""
        direction = torch.as_tensor(reference, dtype=torch.float64).reshape(-1)
        if not torch.isfinite(direction).all():
            raise ValueError("reference contains NaN or infinity")
        norm = torch.linalg.vector_norm(direction)
        if float(norm) == 0:
            raise ValueError("reference direction must be non-zero")
        output: Dict[int, float] = {}
        for client_id, update in updates.items():
            value = torch.as_tensor(update, dtype=torch.float64).reshape(-1)
            if not torch.isfinite(value).all():
                raise ValueError("updates contain NaN or infinity")
            if value.numel() != direction.numel():
                raise ValueError("updates and reference have different dimensions")
            value_norm = torch.linalg.vector_norm(value)
            output[int(client_id)] = 0.0 if float(value_norm) == 0 else float(torch.dot(value, direction) / (value_norm * norm))
        return output

    @staticmethod
    def valid_clients(scores: Mapping[int, float], threshold: float) -> Tuple[int, ...]:
        if not -1.0 <= threshold <= 1.0:
            raise ValueError("cosine threshold must be between -1 and 1")
        return tuple(sorted(int(client_id) for client_id, score in scores.items() if float(score) >= threshold))
