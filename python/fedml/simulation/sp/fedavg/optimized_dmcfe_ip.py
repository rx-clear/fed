"""Optimized fixed-point DMCFE-IP aggregation primitives.

The default backend is a vectorized reference backend.  It exercises the
same weighted inner-product interface as a real DMCFE implementation while
remaining explicit about its lack of cryptographic security.  A production
backend can implement ``WeightedInnerProductBackend`` without changing the
parameter aggregation or personalization code.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Mapping, Optional, Protocol, Sequence, Tuple

import torch


class WeightedInnerProductBackend(Protocol):
    """Backend contract for weighted vector aggregation."""

    name: str
    is_secure: bool

    def weighted_sum(
        self,
        vectors: Sequence[torch.Tensor],
        weights: Sequence[int],
        *,
        label: str,
        chunk_size: int,
    ) -> torch.Tensor:
        """Return ``sum_i weights[i] * vectors[i]``."""


class ChunkedReferenceBackend:
    """CPU reference backend with bounded peak memory.

    The implementation deliberately performs no encryption.  Chunking avoids
    materialising a full ``clients x coordinates`` stack and is useful for
    measuring the aggregation cost independently of a cryptographic backend.
    """

    name = "chunked-reference"
    is_secure = False

    def weighted_sum(
        self,
        vectors: Sequence[torch.Tensor],
        weights: Sequence[int],
        *,
        label: str,
        chunk_size: int,
    ) -> torch.Tensor:
        del label
        if not vectors:
            raise ValueError("vectors must not be empty")
        if len(vectors) != len(weights):
            raise ValueError("vectors and weights must have the same length")
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")

        flat_vectors = [vector.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
                        for vector in vectors]
        dimension = flat_vectors[0].numel()
        if any(vector.numel() != dimension for vector in flat_vectors):
            raise ValueError("all vectors must have the same number of elements")
        max_int = 1 << 63
        absolute_weight_sum = sum(abs(int(weight)) for weight in weights)
        max_coordinate = max(
            (int(vector.abs().max().item()) for vector in flat_vectors if vector.numel()),
            default=0,
        )
        if max_coordinate * absolute_weight_sum >= max_int:
            raise OverflowError("weighted integer sum does not fit in int64")

        result = torch.zeros(dimension, dtype=torch.int64)
        for start in range(0, dimension, int(chunk_size)):
            stop = min(start + int(chunk_size), dimension)
            chunk_sum = torch.zeros(stop - start, dtype=torch.int64)
            for vector, weight in zip(flat_vectors, weights):
                chunk_sum.add_(vector[start:stop] * int(weight))
            result[start:stop] = chunk_sum
        return result.reshape(vectors[0].shape)


class ToyMCFEBackend:
    """Small MCFE adapter for protocol-level correctness tests.

    It uses the repository's cached ``MCFE`` implementation coordinate by
    coordinate.  The backend is intentionally marked insecure because the
    current implementation uses reduced toy parameters and is not a reviewed
    production primitive.
    """

    name = "toy-mcfe"
    is_secure = False

    def __init__(self, modulus_bits: int = 256) -> None:
        from .dmcfe.mcfe import MCFE

        if isinstance(modulus_bits, bool) or not isinstance(modulus_bits, int) or modulus_bits < 64:
            raise ValueError("modulus_bits must be an integer >= 64")
        self.mcfe = MCFE(modulus_bits=modulus_bits)
        self._secret_keys: list[int] = []

    def _ensure_keys(self, count: int) -> None:
        while len(self._secret_keys) < count:
            self._secret_keys.append(self.mcfe.keygen())

    def weighted_sum(
        self,
        vectors: Sequence[torch.Tensor],
        weights: Sequence[int],
        *,
        label: str,
        chunk_size: int,
    ) -> torch.Tensor:
        if not vectors:
            raise ValueError("vectors must not be empty")
        flat = [vector.detach().cpu().to(torch.int64).reshape(-1) for vector in vectors]
        dimension = flat[0].numel()
        if any(vector.numel() != dimension for vector in flat):
            raise ValueError("all vectors must have the same number of elements")
        self._ensure_keys(len(flat))
        derived_key = self.mcfe.derive_key(
            self._secret_keys[: len(flat)],
            weights,
        )
        output = torch.empty(dimension, dtype=torch.int64)
        for start in range(0, dimension, max(1, int(chunk_size))):
            stop = min(start + max(1, int(chunk_size)), dimension)
            for index in range(start, stop):
                ciphertexts = [
                    self.mcfe.encrypt(self._secret_keys[client], int(vector[index]), label)
                    for client, vector in enumerate(flat)
                ]
                output[index] = self.mcfe.decrypt(
                    ciphertexts,
                    weights,
                    derived_key,
                    label,
                )
        return output.reshape(vectors[0].shape)


@dataclass(frozen=True)
class DMCFEAggregationDiagnostics:
    """Metrics needed to report numerical and runtime behavior."""

    backend: str
    backend_secure: bool
    round_idx: int
    clients: int
    total_samples: int
    coordinates: int
    chunks: int
    chunk_size: int
    scale: int
    clip_bound: float
    clipped_fraction: float
    max_abs_error: float
    mean_abs_error: float
    encode_seconds: float
    aggregate_seconds: float
    total_aggregation_seconds: float


@dataclass(frozen=True)
class DMCFEAggregationResult:
    """Aggregated deltas and their numerical diagnostics."""

    deltas: Mapping[str, torch.Tensor]
    diagnostics: DMCFEAggregationDiagnostics


class OptimizedDMCFEIPAggregator:
    """Aggregate model deltas using fixed-point weighted inner products.

    ``local_deltas`` contains ``(sample_count, delta_state)`` pairs.  Only
    floating-point state is quantized; integer buffers must agree across
    clients and are copied unchanged.  The operation is full-dimensional for
    every key in ``key_order``; there is no plaintext suffix fallback.
    """

    def __init__(
        self,
        *,
        scale: int = 1 << 14,
        clip_bound: float = 8.0,
        chunk_size: int = 1 << 16,
        backend: Optional[WeightedInnerProductBackend] = None,
    ) -> None:
        if isinstance(scale, bool) or not isinstance(scale, int) or scale <= 0:
            raise ValueError("scale must be a positive integer")
        if not math.isfinite(float(clip_bound)) or clip_bound <= 0:
            raise ValueError("clip_bound must be finite and positive")
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer")
        self.scale = int(scale)
        self.clip_bound = float(clip_bound)
        self.chunk_size = int(chunk_size)
        self.backend = backend or ChunkedReferenceBackend()

    @staticmethod
    def _validate_weights(local_deltas: Sequence[Tuple[int, Mapping[str, torch.Tensor]]]) -> list[int]:
        weights = []
        for sample_count, _ in local_deltas:
            if isinstance(sample_count, bool) or int(sample_count) != sample_count or int(sample_count) <= 0:
                raise ValueError("sample counts must be positive integers")
            weights.append(int(sample_count))
        return weights

    def _quantize(self, value: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        value = value.detach().to(device="cpu", dtype=torch.float64)
        if not torch.isfinite(value).all():
            raise ValueError("DMCFE-IP does not accept NaN or infinity")
        clipped = int((value.abs() > self.clip_bound).sum().item())
        value = value.clamp(-self.clip_bound, self.clip_bound)
        quantized = torch.round(value * self.scale)
        if quantized.numel() and float(quantized.abs().max().item()) >= float(1 << 63):
            raise OverflowError("quantized update does not fit in int64")
        return quantized.to(torch.int64), clipped, value.numel()

    def aggregate_vectors(
        self,
        vectors: Sequence[torch.Tensor],
        weights: Sequence[int],
        *,
        round_idx: int = 0,
        label: str = "dmcfe-ip",
    ) -> tuple[torch.Tensor, DMCFEAggregationDiagnostics]:
        total_start = time.perf_counter()
        if not vectors:
            raise ValueError("vectors must not be empty")
        if len(vectors) != len(weights):
            raise ValueError("vectors and weights must have the same length")
        if int(round_idx) != round_idx or int(round_idx) < 0:
            raise ValueError("round_idx must be a non-negative integer")
        weight_values = [int(weight) for weight in weights]
        if any(weight <= 0 for weight in weight_values):
            raise ValueError("weights must be positive integers")

        quantized = []
        clipped = 0
        coordinates = 0
        original = []
        for vector in vectors:
            value = vector.detach().to(device="cpu", dtype=torch.float64)
            if not torch.isfinite(value).all():
                raise ValueError("DMCFE-IP does not accept NaN or infinity")
            original.append(value.reshape(-1))
            quantized_vector, clipped_count, total = self._quantize(value)
            quantized.append(quantized_vector)
            clipped += clipped_count
            coordinates = max(coordinates, total)

        encode_seconds = time.perf_counter() - total_start
        aggregate_start = time.perf_counter()
        weighted_sum = self.backend.weighted_sum(
            quantized,
            weight_values,
            label=f"{label}:round={int(round_idx)}",
            chunk_size=self.chunk_size,
        )
        aggregate_seconds = time.perf_counter() - aggregate_start
        total_samples = sum(weight_values)
        aggregate = weighted_sum.to(torch.float64) / float(self.scale * total_samples)
        plain = sum(
            (value * weight for value, weight in zip(original, weight_values)),
            torch.zeros_like(original[0]),
        ) / float(total_samples)
        error = (aggregate.reshape(-1) - plain).abs()
        chunks = max(1, math.ceil(coordinates / self.chunk_size))
        diagnostics = DMCFEAggregationDiagnostics(
            backend=self.backend.name,
            backend_secure=bool(self.backend.is_secure),
            round_idx=int(round_idx),
            clients=len(vectors),
            total_samples=total_samples,
            coordinates=coordinates,
            chunks=chunks,
            chunk_size=self.chunk_size,
            scale=self.scale,
            clip_bound=self.clip_bound,
            clipped_fraction=clipped / float(max(1, len(vectors) * coordinates)),
            max_abs_error=float(error.max().item()) if error.numel() else 0.0,
            mean_abs_error=float(error.mean().item()) if error.numel() else 0.0,
            encode_seconds=encode_seconds,
            aggregate_seconds=aggregate_seconds,
            total_aggregation_seconds=time.perf_counter() - total_start,
        )
        return aggregate.reshape(vectors[0].shape), diagnostics

    def aggregate_deltas(
        self,
        local_deltas: Sequence[Tuple[int, Mapping[str, torch.Tensor]]],
        *,
        key_order: Sequence[str] | None = None,
        round_idx: int = 0,
    ) -> DMCFEAggregationResult:
        total_start = time.perf_counter()
        if not local_deltas:
            raise ValueError("local_deltas must not be empty")
        weights = self._validate_weights(local_deltas)
        first_state = local_deltas[0][1]
        keys = list(key_order) if key_order is not None else list(first_state.keys())
        if not keys:
            raise ValueError("at least one parameter key is required")
        expected = set(keys)
        if set(first_state.keys()) != expected:
            raise ValueError("first client delta keys do not match key_order")
        if any(set(state.keys()) != expected for _, state in local_deltas[1:]):
            raise ValueError("client delta keys do not match key_order")

        output: dict[str, torch.Tensor] = {}
        max_error = 0.0
        weighted_error_sum = 0.0
        clipped = 0.0
        coordinates = 0
        chunks = 0
        encode_seconds = 0.0
        aggregate_seconds = 0.0
        for key in keys:
            reference_original = local_deltas[0][1][key]
            values = [state[key].detach().cpu() for _, state in local_deltas]
            reference = values[0]
            if not torch.is_floating_point(reference):
                if any(not torch.equal(value, reference) for value in values[1:]):
                    raise ValueError(f"integer parameter {key} differs across clients")
                output[key] = reference.clone().to(device=reference_original.device)
                continue
            aggregate, diagnostics = self.aggregate_vectors(
                values,
                weights,
                round_idx=round_idx,
                label=f"dmcfe-ip:{key}",
            )
            output[key] = aggregate.to(
                device=reference_original.device,
                dtype=reference_original.dtype,
            )
            max_error = max(max_error, diagnostics.max_abs_error)
            weighted_error_sum += diagnostics.mean_abs_error * diagnostics.coordinates
            clipped += diagnostics.clipped_fraction * len(values) * diagnostics.coordinates
            coordinates += diagnostics.coordinates
            chunks += diagnostics.chunks
            encode_seconds += diagnostics.encode_seconds
            aggregate_seconds += diagnostics.aggregate_seconds

        total_float_values = max(1, len(local_deltas) * coordinates)
        diagnostics = DMCFEAggregationDiagnostics(
            backend=self.backend.name,
            backend_secure=bool(self.backend.is_secure),
            round_idx=int(round_idx),
            clients=len(local_deltas),
            total_samples=sum(weights),
            coordinates=coordinates,
            chunks=chunks,
            chunk_size=self.chunk_size,
            scale=self.scale,
            clip_bound=self.clip_bound,
            clipped_fraction=clipped / float(total_float_values),
            max_abs_error=max_error,
            mean_abs_error=weighted_error_sum / float(max(1, coordinates)),
            encode_seconds=encode_seconds,
            aggregate_seconds=aggregate_seconds,
            total_aggregation_seconds=time.perf_counter() - total_start,
        )
        return DMCFEAggregationResult(output, diagnostics)
