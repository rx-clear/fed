"""Runnable research baselines for the SP comparison experiment.

BatchCrypt is represented by Paillier additive HE with fixed-point batch
packing.  Masking is represented by pairwise additive masks over the weighted
update.  Both implementations are intended for reproducible simulation, not
as replacements for the original cross-silo deployments.
"""

from __future__ import annotations

import hashlib
import math
import secrets
from typing import Dict, List, Optional

import torch
from sympy import randprime

from .dmcfe.plaintext_packing import PackedLayout, pack_signed, unpack_aggregate
from .quantization import dequantize, quantize


class Paillier:
    """Small additive Paillier implementation used by BatchCryptPrototype."""

    def __init__(self, modulus_bits: int = 256):
        half = int(modulus_bits) // 2
        lower, upper = 1 << (half - 1), 1 << half
        p = int(randprime(lower, upper))
        q = int(randprime(lower, upper))
        while p == q:
            q = int(randprime(lower, upper))
        self.n = p * q
        self.n2 = self.n * self.n
        self.g = self.n + 1
        lam = math.lcm(p - 1, q - 1)
        self._lambda_value = lam
        l_value = ((pow(self.g, lam, self.n2) - 1) // self.n) % self.n
        self.mu = pow(l_value, -1, self.n)

    def encrypt(self, message: int) -> int:
        message = int(message) % self.n
        while True:
            r = secrets.randbelow(self.n - 1) + 1
            if math.gcd(r, self.n) == 1:
                break
        return (pow(self.g, message, self.n2) * pow(r, self.n, self.n2)) % self.n2

    def decrypt(self, ciphertext: int) -> int:
        value = ((pow(int(ciphertext), self._lambda_value, self.n2) - 1) // self.n) % self.n
        decoded = (value * self.mu) % self.n
        return decoded if decoded <= self.n // 2 else decoded - self.n


class BatchCryptPrototype:
    """Quantize, batch-pack, Paillier-encrypt, and homomorphically aggregate."""

    def __init__(self, precision: int = 14, modulus_bits: int = 256):
        self.precision = int(precision)
        self.modulus_bits = int(modulus_bits)

    def aggregate(self, w_locals):
        if not w_locals:
            raise ValueError("w_locals must not be empty")
        total_weight = sum(int(weight) for weight, _ in w_locals)
        result: Dict[str, torch.Tensor] = {}
        for key in w_locals[0][1]:
            tensors = [local[key] for _, local in w_locals]
            if not tensors[0].is_floating_point():
                result[key] = tensors[0].detach().clone()
                continue
            flat = [quantize(tensor.detach().cpu().reshape(-1), self.precision) for tensor in tensors]
            max_abs = max(int(value.abs().max().item()) for value in flat) if flat else 0
            value_bits = max(8, max_abs.bit_length() + 2)
            layout = PackedLayout.for_clients(value_bits, len(flat), self.modulus_bits)
            paillier = Paillier(self.modulus_bits)
            # The prototype stores lambda explicitly; production code should
            # use a vetted Paillier library and protected key storage.
            encrypted_sum = 1
            for start in range(0, flat[0].numel(), layout.slots):
                packed_ciphertexts: List[int] = []
                for values, (weight, _) in zip(flat, w_locals):
                    packed = pack_signed(values[start:start + layout.slots].tolist(), layout)
                    packed_ciphertexts.append(paillier.encrypt(packed))
                for ciphertext, (weight, _) in zip(packed_ciphertexts, w_locals):
                    encrypted_sum = (encrypted_sum * pow(ciphertext, int(weight), paillier.n2)) % paillier.n2
                # Decrypt each batch independently; reset the product for the
                # next batch so ciphertexts cannot mix different slots.
                decoded = paillier.decrypt(encrypted_sum)
                if start == 0:
                    recovered = []
                recovered.extend(unpack_aggregate(decoded, layout, total_weight, min(layout.slots, flat[0].numel() - start)))
                encrypted_sum = 1
            vector = dequantize(torch.tensor(recovered, dtype=torch.int64), self.precision)
            result[key] = (vector / float(total_weight)).reshape(tensors[0].shape).to(tensors[0].device)
        return result


class PairwiseMaskingAggregator:
    """Additive pairwise mask baseline whose masks cancel at the server."""

    def __init__(self, seed: Optional[int] = None):
        self.seed = secrets.randbits(128) if seed is None else int(seed)

    def _mask(self, key: str, left: int, right: int, shape, dtype):
        digest = hashlib.sha256(f"Masking/v1/{self.seed}/{key}/{left}/{right}".encode()).digest()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int.from_bytes(digest[:8], "big"))
        return torch.randn(shape, generator=generator, dtype=dtype)

    def aggregate(self, w_locals):
        if not w_locals:
            raise ValueError("w_locals must not be empty")
        total = float(sum(int(weight) for weight, _ in w_locals))
        masked_updates = []
        for weight, local in w_locals:
            masked_updates.append({
                key: value * (float(weight) / total) if value.is_floating_point() else value.detach().clone()
                for key, value in local.items()
            })
        for left in range(len(w_locals)):
            for right in range(left + 1, len(w_locals)):
                for key, value in w_locals[left][1].items():
                    if not value.is_floating_point():
                        continue
                    mask = self._mask(key, left, right, value.shape, value.dtype).to(value.device)
                    masked_updates[left][key] += mask
                    masked_updates[right][key] -= mask
        result = {}
        for key in w_locals[0][1]:
            values = [update[key] for update in masked_updates]
            result[key] = values[0]
            for value in values[1:]:
                result[key] = result[key] + value
        return result
