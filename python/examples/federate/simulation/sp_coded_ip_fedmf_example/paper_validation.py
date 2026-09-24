"""Executable checks for the mathematical claims in the Coded-IP-FedMF paper.

The checks in this module are deliberately split into two categories:

* protocol checks (MDS recovery, weighted sums, quantisation and the reference
  Byzantine rule), which can be tested numerically; and
* claim-audit entries for primitives that are only represented by transparent
  toy components here (real MCFE/IPFE security and production SecAgg/DP).

The latter entries prevent a successful toy simulation from being mistaken for
evidence of a cryptographic security theorem.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Mapping

import torch

from fedml.simulation.sp.fedavg.coded_ip_fedmf import (
    CodedIPFedMFAggregator,
    MDSCode,
    dequantize,
    normalize_and_quantize,
)
from paper_components import (
    GaussianDPAccountant,
    PairwiseMaskSecAgg,
    clip_and_add_gaussian_noise,
)


@dataclass(frozen=True)
class ValidationCheck:
    name: str
    status: str
    max_error: float
    details: Mapping[str, object]


def _check(
    name: str,
    error: float,
    details: Mapping[str, object],
    tolerance: float = 1e-8,
) -> ValidationCheck:
    return ValidationCheck(
        name=name,
        status="passed" if error <= tolerance else "failed",
        max_error=float(error),
        details=dict(details),
    )


def validate_mds_recovery(
    client_num: int = 8,
    recovery_threshold: int = 5,
    dimension: int = 16,
    seed: int = 101,
    max_subsets: int = 5000,
) -> ValidationCheck:
    """Check every (or a deterministic bounded sample of) online subsets."""
    if max_subsets < 1:
        raise ValueError("max_subsets must be positive")
    code = MDSCode(client_num, recovery_threshold)
    generator = torch.Generator().manual_seed(seed)
    source = torch.randn(recovery_threshold, dimension, generator=generator)
    shares = code.encode(source)
    total_subsets = math.comb(client_num, recovery_threshold)
    if total_subsets <= max_subsets:
        subsets = list(itertools.combinations(range(client_num), recovery_threshold))
    else:
        # Sample unique subsets without materialising C(N,K) combinations.
        subsets_set = set()
        sample_generator = torch.Generator().manual_seed(seed + 1)
        while len(subsets_set) < max_subsets:
            candidate = tuple(sorted(
                int(client_id)
                for client_id in torch.randperm(
                    client_num, generator=sample_generator
                )[:recovery_threshold]
            ))
            subsets_set.add(candidate)
        subsets = sorted(subsets_set)
    errors = []
    for online in subsets:
        recovered = code.decode_sum(shares[list(online)], online)
        errors.append(float((recovered - source.sum(dim=0)).abs().max()))
    return _check(
        "mds_dropout_recovery",
        max(errors, default=float("inf")),
        {"tested_subsets": len(subsets), "total_subsets": total_subsets},
        # Vandermonde interpolation over IEEE-754 reals is not exact.  Keep
        # the measured error in the report while allowing normal float64
        # conditioning error at this small demonstration size.
        tolerance=1e-6,
    )


def validate_weighted_aggregation(
    client_num: int = 8,
    recovery_threshold: int = 5,
    dimension: int = 16,
    seed: int = 103,
) -> ValidationCheck:
    """Verify the dynamic per-client weighted sum advertised by the paper."""
    generator = torch.Generator().manual_seed(seed)
    updates = {
        client_id: torch.randn(dimension, generator=generator, dtype=torch.float64)
        for client_id in range(client_num)
    }
    weights = torch.linspace(0.25, 1.25, client_num, dtype=torch.float64)
    aggregator = CodedIPFedMFAggregator(client_num, recovery_threshold, seed=seed)
    online = tuple(range(client_num - 1, client_num - recovery_threshold - 1, -1))
    result = aggregator.aggregate_weighted_updates(updates, weights, online)
    expected = sum(
        (updates[client_id].to(torch.float64) * weights[client_id])
        for client_id in range(client_num)
    )
    error = float((result.aggregate.to(torch.float64) - expected).abs().max())
    return _check(
        "weighted_sum_recovery",
        error,
        {"online_clients": list(online), "recovery_threshold": recovery_threshold},
    )


def validate_share_upload_dropout(
    client_num: int = 8,
    recovery_threshold: int = 5,
    dimension: int = 16,
    seed: int = 105,
) -> ValidationCheck:
    """Verify recovery when the server receives only a subset of shares.

    The source-shard construction remains a local reference operation.  This
    check covers the transport boundary (missing ciphertext uploads), while
    ``claim_audit`` keeps pre-upload client computation explicitly separate.
    """
    generator = torch.Generator().manual_seed(seed)
    updates = {
        client_id: torch.randn(dimension, generator=generator, dtype=torch.float64)
        for client_id in range(client_num)
    }
    aggregator = CodedIPFedMFAggregator(client_num, recovery_threshold, seed=seed)
    coded_round = aggregator.encode_client_updates(updates, label="upload-dropout")
    online = tuple(sorted(
        int(client_id)
        for client_id in torch.randperm(client_num, generator=generator)[:recovery_threshold]
    ))
    received = aggregator.collect_uploaded_shares(coded_round, online)
    result = aggregator.aggregate(received, online)
    expected = sum(updates.values(), torch.zeros(dimension, dtype=torch.float64))
    error = float((result.aggregate.to(torch.float64) - expected).abs().max())
    return _check(
        "share_upload_dropout_recovery",
        error,
        {
            "online_clients": list(online),
            "received_ciphertexts": len(received.ciphertexts),
            "recovery_threshold": recovery_threshold,
        },
        tolerance=1e-6,
    )


def validate_byzantine_filter(
    client_num: int = 6,
    recovery_threshold: int = 4,
    dimension: int = 12,
    seed: int = 107,
    score_threshold: float = 0.5,
) -> ValidationCheck:
    """Exercise the paper's cosine-score acceptance rule.

    This uses the explicitly plaintext reference filter.  It verifies the
    acceptance and aggregation semantics, not encrypted score privacy.
    """
    generator = torch.Generator().manual_seed(seed)
    reference = torch.randn(dimension, generator=generator, dtype=torch.float64)
    updates = {
        client_id: reference
        + 0.01 * torch.randn(dimension, generator=generator, dtype=torch.float64)
        for client_id in range(client_num - 1)
    }
    malicious_id = client_num - 1
    updates[malicious_id] = -2.0 * reference
    aggregator = CodedIPFedMFAggregator(client_num, recovery_threshold, seed=seed)
    online = tuple(range(recovery_threshold))
    result = aggregator.aggregate_with_byzantine_filter(
        updates,
        online,
        reference,
        score_threshold,
        label="paper-validation-byzantine",
    )
    expected_ids = tuple(range(client_num - 1))
    expected = sum((updates[i] for i in expected_ids), torch.zeros_like(reference))
    error = float((result.aggregate - expected).abs().max())
    accepted_ok = result.accepted_client_ids == expected_ids
    aligned_attack_score = CodedIPFedMFAggregator.score_updates(
        {malicious_id: 10.0 * reference}, reference
    )[malicious_id]
    if not accepted_ok:
        error = float("inf")
    return _check(
        "reference_byzantine_filter",
        error,
        {
            "accepted_clients": list(result.accepted_client_ids or ()),
            "malicious_client": malicious_id,
            "score_threshold": score_threshold,
            "aligned_scaling_attack_score": aligned_attack_score,
            "aligned_scaling_attack_would_pass": aligned_attack_score >= score_threshold,
        },
    )


def validate_quantization(
    dimension: int = 32,
    scale: int = 1 << 14,
    seed: int = 109,
) -> ValidationCheck:
    """Bound the fixed-point round-trip error used by the protocol interface."""
    generator = torch.Generator().manual_seed(seed)
    value = torch.randn(dimension, generator=generator, dtype=torch.float64)
    restored = dequantize(normalize_and_quantize(value, scale=scale), scale=scale)
    error = float((restored - value.to(torch.float64)).abs().max())
    return _check(
        "fixed_point_round_trip",
        error,
        {"scale": scale, "theoretical_rounding_bound": 0.5 / scale},
        tolerance=0.5 / scale + 1e-12,
    )


def validate_secagg_dropout(
    client_num: int = 8,
    dimension: int = 16,
    seed: int = 113,
) -> ValidationCheck:
    """Check toy pairwise-mask cancellation and dropout reconstruction."""
    generator = torch.Generator().manual_seed(seed)
    updates = {
        client_id: torch.randn(dimension, generator=generator, dtype=torch.float64)
        for client_id in range(client_num)
    }
    online = tuple(range(1, client_num, 2))
    if not online:
        online = (0,)
    secagg = PairwiseMaskSecAgg(client_num, seed=seed)
    masked = secagg.mask_updates(updates)
    result = secagg.aggregate({client_id: masked[client_id] for client_id in online}, online)
    expected = sum((updates[i] for i in online), torch.zeros(dimension, dtype=torch.float64))
    error = float((result.aggregate - expected).abs().max())
    return _check(
        "toy_secagg_dropout_recovery",
        error,
        {
            "online_clients": list(online),
            "dropped_clients": list(result.dropped_client_ids),
            "communication_rounds": result.communication_rounds,
            "recovery_messages": result.recovery_messages,
        },
    )


def validate_dp_accountant(
    dimension: int = 16,
    seed: int = 127,
) -> ValidationCheck:
    """Check finite Gaussian noise and repeatable accountant composition."""
    accountant = GaussianDPAccountant(
        noise_multiplier=1.2, sampling_rate=0.25, delta=1e-5
    )
    accountant.step(10)
    generator = torch.Generator().manual_seed(seed)
    value = torch.ones(dimension, dtype=torch.float64)
    noisy = clip_and_add_gaussian_noise(
        value, clip_norm=1.0, noise_multiplier=accountant.noise_multiplier,
        generator=generator,
    )
    if not torch.isfinite(noisy).all() or not math.isfinite(accountant.epsilon):
        error = float("inf")
    else:
        error = 0.0
    return _check(
        "toy_dp_accountant",
        error,
        {
            **accountant.report(),
            "noise_l2_error": float(torch.linalg.vector_norm(noisy - value)),
        },
    )


def claim_audit() -> List[Dict[str, str]]:
    """Return an explicit implementation status for each paper claim."""
    return [
        {
            "claim": "MDS recovery from any K online shares",
            "status": "verified_numerically",
            "evidence": "validate_mds_recovery",
        },
        {
            "claim": "Dynamic weighted aggregation",
            "status": "verified_numerically",
            "evidence": "validate_weighted_aggregation",
        },
        {
            "claim": "Cosine-score Byzantine filtering",
            "status": "verified_reference_only",
            "evidence": "validate_byzantine_filter; aligned scaling attacks still pass",
        },
        {
            "claim": "Byzantine robustness against arbitrary model poisoning",
            "status": "not_guaranteed",
            "evidence": "cosine threshold accepts any positive scalar multiple of the reference",
        },
        {
            "claim": "IND-CPA / sta-IND security under SXDH or DDH",
            "status": "not_implemented",
            "evidence": "ReferenceInnerProduct.is_secure is false",
        },
        {
            "claim": "SecAgg mask recovery with client dropouts",
            "status": "verified_toy_only",
            "evidence": "validate_secagg_dropout; PairwiseMaskSecAgg.is_secure is false",
        },
        {
            "claim": "Differential-privacy budget / CDP guarantee",
            "status": "verified_toy_only",
            "evidence": "validate_dp_accountant; use a reviewed accountant for claims",
        },
        {
            "claim": "Zero precision loss while adding non-zero DP noise",
            "status": "inconsistent_without_further_definition",
            "evidence": "validate_dp_accountant reports non-zero noise_l2_error",
        },
        {
            "claim": "Dropout before client upload",
            "status": "not_modelled",
            "evidence": "source-shard construction is central in the reference simulator",
        },
        {
            "claim": "Recovery after share upload loss",
            "status": "verified_reference_only",
            "evidence": "validate_share_upload_dropout; collect_uploaded_shares models missing ciphertexts",
        },
        {
            "claim": "MovieLens-1M/10M accuracy and baseline speedup",
            "status": "requires_experiment",
            "evidence": "run_full_sweep.py is synthetic unless --ratings is supplied",
        },
    ]


def run_suite(
    client_num: int = 8,
    recovery_threshold: int = 5,
    dimension: int = 16,
    seed: int = 101,
) -> Dict[str, object]:
    """Run all executable checks and return a JSON-serialisable report."""
    checks = [
        validate_mds_recovery(client_num, recovery_threshold, dimension, seed),
        validate_weighted_aggregation(client_num, recovery_threshold, dimension, seed + 2),
        validate_share_upload_dropout(
            client_num, recovery_threshold, dimension, seed + 3
        ),
        validate_byzantine_filter(
            max(client_num, recovery_threshold + 1),
            recovery_threshold,
            dimension,
            seed + 5,
        ),
        validate_quantization(dimension, seed=seed + 7),
        validate_secagg_dropout(client_num, dimension, seed + 9),
        validate_dp_accountant(dimension, seed + 11),
    ]
    return {
        "checks": [asdict(check) for check in checks],
        "all_numeric_checks_passed": all(check.status == "passed" for check in checks),
        "claim_audit": claim_audit(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--client-num", type=int, default=8)
    parser.add_argument("--recovery-threshold", type=int, default=5)
    parser.add_argument("--dimension", type=int, default=16)
    parser.add_argument("--seed", type=int, default=101)
    args = parser.parse_args()
    report = run_suite(**vars(args))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
