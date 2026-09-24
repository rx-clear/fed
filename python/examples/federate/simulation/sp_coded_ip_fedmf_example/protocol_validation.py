"""Focused protocol diagnostics for the claims made in the paper.

This complements ``paper_validation.py`` with checks that should be reported
alongside accuracy curves: conditioning of the real-valued Vandermonde code,
arbitrary (not prefix) online sets, and the distinction between a client
dropping before encoding versus a share dropping after central encoding.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from typing import Dict

import torch

from fedml.simulation.sp.fedavg.coded_ip_fedmf import CodedIPFedMFAggregator, MDSCode


def condition_number_report(
    client_num: int, recovery_threshold: int, max_subsets: int = 5000
) -> Dict[str, float]:
    """Return condition numbers for the full and threshold-row Vandermonde matrices."""
    if max_subsets < 1:
        raise ValueError("max_subsets must be positive")
    code = MDSCode(client_num, recovery_threshold)
    full_condition = float(torch.linalg.cond(code.generator_matrix))
    total_subsets = math.comb(client_num, recovery_threshold)
    if total_subsets <= max_subsets:
        subsets = itertools.combinations(range(client_num), recovery_threshold)
    else:
        # Use a deterministic sample when C(N,K) is too large for an
        # exhaustive condition-number scan.
        generator = torch.Generator().manual_seed(client_num + recovery_threshold)
        sampled = set()
        while len(sampled) < max_subsets:
            sampled.add(tuple(sorted(
                int(client_id)
                for client_id in torch.randperm(client_num, generator=generator)[:recovery_threshold]
            )))
        subsets = sampled
    subset_conditions = []
    for online in subsets:
        subset_conditions.append(float(torch.linalg.cond(code.generator_matrix[list(online)])))
    return {
        "full_generator_condition": full_condition,
        "minimum_subset_condition": min(subset_conditions),
        "maximum_subset_condition": max(subset_conditions),
        "mean_subset_condition": sum(subset_conditions) / len(subset_conditions),
    }


def validate_upload_dropout(
    client_num: int = 8,
    recovery_threshold: int = 5,
    dimension: int = 16,
) -> Dict[str, object]:
    """Distinguish share loss from a client dropping before source creation."""
    updates = {i: torch.ones(dimension) for i in range(client_num)}
    aggregator = CodedIPFedMFAggregator(client_num, recovery_threshold, seed=211)
    coded_round = aggregator.encode_client_updates(updates, label="diagnostic")
    online = tuple(range(client_num - recovery_threshold, client_num))
    received = aggregator.collect_uploaded_shares(coded_round, online)
    share_result = aggregator.aggregate(received, online)
    expected = torch.full((dimension,), float(client_num))
    missing_id = client_num - 1
    incomplete = dict(updates)
    del incomplete[missing_id]
    try:
        aggregator.encode_client_updates(incomplete)
    except ValueError as exc:
        return {
            "pre_upload_dropout_modelled": False,
            "share_upload_dropout_modelled": True,
            "share_upload_max_error": float((share_result.aggregate - expected).abs().max()),
            "received_uploads": len(received.ciphertexts),
            "missing_client": missing_id,
            "observed_error": str(exc),
        }
    return {
        "pre_upload_dropout_modelled": True,
        "share_upload_dropout_modelled": True,
        "share_upload_max_error": float((share_result.aggregate - expected).abs().max()),
        "received_uploads": len(received.ciphertexts),
        "missing_client": missing_id,
        "observed_error": None,
    }


def aligned_poisoning_demo(dimension: int = 16) -> Dict[str, object]:
    """Demonstrate the aligned-scaling attack against a cosine-only filter."""
    reference = torch.ones(dimension, dtype=torch.float64)
    scores = CodedIPFedMFAggregator.score_updates(
        {0: reference, 1: 20.0 * reference}, reference
    )
    return {
        "honest_score": scores[0],
        "aligned_attack_score": scores[1],
        "passes_threshold_0_5": scores[1] >= 0.5,
    }


def run_diagnostics(
    client_num: int = 8,
    recovery_threshold: int = 5,
    dimension: int = 16,
) -> Dict[str, object]:
    if client_num < 1 or recovery_threshold < 1 or recovery_threshold > client_num:
        raise ValueError("require 1 <= recovery_threshold <= client_num")
    return {
        "parameters": {
            "client_num": client_num,
            "recovery_threshold": recovery_threshold,
            "dimension": dimension,
        },
        "conditioning": condition_number_report(client_num, recovery_threshold),
        "upload_dropout": validate_upload_dropout(
            client_num, recovery_threshold, dimension
        ),
        "aligned_poisoning": aligned_poisoning_demo(dimension),
        "interpretation": {
            "real_vandermonde_is_numerically_approximate": True,
            "cosine_filter_is_not_byzantine_complete": True,
            "share_dropout_and_upload_dropout_are_distinct": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--client-num", type=int, default=8)
    parser.add_argument("--recovery-threshold", type=int, default=5)
    parser.add_argument("--dimension", type=int, default=16)
    args = parser.parse_args()
    print(json.dumps(run_diagnostics(**vars(args)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
