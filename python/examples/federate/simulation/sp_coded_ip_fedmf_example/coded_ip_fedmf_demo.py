"""Run a small Coded-IP-FedMF dropout simulation without downloading data."""

from __future__ import annotations

import argparse
import json

import torch

from fedml.simulation.sp.fedavg.coded_ip_fedmf import (
    CodedIPFedMFAggregator,
    normalize_and_quantize,
    dequantize,
)


def run_demo(client_num: int, recovery_threshold: int, dropout: int, dimension: int) -> dict:
    if dropout < 0 or dropout >= client_num:
        raise ValueError("dropout must be in [0, client_num)")
    if client_num - dropout < recovery_threshold:
        raise ValueError("dropout leaves fewer than recovery_threshold clients")

    generator = torch.Generator().manual_seed(7)
    updates = {
        client_id: torch.randn(dimension, generator=generator) * 0.1
        for client_id in range(client_num)
    }
    expected = torch.stack(list(updates.values())).sum(dim=0)

    aggregator = CodedIPFedMFAggregator(client_num, recovery_threshold, seed=7)
    coded_round = aggregator.encode_client_updates(updates)
    online = tuple(range(client_num - dropout))
    result = aggregator.aggregate(coded_round, online)
    error = float((result.aggregate - expected).abs().max())

    quantized = normalize_and_quantize(expected)
    quantization_error = float((dequantize(quantized) - expected).abs().max())
    return {
        "client_num": client_num,
        "recovery_threshold": recovery_threshold,
        "dropout": dropout,
        "online_clients": list(result.online_client_ids),
        "max_recovery_error": error,
        "max_quantization_error": quantization_error,
        "reference_backend_secure": aggregator.ipfe.is_secure,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--client-num", type=int, default=8)
    parser.add_argument("--recovery-threshold", type=int, default=5)
    parser.add_argument("--dropout", type=int, default=2)
    parser.add_argument("--dimension", type=int, default=16)
    args = parser.parse_args()
    print(json.dumps(run_demo(**vars(args)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
