"""Bounded, non-secure timing probe for the repository's toy MCFE primitive.

The measured integer payloads are produced by this script. They are not FedML
wire messages, and these timings must not be reported as production DMCFE-IP
or full-model federated-training overhead.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import sympy

from fedml.simulation.sp.fedavg.dmcfe.mcfe import MCFE


TIMING_FIELDS = (
    "prepare_encryption_seconds", "encryption_seconds", "derive_key_seconds",
    "prepare_decryption_seconds", "decryption_seconds", "payload_encode_seconds",
    "round_seconds",
)
PAYLOAD_FIELDS = ("ciphertext_payload_bytes", "derived_key_payload_bytes")
TRIAL_FIELDS = (
    "seed", "dimension", "trial", "clients", "modulus_bits", "max_abs_error",
    *TIMING_FIELDS, *PAYLOAD_FIELDS,
)
SETUP_FIELDS = ("seed", "dimension", "modulus_bits", "setup_seconds", "keygen_seconds")
SUMMARY_FIELDS = (
    "dimension", "clients", "modulus_bits", "seeds", "repeats_per_seed",
    "metric", "unit", "mean_seed_median", "sd_seed_median",
)


def parse_integers(raw: str, *, name: str, allow_zero: bool = False) -> list[int]:
    """Parse a short comma-separated set of integers without silent duplicates."""
    try:
        values = [int(part.strip()) for part in raw.split(",")]
    except ValueError as exc:
        raise ValueError(f"{name} must be comma-separated integers") from exc
    lower = 0 if allow_zero else 1
    if not values or any(value < lower for value in values) or len(values) != len(set(values)):
        raise ValueError(f"{name} must contain distinct integers >= {lower}")
    return values


def signed_integer_bytes(value: int) -> bytes:
    """Return the shortest signed big-endian two's-complement encoding."""
    width = 1
    while True:
        try:
            return int(value).to_bytes(width, byteorder="big", signed=True)
        except OverflowError:
            width += 1


def make_messages(seed: int, clients: int, dimension: int) -> list[list[int]]:
    rng = random.Random(seed)
    return [[rng.randint(-256, 256) for _ in range(dimension)] for _ in range(clients)]


def measure_trial(
    mcfe: MCFE,
    secret_keys: list[int],
    weights: list[int],
    messages: list[list[int]],
    *,
    label: str,
) -> dict[str, float | int]:
    """Time one fresh-label weighted sum and check every decrypted coordinate."""
    if not secret_keys or len(secret_keys) != len(weights) or len(weights) != len(messages):
        raise ValueError("keys, weights, and message vectors must have equal nonzero length")
    dimension = len(messages[0])
    if dimension < 1 or any(len(vector) != dimension for vector in messages):
        raise ValueError("message vectors must have equal positive dimension")

    round_start = time.perf_counter()
    start = time.perf_counter()
    contexts = [mcfe.prepare_encryption(secret, label) for secret in secret_keys]
    prepare_encryption_seconds = time.perf_counter() - start

    start = time.perf_counter()
    ciphertexts = [
        [mcfe.encrypt_prepared(contexts[client], messages[client][coordinate])
         for client in range(len(messages))]
        for coordinate in range(dimension)
    ]
    encryption_seconds = time.perf_counter() - start

    start = time.perf_counter()
    derived_key = mcfe.derive_key(secret_keys, weights)
    derive_key_seconds = time.perf_counter() - start

    start = time.perf_counter()
    mcfe.prepare_decryption(derived_key, label)
    prepare_decryption_seconds = time.perf_counter() - start

    start = time.perf_counter()
    decoded = [mcfe.decrypt(row, weights, derived_key, label) for row in ciphertexts]
    decryption_seconds = time.perf_counter() - start
    expected = [
        sum(weights[client] * messages[client][coordinate] for client in range(len(messages)))
        for coordinate in range(dimension)
    ]
    max_abs_error = max(abs(actual - target) for actual, target in zip(decoded, expected))
    if max_abs_error:
        raise ArithmeticError(f"toy MCFE weighted sum differs by {max_abs_error}")

    start = time.perf_counter()
    ciphertext_width = (mcfe.N2.bit_length() + 7) // 8
    ciphertext_payload = b"".join(
        int(value).to_bytes(ciphertext_width, byteorder="big", signed=False)
        for row in ciphertexts for value in row
    )
    derived_key_payload = signed_integer_bytes(derived_key)
    payload_encode_seconds = time.perf_counter() - start
    return {
        "max_abs_error": max_abs_error,
        "prepare_encryption_seconds": prepare_encryption_seconds,
        "encryption_seconds": encryption_seconds,
        "derive_key_seconds": derive_key_seconds,
        "prepare_decryption_seconds": prepare_decryption_seconds,
        "decryption_seconds": decryption_seconds,
        "payload_encode_seconds": payload_encode_seconds,
        "round_seconds": time.perf_counter() - round_start,
        "ciphertext_payload_bytes": len(ciphertext_payload),
        "derived_key_payload_bytes": len(derived_key_payload),
    }


def summarize(
    trials: list[dict[str, float | int]],
    setups: list[dict[str, float | int]],
    *,
    clients: int,
    modulus_bits: int,
    repeats: int,
) -> list[dict[str, float | int | str]]:
    """Treat one seed median as one observation for each timing metric."""
    by_dimension_seed: dict[tuple[int, int], list[dict[str, float | int]]] = defaultdict(list)
    for row in trials:
        by_dimension_seed[(int(row["dimension"]), int(row["seed"]))].append(row)
    setup_by_dimension_seed = {
        (int(row["dimension"]), int(row["seed"])): row for row in setups
    }
    dimensions = sorted({dimension for dimension, _ in by_dimension_seed})
    output: list[dict[str, float | int | str]] = []
    for dimension in dimensions:
        seeds = sorted(seed for dim, seed in by_dimension_seed if dim == dimension)
        for metric in (*TIMING_FIELDS, *PAYLOAD_FIELDS, "setup_seconds", "keygen_seconds"):
            values = []
            for seed in seeds:
                if metric in ("setup_seconds", "keygen_seconds"):
                    value = float(setup_by_dimension_seed[(dimension, seed)][metric])
                else:
                    value = statistics.median(
                        float(row[metric]) for row in by_dimension_seed[(dimension, seed)]
                    )
                values.append(value)
            output.append({
                "dimension": dimension,
                "clients": clients,
                "modulus_bits": modulus_bits,
                "seeds": len(seeds),
                "repeats_per_seed": repeats,
                "metric": metric,
                "unit": "bytes" if metric in PAYLOAD_FIELDS else "seconds",
                "mean_seed_median": statistics.mean(values),
                "sd_seed_median": statistics.stdev(values) if len(values) > 1 else "",
            })
    return output


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimensions", default="1,8,32")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--clients", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--modulus-bits", type=int, default=256)
    parser.add_argument("--output-dir", type=Path, default=Path("results/toy_mcfe_probe_v1"))
    args = parser.parse_args(argv)
    try:
        dimensions = parse_integers(args.dimensions, name="dimensions")
        seeds = parse_integers(args.seeds, name="seeds", allow_zero=True)
        if max(dimensions) > 256 or len(seeds) > 20 or not 2 <= args.clients <= 8:
            raise ValueError("toy probe limits: dimensions <= 256, <= 20 seeds, 2-8 clients")
        if not 1 <= args.repeats <= 100 or not 0 <= args.warmups <= 100:
            raise ValueError("repeats must be 1-100 and warmups 0-100")
        if not 64 <= args.modulus_bits <= 512:
            raise ValueError("toy modulus_bits must be 64-512")
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            raise ValueError(f"output directory is not empty: {args.output_dir}")
    except ValueError as exc:
        parser.error(str(exc))

    trials: list[dict[str, float | int]] = []
    setups: list[dict[str, float | int]] = []
    weights = [32 + 7 * client for client in range(args.clients)]
    for dimension in dimensions:
        for seed in seeds:
            start = time.perf_counter()
            mcfe = MCFE(modulus_bits=args.modulus_bits)
            setup_seconds = time.perf_counter() - start
            start = time.perf_counter()
            secret_keys = [mcfe.keygen() for _ in range(args.clients)]
            keygen_seconds = time.perf_counter() - start
            setups.append({
                "seed": seed, "dimension": dimension, "modulus_bits": args.modulus_bits,
                "setup_seconds": setup_seconds, "keygen_seconds": keygen_seconds,
            })
            messages = make_messages(seed, args.clients, dimension)
            for trial in range(-args.warmups, args.repeats):
                label = f"toy-mcfe-probe:v1:dim={dimension}:seed={seed}:trial={trial}"
                measurement = measure_trial(mcfe, secret_keys, weights, messages, label=label)
                if trial >= 0:
                    trials.append({
                        "seed": seed, "dimension": dimension, "trial": trial,
                        "clients": args.clients, "modulus_bits": args.modulus_bits,
                        **measurement,
                    })
            print(f"[OK] dimension={dimension} seed={seed}: {args.repeats} exact trials", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "trials.csv", TRIAL_FIELDS, trials)
    write_csv(args.output_dir / "setup.csv", SETUP_FIELDS, setups)
    write_csv(args.output_dir / "summary.csv", SUMMARY_FIELDS, summarize(
        trials, setups, clients=args.clients, modulus_bits=args.modulus_bits, repeats=args.repeats,
    ))
    source = Path(sys.modules[MCFE.__module__].__file__)
    metadata = {
        "backend": "toy-mcfe", "backend_secure": False,
        "claim_boundary": "small_correctness_probe_only",
        "dimensions": dimensions, "seeds": seeds, "clients": args.clients,
        "repeats": args.repeats, "warmups": args.warmups,
        "modulus_bits": args.modulus_bits, "weights": weights,
        "message_generation": "random.Random(seed), uniform integers [-256, 256]",
        "label_policy": "fresh label per trial; one label shared by all coordinates",
        "ciphertext_payload_encoding": "fixed-width unsigned big-endian, width=ceil(bit_length(N^2)/8)",
        "derived_key_payload_encoding": "minimal signed big-endian two's complement",
        "payload_excludes": ["metadata", "network framing", "key exchange", "transport"],
        "python": sys.version, "platform": platform.platform(), "sympy": sympy.__version__,
        "mcfe_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "benchmark_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (args.output_dir / "environment.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(trials)} trials to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
