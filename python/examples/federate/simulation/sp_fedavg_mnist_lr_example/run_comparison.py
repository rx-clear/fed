"""Run MNIST comparisons without discarding completed runs.

``Ours`` remains a CLI alias for the fixed-point reference aggregation path.
The resulting method label states what was actually implemented.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Callable

import yaml


METHODS = {
    "FedAvg": {"optimizer": "FedAvg", "dmcfe": False, "slug": "fedavg", "label": "FedAvg", "backend": "plaintext"},
    "FedRep": {"optimizer": "FedRep", "dmcfe": False, "slug": "fedrep", "label": "FedRep", "backend": "plaintext"},
    "Ours": {"optimizer": "FedRep", "dmcfe": True, "slug": "fedrep_reference", "label": "FedRep+FixedPointReference", "backend": "chunked-reference"},
    "BatchCrypt": {"optimizer": "FedAvg", "dmcfe": False, "slug": "batchcrypt", "label": "BatchCrypt", "backend": "BatchCrypt", "privacy": "BatchCrypt"},
    "Masking": {"optimizer": "FedAvg", "dmcfe": False, "slug": "masking", "label": "Masking", "backend": "Masking", "privacy": "Masking"},
    "HybridAlpha": None,
    "CryptoFE": None,
}
REFERENCES = {
    "HybridAlpha": "https://arxiv.org/abs/1912.05897",
    "CryptoFE": "https://doi.org/10.1109/GLOBECOM48099.2022.10001080",
}
MANIFEST_FIELDS = [
    "run_id", "requested_method", "method", "seed", "rounds", "status", "rows",
    "result_file", "partial_file", "config_hash", "matrix_id", "config_snapshot",
    "environment_snapshot", "backend", "backend_secure", "exit_code", "error", "reason", "reference",
]
REQUIRED_METRICS = {
    "run_id", "method", "seed", "round", "test_acc", "test_loss",
    "config_hash", "matrix_id", "backend", "backend_secure",
}


def _hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)):
            raise ValueError(f"Duplicate CSV columns: {path}")
        return fields, list(reader)


def _config(base: dict, method: str, seed: int, rounds: int,
            output: Path, run_id: str, matrix_id: str) -> dict:
    spec = METHODS[method]
    config = copy.deepcopy(base)
    config.setdefault("common_args", {})["random_seed"] = seed
    config["common_args"]["experiment_run_id"] = run_id
    config["common_args"]["experiment_method"] = spec["label"]
    config["common_args"]["experiment_matrix_id"] = matrix_id
    config["common_args"]["mlops_run_name"] = run_id
    config.setdefault("tracking_args", {})["run_name"] = run_id
    train = config.setdefault("train_args", {})
    train.update({
        "federated_optimizer": spec["optimizer"],
        "enable_dmcfe": spec["dmcfe"],
        "privacy_aggregation": spec.get("privacy"),
        "comm_round": rounds,
        "run_dmcfe_smoke_tests": False,
        "experiment_result_file": str(output),
    })
    if spec["dmcfe"]:
        train.update({"dmcfe_aggregation_mode": "optimized", "dmcfe_backend": "chunked_reference"})
    config.setdefault("validation_args", {})["frequency_of_the_test"] = 1
    config["common_args"]["experiment_config_hash"] = _hash(config)
    return config


def _validate_result(path: Path, record: dict[str, object]) -> tuple[list[str], list[dict[str, str]]]:
    fields, rows = _read_csv(path)
    missing = REQUIRED_METRICS - set(fields)
    if missing:
        raise ValueError(f"{path}: missing fields {sorted(missing)}")
    rounds = int(record["rounds"])
    if len(rows) != rounds:
        raise ValueError(f"{path}: expected {rounds} rows, found {len(rows)}")
    seen = set()
    for row in rows:
        try:
            round_idx = int(row["round"])
            accuracy = float(row["test_acc"])
            loss = float(row["test_loss"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path}: invalid round or test metric") from exc
        if round_idx in seen or not 0 <= round_idx < rounds:
            raise ValueError(f"{path}: duplicate or out-of-range round {round_idx}")
        seen.add(round_idx)
        if not math.isfinite(accuracy) or not 0 <= accuracy <= 1 or not math.isfinite(loss) or loss < 0:
            raise ValueError(f"{path}: invalid test metrics at round {round_idx}")
        for key in ("run_id", "method", "seed", "config_hash", "matrix_id", "backend"):
            if row[key] != str(record[key]):
                raise ValueError(f"{path}: {key} mismatch at round {round_idx}")
        if row["backend_secure"] != "False":
            raise ValueError(f"{path}: unexpected secure-backend claim")
        if record["backend"] == "chunked-reference":
            for key in ("coordinates", "chunks", "clipped_fraction", "scale", "clip_bound", "encode_seconds", "aggregate_seconds"):
                if not row.get(key):
                    raise ValueError(f"{path}: missing {key} at round {round_idx}")
            if int(row["coordinates"]) <= 0 or not 0 <= float(row["clipped_fraction"]) <= 1:
                raise ValueError(f"{path}: invalid aggregation diagnostics")
    if seen != set(range(rounds)):
        raise ValueError(f"{path}: incomplete round sequence")
    return fields, sorted(rows, key=lambda row: int(row["round"]))


def _load_manifest(path: Path) -> list[dict[str, str]]:
    return _read_csv(path)[1] if path.exists() else []


def _save_manifest(path: Path, records: list[dict[str, object]]) -> None:
    _atomic_csv(path, MANIFEST_FIELDS, records)


def _upsert(records: list[dict[str, object]], new: dict[str, object]) -> None:
    for index, record in enumerate(records):
        if (record.get("run_id") and record["run_id"] == new["run_id"]) or (
            not record.get("run_id") and new.get("status") == "not_implemented"
            and record.get("method") == new.get("method")
            and record.get("status") == "not_implemented"
        ):
            records[index] = new
            return
    records.append(new)


def _merge(result_dir: Path, records: list[dict[str, object]]) -> None:
    fields = ["run_id", "method", "seed", "round", "test_acc", "test_loss"]
    merged_rows: list[dict[str, str]] = []
    completed = sorted(
        (record for record in records if record.get("status") == "completed" and record.get("run_id")),
        key=lambda record: (str(record.get("matrix_id")), str(record.get("method")), int(record.get("seed", 0))),
    )
    for record in completed:
        source = Path(str(record["result_file"]))
        row_fields, rows = _validate_result(source, record)
        fields.extend(field for field in row_fields if field not in fields)
        merged_rows.extend(rows)
    _atomic_csv(result_dir / "comparison_metrics.csv", fields, merged_rows)


def _source_hashes() -> dict[str, str]:
    example_root = Path(__file__).resolve().parent
    source_root = Path(__file__).resolve().parents[4]
    source_paths = [
        example_root / "run_comparison.py",
        example_root / "torch_fedavg_mnist_lr_one_line_example.py",
        source_root / "fedml/simulation/sp/fedavg/fedavg_api.py",
        source_root / "fedml/simulation/sp/fedavg/experiment_recorder.py",
        source_root / "fedml/simulation/sp/fedavg/optimized_dmcfe_ip.py",
        source_root / "fedml/simulation/sp/fedavg/client.py",
        source_root / "fedml/simulation/sp/fedrep/fedrep_api.py",
        source_root / "fedml/simulation/sp/fedrep/client.py",
        source_root / "fedml/model/model_hub.py",
        source_root / "fedml/model/cv/cnn.py",
        source_root / "fedml/data/MNIST/data_loader.py",
        source_root / "fedml/ml/trainer/my_model_trainer_classification.py",
        source_root / "fedml/simulation/simulator.py",
    ]
    return {
        str(path.relative_to(source_root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_paths if path.exists()
    }


def _environment(root: Path) -> dict[str, object]:
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
            text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        head = None
    versions = {}
    for package in ("torch", "fedml", "PyYAML", "matplotlib"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    try:
        import torch
        cuda = {"available": torch.cuda.is_available(), "runtime": torch.version.cuda}
        if cuda["available"]:
            cuda["device_0"] = torch.cuda.get_device_name(0)
    except (ImportError, RuntimeError, AssertionError) as exc:
        cuda = {"error": str(exc)}
    return {"python": sys.executable, "python_version": sys.version, "platform": platform.platform(),
            "packages": versions, "cuda": cuda, "git_head": head,
            "source_sha256": _source_hashes()}


def _launch(root: Path, config_path: Path) -> int:
    return subprocess.run(
        [sys.executable, "torch_fedavg_mnist_lr_one_line_example.py", "--cf", config_path.name],
        cwd=root, check=False,
    ).returncode


def run_matrix(root: Path, base: dict, methods: list[str], seeds: list[int], rounds: int,
               result_dir: Path, *, resume: bool = False, force: bool = False,
               launch: Callable[[Path, Path], int] = _launch) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = result_dir / "comparison_manifest.csv"
    records: list[dict[str, object]] = _load_manifest(manifest_path)
    merged_path = result_dir / "comparison_metrics.csv"
    if (merged_path.exists() and not force
            and not any(record.get("run_id") and record.get("config_hash")
                        for record in records)):
        raise ValueError("existing merged CSV has no resumable manifest; choose a new --output-dir or --force")
    matrix_id = _hash({"base": base, "rounds": rounds, "source_sha256": _source_hashes()})[:16]
    existing_matrices = {
        str(record.get("matrix_id")) for record in records
        if record.get("status") == "completed" and record.get("matrix_id")
    }
    if existing_matrices and existing_matrices != {matrix_id}:
        raise ValueError("output directory contains another experiment matrix; choose a new --output-dir")
    jobs = []
    for method in methods:
        spec = METHODS[method]
        for seed in seeds:
            if spec is None:
                _upsert(records, {"run_id": f"{method.lower()}_seed{seed}_{rounds}r",
                                  "requested_method": method, "method": method, "seed": seed,
                                  "rounds": rounds, "status": "not_implemented", "rows": 0,
                                  "reason": "No implementation exists in this repository",
                                  "reference": REFERENCES.get(method, "")})
                continue
            run_id = f"{spec['slug']}_seed{seed}_{rounds}r"
            output = result_dir / f"{run_id}.csv"
            snapshot = result_dir / "configs" / f"{run_id}.yaml"
            environment = result_dir / "configs" / f"{run_id}.environment.json"
            config = _config(base, method, seed, rounds, output, run_id, matrix_id)
            record = {
                "run_id": run_id, "requested_method": method, "method": spec["label"],
                "seed": seed, "rounds": rounds, "status": "pending", "rows": 0,
                "result_file": str(output), "partial_file": "", "config_hash": config["common_args"]["experiment_config_hash"],
                "matrix_id": matrix_id, "config_snapshot": str(snapshot),
                "environment_snapshot": str(environment), "backend": spec["backend"],
                "backend_secure": False, "exit_code": "", "error": "", "reason": "", "reference": "",
            }
            previous = next((item for item in records if item.get("run_id") == run_id), None)
            if previous and previous.get("config_hash") != record["config_hash"] and not force:
                raise ValueError(f"{run_id}: existing run uses another config; choose a new --output-dir or --force")
            if output.exists() and not (resume or force):
                raise ValueError(f"{output} exists; use --resume or --force")
            if output.exists() and resume and not force:
                _validate_result(output, record)
                if not previous or previous.get("status") != "completed":
                    record.update({"status": "completed", "rows": rounds, "exit_code": 0})
                    _upsert(records, record)
                print(f"[SKIP] {run_id}: validated completed result", flush=True)
                continue
            jobs.append((record, config, output, snapshot, environment))

    for record, _, _, _, _ in jobs:
        _upsert(records, record)
    _save_manifest(manifest_path, records)
    _merge(result_dir, records)
    for record, config, output, snapshot, environment in jobs:
        run_id = str(record["run_id"])
        partial = result_dir / f".{run_id}.{uuid.uuid4().hex}.partial.csv"
        runtime = copy.deepcopy(config)
        runtime["train_args"]["experiment_result_file"] = str(partial)
        runtime_config = root / f".comparison_{run_id}_{uuid.uuid4().hex}.yaml"
        record["status"] = "running"
        record["partial_file"] = str(partial)
        _save_manifest(manifest_path, records)
        print(f"[RUN] {run_id}: {rounds} rounds", flush=True)
        try:
            runtime_config.write_text(yaml.safe_dump(runtime, sort_keys=False), encoding="utf-8")
            exit_code = launch(root, runtime_config)
            record["exit_code"] = exit_code
            if exit_code != 0:
                raise RuntimeError(f"training exited with code {exit_code}")
            _validate_result(partial, record)
            _atomic_text(snapshot, yaml.safe_dump(config, sort_keys=False))
            _atomic_text(environment, json.dumps(_environment(root), indent=2, ensure_ascii=False) + "\n")
            os.replace(partial, output)
            record.update({"status": "completed", "rows": rounds, "partial_file": "", "error": ""})
            _save_manifest(manifest_path, records)
            _merge(result_dir, records)
            print(f"[OK] {run_id}: {rounds} validated rows", flush=True)
        except (OSError, RuntimeError, ValueError) as exc:
            record.update({"status": "failed", "error": str(exc)})
            _save_manifest(manifest_path, records)
            _merge(result_dir, records)
            raise
        finally:
            runtime_config.unlink(missing_ok=True)
    _merge(result_dir, records)
    print(f"Metrics: {result_dir / 'comparison_metrics.csv'}", flush=True)
    print(f"Manifest: {manifest_path}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="fedml_config.yaml", help="base FedML YAML")
    parser.add_argument("--methods", default="FedAvg,FedRep,Ours", help="comma-separated methods, or all")
    parser.add_argument("--seeds", default="0", help="comma-separated integer seeds")
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--output-dir", default="results/comparison")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--resume", action="store_true", help="validate and skip completed runs")
    group.add_argument("--force", action="store_true", help="replace selected completed runs after a valid rerun")
    args = parser.parse_args(argv)
    methods = list(METHODS) if args.methods.strip().lower() == "all" else [item.strip() for item in args.methods.split(",") if item.strip()]
    unknown = [method for method in methods if method not in METHODS]
    if not methods or unknown or len(methods) != len(set(methods)):
        parser.error(f"invalid or duplicate methods: {unknown or methods}")
    try:
        seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    except ValueError:
        parser.error("--seeds must contain comma-separated integers")
    if not seeds or len(seeds) != len(set(seeds)) or args.rounds < 1:
        parser.error("provide unique seeds and a positive --rounds")
    root = Path(__file__).resolve().parent
    config_path = (root / args.config).resolve()
    result_dir = (root / args.output_dir).resolve()
    try:
        base = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(base, dict):
            raise ValueError("base config must be a YAML mapping")
        run_matrix(root, base, methods, seeds, args.rounds, result_dir,
                   resume=args.resume, force=args.force)
    except (OSError, RuntimeError, ValueError, yaml.YAMLError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
