"""Run the available 50-round comparison methods and build one metrics CSV.

The repository implements FedAvg, FedRep, FedRep+DMCFE, BatchCrypt, and
pairwise Masking for the single-process research comparison. HybridAlpha and
CryptoFE are recorded as unavailable instead of being silently replaced by a
different algorithm.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import yaml


METHODS = {
    "FedAvg": {"federated_optimizer": "FedAvg", "enable_dmcfe": False, "slug": "fedavg"},
    "FedRep": {"federated_optimizer": "FedRep", "enable_dmcfe": False, "slug": "fedrep"},
    "Ours": {"federated_optimizer": "FedRep", "enable_dmcfe": True, "slug": "fedrep_dmcfe"},
    "BatchCrypt": {"federated_optimizer": "FedAvg", "enable_dmcfe": False, "privacy_aggregation": "BatchCrypt", "slug": "batchcrypt"},
    "Masking": {"federated_optimizer": "FedAvg", "enable_dmcfe": False, "privacy_aggregation": "Masking", "slug": "masking"},
    "HybridAlpha": None,
    "CryptoFE": None,
}

REFERENCES = {
    "HybridAlpha": "https://arxiv.org/abs/1912.05897",
    "CryptoFE": "https://doi.org/10.1109/GLOBECOM48099.2022.10001080",
}


def _write_config(
    base: dict,
    path: Path,
    method: str,
    output_file: str,
    seed: int,
) -> None:
    config = yaml.safe_load(yaml.safe_dump(base))
    config.setdefault("common_args", {})["random_seed"] = int(seed)
    train_args = config.setdefault("train_args", {})
    spec = METHODS[method]
    train_args.update(
        {
            "federated_optimizer": spec["federated_optimizer"],
            "enable_dmcfe": spec["enable_dmcfe"],
            "privacy_aggregation": spec.get("privacy_aggregation"),
            "comm_round": 50,
            "run_dmcfe_smoke_tests": False,
            "experiment_result_file": output_file,
        }
    )
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def _append_metrics(source: Path, target: Path, method: str, seed: int) -> int:
    with source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return 0
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = ["method", "seed"] + [key for key in rows[0] if key not in {"method", "seed"}]
    write_header = not target.exists()
    with target.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({"method": method, "seed": int(seed), **row})
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="fedml_config.yaml", help="base FedML YAML")
    parser.add_argument(
        "--methods",
        default="FedAvg,FedRep,Ours",
        help="comma-separated methods; use all to include unavailable entries in the manifest",
    )
    parser.add_argument(
        "--seeds",
        default="0",
        help="comma-separated integer random seeds (for example: 0,1,2)",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    base_path = (root / args.config).resolve()
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    requested = list(METHODS) if args.methods.strip().lower() == "all" else [x.strip() for x in args.methods.split(",") if x.strip()]
    try:
        seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    except ValueError as exc:
        raise SystemExit("--seeds must be a comma-separated list of integers") from exc
    if not seeds:
        raise SystemExit("--seeds must contain at least one integer")
    unknown = [method for method in requested if method not in METHODS]
    if unknown:
        raise SystemExit("Unknown method(s): " + ", ".join(unknown))

    result_dir = root / "results" / "comparison"
    result_dir.mkdir(parents=True, exist_ok=True)
    merged = result_dir / "comparison_metrics.csv"
    if merged.exists():
        merged.unlink()
    manifest = result_dir / "comparison_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        manifest_writer = csv.DictWriter(
            handle,
            fieldnames=["method", "seed", "status", "rows", "result_file", "reason", "reference"],
        )
        manifest_writer.writeheader()
        for method in requested:
            spec = METHODS[method]
            if spec is None:
                for seed in seeds:
                    manifest_writer.writerow({"method": method, "seed": seed, "status": "not_implemented", "rows": 0, "result_file": "", "reason": "No implementation exists in this repository", "reference": REFERENCES.get(method, "")})
                print(f"[SKIP] {method}: no implementation in this repository")
                continue
            slug = spec["slug"]
            for seed in seeds:
                run_slug = slug if len(seeds) == 1 else f"{slug}_seed{seed}"
                result_file = f"results/comparison/{run_slug}.csv"
                generated_config = root / f".comparison_{run_slug}.yaml"
                output_path = root / result_file
                if output_path.exists():
                    output_path.unlink()
                _write_config(base, generated_config, method, result_file, seed)
                print(f"[RUN] {method} seed={seed}: 50 rounds")
                try:
                    subprocess.run(
                        [sys.executable, "torch_fedavg_mnist_lr_one_line_example.py", "--cf", generated_config.name],
                        cwd=str(root),
                        check=True,
                    )
                finally:
                    generated_config.unlink(missing_ok=True)
                rows = _append_metrics(output_path, merged, method, seed)
                manifest_writer.writerow({"method": method, "seed": seed, "status": "completed", "rows": rows, "result_file": result_file, "reason": "", "reference": ""})
                print(f"[OK] {method} seed={seed}: {rows} metric rows")
    print(f"Merged metrics: {merged}")
    print(f"Manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
