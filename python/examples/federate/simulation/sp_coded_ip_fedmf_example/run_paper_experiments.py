"""Run the reproducible experiment bundle used to audit the paper claims.

The command writes one directory containing numeric protocol checks, synthetic
dropout/accuracy sweeps, confidence intervals, and communication measurements.
It is intentionally self-contained so a result directory can be archived with
the command-line parameters that produced it.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional

# The examples are also intended to be run as standalone scripts.  Put their
# directory on the import path before importing sibling modules.
_EXAMPLE_DIR = Path(__file__).resolve().parent
if str(_EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLE_DIR))

from communication_benchmark import benchmark as run_benchmark  # noqa: E402
from full_experiment import run_comparison  # noqa: E402
from run_baselines import run_baseline_sweep, summarize_baselines  # noqa: E402
from movielens import load_ratings, make_synthetic_ratings  # noqa: E402
from paper_validation import run_suite  # noqa: E402
from protocol_validation import run_diagnostics  # noqa: E402
from run_experiments import run_sweep  # noqa: E402
from statistical_report import summarize as summarize_statistics  # noqa: E402
from summarize_results import summarize as summarize_final  # noqa: E402


DROPOUT_RATES = (0.0, 0.1, 0.2, 0.3)


def _write_rows(rows: List[Mapping[str, object]], path: Path) -> None:
    if not rows:
        raise ValueError(f"cannot write an empty result: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _full_sweep(
    ratings,
    rounds: int,
    repeats: int,
    seed: int,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for repeat in range(repeats):
        for rate in DROPOUT_RATES:
            run_rows = run_comparison(
                ratings,
                rounds=rounds,
                dropout_rate=rate,
                seed=seed + repeat,
            )
            for row in run_rows:
                row = dict(row)
                row["repeat"] = repeat
                rows.append(row)
    return rows


def run_all(
    output_dir: Path,
    *,
    ratings_path: Optional[str] = None,
    client_num: int = 12,
    rounds: int = 20,
    repeats: int = 3,
    seed: int = 17,
    ldp_noise_multiplier: float = 1.0,
    make_plot: bool = False,
) -> Dict[str, object]:
    """Run all non-optional experiments and return a manifest."""
    if client_num < 2:
        raise ValueError("client_num must be at least two for the paired experiments")
    if rounds < 1 or repeats < 1:
        raise ValueError("rounds and repeats must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)

    validation = run_suite(
        client_num=client_num,
        recovery_threshold=max(1, client_num - max(1, int(round(client_num * 0.2)))),
        dimension=16,
        seed=seed,
    )
    validation_path = output_dir / "paper_validation.json"
    validation_path.write_text(
        json.dumps(validation, indent=2, sort_keys=True), encoding="utf-8"
    )
    diagnostics = run_diagnostics(
        client_num=client_num,
        recovery_threshold=max(1, client_num - max(1, int(round(client_num * 0.2)))),
        dimension=16,
    )
    diagnostics_path = output_dir / "protocol_diagnostics.json"
    diagnostics_path.write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True), encoding="utf-8"
    )

    sweep_rows = run_sweep(
        client_num=max(client_num, 2),
        recovery_threshold=max(1, client_num - max(1, int(round(client_num * 0.3)))),
        dimension=128,
        dropout_rates=DROPOUT_RATES,
        repeats=repeats,
        seed=seed,
    )
    sweep_path = output_dir / "coded_ip_fedmf_sweep.csv"
    _write_rows(sweep_rows, sweep_path)

    ratings = (
        load_ratings(ratings_path, client_num)
        if ratings_path
        else make_synthetic_ratings(client_num, 40, 8, 15, seed=seed)
    )
    baseline_rows = run_baseline_sweep(
        ratings,
        rounds=rounds,
        repeats=repeats,
        seed=seed,
        ldp_noise_multiplier=ldp_noise_multiplier,
    )
    baseline_path = output_dir / "baseline_comparison.csv"
    _write_rows(baseline_rows, baseline_path)
    baseline_summary_path = output_dir / "baseline_summary.json"
    baseline_summary_path.write_text(
        json.dumps(summarize_baselines(baseline_rows), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    baseline_statistics_path = output_dir / "baseline_confidence_intervals.csv"
    _write_rows(summarize_statistics(baseline_path), baseline_statistics_path)
    full_rows = _full_sweep(ratings, rounds, repeats, seed)
    full_path = output_dir / "full_fedmf_sweep.csv"
    _write_rows(full_rows, full_path)

    summary_rows = summarize_final(full_path)
    summary_path = output_dir / "full_fedmf_summary.csv"
    _write_rows(summary_rows, summary_path)
    statistics_rows = summarize_statistics(full_path)
    statistics_path = output_dir / "full_fedmf_confidence_intervals.csv"
    _write_rows(statistics_rows, statistics_path)

    benchmark_path = output_dir / "communication_benchmark.csv"
    _write_rows(run_benchmark(seed=seed), benchmark_path)

    plot_path: Optional[Path] = None
    if make_plot:
        plot_path = output_dir / "rmse_curves.png"
        plot_command = [
            sys.executable,
            str(Path(__file__).with_name("plot_results.py")),
            str(full_path),
            "--output",
            str(plot_path),
        ]
        subprocess.run(plot_command, check=True)

    manifest = {
        "parameters": {
            "ratings_path": ratings_path,
            "client_num": client_num,
            "rounds": rounds,
            "repeats": repeats,
            "seed": seed,
            "ldp_noise_multiplier": ldp_noise_multiplier,
            "dropout_rates": list(DROPOUT_RATES),
        },
        "numeric_checks_passed": validation["all_numeric_checks_passed"],
        "claim_audit_status_counts": {
            status: sum(
                1 for claim in validation["claim_audit"] if claim["status"] == status
            )
            for status in sorted({claim["status"] for claim in validation["claim_audit"]})
        },
        "security_scope": "reference_only; cryptographic security claims are not verified",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "outputs": {
            "paper_validation": str(validation_path),
            "protocol_diagnostics": str(diagnostics_path),
            "coded_sweep": str(sweep_path),
            "baseline_comparison": str(baseline_path),
            "baseline_summary": str(baseline_summary_path),
            "baseline_confidence_intervals": str(baseline_statistics_path),
            "full_sweep": str(full_path),
            "summary": str(summary_path),
            "confidence_intervals": str(statistics_path),
            "communication_benchmark": str(benchmark_path),
            "plot": str(plot_path) if plot_path else None,
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/paper"))
    parser.add_argument("--ratings")
    parser.add_argument("--client-num", type=int, default=12)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--ldp-noise-multiplier", type=float, default=1.0)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    manifest = run_all(
        args.output_dir,
        ratings_path=args.ratings,
        client_num=args.client_num,
        rounds=args.rounds,
        repeats=args.repeats,
        seed=args.seed,
        ldp_noise_multiplier=args.ldp_noise_multiplier,
        make_plot=args.plot,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
