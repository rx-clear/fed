"""Summarize matched-seed MNIST comparisons and plot mean ± sample SD.

Seeds are independent comparison units. Communication rounds describe each
run's trajectory and are never counted as independent replicates.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


def _read_runs(path: Path) -> dict[tuple[str, int], list[dict[str, float | int | str]]]:
    if not path.exists():
        raise ValueError(f"Input file not found: {path}. Run run_comparison.py first.")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"method", "seed", "round", "test_acc", "test_loss", "matrix_id", "backend", "backend_secure"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: missing columns {sorted(required - set(reader.fieldnames or []))}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: no completed metric rows")
    matrix_ids = {row["matrix_id"] for row in rows}
    if len(matrix_ids) != 1 or not next(iter(matrix_ids)):
        raise ValueError("Comparison contains multiple or missing matrix IDs")
    grouped = defaultdict(list)
    for row in rows:
        try:
            seed = int(row["seed"])
            round_idx = int(row["round"])
            accuracy = float(row["test_acc"])
            loss = float(row["test_loss"])
        except (TypeError, ValueError) as exc:
            raise ValueError("Comparison contains a non-numeric seed, round, or metric") from exc
        if not row["method"] or round_idx < 0 or not math.isfinite(accuracy) or not 0 <= accuracy <= 1:
            raise ValueError("Comparison contains an invalid method, round, or accuracy")
        if not math.isfinite(loss) or loss < 0:
            raise ValueError("Comparison contains an invalid test loss")
        if row["backend_secure"] != "False":
            raise ValueError("Comparison contains an unsupported secure-backend claim")
        grouped[(row["method"], seed)].append({
            "round": round_idx, "test_acc": accuracy, "test_loss": loss,
            "backend": row["backend"], "matrix_id": row["matrix_id"],
        })
    expected_rounds = None
    by_method = defaultdict(set)
    for (method, seed), run in grouped.items():
        run.sort(key=lambda item: item["round"])
        rounds = [item["round"] for item in run]
        if rounds != list(range(len(run))):
            raise ValueError(f"{method} seed={seed}: missing or repeated rounds")
        if expected_rounds is None:
            expected_rounds = len(run)
        elif expected_rounds != len(run):
            raise ValueError("Runs have different round counts")
        if len({item["backend"] for item in run}) != 1:
            raise ValueError(f"{method} seed={seed}: backend changes between rounds")
        by_method[method].add(seed)
    if len({tuple(sorted(seeds)) for seeds in by_method.values()}) != 1:
        raise ValueError("Methods have unmatched seeds; complete the matrix before analysis")
    return dict(grouped)


def _mean_sd(values: list[float]) -> tuple[float, float | str]:
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else ""


def summarize(runs: dict[tuple[str, int], list[dict]], accuracy_target: float = 0.75):
    by_round = defaultdict(list)
    seed_rows = []
    for (method, seed), run in sorted(runs.items()):
        backend = run[0]["backend"]
        for item in run:
            by_round[(method, item["round"])].append(item)
        target_round = next((item["round"] for item in run if item["test_acc"] >= accuracy_target), "")
        seed_rows.append({
            "method": method, "seed": seed, "backend": backend,
            "backend_secure": False, "rounds": len(run),
            "final_test_acc": run[-1]["test_acc"],
            "final_test_loss": run[-1]["test_loss"],
            "mean_test_acc": statistics.mean(item["test_acc"] for item in run),
            "best_test_acc": max(item["test_acc"] for item in run),
            "first_round_at_target": target_round,
            "accuracy_target": accuracy_target,
            "claim_boundary": "fixed_point_reference_only" if backend == "chunked-reference" else "algorithmic_baseline",
        })
    round_rows = []
    for (method, round_idx), items in sorted(by_round.items()):
        acc_mean, acc_sd = _mean_sd([item["test_acc"] for item in items])
        loss_mean, loss_sd = _mean_sd([item["test_loss"] for item in items])
        round_rows.append({"method": method, "round": round_idx, "n_seeds": len(items),
                           "test_acc_mean": acc_mean, "test_acc_sd": acc_sd,
                           "test_loss_mean": loss_mean, "test_loss_sd": loss_sd})
    baseline = {row["seed"]: row for row in seed_rows if row["method"] == "FedAvg"}
    paired_rows = []
    for row in seed_rows:
        if row["method"] == "FedAvg" or row["seed"] not in baseline:
            continue
        control = baseline[row["seed"]]
        paired_rows.append({
            "method": row["method"], "baseline": "FedAvg", "seed": row["seed"],
            "delta_final_test_acc": row["final_test_acc"] - control["final_test_acc"],
            "delta_final_test_loss": row["final_test_loss"] - control["final_test_loss"],
        })
    paired_summary = []
    for method in sorted({row["method"] for row in paired_rows}):
        items = [row for row in paired_rows if row["method"] == method]
        acc_mean, acc_sd = _mean_sd([row["delta_final_test_acc"] for row in items])
        loss_mean, loss_sd = _mean_sd([row["delta_final_test_loss"] for row in items])
        paired_summary.append({
            "method": method, "baseline": "FedAvg", "n_paired_seeds": len(items),
            "mean_delta_final_test_acc": acc_mean, "sd_delta_final_test_acc": acc_sd,
            "mean_delta_final_test_loss": loss_mean, "sd_delta_final_test_loss": loss_sd,
        })
    return round_rows, seed_rows, paired_rows, paired_summary


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_curves(round_rows: list[dict], output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    methods = sorted({row["method"] for row in round_rows})
    for metric, label in (("test_acc", "Test accuracy"), ("test_loss", "Test loss")):
        figure, axis = plt.subplots(figsize=(8, 5))
        for method in methods:
            items = [row for row in round_rows if row["method"] == method]
            rounds = [row["round"] for row in items]
            means = [row[f"{metric}_mean"] for row in items]
            line, = axis.plot(rounds, means, label=f"{method} (n={items[0]['n_seeds']})")
            if items[0][f"{metric}_sd"] != "":
                lower = [max(0.0, mean - row[f"{metric}_sd"]) for mean, row in zip(means, items)]
                upper = [min(1.0, mean + row[f"{metric}_sd"]) if metric == "test_acc"
                         else mean + row[f"{metric}_sd"] for mean, row in zip(means, items)]
                axis.fill_between(rounds, lower, upper, color=line.get_color(), alpha=0.16)
        axis.set_xlabel("Communication round")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_dir / f"{metric}.png", dpi=160)
        plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/comparison/comparison_metrics.csv")
    parser.add_argument("--output-dir", default="results/comparison/analysis")
    parser.add_argument("--plots-dir", default=None)
    parser.add_argument("--accuracy-target", type=float, default=0.75)
    args = parser.parse_args(argv)
    if not 0 <= args.accuracy_target <= 1:
        parser.error("--accuracy-target must be within [0, 1]")
    try:
        runs = _read_runs(Path(args.input))
        round_rows, seed_rows, paired_rows, paired_summary = summarize(runs, args.accuracy_target)
        output = Path(args.output_dir)
        _write_csv(output / "round_summary.csv", list(round_rows[0]), round_rows)
        _write_csv(output / "seed_summary.csv", list(seed_rows[0]), seed_rows)
        _write_csv(output / "paired_differences.csv",
                   ["method", "baseline", "seed", "delta_final_test_acc", "delta_final_test_loss"], paired_rows)
        _write_csv(output / "paired_summary.csv",
                   ["method", "baseline", "n_paired_seeds", "mean_delta_final_test_acc",
                    "sd_delta_final_test_acc", "mean_delta_final_test_loss", "sd_delta_final_test_loss"], paired_summary)
        plots = Path(args.plots_dir) if args.plots_dir else output / "plots"
        plot_curves(round_rows, plots)
    except (OSError, ValueError, ImportError) as exc:
        parser.exit(1, f"[ERROR] {exc}\n")
    print(f"Analyzed {len(runs)} runs; summaries: {output}; plots: {plots}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
