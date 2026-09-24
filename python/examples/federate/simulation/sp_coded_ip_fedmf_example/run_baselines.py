"""Run paired centralized/coded/plain/LDP matrix-factorisation baselines."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Mapping

from baselines import run_baseline_comparison
from movielens import load_ratings, make_synthetic_ratings


def _write_rows(rows: List[Mapping[str, object]], path: Path) -> None:
    if not rows:
        raise ValueError("cannot write an empty result")
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


def run_baseline_sweep(
    ratings,
    *,
    rounds: int = 20,
    repeats: int = 3,
    dropout_rates=(0.0, 0.1, 0.2, 0.3),
    seed: int = 17,
    ldp_noise_multiplier: float = 1.0,
) -> List[Dict[str, object]]:
    if (
        isinstance(rounds, bool)
        or not isinstance(rounds, int)
        or rounds < 1
        or isinstance(repeats, bool)
        or not isinstance(repeats, int)
        or repeats < 1
    ):
        raise ValueError("rounds and repeats must be positive")
    dropout_rates = tuple(float(rate) for rate in dropout_rates)
    if not dropout_rates:
        raise ValueError("dropout_rates must not be empty")
    if any(not math.isfinite(rate) or not 0.0 <= rate < 1.0 for rate in dropout_rates):
        raise ValueError("dropout rates must be finite and in [0, 1)")
    rows: List[Dict[str, object]] = []
    for repeat in range(repeats):
        for rate in dropout_rates:
            result_rows = run_baseline_comparison(
                ratings,
                rounds=rounds,
                dropout_rate=float(rate),
                ldp_noise_multiplier=ldp_noise_multiplier,
                seed=seed + repeat,
            )
            for row in result_rows:
                row = dict(row)
                row["repeat"] = repeat
                rows.append(row)
    return rows


def summarize_baselines(rows: List[Mapping[str, object]]) -> List[Dict[str, object]]:
    """Return final-round means grouped by dropout rate."""
    groups: Dict[float, List[Mapping[str, object]]] = {}
    for row in rows:
        groups.setdefault(float(row["dropout_rate"]), []).append(row)
    summary = []
    metric_names = (
        "centralized_rmse", "coded_rmse", "plaintext_online_rmse", "ldp_online_rmse",
        "centralized_mae", "coded_mae", "plaintext_online_mae", "ldp_online_mae",
    )
    for rate, rate_rows in sorted(groups.items()):
        final_round = max(int(row["round"]) for row in rate_rows)
        final_rows = [row for row in rate_rows if int(row["round"]) == final_round]
        summary_row: Dict[str, object] = {
            "dropout_rate": rate,
            "round": final_round,
            "online_clients": int(final_rows[-1]["online_clients"]),
            "max_coded_vs_central_error": max(
                float(row["coded_vs_central_max_error"]) for row in final_rows
            ),
            "mean_ldp_epsilon": sum(float(row["ldp_epsilon"]) for row in final_rows)
            / len(final_rows),
        }
        for name in metric_names:
            summary_row[f"final_{name}"] = sum(
                float(row[name]) for row in final_rows
            ) / len(final_rows)
        summary.append(summary_row)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratings", help="MovieLens ratings.dat or ratings.csv")
    parser.add_argument("--client-num", type=int, default=12)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--dropout-rate", type=float, default=None)
    parser.add_argument("--ldp-noise-multiplier", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=Path("results/baselines.csv"))
    args = parser.parse_args()
    ratings = (
        load_ratings(args.ratings, args.client_num)
        if args.ratings
        else make_synthetic_ratings(args.client_num, 40, 8, 15)
    )
    rates = (args.dropout_rate,) if args.dropout_rate is not None else (0.0, 0.1, 0.2, 0.3)
    rows = run_baseline_sweep(
        ratings,
        rounds=args.rounds,
        repeats=args.repeats,
        dropout_rates=rates,
        ldp_noise_multiplier=args.ldp_noise_multiplier,
    )
    _write_rows(rows, args.output)
    summary_rows = summarize_baselines(rows)
    summary = {
        "rows": len(rows),
        "output": str(args.output),
        "methods": ["centralized", "coded", "plaintext_online", "ldp_online"],
        "ldp_noise_multiplier": args.ldp_noise_multiplier,
        "final_by_dropout_rate": summary_rows,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
