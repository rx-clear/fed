"""Create mean/std/95% confidence intervals from sweep results."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


_T_CRITICAL_95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
}


def _summary(values):
    values = [float(value) for value in values]
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / max(1, len(values) - 1)
    std = math.sqrt(variance)
    # Use Student's t for the small repeat counts typical of these sweeps.
    critical = _T_CRITICAL_95.get(len(values) - 1, 1.96)
    margin = critical * std / math.sqrt(len(values))
    return mean, std, mean - margin, mean + margin


def summarize(input_path: Path):
    groups = defaultdict(dict)
    with input_path.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        metric_names = [
            name for name in (reader.fieldnames or [])
            if name.endswith("_rmse") or name.endswith("_mae") or name.endswith("_error")
        ]
        if not metric_names:
            raise ValueError("input contains no recognised RMSE/MAE/error metrics")
        for row in reader:
            key = (float(row["dropout_rate"]), int(row["round"]))
            for metric in metric_names:
                groups[key].setdefault(metric, []).append(row[metric])
    result = []
    for (dropout_rate, round_idx), metrics in sorted(groups.items()):
        row = {"dropout_rate": dropout_rate, "round": round_idx}
        for metric, values in metrics.items():
            mean, std, lower, upper = _summary(values)
            row.update({f"{metric}_mean": mean, f"{metric}_std": std,
                        f"{metric}_ci95_low": lower, f"{metric}_ci95_high": upper})
        result.append(row)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = summarize(args.input)
    if not rows:
        raise SystemExit("input contains no rows")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} confidence-interval rows to {args.output}")


if __name__ == "__main__":
    main()
