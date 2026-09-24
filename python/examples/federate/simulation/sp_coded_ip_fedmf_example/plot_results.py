"""Plot RMSE curves from the full MF sweep CSV."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/rmse_curves.png"))
    args = parser.parse_args()
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("Install matplotlib to plot results: pip install matplotlib") from exc
    curves = defaultdict(lambda: {"coded": [], "plaintext": []})
    with args.input.open("r", newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            rate = float(row["dropout_rate"])
            curves[rate]["coded"].append((int(row["round"]), float(row["coded_rmse"])))
            curves[rate]["plaintext"].append((int(row["round"]), float(row["plaintext_rmse"])))
    for rate, values in sorted(curves.items()):
        coded_by_round = defaultdict(list)
        plaintext_by_round = defaultdict(list)
        for round_idx, value in values["coded"]:
            coded_by_round[round_idx].append(value)
        for round_idx, value in values["plaintext"]:
            plaintext_by_round[round_idx].append(value)
        coded = sorted((round_idx, sum(points) / len(points))
                       for round_idx, points in coded_by_round.items())
        plaintext = sorted((round_idx, sum(points) / len(points))
                           for round_idx, points in plaintext_by_round.items())
        plt.plot([x for x, _ in coded], [y for _, y in coded],
                 label=f"Coded-IP {rate:.0%}")
        plt.plot([x for x, _ in plaintext], [y for _, y in plaintext],
                 linestyle="--", label=f"Online plaintext {rate:.0%}")
    plt.xlabel("Communication round")
    plt.ylabel("RMSE")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=180)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
