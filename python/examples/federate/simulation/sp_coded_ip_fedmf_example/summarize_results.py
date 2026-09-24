"""Summarise full_fedmf_sweep.csv into paper-ready CSV rows."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def summarize(input_path: Path):
    groups = defaultdict(list)
    with input_path.open("r", newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            groups[float(row["dropout_rate"])].append(row)
    output = []
    for rate, rows in sorted(groups.items()):
        final_round = max(int(row["round"]) for row in rows)
        final_rows = [row for row in rows if int(row["round"]) == final_round]
        last = final_rows[-1]

        def mean_metric(name):
            return sum(float(row[name]) for row in final_rows) / len(final_rows)

        output.append({
            "dropout_rate": rate,
            "online_clients": int(last["online_clients"]),
            "final_coded_rmse": mean_metric("coded_rmse"),
            "final_plaintext_rmse": mean_metric("plaintext_rmse"),
            "final_coded_mae": mean_metric("coded_mae"),
            "final_plaintext_mae": mean_metric("plaintext_mae"),
            "max_recovery_error": max(float(r["recovery_error"]) for r in rows),
            "coded_communication_rounds": int(last["coded_communication_rounds"]),
            "secagg_baseline_rounds": int(last["secagg_baseline_rounds"]),
        })
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = summarize(args.input)
    if not rows:
        raise SystemExit("input contains no rows")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    else:
        print("dropout_rate,final_coded_rmse,final_plaintext_rmse,max_recovery_error")
        for row in rows:
            print(f"{row['dropout_rate']},{row['final_coded_rmse']},"
                  f"{row['final_plaintext_rmse']},{row['max_recovery_error']}")


if __name__ == "__main__":
    main()
