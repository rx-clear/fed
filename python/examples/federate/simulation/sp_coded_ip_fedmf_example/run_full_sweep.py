"""Run the complete MF comparison at several dropout rates."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from full_experiment import run_comparison
from movielens import load_ratings, make_synthetic_ratings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratings")
    parser.add_argument("--client-num", type=int, default=12)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path, default=Path("results/full_fedmf_sweep.csv"))
    args = parser.parse_args()
    if args.client_num < 1:
        raise SystemExit("--client-num must be positive")
    if args.rounds < 1:
        raise SystemExit("--rounds must be positive")
    if args.repeats < 1:
        raise SystemExit("--repeats must be positive")
    ratings = (load_ratings(args.ratings, args.client_num) if args.ratings else
               make_synthetic_ratings(args.client_num, 40, 8, 15, seed=args.seed))
    rows = []
    for repeat in range(args.repeats):
        for rate in (0.0, 0.1, 0.2, 0.3):
            repeat_rows = run_comparison(
                ratings, rounds=args.rounds, dropout_rate=rate, seed=args.seed + repeat
            )
            for row in repeat_rows:
                row["repeat"] = repeat
            rows.extend(repeat_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
