"""Compatibility entry point for comparison analysis and uncertainty plots."""

from __future__ import annotations

import argparse
from pathlib import Path

from analyze_comparison import main as analyze_main


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/comparison/comparison_metrics.csv")
    parser.add_argument("--output-dir", default="results/comparison/plots")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    return analyze_main([
        "--input", args.input,
        "--output-dir", str(output_dir.parent / "analysis"),
        "--plots-dir", str(output_dir),
    ])


if __name__ == "__main__":
    raise SystemExit(main())
