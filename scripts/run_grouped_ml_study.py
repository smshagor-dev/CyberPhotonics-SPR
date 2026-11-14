"""Produce evidence-bound, split-locked ML baseline and seed-ablation reports."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sprpcf.validation.publication_ml import run_grouped_ml_study


def _seeds(text: str) -> tuple[int, ...]:
    try:
        values = tuple(int(value.strip()) for value in text.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Supply comma-separated integer seeds") from exc
    if len(values) < 2 or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("At least two distinct seeds required")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Run group-disjoint paired forward-ML benchmarks.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=_seeds, default=(7, 17, 29))
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--trees", type=int, default=60)
    parser.add_argument("--bootstrap-resamples", type=int, default=500)
    parser.add_argument("--ablation-runs", type=Path, help="Optional existing tandem physics-loss ablation_runs.csv")
    parser.add_argument("--allow-software-only", action="store_true",
                        help="Explicitly allow synthetic/mock data for software-only evaluation")
    args = parser.parse_args()
    result = run_grouped_ml_study(
        args.data, args.out, seeds=args.seeds, test_fraction=args.test_fraction,
        trees=args.trees, bootstrap_resamples=args.bootstrap_resamples,
        allow_software_only=args.allow_software_only, ablation_runs=args.ablation_runs,
    )
    print(json.dumps({
        "study": result["study"],
        "dataset_evidence_class": result["source"]["evidence_class"],
        "observations": result["observations"],
        "publication_ready": result["publication_ready"],
        "outputs": str(args.out),
        "scientific_blockers": result["scientific_blockers"],
    }, indent=2))


if __name__ == "__main__":
    main()
