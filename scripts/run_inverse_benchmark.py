"""Run a source-bound, matched forward-surrogate-query inverse-design comparison."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sprpcf.validation.inverse_benchmark import run_inverse_benchmark


def _parse_seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Comma-separated integer seeds required") from exc
    if len(seeds) < 2 or len(seeds) != len(set(seeds)):
        raise argparse.ArgumentTypeError("At least two distinct generation seeds required")
    return seeds


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare tandem vs retrieval/random inverse designs using identical surrogate-query budgets."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Trained checkpoint with dataset and geometry split provenance")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=_parse_seeds, default=(7, 17, 29))
    parser.add_argument("--candidates-per-target", type=int, default=16)
    parser.add_argument("--latent-scale", type=float, default=0.10)
    parser.add_argument("--bootstrap-resamples", type=int, default=500)
    parser.add_argument("--allow-software-only", action="store_true",
                        help="Permit labelled synthetic data; results remain software-only")
    args = parser.parse_args()
    report = run_inverse_benchmark(
        args.data, args.checkpoint, args.out,
        seeds=args.seeds, candidates_per_target=args.candidates_per_target,
        latent_scale=args.latent_scale, bootstrap_resamples=args.bootstrap_resamples,
        allow_software_only=args.allow_software_only,
    )
    print(json.dumps({
        "experiment": report["experiment"],
        "source": report["evidence_class"],
        "split_sha256": report["split_sha256"],
        "targets": report["heldout_geometry_groups"],
        "physics_solver_evaluations": report["physics_solver_evaluations"],
        "scientific_submission_ready": report["scientific_submission_ready"],
        "report_dir": str(args.out),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
