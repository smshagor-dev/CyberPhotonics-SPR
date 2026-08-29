from __future__ import annotations

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import argparse
import json
from pathlib import Path

from sprpcf.ml.dataset import read_table
from sprpcf.ml.multiobjective import optimize_target_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Run calibrated Pareto inverse design for PCF-SPR sensing targets.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--reference-data", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("outputs/multiobjective"))
    parser.add_argument("--candidates-per-target", type=int, default=128)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--latent-scale", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    result = optimize_target_table(
        args.checkpoint,
        read_table(args.targets),
        args.reference_data,
        candidates_per_target=args.candidates_per_target,
        confidence=args.confidence,
        latent_scale=args.latent_scale,
        seed=args.seed,
        device=args.device,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    candidates_path = args.out / "pareto_candidates.csv"
    selected_path = args.out / "selected_designs.csv"
    calibration_path = args.out / "calibration.json"
    result.candidates.to_csv(candidates_path, index=False)
    result.selected.to_csv(selected_path, index=False)
    calibration_path.write_text(json.dumps(result.calibration, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "targets": int(len(result.selected)),
                "candidates": int(len(result.candidates)),
                "pareto_candidates": str(candidates_path),
                "selected_designs": str(selected_path),
                "calibration": str(calibration_path),
                **result.calibration,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
