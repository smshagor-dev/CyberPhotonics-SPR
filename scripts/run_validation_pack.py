from __future__ import annotations

import argparse
import json
from pathlib import Path

from sprpcf.validation.benchmark import run_validation_pack


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the CyberPhotonics-SPR scientific validation pack.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("outputs/validation"))
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--mc-samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    summary = run_validation_pack(
        data_path=args.data,
        checkpoint_path=args.checkpoint,
        output_dir=args.out,
        bootstrap_resamples=args.bootstrap_resamples,
        mc_samples=args.mc_samples,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
