from __future__ import annotations

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import argparse
import json
from pathlib import Path

from sprpcf.validation.ablation import run_ablation_study


def _parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seeds:
        raise argparse.ArgumentTypeError("At least one integer seed is required.")
    return seeds


def main() -> None:
    parser = argparse.ArgumentParser(description="Run physics-loss ablation across deterministic seeds.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("outputs/ablation"))
    parser.add_argument("--seeds", type=_parse_seeds, default=(7, 17, 29))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--dispersion-weight", type=float, default=0.0)
    parser.add_argument("--mc-samples", type=int, default=16)
    args = parser.parse_args()

    result = run_ablation_study(
        data_path=args.data,
        output_dir=args.out,
        seeds=args.seeds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device_name=args.device,
        physics_alpha=args.alpha,
        physics_beta=args.beta,
        dispersion_weight=args.dispersion_weight,
        mc_samples=args.mc_samples,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
