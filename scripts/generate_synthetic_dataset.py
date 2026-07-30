from __future__ import annotations

import argparse
from pathlib import Path

from sprpcf.simulation.comsol_sweep import write_dataset
from sprpcf.simulation.synthetic import build_synthetic_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic PCF-SPR spectra for pipeline validation.")
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--wavelengths", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    frame = build_synthetic_dataset(args.samples, wavelengths=args.wavelengths, seed=args.seed)
    write_dataset(frame, args.out)
    print(f"Wrote {len(frame)} samples to {args.out}")


if __name__ == "__main__":
    main()
