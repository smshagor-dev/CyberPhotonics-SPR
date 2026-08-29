from __future__ import annotations

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import argparse
from pathlib import Path

from sprpcf.simulation.comsol_sweep import write_dataset
from sprpcf.simulation.synthetic import DEFAULT_ANALYTE_RI, build_synthetic_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic fixed-geometry RI sweeps for pipeline validation.")
    parser.add_argument("--samples", type=int, default=100, help="Number of base geometries; output has five RI rows per geometry.")
    parser.add_argument("--wavelengths", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    frame = build_synthetic_dataset(args.samples, wavelengths=args.wavelengths, seed=args.seed)
    write_dataset(
        frame,
        args.out,
        metadata={
            "source": "synthetic",
            "seed": args.seed,
            "base_geometries": args.samples,
            "wavelength_samples": args.wavelengths,
            "analyte_ri_values": list(DEFAULT_ANALYTE_RI),
        },
    )
    print(f"Wrote {len(frame)} rows ({args.samples} base geometries) to {args.out}")


if __name__ == "__main__":
    main()
