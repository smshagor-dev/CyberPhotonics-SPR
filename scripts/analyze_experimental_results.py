from __future__ import annotations

import argparse
import json
from pathlib import Path

from sprpcf.validation.experiment import analyze_experimental_measurements


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze RI-labelled calibrated experimental spectra without qualifying them as physical evidence."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = analyze_experimental_measurements(args.manifest, args.out)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
