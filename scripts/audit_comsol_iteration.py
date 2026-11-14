from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sprpcf.validation.physics_audit import audit_comsol_iteration


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit archived COMSOL spectra and verification tables for consistency. "
            "A passing audit does not establish electromagnetic or experimental correctness."
        )
    )
    parser.add_argument("--iteration", type=Path, required=True, help="Closed-loop iteration directory")
    parser.add_argument("--out", type=Path, required=True, help="Write machine-readable audit report here")
    parser.add_argument("--metric-tolerance-nm", type=float, default=0.05)
    parser.add_argument("--strict", action="store_true", help="Exit 2 when the audit detects inconsistencies")
    args = parser.parse_args()

    report = audit_comsol_iteration(args.iteration, metric_tolerance_nm=args.metric_tolerance_nm)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    if args.strict and not report["consistency_ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
