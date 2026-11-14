"""Run a fail-closed numerical-refinement consistency check on archived COMSOL evidence."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sprpcf.validation.convergence import audit_refinement_archive


def main() -> None:
    parser = argparse.ArgumentParser(description="Check COMSOL mesh and spectral refinement archives; never certify physics.")
    parser.add_argument("--archive", type=Path, required=True, help="Directory containing refinement_manifest.json and artifacts")
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--resonance-tol-nm", type=float, default=0.1)
    parser.add_argument("--fwhm-rel-tol", type=float, default=0.01)
    parser.add_argument("--loss-rel-tol", type=float, default=0.02)
    parser.add_argument("--strict", action="store_true", help="Exit 2 unless archived refinement is consistent")
    args = parser.parse_args()
    report = audit_refinement_archive(
        args.archive,
        resonance_tol_nm=args.resonance_tol_nm,
        fwhm_rel_tol=args.fwhm_rel_tol,
        loss_rel_tol=args.loss_rel_tol,
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True, indent=2, allow_nan=False))
    if args.strict and not report["convergence_consistent"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
