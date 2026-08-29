from __future__ import annotations

import argparse
import json
import sys

from .dependencies import dependency_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit CyberPhotonics-SPR runtime dependencies.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = dependency_report()
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print("CyberPhotonics-SPR Environment Doctor")
        print("=" * 42)
        for row in report:
            state = "READY" if row["available"] else "MISSING"
            required = "required" if row["required"] else "optional"
            print(f"[{state:7}] {row['label']} ({required})")
            if not row["available"]:
                missing = ", ".join(str(value) for value in row["missing"])
                print(f"          Missing: {missing}")
                print(f"          Repair : {row['install']}")
        print()
        print(f"Python: {sys.executable}")

    required_missing = any(bool(row["required"]) and not bool(row["available"]) for row in report)
    return 2 if required_missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
