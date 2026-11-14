"""Inspect local research artifacts without treating files as independent scientific proof."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sprpcf.evidence.inventory import build_evidence_inventory, evidence_inventory_markdown


def main() -> None:
    parser = argparse.ArgumentParser(description="Inventory datasets, checkpoints and physical-evidence candidates.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--external-root", type=Path, action="append", default=[],
                        help="Additional local experiment archive to scan; repeat as needed.")
    parser.add_argument("--evidence-registry", type=Path,
                        help="Qualified evidence registry; absent defaults to outputs/evidence/evidence_registry.json.")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    parser.add_argument("--fail-on-blockers", action="store_true",
                        help="Exit code 2 when any listed submission requirement remains blocked.")
    args = parser.parse_args()
    report = build_evidence_inventory(
        args.repo_root, external_roots=args.external_root, evidence_registry=args.evidence_registry,
    )
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    print(payload, end="")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(payload, encoding="utf-8")
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(evidence_inventory_markdown(report), encoding="utf-8")
    if args.fail_on_blockers and any(row["status"] == "blocked" for row in report["requirements"]):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
