from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from sprpcf.validation.campaign import (
    campaign_status,
    campaign_status_markdown,
    initialize_campaign,
    stable_release_gate,
)
from sprpcf.validation.preflight import build_campaign_preflight, campaign_preflight_markdown


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan, preflight, inspect, and gate the CyberPhotonics-SPR Real Validation Campaign."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    preflight = sub.add_parser("preflight", help="Validate real execution inputs and metadata before acquisition.")
    preflight.add_argument("--config", type=Path, required=True)
    preflight.add_argument("--json-out", type=Path)
    preflight.add_argument("--markdown-out", type=Path)
    preflight.add_argument("--strict", action="store_true")

    init = sub.add_parser("init", help="Create a hash-bound campaign manifest and reviewer-facing runbook.")
    init.add_argument("--config", type=Path, required=True)
    init.add_argument("--out", type=Path, required=True)
    init.add_argument("--overwrite", action="store_true")

    status = sub.add_parser("status", help="Inspect campaign artifacts and qualified evidence.")
    status.add_argument("--campaign", type=Path, required=True)
    status.add_argument("--evidence-registry", type=Path)
    status.add_argument("--json-out", type=Path)
    status.add_argument("--markdown-out", type=Path)

    gate = sub.add_parser("gate", help="Require full physical evidence plus a stable project version.")
    gate.add_argument("--campaign", type=Path, required=True)
    gate.add_argument("--repo-root", type=Path, default=Path("."))
    gate.add_argument("--expected-version", default="1.0.0")
    gate.add_argument("--json-out", type=Path)
    gate.add_argument("--strict", action="store_true")
    return parser


def _write(path: Path | None, text: str) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _normalize_runbook_headings(path: Path) -> None:
    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"^(## )[^\s]+ — ", r"\1", text, flags=re.MULTILINE)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "preflight":
        report = build_campaign_preflight(args.config)
        _write(args.markdown_out, campaign_preflight_markdown(report))
    elif args.command == "init":
        report = initialize_campaign(args.config, args.out, overwrite=args.overwrite)
        _normalize_runbook_headings(args.out / "RUNBOOK.md")
    elif args.command == "status":
        report = campaign_status(args.campaign, evidence_registry=args.evidence_registry)
        _write(args.markdown_out, campaign_status_markdown(report))
    else:
        report = stable_release_gate(
            args.campaign,
            repo_root=args.repo_root,
            expected_version=args.expected_version,
        )

    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(payload, end="")
    _write(getattr(args, "json_out", None), payload)

    if args.command == "preflight" and args.strict and not report["ready"]:
        raise SystemExit(2)
    if args.command == "gate" and args.strict and not report["ready_for_stable_release"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
