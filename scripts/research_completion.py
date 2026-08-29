from __future__ import annotations

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import argparse
import json
from pathlib import Path

from sprpcf.validation.completion import (
    qualify_available_evidence,
    refresh_completion_outputs,
    research_completion_markdown,
    research_completion_status,
    run_real_comsol_validation,
)


def _write(path: Path | None, text: str) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Drive Real COMSOL Validation through Stable Release without fabricating missing physical evidence."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    status = sub.add_parser("status", help="Show the six-work completion state and next commands.")
    status.add_argument("--campaign", type=Path, required=True)
    status.add_argument("--repo-root", type=Path, default=Path("."))
    status.add_argument("--json-out", type=Path)
    status.add_argument("--markdown-out", type=Path)

    comsol = sub.add_parser("run-comsol", help="Run the configured real COMSOL closed-loop command locally.")
    comsol.add_argument("--campaign", type=Path, required=True)
    comsol.add_argument("--repo-root", type=Path, default=Path("."))

    qualify = sub.add_parser("qualify-ready", help="Qualify any real stage whose required artifacts are already present.")
    qualify.add_argument("--campaign", type=Path, required=True)

    refresh = sub.add_parser(
        "refresh",
        help="Analyze supplied measurements, rebuild finalization/results packages, and recompute stable-release readiness.",
    )
    refresh.add_argument("--campaign", type=Path, required=True)
    refresh.add_argument("--repo-root", type=Path, default=Path("."))
    refresh.add_argument("--journal")
    refresh.add_argument("--manuscript", type=Path)
    refresh.add_argument("--no-replace", action="store_true")

    advance = sub.add_parser(
        "advance",
        help="Qualify all currently available real artifacts, refresh derived packages, then print the new completion state.",
    )
    advance.add_argument("--campaign", type=Path, required=True)
    advance.add_argument("--repo-root", type=Path, default=Path("."))
    advance.add_argument("--journal")
    advance.add_argument("--manuscript", type=Path)
    advance.add_argument("--strict-complete", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "run-comsol":
        report = run_real_comsol_validation(args.campaign, repo_root=args.repo_root)
        print(json.dumps(report, indent=2, sort_keys=True))
        if not report["ok"]:
            raise SystemExit(report["returncode"] or 2)
        return
    if args.command == "qualify-ready":
        report = qualify_available_evidence(args.campaign)
        print(json.dumps(report, indent=2, sort_keys=True))
        if not report["validation"].get("ok"):
            raise SystemExit(2)
        return
    if args.command == "refresh":
        report = refresh_completion_outputs(
            args.campaign,
            repo_root=args.repo_root,
            journal=args.journal,
            manuscript=args.manuscript,
            replace=not args.no_replace,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    if args.command == "advance":
        qualification = qualify_available_evidence(args.campaign)
        refresh = refresh_completion_outputs(
            args.campaign,
            repo_root=args.repo_root,
            journal=args.journal,
            manuscript=args.manuscript,
            replace=True,
        )
        status = research_completion_status(args.campaign, repo_root=args.repo_root)
        report = {"qualification": qualification, "refresh": refresh, "status": status}
        print(json.dumps(report, indent=2, sort_keys=True))
        if args.strict_complete and not status["complete"]:
            raise SystemExit(2)
        return

    report = research_completion_status(args.campaign, repo_root=args.repo_root)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(payload, end="")
    _write(args.json_out, payload)
    _write(args.markdown_out, research_completion_markdown(report))


if __name__ == "__main__":
    main()
