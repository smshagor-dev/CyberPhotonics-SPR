from __future__ import annotations

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import argparse
import json
from pathlib import Path

from sprpcf.utils.stable_release import (
    apply_stable_version,
    build_stable_release_plan,
    stable_release_plan_markdown,
    validate_stable_release_certificate,
)


def _write(path: Path | None, text: str) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gate stable-version promotion on qualified evidence, finalization, and manuscript Results readiness."
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--finalization-dir", type=Path)
    parser.add_argument("--paper-results-dir", type=Path)
    parser.add_argument("--target-version", default="1.0.0")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Update pyproject/package/CITATION metadata and write STABLE_RELEASE_EVIDENCE.json only if the strict gate passes.",
    )
    parser.add_argument("--validate-certificate", action="store_true")
    args = parser.parse_args()

    if args.validate_certificate:
        report = validate_stable_release_certificate(args.repo_root, expected_version=args.target_version)
        print(json.dumps(report, indent=2, sort_keys=True))
        if args.strict and not report["ok"]:
            raise SystemExit(2)
        return

    if args.finalization_dir is None or args.paper_results_dir is None:
        parser.error("--finalization-dir and --paper-results-dir are required unless --validate-certificate is used.")
    plan = build_stable_release_plan(
        repo_root=args.repo_root,
        finalization_dir=args.finalization_dir,
        paper_results_dir=args.paper_results_dir,
        target_version=args.target_version,
    )
    _write(args.json_out, json.dumps(plan, indent=2, sort_keys=True) + "\n")
    _write(args.markdown_out, stable_release_plan_markdown(plan))
    result: dict[str, object] = {"plan": plan}
    if args.apply:
        result["certificate"] = apply_stable_version(plan, repo_root=args.repo_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.strict and not plan["ready_for_promotion"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
