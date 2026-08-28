from __future__ import annotations

import argparse
import json
from pathlib import Path

from sprpcf.utils.release import validate_release


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate research-release metadata and repository hygiene.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--expected-version")
    parser.add_argument("--json-out", type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = validate_release(args.repo_root, expected_version=args.expected_version)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(payload, end="")
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(payload, encoding="utf-8")
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
