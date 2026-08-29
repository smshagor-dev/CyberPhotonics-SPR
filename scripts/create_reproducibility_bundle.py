from __future__ import annotations

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import argparse
import json
from pathlib import Path

import yaml

from sprpcf.utils.reproducibility import create_reproducibility_bundle


def parse_artifact(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Artifact must use ROLE=PATH syntax.")
    role, raw_path = value.split("=", 1)
    if not role.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("Artifact must use non-empty ROLE=PATH syntax.")
    return role.strip(), Path(raw_path.strip())


def load_config(path: Path | None) -> dict:
    if path is None:
        return {}
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
    else:
        payload = yaml.safe_load(text)
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError("Config snapshot must decode to a mapping/object.")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a research provenance and environment bundle.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--name", required=True, help="Experiment/run name.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--artifact", action="append", type=parse_artifact, default=[], help="Repeatable ROLE=PATH artifact.")
    parser.add_argument("--data", type=Path, help="Convenience alias for --artifact dataset=PATH.")
    parser.add_argument("--checkpoint", type=Path, help="Convenience alias for --artifact checkpoint=PATH.")
    parser.add_argument("--config", type=Path, help="YAML/JSON configuration to snapshot into the manifest.")
    parser.add_argument("--notes")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    artifacts = list(args.artifact)
    if args.data is not None:
        artifacts.append(("dataset", args.data))
    if args.checkpoint is not None:
        artifacts.append(("checkpoint", args.checkpoint))
    if args.config is not None:
        artifacts.append(("config", args.config))
    manifest = create_reproducibility_bundle(
        args.out,
        experiment_name=args.name,
        seed=args.seed,
        artifacts=artifacts,
        config=load_config(args.config),
        repo_root=args.repo_root,
        notes=args.notes,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
