from __future__ import annotations

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import argparse
import json
from pathlib import Path

from sprpcf.evidence.qualification import (
    qualify_comsol_iteration,
    qualify_device_benchmark,
    qualify_experimental_sensor,
    validate_evidence_registry,
    write_evidence_record,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Register and validate hash-bound physical research evidence.")
    parser.add_argument("--registry", type=Path, default=Path("outputs/evidence/evidence_registry.json"))
    sub = parser.add_subparsers(dest="command", required=True)

    comsol = sub.add_parser("comsol", help="Qualify a real COMSOL closed-loop iteration.")
    comsol.add_argument("--iteration-dir", type=Path, required=True)
    comsol.add_argument("--model", type=Path)
    comsol.add_argument("--config", type=Path)
    comsol.add_argument("--label", default="COMSOL closed-loop evidence")

    experimental = sub.add_parser("experimental", help="Qualify measured sensor evidence.")
    experimental.add_argument("--raw-data", type=Path, action="append", required=True)
    experimental.add_argument("--protocol", type=Path, required=True)
    experimental.add_argument("--calibration", type=Path, required=True)
    experimental.add_argument("--instrument-id", required=True)
    experimental.add_argument("--acquired-at", required=True, help="ISO-8601 timestamp including timezone or Z.")
    experimental.add_argument("--label", default="Experimental sensor evidence")

    device = sub.add_parser("device", help="Qualify an exact-device runtime benchmark.")
    device.add_argument("--benchmark", type=Path, required=True)
    device.add_argument("--model", type=Path, required=True)
    device.add_argument("--device-name", required=True)
    device.add_argument("--os-name", required=True)
    device.add_argument("--runtime", required=True)
    device.add_argument("--accelerator")
    device.add_argument("--label", default="Exact-device runtime benchmark")

    validate = sub.add_parser("validate", help="Validate the registry and all registered artifact hashes.")
    validate.add_argument("--no-file-check", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "validate":
        report = validate_evidence_registry(args.registry, verify_files=not args.no_file_check)
        print(json.dumps(report, indent=2, sort_keys=True))
        if not report["ok"]:
            raise SystemExit(2)
        return

    if args.command == "comsol":
        record = qualify_comsol_iteration(
            args.iteration_dir,
            registry_path=args.registry,
            model_path=args.model,
            config_path=args.config,
            label=args.label,
        )
    elif args.command == "experimental":
        record = qualify_experimental_sensor(
            args.raw_data,
            protocol_path=args.protocol,
            calibration_path=args.calibration,
            instrument_id=args.instrument_id,
            acquired_at=args.acquired_at,
            registry_path=args.registry,
            label=args.label,
        )
    else:
        record = qualify_device_benchmark(
            args.benchmark,
            model_path=args.model,
            device_name=args.device_name,
            os_name=args.os_name,
            runtime=args.runtime,
            accelerator=args.accelerator,
            registry_path=args.registry,
            label=args.label,
        )

    report = write_evidence_record(args.registry, record)
    print(
        json.dumps(
            {
                "registry": str(args.registry),
                "record_id": record["record_id"],
                "evidence_class": record["evidence_class"],
                "validation": report,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
