from __future__ import annotations

import json
import re
import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

from sprpcf.evidence.qualification import PHYSICAL_EVIDENCE_CLASSES, validate_evidence_registry
from sprpcf.utils.reproducibility import sha256_file

CAMPAIGN_SCHEMA_VERSION = 1
_REQUIRED_SECTIONS = ("comsol", "experiment", "device")
_STAGE_CLASS = {
    "comsol": "comsol_physics",
    "experiment": "experimental_sensor",
    "device": "device_benchmark",
}
_PRERELEASE_PATTERN = re.compile(r"(?:a|b|rc)\d+$")


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"Invalid campaign YAML {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Campaign configuration must contain a YAML mapping.")
    return payload


def _require_mapping(payload: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = payload.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"Campaign section {name!r} must be a mapping.")
    return dict(value)


def validate_campaign_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if int(config.get("schema_version", 0)) != CAMPAIGN_SCHEMA_VERSION:
        raise ValueError(f"campaign schema_version must be {CAMPAIGN_SCHEMA_VERSION}.")
    campaign_id = str(config.get("campaign_id") or "").strip()
    if not campaign_id:
        raise ValueError("campaign_id is required.")
    if any(char in campaign_id for char in "/\\"):
        raise ValueError("campaign_id must not contain path separators.")

    normalized: dict[str, Any] = {
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "campaign_id": campaign_id,
        "registry": str(config.get("registry") or "outputs/evidence/evidence_registry.json"),
        "reviewer_package": str(config.get("reviewer_package") or "outputs/reviewer_package"),
        "submission_package": str(config.get("submission_package") or "outputs/submission_package"),
    }
    for section_name in _REQUIRED_SECTIONS:
        normalized[section_name] = _require_mapping(config, section_name)

    comsol = normalized["comsol"]
    required_comsol = ("checkpoint", "targets", "base_data", "model", "config", "output_dir")
    missing = [name for name in required_comsol if not str(comsol.get(name) or "").strip()]
    if missing:
        raise ValueError(f"comsol section missing required values: {missing}")

    experiment = normalized["experiment"]
    required_experiment = ("protocol", "calibration", "instrument_id", "acquired_at")
    missing = [name for name in required_experiment if not str(experiment.get(name) or "").strip()]
    if missing:
        raise ValueError(f"experiment section missing required values: {missing}")
    raw_data = experiment.get("raw_data", [])
    if not isinstance(raw_data, list):
        raise ValueError("experiment.raw_data must be a list of planned measured-data paths.")
    experiment["raw_data"] = [str(value) for value in raw_data]

    device = normalized["device"]
    required_device = ("benchmark", "model", "device_name", "os_name", "runtime")
    missing = [name for name in required_device if not str(device.get(name) or "").strip()]
    if missing:
        raise ValueError(f"device section missing required values: {missing}")

    return normalized


def load_campaign_config(path: str | Path) -> dict[str, Any]:
    return validate_campaign_config(_read_yaml(Path(path)))


def _quote(value: Any) -> str:
    return shlex.quote(str(value))


def _qualify_commands(config: Mapping[str, Any]) -> dict[str, str]:
    registry = _quote(config["registry"])
    comsol = config["comsol"]
    experiment = config["experiment"]
    device = config["device"]

    comsol_command = " ".join(
        [
            "python scripts/register_evidence.py",
            "--registry", registry,
            "comsol",
            "--iteration-dir", _quote(comsol["output_dir"]),
            "--model", _quote(comsol["model"]),
            "--config", _quote(comsol["config"]),
        ]
    )

    raw = " ".join(f"--raw-data {_quote(path)}" for path in experiment.get("raw_data", []))
    experiment_command = " ".join(
        part
        for part in [
            "python scripts/register_evidence.py",
            "--registry", registry,
            "experimental",
            raw,
            "--protocol", _quote(experiment["protocol"]),
            "--calibration", _quote(experiment["calibration"]),
            "--instrument-id", _quote(experiment["instrument_id"]),
            "--acquired-at", _quote(experiment["acquired_at"]),
        ]
        if part
    )

    device_command = " ".join(
        part
        for part in [
            "python scripts/register_evidence.py",
            "--registry", registry,
            "device",
            "--benchmark", _quote(device["benchmark"]),
            "--model", _quote(device["model"]),
            "--device-name", _quote(device["device_name"]),
            "--os-name", _quote(device["os_name"]),
            "--runtime", _quote(device["runtime"]),
            (
                "--accelerator " + _quote(device["accelerator"])
                if str(device.get("accelerator") or "").strip()
                else ""
            ),
        ]
        if part
    )
    return {"comsol": comsol_command, "experiment": experiment_command, "device": device_command}


def _execution_commands(config: Mapping[str, Any]) -> dict[str, str]:
    comsol = config["comsol"]
    command = " ".join(
        [
            "python scripts/run_comsol_closed_loop.py",
            "--backend comsol",
            "--checkpoint", _quote(comsol["checkpoint"]),
            "--targets", _quote(comsol["targets"]),
            "--base-data", _quote(comsol["base_data"]),
            "--out", _quote(comsol["output_dir"]),
            "--comsol-model", _quote(comsol["model"]),
            "--comsol-config", _quote(comsol["config"]),
            "--passes", _quote(comsol.get("passes", 32)),
            "--ri-span", _quote(comsol.get("ri_span", 0.04)),
            "--ri-points", _quote(comsol.get("ri_points", 5)),
            "--seed", _quote(comsol.get("seed", 7)),
        ]
    )
    return {"comsol": command}


def initialize_campaign(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    source = Path(config_path).resolve()
    config = load_campaign_config(source)
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()) and not overwrite:
        raise FileExistsError(f"Campaign output must be empty or use overwrite=True: {out}")
    out.mkdir(parents=True, exist_ok=True)

    snapshot = out / "campaign_config.snapshot.yaml"
    snapshot.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    commands = _qualify_commands(config)
    manifest = {
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "campaign_id": config["campaign_id"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_config": source.name,
        "snapshot": snapshot.name,
        "snapshot_sha256": sha256_file(snapshot),
        "registry": str(config["registry"]),
        "required_evidence_classes": list(PHYSICAL_EVIDENCE_CLASSES),
        "stages": {
            name: {
                "required_evidence_class": _STAGE_CLASS[name],
                "status": "pending",
                "qualification_command": commands[name],
            }
            for name in _REQUIRED_SECTIONS
        },
        "scientific_boundary": (
            "Campaign initialization creates plans and provenance only. It does not create COMSOL, "
            "experimental-sensor, or exact-device evidence."
        ),
    }
    (out / "campaign_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out / "RUNBOOK.md").write_text(campaign_runbook(config), encoding="utf-8")
    (out / "EXPERIMENT_PROTOCOL_TEMPLATE.md").write_text(
        "# Experimental Protocol\n\n"
        "- Instrument ID: REPLACE\n"
        "- Operator: REPLACE\n"
        "- Acquisition timestamp/timezone: REPLACE\n"
        "- Analyte preparation: REPLACE\n"
        "- Temperature/environment: REPLACE\n"
        "- Optical source and spectrometer settings: REPLACE\n"
        "- Dark/reference acquisition: REPLACE\n"
        "- Replicates and ordering: REPLACE\n"
        "- Raw-data file mapping: REPLACE\n"
        "- Deviations/notes: REPLACE\n",
        encoding="utf-8",
    )
    (out / "CALIBRATION_RECORD_TEMPLATE.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "instrument_id": "REPLACE",
                "wavelength_calibration": "REPLACE",
                "dark_reference": "REPLACE",
                "calibrated_at": "REPLACE_WITH_TIMEZONE_AWARE_ISO8601",
                "notes": "REPLACE",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (out / "DEVICE_METADATA_TEMPLATE.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "device_name": "REPLACE",
                "os_name": "REPLACE",
                "runtime": "REPLACE",
                "accelerator": "REPLACE_IF_APPLICABLE",
                "model_file": str(config["device"]["model"]),
                "benchmark_file": str(config["device"]["benchmark"]),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def _planned_artifacts(config: Mapping[str, Any], stage: str) -> list[Path]:
    if stage == "comsol":
        root = Path(config["comsol"]["output_dir"])
        return [root / "iteration_manifest.json", root / "simulation_results.csv", root / "verification.csv"]
    if stage == "experiment":
        experiment = config["experiment"]
        return [
            *[Path(value) for value in experiment.get("raw_data", [])],
            Path(experiment["protocol"]),
            Path(experiment["calibration"]),
        ]
    device = config["device"]
    return [Path(device["benchmark"]), Path(device["model"])]


def campaign_status(
    campaign_dir: str | Path,
    *,
    evidence_registry: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(campaign_dir)
    manifest_path = root / "campaign_manifest.json"
    snapshot_path = root / "campaign_config.snapshot.yaml"
    if not manifest_path.is_file() or not snapshot_path.is_file():
        raise FileNotFoundError("Campaign directory is missing campaign_manifest.json or campaign_config.snapshot.yaml.")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = validate_campaign_config(_read_yaml(snapshot_path))
    expected_snapshot_hash = str(manifest.get("snapshot_sha256") or "")
    actual_snapshot_hash = sha256_file(snapshot_path)
    snapshot_ok = expected_snapshot_hash == actual_snapshot_hash

    registry_path = Path(evidence_registry or config["registry"])
    registry = validate_evidence_registry(registry_path, verify_files=True)
    qualified = set(registry.get("evidence_classes", [])) if registry.get("ok") else set()

    stages: dict[str, Any] = {}
    for stage in _REQUIRED_SECTIONS:
        evidence_class = _STAGE_CLASS[stage]
        planned = _planned_artifacts(config, stage)
        existing = [str(path) for path in planned if path.is_file()]
        if evidence_class in qualified:
            status = "qualified"
        elif existing:
            status = "awaiting_qualification"
        else:
            status = "pending"
        stages[stage] = {
            "status": status,
            "required_evidence_class": evidence_class,
            "planned_artifacts": [str(path) for path in planned],
            "existing_artifacts": existing,
        }

    missing = [value for value in PHYSICAL_EVIDENCE_CLASSES if value not in qualified]
    return {
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "campaign_id": config["campaign_id"],
        "campaign_dir": str(root),
        "snapshot_integrity": snapshot_ok,
        "registry": str(registry_path),
        "registry_validation": registry,
        "qualified_evidence_classes": sorted(qualified),
        "missing_evidence_classes": missing,
        "stages": stages,
        "complete": snapshot_ok and registry.get("ok") is True and not missing,
    }


def stable_release_gate(
    campaign_dir: str | Path,
    *,
    repo_root: str | Path = ".",
    expected_version: str = "1.0.0",
) -> dict[str, Any]:
    status = campaign_status(campaign_dir)
    from sprpcf.utils.readiness import build_readiness_report

    config = load_campaign_config(Path(campaign_dir) / "campaign_config.snapshot.yaml")
    readiness = build_readiness_report(
        repo_root,
        profile="full",
        expected_version=expected_version,
        reviewer_package=config["reviewer_package"],
        submission_package=config["submission_package"],
        evidence_registry=config["registry"],
    )
    version = str(readiness.get("sprpcf_version") or "")
    stable_version = bool(version) and _PRERELEASE_PATTERN.search(version) is None
    blockers: list[str] = []
    if not status["complete"]:
        blockers.append("real-validation campaign is incomplete")
    if not readiness["ready"]:
        blockers.append("whole-system full readiness failed")
    if not stable_version:
        blockers.append(f"project version {version!r} is a prerelease")

    return {
        "schema_version": 1,
        "campaign": status,
        "readiness": readiness,
        "stable_version": stable_version,
        "ready_for_stable_release": not blockers,
        "blockers": blockers,
    }


def campaign_runbook(config: Mapping[str, Any]) -> str:
    execute = _execution_commands(config)
    qualify = _qualify_commands(config)
    lines = [
        f"# Real Validation Campaign — {config['campaign_id']}",
        "",
        "This runbook separates software orchestration from external physical evidence.",
        "Nothing is considered physical evidence until it is registered by the qualification registry.",
        "",
        "## Real COMSOL Validation",
        "",
        "Run the real COMSOL-backed closed loop:",
        "",
        "```bash",
        execute["comsol"],
        "```",
        "",
        "Then qualify the generated COMSOL artifacts:",
        "",
        "```bash",
        qualify["comsol"],
        "```",
        "",
        "## Experimental Sensor Validation",
        "",
        "Acquire raw measured spectra using the documented protocol and calibration. Preserve the raw frames before preprocessing.",
        "After the planned raw files exist, register them:",
        "",
        "```bash",
        qualify["experiment"],
        "```",
        "",
        "## Exact-Device Benchmark",
        "",
        "Run `scripts/run_hardware_pipeline.py` on the exact target device with benchmarking enabled, then register:",
        "",
        "```bash",
        qualify["device"],
        "```",
        "",
        "## Evidence Qualification Status",
        "",
        "```bash",
        "python scripts/validation_campaign.py status --campaign <campaign-dir>",
        "```",
        "",
        "## Stable Release Gate",
        "",
        "The stable gate intentionally requires all three qualified physical evidence classes and a non-prerelease project version:",
        "",
        "```bash",
        "python scripts/validation_campaign.py gate --campaign <campaign-dir> --expected-version 1.0.0 --strict",
        "```",
        "",
        "Synthetic/replay outputs never satisfy the physical evidence gate.",
        "",
    ]
    return "\n".join(lines)


def campaign_status_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        f"# Validation Campaign Status — {report['campaign_id']}",
        "",
        f"- Complete: **{'YES' if report['complete'] else 'NO'}**",
        f"- Snapshot integrity: **{'PASS' if report['snapshot_integrity'] else 'FAIL'}**",
        f"- Registry valid: **{'YES' if report['registry_validation'].get('ok') else 'NO'}**",
        "",
        "| Stage | Status | Required evidence |",
        "|---|---|---|",
    ]
    for name, stage in report["stages"].items():
        lines.append(f"| {name} | {stage['status']} | `{stage['required_evidence_class']}` |")
    lines.extend(["", "Missing evidence:"])
    missing = report.get("missing_evidence_classes", [])
    lines.extend(f"- `{value}`" for value in missing) if missing else lines.append("- none")
    lines.extend(["", "Physical completion requires hash-validated registry records; package flags alone are not accepted.", ""])
    return "\n".join(lines)
