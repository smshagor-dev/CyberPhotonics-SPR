from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from sprpcf.utils.reproducibility import sha256_file

PHYSICAL_EVIDENCE_CLASSES = (
    "comsol_physics",
    "experimental_sensor",
    "device_benchmark",
)
_REQUIRED_BENCHMARK_KEYS = (
    "iterations",
    "latency_ms_p50",
    "latency_ms_p95",
    "latency_ms_p99",
    "throughput_fps",
)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSON artifact {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return payload


def _resolve_artifact(value: str | Path | None, *, fallback_root: Path, fallback_name: str | None = None) -> Path:
    candidates: list[Path] = []
    if value is not None:
        supplied = Path(value)
        candidates.append(supplied)
        if not supplied.is_absolute():
            candidates.append(fallback_root / supplied)
    if fallback_name is not None:
        candidates.append(fallback_root / fallback_name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    display = value if value is not None else fallback_name
    raise FileNotFoundError(f"Required evidence artifact is unavailable: {display}")


def _artifact(path: Path, role: str, *, registry_parent: Path) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        stored = resolved.relative_to(registry_parent.resolve()).as_posix()
    except ValueError:
        stored = str(resolved)
    return {
        "role": role,
        "path": stored,
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _record_id(evidence_class: str, artifacts: Sequence[Mapping[str, Any]], metadata: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        {
            "evidence_class": evidence_class,
            "artifacts": [{"role": row["role"], "sha256": row["sha256"]} for row in artifacts],
            "metadata": dict(metadata),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()[:20]


def _new_record(
    evidence_class: str,
    label: str,
    artifacts: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    if evidence_class not in PHYSICAL_EVIDENCE_CLASSES:
        raise ValueError(f"Unsupported qualified evidence class: {evidence_class}")
    artifact_rows = [dict(row) for row in artifacts]
    metadata_row = dict(metadata)
    return {
        "record_id": _record_id(evidence_class, artifact_rows, metadata_row),
        "evidence_class": evidence_class,
        "label": label,
        "qualification": "manifest_and_hash_validated",
        "qualified": True,
        "artifacts": artifact_rows,
        "metadata": metadata_row,
    }


def _manifest_value_path(manifest: Mapping[str, Any], section: str, name: str) -> str | None:
    values = manifest.get(section, {})
    if not isinstance(values, Mapping):
        return None
    value = values.get(name)
    return str(value) if value else None


def qualify_comsol_iteration(
    iteration_dir: str | Path,
    *,
    registry_path: str | Path,
    model_path: str | Path | None = None,
    config_path: str | Path | None = None,
    label: str = "COMSOL closed-loop evidence",
) -> dict[str, Any]:
    iteration = Path(iteration_dir).resolve()
    manifest_path = iteration / "iteration_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing closed-loop manifest: {manifest_path}")
    manifest = _read_json(manifest_path)
    if str(manifest.get("backend", "")).lower() != "comsol":
        raise ValueError("COMSOL evidence qualification requires iteration_manifest backend='comsol'.")
    if str(manifest.get("evidence_class", "")).lower() != "comsol_physics":
        raise ValueError("COMSOL evidence qualification requires evidence_class='comsol_physics'.")

    manifest_model = _manifest_value_path(manifest, "inputs", "comsol_model")
    manifest_config = _manifest_value_path(manifest, "inputs", "comsol_config")
    model = _resolve_artifact(model_path or manifest_model, fallback_root=iteration)
    config = _resolve_artifact(config_path or manifest_config, fallback_root=iteration)
    simulation = _resolve_artifact(
        _manifest_value_path(manifest, "outputs", "simulation_results"),
        fallback_root=iteration,
        fallback_name="simulation_results.csv",
    )
    verification = _resolve_artifact(
        _manifest_value_path(manifest, "outputs", "verification"),
        fallback_root=iteration,
        fallback_name="verification.csv",
    )

    inputs = manifest.get("inputs", {})
    if not isinstance(inputs, Mapping):
        raise ValueError("Closed-loop manifest inputs must be an object.")
    expected_model_hash = str(inputs.get("comsol_model_sha256") or "")
    expected_config_hash = str(inputs.get("comsol_config_sha256") or "")
    if not expected_model_hash or sha256_file(model) != expected_model_hash:
        raise ValueError("COMSOL model hash does not match iteration_manifest.json.")
    if not expected_config_hash or sha256_file(config) != expected_config_hash:
        raise ValueError("COMSOL config hash does not match iteration_manifest.json.")

    with verification.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        required = {"target_id", "accepted", "reason"}
        if not required.issubset(fields):
            raise ValueError(f"verification.csv missing required columns: {sorted(required - fields)}")

    registry_parent = Path(registry_path).resolve().parent
    artifacts = [
        _artifact(manifest_path, "iteration_manifest", registry_parent=registry_parent),
        _artifact(model, "comsol_model", registry_parent=registry_parent),
        _artifact(config, "comsol_config", registry_parent=registry_parent),
        _artifact(simulation, "simulation_results", registry_parent=registry_parent),
        _artifact(verification, "verification", registry_parent=registry_parent),
    ]
    sidecar = simulation.with_suffix(simulation.suffix + ".meta.json")
    if sidecar.is_file():
        artifacts.append(_artifact(sidecar, "simulation_metadata", registry_parent=registry_parent))

    metadata = {
        "backend": "comsol",
        "selected_targets": int(manifest.get("selected_targets", 0)),
        "accepted_targets": int(manifest.get("accepted_targets", 0)),
        "ri_points": int(manifest.get("ri_points", 0)),
        "seed": int(manifest.get("seed", 0)),
    }
    return _new_record("comsol_physics", label, artifacts, metadata)


def _parse_timestamp(value: str) -> str:
    normalized = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError("acquired_at must be an ISO-8601 timestamp with timezone.") from exc
    if parsed.tzinfo is None:
        raise ValueError("acquired_at must include a timezone offset or Z.")
    return parsed.isoformat()


def qualify_experimental_sensor(
    raw_data: Sequence[str | Path],
    *,
    protocol_path: str | Path,
    calibration_path: str | Path,
    instrument_id: str,
    acquired_at: str,
    registry_path: str | Path,
    label: str = "Experimental sensor evidence",
) -> dict[str, Any]:
    if not raw_data:
        raise ValueError("At least one raw measured-data artifact is required.")
    if not instrument_id.strip():
        raise ValueError("instrument_id is required for experimental evidence.")
    acquired = _parse_timestamp(acquired_at)
    protocol = _resolve_artifact(protocol_path, fallback_root=Path.cwd())
    calibration = _resolve_artifact(calibration_path, fallback_root=Path.cwd())
    measured = [_resolve_artifact(path, fallback_root=Path.cwd()) for path in raw_data]

    registry_parent = Path(registry_path).resolve().parent
    artifacts = [
        *[
            _artifact(path, f"raw_measured_data_{index:03d}", registry_parent=registry_parent)
            for index, path in enumerate(measured, start=1)
        ],
        _artifact(protocol, "experimental_protocol", registry_parent=registry_parent),
        _artifact(calibration, "calibration", registry_parent=registry_parent),
    ]
    metadata = {
        "instrument_id": instrument_id.strip(),
        "acquired_at": acquired,
        "raw_artifact_count": len(measured),
        "source": "measured_sensor",
    }
    return _new_record("experimental_sensor", label, artifacts, metadata)


def qualify_device_benchmark(
    benchmark_path: str | Path,
    *,
    model_path: str | Path,
    device_name: str,
    os_name: str,
    runtime: str,
    registry_path: str | Path,
    accelerator: str | None = None,
    label: str = "Exact-device runtime benchmark",
) -> dict[str, Any]:
    if not device_name.strip() or not os_name.strip() or not runtime.strip():
        raise ValueError("device_name, os_name, and runtime are required for device benchmark evidence.")
    benchmark = _resolve_artifact(benchmark_path, fallback_root=Path.cwd())
    model = _resolve_artifact(model_path, fallback_root=Path.cwd())
    payload = _read_json(benchmark)
    missing = [name for name in _REQUIRED_BENCHMARK_KEYS if name not in payload]
    if missing:
        raise ValueError(f"Benchmark JSON missing required fields: {missing}")
    if int(payload["iterations"]) < 1:
        raise ValueError("Benchmark iterations must be >= 1.")

    registry_parent = Path(registry_path).resolve().parent
    artifacts = [
        _artifact(benchmark, "benchmark_json", registry_parent=registry_parent),
        _artifact(model, "deployed_model", registry_parent=registry_parent),
    ]
    metadata = {
        "device_name": device_name.strip(),
        "os_name": os_name.strip(),
        "runtime": runtime.strip(),
        "accelerator": accelerator.strip() if accelerator else None,
        "iterations": int(payload["iterations"]),
        "benchmark_scope": "exact_named_device",
    }
    return _new_record("device_benchmark", label, artifacts, metadata)


def _artifact_path(registry_path: Path, stored: str) -> Path:
    candidate = Path(stored)
    if candidate.is_absolute():
        return candidate
    return (registry_path.resolve().parent / candidate).resolve()


def validate_evidence_registry(registry_path: str | Path, *, verify_files: bool = True) -> dict[str, Any]:
    path = Path(registry_path)
    errors: list[str] = []
    if not path.is_file():
        return {"ok": False, "errors": [f"Evidence registry not found: {path}"], "evidence_classes": [], "records": 0}
    try:
        payload = _read_json(path)
    except ValueError as exc:
        return {"ok": False, "errors": [str(exc)], "evidence_classes": [], "records": 0}
    if payload.get("schema_version") != 1:
        errors.append("Evidence registry schema_version must be 1.")
    records = payload.get("records")
    if not isinstance(records, list):
        errors.append("Evidence registry records must be a list.")
        records = []

    seen: set[str] = set()
    classes: set[str] = set()
    for index, record in enumerate(records):
        prefix = f"record[{index}]"
        if not isinstance(record, Mapping):
            errors.append(f"{prefix} must be an object.")
            continue
        record_id = str(record.get("record_id") or "")
        evidence_class = str(record.get("evidence_class") or "")
        metadata = record.get("metadata")
        artifacts = record.get("artifacts")
        if not record_id:
            errors.append(f"{prefix} missing record_id.")
        elif record_id in seen:
            errors.append(f"Duplicate record_id: {record_id}")
        seen.add(record_id)
        if evidence_class not in PHYSICAL_EVIDENCE_CLASSES:
            errors.append(f"{prefix} has unsupported evidence_class {evidence_class!r}.")
        if record.get("qualified") is not True or record.get("qualification") != "manifest_and_hash_validated":
            errors.append(f"{prefix} is not qualified by the registry contract.")
        if not isinstance(metadata, Mapping):
            errors.append(f"{prefix}.metadata must be an object.")
            metadata = {}
        if not isinstance(artifacts, list) or not artifacts:
            errors.append(f"{prefix} must contain at least one artifact.")
            continue

        record_ok = True
        canonical_artifacts: list[Mapping[str, Any]] = []
        for artifact_index, artifact in enumerate(artifacts):
            artifact_prefix = f"{prefix}.artifacts[{artifact_index}]"
            if not isinstance(artifact, Mapping):
                errors.append(f"{artifact_prefix} must be an object.")
                record_ok = False
                continue
            canonical_artifacts.append(artifact)
            expected_hash = str(artifact.get("sha256") or "")
            if len(expected_hash) != 64 or any(char not in "0123456789abcdef" for char in expected_hash.lower()):
                errors.append(f"{artifact_prefix} has invalid sha256.")
                record_ok = False
                continue
            if verify_files:
                stored = str(artifact.get("path") or "")
                actual = _artifact_path(path, stored)
                if not actual.is_file():
                    errors.append(f"{artifact_prefix} file missing: {stored}")
                    record_ok = False
                elif sha256_file(actual) != expected_hash:
                    errors.append(f"{artifact_prefix} hash mismatch: {stored}")
                    record_ok = False

        if evidence_class in PHYSICAL_EVIDENCE_CLASSES:
            expected_record_id = _record_id(evidence_class, canonical_artifacts, metadata)
            if record_id != expected_record_id:
                errors.append(f"{prefix} record_id mismatch; class/metadata/artifact hashes were modified.")
                record_ok = False
        if record_ok and evidence_class in PHYSICAL_EVIDENCE_CLASSES:
            classes.add(evidence_class)

    return {
        "ok": not errors,
        "errors": errors,
        "evidence_classes": sorted(classes),
        "records": len(records),
    }


def write_evidence_record(registry_path: str | Path, record: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(registry_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        payload = _read_json(path)
        if payload.get("schema_version") != 1 or not isinstance(payload.get("records"), list):
            raise ValueError("Existing evidence registry has an unsupported structure.")
    else:
        payload = {"schema_version": 1, "records": []}

    records = [row for row in payload["records"] if isinstance(row, Mapping)]
    record_id = str(record.get("record_id") or "")
    records = [row for row in records if str(row.get("record_id") or "") != record_id]
    records.append(dict(record))
    records.sort(key=lambda row: (str(row.get("evidence_class", "")), str(row.get("record_id", ""))))
    payload["records"] = records
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result = validate_evidence_registry(path, verify_files=True)
    if not result["ok"]:
        raise RuntimeError("Evidence registry failed validation after write: " + "; ".join(result["errors"]))
    return result
