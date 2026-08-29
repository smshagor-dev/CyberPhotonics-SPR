from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from sprpcf.validation.campaign import load_campaign_config


def _check(name: str, ok: bool, detail: str, *, required: bool = True) -> dict[str, Any]:
    return {
        "name": name,
        "status": "pass" if ok else "fail",
        "required": required,
        "detail": detail,
    }


def _is_placeholder(value: Any) -> bool:
    return "REPLACE" in str(value).upper()


def _existing_file_check(name: str, value: Any) -> dict[str, Any]:
    path = Path(str(value))
    placeholder = _is_placeholder(value)
    exists = path.is_file() if not placeholder else False
    detail = f"path={path}, placeholder={placeholder}, exists={exists}"
    return _check(name, not placeholder and exists, detail)


def _metadata_check(name: str, value: Any) -> dict[str, Any]:
    text = str(value or "").strip()
    placeholder = _is_placeholder(text)
    return _check(name, bool(text) and not placeholder, f"value={text!r}, placeholder={placeholder}")


def _timestamp_check(value: Any) -> dict[str, Any]:
    text = str(value or "").strip()
    if _is_placeholder(text):
        return _check("experiment:acquired_at", False, "timestamp is still a placeholder")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return _check("experiment:acquired_at", False, "timestamp is not valid ISO-8601")
    aware = parsed.tzinfo is not None and parsed.utcoffset() is not None
    return _check(
        "experiment:acquired_at",
        aware,
        "timezone-aware ISO-8601 timestamp" if aware else "timestamp must include a timezone offset or Z",
    )


def _writable_destination_check(name: str, value: Any) -> dict[str, Any]:
    path = Path(str(value))
    if _is_placeholder(value):
        return _check(name, False, f"destination contains a placeholder: {path}")
    candidate = path if path.suffix == "" else path.parent
    probe = candidate
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    writable = probe.exists() and probe.is_dir() and os.access(probe, os.W_OK)
    return _check(name, writable, f"destination={path}, nearest_existing_parent={probe}, writable={writable}")


def build_campaign_preflight(config_path: str | Path) -> dict[str, Any]:
    """Validate that a real-validation configuration is ready for external execution.

    The preflight checks configuration, metadata, source artifacts, and output destinations only.
    It never treats planned paths or synthetic/replay artifacts as physical evidence.
    """
    path = Path(config_path)
    config = load_campaign_config(path)
    checks: list[dict[str, Any]] = []

    comsol = config["comsol"]
    for key in ("checkpoint", "targets", "base_data", "model", "config"):
        checks.append(_existing_file_check(f"comsol:{key}", comsol[key]))
    checks.append(
        _check(
            "comsol:model_suffix",
            str(comsol["model"]).lower().endswith(".mph") and not _is_placeholder(comsol["model"]),
            f"model={comsol['model']}",
        )
    )
    try:
        passes = int(comsol.get("passes", 32))
    except (TypeError, ValueError):
        passes = 0
    checks.append(_check("comsol:passes", passes >= 1, f"passes={passes}"))
    try:
        ri_points = int(comsol.get("ri_points", 5))
    except (TypeError, ValueError):
        ri_points = 0
    checks.append(
        _check(
            "comsol:ri_points",
            ri_points >= 3 and ri_points % 2 == 1,
            f"ri_points={ri_points}; fixed-geometry RI validation requires an odd target-centered sweep with >=3 points",
        )
    )
    try:
        ri_span = float(comsol.get("ri_span", 0.04))
    except (TypeError, ValueError):
        ri_span = 0.0
    checks.append(_check("comsol:ri_span", ri_span > 0.0, f"ri_span={ri_span}"))
    checks.append(_writable_destination_check("comsol:output_dir", comsol["output_dir"]))

    experiment = config["experiment"]
    checks.append(_metadata_check("experiment:instrument_id", experiment["instrument_id"]))
    checks.append(_timestamp_check(experiment["acquired_at"]))
    checks.append(_existing_file_check("experiment:protocol", experiment["protocol"]))
    checks.append(_existing_file_check("experiment:calibration", experiment["calibration"]))
    raw_data = experiment.get("raw_data", [])
    checks.append(
        _check(
            "experiment:raw_data_plan",
            isinstance(raw_data, list) and bool(raw_data) and not any(_is_placeholder(value) for value in raw_data),
            f"planned_raw_artifacts={len(raw_data) if isinstance(raw_data, list) else 0}",
        )
    )
    if isinstance(raw_data, list):
        for index, raw_path in enumerate(raw_data, start=1):
            checks.append(_writable_destination_check(f"experiment:raw_destination:{index}", raw_path))

    device = config["device"]
    checks.append(_metadata_check("device:device_name", device["device_name"]))
    checks.append(_metadata_check("device:os_name", device["os_name"]))
    checks.append(_metadata_check("device:runtime", device["runtime"]))
    if str(device.get("accelerator") or "").strip():
        checks.append(_metadata_check("device:accelerator", device["accelerator"]))
    checks.append(_existing_file_check("device:model", device["model"]))
    runtime = str(device["runtime"]).strip().lower()
    if runtime == "litert":
        checks.append(
            _check(
                "device:model_suffix",
                str(device["model"]).lower().endswith(".tflite"),
                f"LiteRT model={device['model']}",
            )
        )
    checks.append(_writable_destination_check("device:benchmark", device["benchmark"]))

    checks.append(_writable_destination_check("campaign:registry", config["registry"]))
    checks.append(_writable_destination_check("campaign:reviewer_package", config["reviewer_package"]))
    checks.append(_writable_destination_check("campaign:submission_package", config["submission_package"]))

    required_failures = [row for row in checks if row["required"] and row["status"] == "fail"]
    stage_ready = {
        stage: not any(
            row["required"] and row["status"] == "fail" and row["name"].startswith(stage + ":")
            for row in checks
        )
        for stage in ("comsol", "experiment", "device")
    }

    return {
        "schema_version": 1,
        "campaign_id": config["campaign_id"],
        "config": str(path),
        "ready": not required_failures,
        "stage_ready": stage_ready,
        "checks": checks,
        "required_failures": required_failures,
        "scientific_boundary": (
            "Preflight readiness means required inputs and metadata are present for external execution. "
            "It is not COMSOL, experimental-sensor, or exact-device evidence."
        ),
    }


def campaign_preflight_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        f"# Real Validation Campaign Preflight — {report['campaign_id']}",
        "",
        f"- Ready: **{'YES' if report['ready'] else 'NO'}**",
        f"- COMSOL execution ready: **{'YES' if report['stage_ready']['comsol'] else 'NO'}**",
        f"- Experimental acquisition ready: **{'YES' if report['stage_ready']['experiment'] else 'NO'}**",
        f"- Exact-device benchmark ready: **{'YES' if report['stage_ready']['device'] else 'NO'}**",
        "",
        "| Check | Status | Detail |",
        "|---|---|---|",
    ]
    for row in report["checks"]:
        detail = str(row["detail"]).replace("|", "\\|").replace("\n", " ")
        lines.append(f"| `{row['name']}` | {row['status']} | {detail} |")
    lines.extend(["", str(report["scientific_boundary"]), ""])
    return "\n".join(lines)
