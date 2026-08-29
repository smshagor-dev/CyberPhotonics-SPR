from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

from sprpcf.evidence.qualification import (
    qualify_comsol_iteration,
    qualify_device_benchmark,
    qualify_experimental_sensor,
    validate_evidence_registry,
    write_evidence_record,
)
from sprpcf.publication.finalization import (
    build_evidence_finalization_package,
    validate_finalization_package,
)
from sprpcf.publication.results import build_paper_results_package, validate_paper_results_package
from sprpcf.utils.stable_release import build_stable_release_plan
from sprpcf.validation.campaign import campaign_status, load_campaign_config
from sprpcf.validation.experiment import analyze_experimental_measurements

_WORK_NAMES = (
    "Real COMSOL Validation",
    "Experimental Sensor Validation",
    "Exact-Device Benchmark",
    "Evidence-Aware Finalization",
    "Paper Results Finalization",
    "Stable Release",
)


def _snapshot(campaign_dir: Path) -> Path:
    path = campaign_dir / "campaign_config.snapshot.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Campaign snapshot not found: {path}")
    return path


def _is_placeholder(value: Any) -> bool:
    text = str(value or "").strip().upper()
    return not text or "REPLACE" in text


def _resolve_config_path(snapshot: Path, value: str | Path) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    repo_candidate = Path.cwd() / candidate
    if repo_candidate.exists() or candidate.parts[:1] in {("outputs",), ("models",), ("data",), ("configs",)}:
        return repo_candidate
    return (snapshot.parent / candidate).resolve()


def completion_paths(campaign_dir: str | Path) -> dict[str, Path]:
    root = Path(campaign_dir)
    snapshot = _snapshot(root)
    config = load_campaign_config(snapshot)
    experiment = config["experiment"]
    measurement_manifest = experiment.get("measurement_manifest") or (root / "experimental_measurements.yaml")
    analysis_output = experiment.get("analysis_output_dir") or (root / "experimental_analysis")
    finalization = config.get("finalization_package") or (root / "finalization")
    paper_results = config.get("paper_results_package") or (root / "paper_results")
    return {
        "snapshot": snapshot,
        "registry": _resolve_config_path(snapshot, config["registry"]),
        "measurement_manifest": _resolve_config_path(snapshot, measurement_manifest),
        "experimental_analysis": _resolve_config_path(snapshot, analysis_output),
        "finalization": _resolve_config_path(snapshot, finalization),
        "paper_results": _resolve_config_path(snapshot, paper_results),
    }


def _qualification_command(config: Mapping[str, Any], stage: str) -> str:
    registry = shlex.quote(str(config["registry"]))
    if stage == "comsol":
        values = config["comsol"]
        return " ".join(
            [
                "python scripts/register_evidence.py --registry",
                registry,
                "comsol --iteration-dir",
                shlex.quote(str(values["output_dir"])),
                "--model",
                shlex.quote(str(values["model"])),
                "--config",
                shlex.quote(str(values["config"])),
            ]
        )
    if stage == "experiment":
        values = config["experiment"]
        raw = " ".join(f"--raw-data {shlex.quote(str(value))}" for value in values.get("raw_data", []))
        return " ".join(
            [
                "python scripts/register_evidence.py --registry",
                registry,
                "experimental",
                raw,
                "--protocol",
                shlex.quote(str(values["protocol"])),
                "--calibration",
                shlex.quote(str(values["calibration"])),
                "--instrument-id",
                shlex.quote(str(values["instrument_id"])),
                "--acquired-at",
                shlex.quote(str(values["acquired_at"])),
            ]
        )
    values = config["device"]
    accelerator = (
        f" --accelerator {shlex.quote(str(values['accelerator']))}"
        if not _is_placeholder(values.get("accelerator"))
        else ""
    )
    return (
        "python scripts/register_evidence.py --registry "
        + registry
        + " device --benchmark "
        + shlex.quote(str(values["benchmark"]))
        + " --model "
        + shlex.quote(str(values["model"]))
        + " --device-name "
        + shlex.quote(str(values["device_name"]))
        + " --os-name "
        + shlex.quote(str(values["os_name"]))
        + " --runtime "
        + shlex.quote(str(values["runtime"]))
        + accelerator
    )


def _comsol_execution_args(config: Mapping[str, Any]) -> list[str]:
    values = config["comsol"]
    return [
        sys.executable,
        "scripts/run_comsol_closed_loop.py",
        "--backend",
        "comsol",
        "--checkpoint",
        str(values["checkpoint"]),
        "--targets",
        str(values["targets"]),
        "--base-data",
        str(values["base_data"]),
        "--out",
        str(values["output_dir"]),
        "--comsol-model",
        str(values["model"]),
        "--comsol-config",
        str(values["config"]),
        "--passes",
        str(values.get("passes", 32)),
        "--ri-span",
        str(values.get("ri_span", 0.04)),
        "--ri-points",
        str(values.get("ri_points", 5)),
        "--seed",
        str(values.get("seed", 7)),
    ]


def run_real_comsol_validation(campaign_dir: str | Path, *, repo_root: str | Path = ".") -> dict[str, Any]:
    root = Path(campaign_dir)
    config = load_campaign_config(_snapshot(root))
    values = config["comsol"]
    for key in ("checkpoint", "targets", "base_data", "model", "config"):
        if _is_placeholder(values.get(key)):
            raise ValueError(f"COMSOL campaign value {key!r} is still a placeholder.")
    args = _comsol_execution_args(config)
    completed = subprocess.run(args, cwd=Path(repo_root), check=False)
    return {"returncode": completed.returncode, "command": args, "ok": completed.returncode == 0}


def qualify_available_evidence(campaign_dir: str | Path) -> dict[str, Any]:
    root = Path(campaign_dir)
    snapshot = _snapshot(root)
    config = load_campaign_config(snapshot)
    paths = completion_paths(root)
    registry = paths["registry"]
    validation = validate_evidence_registry(registry, verify_files=True)
    qualified = set(validation.get("evidence_classes", [])) if validation.get("ok") else set()
    actions: list[dict[str, Any]] = []

    def record_action(stage: str, status: str, detail: str) -> None:
        actions.append({"stage": stage, "status": status, "detail": detail})

    if "comsol_physics" not in qualified:
        values = config["comsol"]
        iteration = _resolve_config_path(snapshot, values["output_dir"])
        if (iteration / "iteration_manifest.json").is_file():
            try:
                record = qualify_comsol_iteration(
                    iteration,
                    registry_path=registry,
                    model_path=_resolve_config_path(snapshot, values["model"]),
                    config_path=_resolve_config_path(snapshot, values["config"]),
                )
                write_evidence_record(registry, record)
                qualified.add("comsol_physics")
                record_action("Real COMSOL Validation", "qualified", record["record_id"])
            except Exception as exc:  # qualification errors must remain visible without hiding other stage status
                record_action("Real COMSOL Validation", "blocked", str(exc))
        else:
            record_action("Real COMSOL Validation", "pending", "COMSOL iteration artifacts not present")
    else:
        record_action("Real COMSOL Validation", "qualified", "already present in registry")

    if "experimental_sensor" not in qualified:
        values = config["experiment"]
        raw = [_resolve_config_path(snapshot, value) for value in values.get("raw_data", [])]
        protocol = _resolve_config_path(snapshot, values["protocol"])
        calibration = _resolve_config_path(snapshot, values["calibration"])
        metadata_ok = not _is_placeholder(values.get("instrument_id")) and not _is_placeholder(values.get("acquired_at"))
        if raw and all(path.is_file() for path in raw) and protocol.is_file() and calibration.is_file() and metadata_ok:
            try:
                record = qualify_experimental_sensor(
                    raw,
                    protocol_path=protocol,
                    calibration_path=calibration,
                    instrument_id=str(values["instrument_id"]),
                    acquired_at=str(values["acquired_at"]),
                    registry_path=registry,
                )
                write_evidence_record(registry, record)
                qualified.add("experimental_sensor")
                record_action("Experimental Sensor Validation", "qualified", record["record_id"])
            except Exception as exc:
                record_action("Experimental Sensor Validation", "blocked", str(exc))
        else:
            record_action("Experimental Sensor Validation", "pending", "raw data/protocol/calibration/metadata incomplete")
    else:
        record_action("Experimental Sensor Validation", "qualified", "already present in registry")

    if "device_benchmark" not in qualified:
        values = config["device"]
        benchmark = _resolve_config_path(snapshot, values["benchmark"])
        model = _resolve_config_path(snapshot, values["model"])
        metadata_ok = all(not _is_placeholder(values.get(key)) for key in ("device_name", "os_name", "runtime"))
        if benchmark.is_file() and model.is_file() and metadata_ok:
            try:
                accelerator = None if _is_placeholder(values.get("accelerator")) else str(values.get("accelerator"))
                record = qualify_device_benchmark(
                    benchmark,
                    model_path=model,
                    device_name=str(values["device_name"]),
                    os_name=str(values["os_name"]),
                    runtime=str(values["runtime"]),
                    accelerator=accelerator,
                    registry_path=registry,
                )
                write_evidence_record(registry, record)
                qualified.add("device_benchmark")
                record_action("Exact-Device Benchmark", "qualified", record["record_id"])
            except Exception as exc:
                record_action("Exact-Device Benchmark", "blocked", str(exc))
        else:
            record_action("Exact-Device Benchmark", "pending", "benchmark/model/exact-device metadata incomplete")
    else:
        record_action("Exact-Device Benchmark", "qualified", "already present in registry")

    final_validation = validate_evidence_registry(registry, verify_files=True)
    return {"registry": str(registry), "actions": actions, "validation": final_validation}


def _safe_replace_derived(path: Path, marker: str) -> None:
    if not path.exists() or not any(path.iterdir()):
        return
    if not (path / marker).is_file():
        raise ValueError(f"Refusing to replace non-empty derived output without marker {marker}: {path}")
    shutil.rmtree(path)


def refresh_completion_outputs(
    campaign_dir: str | Path,
    *,
    repo_root: str | Path = ".",
    journal: str | None = None,
    manuscript: str | Path | None = None,
    replace: bool = True,
) -> dict[str, Any]:
    root = Path(campaign_dir)
    snapshot = _snapshot(root)
    config = load_campaign_config(snapshot)
    paths = completion_paths(root)

    analysis: dict[str, Any] | None = None
    measurement_manifest = paths["measurement_manifest"]
    if measurement_manifest.is_file():
        if replace:
            _safe_replace_derived(paths["experimental_analysis"], "experimental_summary.json")
        analysis = analyze_experimental_measurements(measurement_manifest, paths["experimental_analysis"])

    registry_validation = validate_evidence_registry(paths["registry"], verify_files=True)
    if not registry_validation.get("ok"):
        return {
            "registry_validation": registry_validation,
            "experimental_analysis": analysis,
            "finalization": None,
            "paper_results": None,
            "stable_release": None,
        }

    finalization = build_evidence_finalization_package(
        paths["finalization"],
        evidence_registry=paths["registry"],
        repo_root=repo_root,
        journal=journal,
        manuscript=manuscript,
        replace=replace,
    )
    paper_results = build_paper_results_package(
        paths["paper_results"],
        evidence_registry=paths["registry"],
        experimental_analysis_dir=paths["experimental_analysis"] if analysis is not None else None,
        replace=replace,
    )
    stable = build_stable_release_plan(
        repo_root=repo_root,
        finalization_dir=paths["finalization"],
        paper_results_dir=paths["paper_results"],
        target_version="1.0.0",
    )
    return {
        "registry_validation": registry_validation,
        "experimental_analysis": analysis,
        "finalization": finalization,
        "paper_results": paper_results,
        "stable_release": stable,
    }


def research_completion_status(
    campaign_dir: str | Path,
    *,
    repo_root: str | Path = ".",
) -> dict[str, Any]:
    root = Path(campaign_dir)
    snapshot = _snapshot(root)
    config = load_campaign_config(snapshot)
    paths = completion_paths(root)
    campaign = campaign_status(root, evidence_registry=paths["registry"])

    analysis_ready = (paths["experimental_analysis"] / "experimental_summary.json").is_file()
    final_validation = (
        validate_finalization_package(paths["finalization"])
        if (paths["finalization"] / "FINALIZATION_MANIFEST.json").is_file()
        else {"ok": False, "ready_for_stable_release": False, "errors": ["finalization package not built"]}
    )
    results_validation = (
        validate_paper_results_package(paths["paper_results"])
        if (paths["paper_results"] / "PAPER_RESULTS_MANIFEST.json").is_file()
        else {"ok": False, "ready_for_manuscript_results": False, "errors": ["paper-results package not built"]}
    )
    stable = None
    if final_validation.get("ok") and results_validation.get("ok"):
        stable = build_stable_release_plan(
            repo_root=repo_root,
            finalization_dir=paths["finalization"],
            paper_results_dir=paths["paper_results"],
            target_version="1.0.0",
        )

    stage_map = {name: row for name, row in campaign.get("stages", {}).items() if isinstance(row, Mapping)}
    works = [
        {
            "name": _WORK_NAMES[0],
            "status": stage_map.get("comsol", {}).get("status", "pending"),
            "next_command": _qualification_command(config, "comsol"),
        },
        {
            "name": _WORK_NAMES[1],
            "status": (
                "qualified_and_analyzed"
                if stage_map.get("experiment", {}).get("status") == "qualified" and analysis_ready
                else stage_map.get("experiment", {}).get("status", "pending")
            ),
            "next_command": (
                f"python scripts/analyze_experimental_results.py --manifest {shlex.quote(str(paths['measurement_manifest']))} "
                f"--out {shlex.quote(str(paths['experimental_analysis']))}"
            ),
        },
        {
            "name": _WORK_NAMES[2],
            "status": stage_map.get("device", {}).get("status", "pending"),
            "next_command": _qualification_command(config, "device"),
        },
        {
            "name": _WORK_NAMES[3],
            "status": "ready" if final_validation.get("ok") else "pending",
            "next_command": "python scripts/research_completion.py refresh --campaign " + shlex.quote(str(root)),
        },
        {
            "name": _WORK_NAMES[4],
            "status": "ready" if results_validation.get("ready_for_manuscript_results") else "pending",
            "next_command": "python scripts/build_paper_results.py --evidence-registry "
            + shlex.quote(str(paths["registry"]))
            + " --experimental-analysis-dir "
            + shlex.quote(str(paths["experimental_analysis"]))
            + " --out "
            + shlex.quote(str(paths["paper_results"])),
        },
        {
            "name": _WORK_NAMES[5],
            "status": "ready" if stable and stable.get("ready_for_promotion") else "blocked",
            "next_command": "python scripts/prepare_stable_release.py --finalization-dir "
            + shlex.quote(str(paths["finalization"]))
            + " --paper-results-dir "
            + shlex.quote(str(paths["paper_results"]))
            + " --target-version 1.0.0 --strict",
        },
    ]
    return {
        "schema_version": 1,
        "campaign": campaign,
        "paths": {key: str(value) for key, value in paths.items()},
        "works": works,
        "finalization_validation": final_validation,
        "paper_results_validation": results_validation,
        "stable_release": stable,
        "complete": bool(stable and stable.get("ready_for_promotion")),
        "scientific_boundary": (
            "External COMSOL execution, laboratory measurement, and exact-device benchmarking remain real-world actions. "
            "This controller can qualify and propagate supplied artifacts but cannot manufacture missing physical evidence."
        ),
    }


def research_completion_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Research Completion Status",
        "",
        "| Work | Status | Next command |",
        "|---|---|---|",
    ]
    for row in report.get("works", []):
        command = str(row.get("next_command", "")).replace("|", "\\|")
        lines.append(f"| {row.get('name')} | `{row.get('status')}` | `{command}` |")
    lines.extend(
        [
            "",
            f"Overall completion gate: **{'pass' if report.get('complete') else 'blocked'}**",
            "",
            str(report.get("scientific_boundary", "")),
            "",
        ]
    )
    return "\n".join(lines)
