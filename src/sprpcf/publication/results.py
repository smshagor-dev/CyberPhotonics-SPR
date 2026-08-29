from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

from sprpcf.evidence.qualification import PHYSICAL_EVIDENCE_CLASSES, validate_evidence_registry
from sprpcf.utils.reproducibility import sha256_file

_REQUIRED_CLASSES = tuple(PHYSICAL_EVIDENCE_CLASSES)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _artifact_path(registry: Path, stored: str) -> Path:
    candidate = Path(stored)
    return candidate if candidate.is_absolute() else (registry.resolve().parent / candidate).resolve()


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def _safe_prepare(output_dir: Path, *, replace: bool) -> None:
    if not output_dir.exists() or not any(output_dir.iterdir()):
        return
    if not replace:
        raise FileExistsError(f"Paper-results output must be empty or absent: {output_dir}")
    if not (output_dir / "PAPER_RESULTS_MANIFEST.json").is_file():
        raise ValueError("Refusing to replace a non-empty directory not created by the paper-results pipeline.")
    shutil.rmtree(output_dir)


def _records_by_class(payload: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    result = {name: [] for name in _REQUIRED_CLASSES}
    records = payload.get("records", [])
    if not isinstance(records, list):
        return result
    for record in records:
        if not isinstance(record, Mapping) or record.get("qualified") is not True:
            continue
        evidence_class = str(record.get("evidence_class") or "")
        if evidence_class in result:
            result[evidence_class].append(dict(record))
    return result


def _artifact_for_role(registry: Path, record: Mapping[str, Any], role: str) -> Path | None:
    artifacts = record.get("artifacts", [])
    if not isinstance(artifacts, list):
        return None
    for artifact in artifacts:
        if isinstance(artifact, Mapping) and str(artifact.get("role") or "") == role:
            path = _artifact_path(registry, str(artifact.get("path") or ""))
            if path.is_file():
                return path
    return None


def _experimental_raw_hashes(records: Sequence[Mapping[str, Any]]) -> set[str]:
    hashes: set[str] = set()
    for record in records:
        artifacts = record.get("artifacts", [])
        if not isinstance(artifacts, list):
            continue
        for artifact in artifacts:
            if not isinstance(artifact, Mapping):
                continue
            role = str(artifact.get("role") or "")
            digest = str(artifact.get("sha256") or "")
            if role.startswith("raw_measured_data_") and digest:
                hashes.add(digest)
    return hashes


def _copy(path: Path, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destination)
    return {
        "path": destination.as_posix(),
        "size_bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
    }


def _package_checksums(root: Path) -> None:
    checksum = root / "checksums.sha256"
    targets = sorted(
        (candidate for candidate in root.rglob("*") if candidate.is_file() and candidate != checksum),
        key=lambda value: value.relative_to(root).as_posix(),
    )
    checksum.write_text(
        "\n".join(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}" for path in targets) + "\n",
        encoding="utf-8",
    )


def _write_summary(path: Path, manifest: Mapping[str, Any]) -> None:
    lines = [
        "# Paper Results Evidence Summary",
        "",
        "This document is generated only from hash-validated qualified evidence and traceable derived analysis.",
        "",
        "## Readiness",
        "",
        f"- COMSOL numerical physics: **{'available' if manifest['comsol']['available'] else 'not supplied'}**",
        f"- Experimental sensor evidence: **{'available' if manifest['experiment']['available'] else 'not supplied'}**",
        f"- Experimental derived analysis linked to qualified raw data: **{'yes' if manifest['experiment']['analysis_bound_to_qualified_raw'] else 'no'}**",
        f"- Exact-device benchmark: **{'available' if manifest['device']['available'] else 'not supplied'}**",
        f"- Ready for evidence-backed manuscript Results refresh: **{'yes' if manifest['ready_for_manuscript_results'] else 'no'}**",
        "",
        "## Evidence-backed values",
        "",
    ]
    experiment = manifest["experiment"]
    metrics = experiment.get("metrics") or {}
    if experiment.get("analysis_bound_to_qualified_raw"):
        lines.extend(
            [
                f"- Experimental sensitivity: `{metrics.get('sensitivity_nm_per_riu')}` nm/RIU",
                f"- Experimental RI-fit R²: `{metrics.get('fit_r2')}`",
                f"- Experimental mean FWHM: `{metrics.get('mean_fwhm_nm')}` nm",
                f"- Experimental FOM: `{metrics.get('fom_per_riu')}` 1/RIU",
                f"- Mean resonance repeatability SD: `{metrics.get('repeatability_mean_sd_nm')}` nm",
                f"- Maximum resonance repeatability SD: `{metrics.get('repeatability_max_sd_nm')}` nm",
            ]
        )
    else:
        lines.append("- Experimental performance values are withheld because derived analysis is not bound to qualified raw measurements.")
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "COMSOL evidence is numerical physics, not laboratory measurement. Experimental values appear only when the analysis raw-file hashes are present in the qualified experimental record. Device metrics appear only from an exact-device benchmark record. Missing evidence is reported as missing rather than replaced with synthetic or surrogate values.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_paper_results_package(
    output_dir: str | Path,
    *,
    evidence_registry: str | Path,
    experimental_analysis_dir: str | Path | None = None,
    replace: bool = False,
) -> dict[str, Any]:
    out = Path(output_dir)
    registry = Path(evidence_registry)
    validation = validate_evidence_registry(registry, verify_files=True)
    if not validation.get("ok"):
        raise ValueError("Qualified evidence registry is invalid: " + "; ".join(validation.get("errors", [])))
    _safe_prepare(out, replace=replace)
    out.mkdir(parents=True, exist_ok=True)

    payload = _read_json(registry)
    records = _records_by_class(payload)
    copied: list[dict[str, Any]] = []

    comsol_rows: list[dict[str, Any]] = []
    for record in records["comsol_physics"]:
        record_id = str(record.get("record_id") or "unknown")
        verification = _artifact_for_role(registry, record, "verification")
        iteration_manifest = _artifact_for_role(registry, record, "iteration_manifest")
        if verification is not None:
            destination = out / "tables" / f"comsol_verification_{record_id}.csv"
            entry = _copy(verification, destination)
            entry.update({"role": "comsol_verification", "record_id": record_id})
            copied.append(entry)
        if iteration_manifest is not None:
            destination = out / "source_evidence" / f"comsol_iteration_{record_id}.json"
            entry = _copy(iteration_manifest, destination)
            entry.update({"role": "comsol_iteration_manifest", "record_id": record_id})
            copied.append(entry)
        metadata = record.get("metadata", {}) if isinstance(record.get("metadata"), Mapping) else {}
        comsol_rows.append(
            {
                "record_id": record_id,
                "selected_targets": metadata.get("selected_targets", ""),
                "accepted_targets": metadata.get("accepted_targets", ""),
                "ri_points": metadata.get("ri_points", ""),
                "seed": metadata.get("seed", ""),
            }
        )
    _write_csv(
        out / "TABLE_COMSOL_EVIDENCE.csv",
        ("record_id", "selected_targets", "accepted_targets", "ri_points", "seed"),
        comsol_rows,
    )

    device_rows: list[dict[str, Any]] = []
    for record in records["device_benchmark"]:
        record_id = str(record.get("record_id") or "unknown")
        benchmark = _artifact_for_role(registry, record, "benchmark_json")
        if benchmark is None:
            continue
        values = _read_json(benchmark)
        metadata = record.get("metadata", {}) if isinstance(record.get("metadata"), Mapping) else {}
        device_rows.append(
            {
                "record_id": record_id,
                "device_name": metadata.get("device_name", ""),
                "os_name": metadata.get("os_name", ""),
                "runtime": metadata.get("runtime", ""),
                "accelerator": metadata.get("accelerator", ""),
                "iterations": values.get("iterations", ""),
                "latency_ms_p50": values.get("latency_ms_p50", ""),
                "latency_ms_p95": values.get("latency_ms_p95", ""),
                "latency_ms_p99": values.get("latency_ms_p99", ""),
                "throughput_fps": values.get("throughput_fps", ""),
                "peak_memory_bytes": values.get("peak_memory_bytes", values.get("memory_peak_bytes", "")),
            }
        )
        destination = out / "source_evidence" / f"device_benchmark_{record_id}.json"
        entry = _copy(benchmark, destination)
        entry.update({"role": "device_benchmark", "record_id": record_id})
        copied.append(entry)
    _write_csv(
        out / "TABLE_DEVICE_BENCHMARK.csv",
        (
            "record_id",
            "device_name",
            "os_name",
            "runtime",
            "accelerator",
            "iterations",
            "latency_ms_p50",
            "latency_ms_p95",
            "latency_ms_p99",
            "throughput_fps",
            "peak_memory_bytes",
        ),
        device_rows,
    )

    experiment_summary: dict[str, Any] | None = None
    analysis_bound = False
    analysis_dir = Path(experimental_analysis_dir) if experimental_analysis_dir is not None else None
    if analysis_dir is not None and (analysis_dir / "experimental_summary.json").is_file():
        candidate = _read_json(analysis_dir / "experimental_summary.json")
        raw_hashes = {
            str(row.get("raw_sha256") or "")
            for row in candidate.get("source_spectra", [])
            if isinstance(row, Mapping) and row.get("raw_sha256")
        }
        qualified_hashes = _experimental_raw_hashes(records["experimental_sensor"])
        analysis_bound = bool(raw_hashes) and raw_hashes.issubset(qualified_hashes)
        if analysis_bound:
            experiment_summary = candidate
            for name in (
                "experimental_summary.json",
                "replicate_metrics.csv",
                "ri_summary.csv",
                "resonance_vs_ri.png",
                "repeatability_by_ri.png",
            ):
                source = analysis_dir / name
                if source.is_file():
                    folder = "figures" if source.suffix.lower() == ".png" else "tables"
                    destination = out / folder / name
                    entry = _copy(source, destination)
                    entry.update({"role": "experimental_analysis"})
                    copied.append(entry)

    experiment_rows: list[dict[str, Any]] = []
    if experiment_summary is not None:
        for metric, unit in (
            ("sensitivity_nm_per_riu", "nm/RIU"),
            ("fit_r2", "dimensionless"),
            ("mean_fwhm_nm", "nm"),
            ("fom_per_riu", "1/RIU"),
            ("repeatability_mean_sd_nm", "nm"),
            ("repeatability_max_sd_nm", "nm"),
        ):
            experiment_rows.append({"metric": metric, "value": experiment_summary.get(metric), "unit": unit})
    _write_csv(out / "TABLE_EXPERIMENTAL_RESULTS.csv", ("metric", "value", "unit"), experiment_rows)

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "evidence_registry": str(registry),
        "evidence_registry_sha256": sha256_file(registry),
        "qualified_evidence_classes": validation.get("evidence_classes", []),
        "comsol": {
            "available": bool(records["comsol_physics"]),
            "record_count": len(records["comsol_physics"]),
            "table_rows": len(comsol_rows),
        },
        "experiment": {
            "available": bool(records["experimental_sensor"]),
            "record_count": len(records["experimental_sensor"]),
            "analysis_supplied": analysis_dir is not None,
            "analysis_bound_to_qualified_raw": analysis_bound,
            "metrics": experiment_summary,
        },
        "device": {
            "available": bool(records["device_benchmark"]),
            "record_count": len(records["device_benchmark"]),
            "table_rows": len(device_rows),
        },
        "copied_artifacts": copied,
        "ready_for_manuscript_results": (
            bool(records["comsol_physics"])
            and bool(records["experimental_sensor"])
            and analysis_bound
            and bool(records["device_benchmark"])
        ),
        "scientific_boundary": (
            "Only qualified physical evidence is summarized. Experimental derived values require raw-hash linkage to the "
            "qualified experimental record; synthetic, surrogate, replay, or workstation results cannot satisfy these gates."
        ),
    }
    (out / "PAPER_RESULTS_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_summary(out / "RESULTS_EVIDENCE_SUMMARY.md", manifest)
    _package_checksums(out)
    return manifest


def validate_paper_results_package(output_dir: str | Path) -> dict[str, Any]:
    root = Path(output_dir)
    required = (
        "PAPER_RESULTS_MANIFEST.json",
        "RESULTS_EVIDENCE_SUMMARY.md",
        "TABLE_COMSOL_EVIDENCE.csv",
        "TABLE_EXPERIMENTAL_RESULTS.csv",
        "TABLE_DEVICE_BENCHMARK.csv",
        "checksums.sha256",
    )
    errors = [f"Missing paper-results file: {name}" for name in required if not (root / name).is_file()]
    checksum = root / "checksums.sha256"
    if checksum.is_file():
        for line_number, line in enumerate(checksum.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                expected, relative = line.split("  ", 1)
            except ValueError:
                errors.append(f"Malformed checksum line {line_number}.")
                continue
            candidate = root / relative
            if not candidate.is_file():
                errors.append(f"Checksum target missing: {relative}")
            elif sha256_file(candidate) != expected:
                errors.append(f"Checksum mismatch: {relative}")
    manifest = _read_json(root / "PAPER_RESULTS_MANIFEST.json") if (root / "PAPER_RESULTS_MANIFEST.json").is_file() else {}
    return {
        "ok": not errors,
        "ready_for_manuscript_results": bool(manifest.get("ready_for_manuscript_results")),
        "errors": errors,
    }
