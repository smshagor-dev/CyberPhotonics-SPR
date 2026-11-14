"""Conservative, reproducible inventory of research artifacts and evidence gaps.

An inventory records file existence and provenance; it never certifies optical physics.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from sprpcf.evidence.qualification import validate_evidence_registry
from sprpcf.utils.reproducibility import git_state, sha256_file

SEARCH_DIRECTORIES = ("data", "models", "outputs", "reports", "physics", "experiments", "evidence")
DATA_SUFFIXES = {".csv", ".parquet"}
MODEL_SUFFIXES = {".pt", ".pth", ".ckpt", ".onnx", ".tflite", ".safetensors"}
CATEGORIES = (
    "dataset", "checkpoint", "comsol_model", "comsol_iteration", "raw_spectrum",
    "raw_measurement_candidate", "device_benchmark_candidate", "research_result",
    "evidence_registry", "other",
)


def _json_object(path: Path) -> dict[str, Any] | None:
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _category(relative: str, path: Path) -> str:
    name = path.name.lower()
    parts = {p.lower() for p in Path(relative).parts}
    if name == "evidence_registry.json":
        return "evidence_registry"
    if name == "iteration_manifest.json":
        return "comsol_iteration"
    if path.suffix.lower() == ".mph":
        return "comsol_model"
    if path.suffix.lower() in MODEL_SUFFIXES:
        return "checkpoint"
    if "benchmark" in name and path.suffix.lower() == ".json":
        return "device_benchmark_candidate"
    if name.startswith("simulation_results") and parts & {"comsol", "closed_loop", "real_validation"}:
        return "raw_spectrum"
    if "raw" in parts and path.suffix.lower() in DATA_SUFFIXES | {".jsonl"}:
        return "raw_measurement_candidate"
    if "data" in parts and path.suffix.lower() in DATA_SUFFIXES:
        return "dataset"
    if "outputs" in parts and path.suffix.lower() in DATA_SUFFIXES | {".json", ".png"}:
        return "research_result"
    return "other"


def _record(path: Path, label: str) -> dict[str, Any]:
    category = _category(label, path)
    record: dict[str, Any] = {
        "path": label,
        "category": category,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "evidence_status": "unverified",
        "declared_source": None,
        "warnings": [],
    }
    if category == "dataset":
        sidecar = path.with_name(path.name + ".meta.json")
        if not sidecar.is_file():
            record["warnings"].append("Missing dataset provenance sidecar")
        else:
            meta = _json_object(sidecar)
            record["sidecar"] = label + ".meta.json"
            if meta is None:
                record["warnings"].append("Malformed dataset provenance sidecar")
            else:
                source = str(meta.get("source") or "").strip().lower()
                record["declared_source"] = source or None
                if str(meta.get("sha256") or "").lower() != record["sha256"]:
                    record["warnings"].append("Dataset SHA-256 does not match sidecar")
                    record["evidence_status"] = "invalid_provenance"
                elif source in {"synthetic", "software_only", "mock", "replay"}:
                    record["evidence_status"] = "software_only"
                elif source in {"comsol", "experimental", "measured_sensor"}:
                    record["evidence_status"] = "declared_physical_unqualified"
                    record["warnings"].append("Source label and hash do not independently validate physics")
                else:
                    record["warnings"].append("Missing or unrecognized dataset source label")
    elif category == "device_benchmark_candidate":
        info = _json_object(path)
        if info is None:
            record["warnings"].append("Benchmark JSON unavailable or malformed")
        elif str(info.get("protocol", "")).lower() in {"mock", "simulation", "replay", "synthetic"}:
            record["declared_source"] = info["protocol"].lower()
            record["evidence_status"] = "software_only"
        else:
            record["warnings"].append("Benchmark not qualified against exact device and model")
    elif category == "comsol_iteration":
        info = _json_object(path)
        if info is None:
            record["warnings"].append("Invalid COMSOL iteration manifest")
        else:
            record["declared_source"] = str(info.get("backend") or "") or None
            if info.get("backend") == "comsol" and info.get("evidence_class") == "comsol_physics":
                record["evidence_status"] = "declared_physical_unqualified"
                record["warnings"].append("Manifest claims require model/hash, archived-spectrum and scientific review")
            else:
                record["evidence_status"] = "software_only" if info.get("backend") == "synthetic" else "unverified"
    return record


def _iter_files(root: Path) -> Iterable[Path]:
    if not root.is_dir() or root.is_symlink():
        return
    for file in sorted(root.rglob("*")):
        if (file.is_file() and not file.is_symlink() and file.name != ".gitkeep"
                and not file.name.endswith(".meta.json")
                and not any(part.startswith(".") for part in file.relative_to(root).parts)):
            yield file


def build_evidence_inventory(
    repo_root: str | Path,
    *,
    external_roots: Iterable[str | Path] = (),
    evidence_registry: str | Path | None = None,
) -> dict[str, Any]:
    """Scan known repository locations and explicitly supplied external evidence roots.

    Findings are filesystem-bound, not proof of a qualified scientific experiment.
    """
    root = Path(repo_root).resolve()
    if not root.is_dir():
        raise ValueError(f"Repository root is not a directory: {root}")
    artifacts: list[dict[str, Any]] = []
    scan_scope: list[dict[str, Any]] = []
    locations = [(root / name, name) for name in SEARCH_DIRECTORIES]
    for index, item in enumerate(external_roots, 1):
        locations.append((Path(item).resolve(), f"external_{index}"))
    for location, label in locations:
        exists = location.is_dir() and not location.is_symlink()
        scan_scope.append({"location": label, "available": exists})
        if not exists:
            continue
        for file in _iter_files(location):
            path_label = f"{label}/{file.relative_to(location).as_posix()}"
            artifacts.append(_record(file, path_label))
    artifacts.sort(key=lambda row: row["path"])

    registry_path = Path(evidence_registry).resolve() if evidence_registry is not None else (
        root / "outputs" / "evidence" / "evidence_registry.json"
    )
    registry = validate_evidence_registry(registry_path)
    classes = set(registry.get("evidence_classes", [])) if registry.get("ok") else set()
    datasets = [a for a in artifacts if a["category"] == "dataset"]
    models = [a for a in artifacts if a["category"] == "checkpoint"]
    results = [a for a in artifacts if a["category"] == "research_result"]
    comsol = [a for a in artifacts if a["category"] in {"comsol_model", "comsol_iteration", "raw_spectrum"}]
    measurements = [a for a in artifacts if a["category"] == "raw_measurement_candidate"]
    benchmarks = [a for a in artifacts if a["category"] == "device_benchmark_candidate"]

    requirements = [
        {
            "id": "dataset_provenance",
            "status": "present" if datasets and all(
                a["evidence_status"] in {"software_only", "declared_physical_unqualified"}
                and a["declared_source"] for a in datasets
            ) else "blocked",
            "detail": "Every dataset requires a source label and matching SHA-256 sidecar; synthetic is software-only.",
            "next_action": "Locate original dataset sources and sidecars; verify source and split identities.",
        },
        {
            "id": "trained_model",
            "status": "present_unverified" if models else "blocked",
            "detail": "A checkpoint file does not establish trained-model accuracy or reproducibility.",
            "next_action": "Archive trained weights, training configuration, dataset hash, seeds and evaluation.",
        },
        {
            "id": "independent_comsol_physics",
            "status": "provenance_only" if "comsol_physics" in classes else "blocked",
            "detail": "A valid registry cannot prove FEM correctness, mesh convergence or actual solver execution.",
            "next_action": "Supply validated .mph model, meshes, raw RI sweeps, audit and independent physics review.",
        },
        {
            "id": "baseline_ablation_statistics",
            "status": "present_unverified" if results else "blocked",
            "detail": "Generated result files require evaluation of their splits, source, seeds and numerical claims.",
            "next_action": "Run baseline, ablation, uncertainty and OOD experiments on frozen group-disjoint data.",
        },
        {
            "id": "claim_traceability",
            "status": "blocked",
            "detail": "File inventory alone cannot attest manuscript numeric claims to qualified experiment records.",
            "next_action": "Build and inspect a claims-to-evidence matrix after independent results are archived.",
        },
    ]
    return {
        "schema_version": 1,
        "git": git_state(root),
        "scope": scan_scope,
        "external_roots_supplied": len(locations) - len(SEARCH_DIRECTORIES),
        "artifacts": artifacts,
        "counts": {
            "artifacts": len(artifacts), "datasets": len(datasets), "checkpoints": len(models),
            "comsol_candidates": len(comsol), "raw_measurement_candidates": len(measurements),
            "benchmark_candidates": len(benchmarks), "research_results": len(results),
        },
        "registry": {
            "provided_or_discovered": registry_path.is_file(),
            "integrity_valid": bool(registry.get("ok")),
            "record_count": registry.get("records", 0),
            "evidence_classes_with_valid_hashes": sorted(classes),
            "errors": registry.get("errors", []),
        },
        "requirements": requirements,
        "scientific_submission_ready": False,
        "limitations": [
            "Inventory searches selected local directories and explicitly supplied external roots; other storage is not inspected.",
            "Registry hashes validate stored provenance, not COMSOL physics, instrument calibration or independent authenticity.",
            "No artifact is upgraded to independently verified physics, sensor or target-device evidence by this inventory.",
            "Real experimental measurements and exact-device benchmarks are required only when the manuscript claims them.",
        ],
    }


def evidence_inventory_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Research Evidence Inventory",
        "",
        f"Git commit: `{report['git']['commit'] or 'unavailable'}`",
        f"Artifacts inventoried: **{report['counts']['artifacts']}**",
        "Scientific submission ready: **NO** (independent review is required)",
        "",
        "## Requirement gaps",
        "",
        "| Requirement | Inventory status | Needed next |",
        "|---|---|---|",
    ]
    for row in report["requirements"]:
        lines.append(f"| {row['id']} | {row['status']} | {row['next_action']} |")
    lines.extend(["", "## Artifacts", "", "| Path | Type | Evidence status | SHA-256 |",
                  "|---|---|---|---|"])
    for row in report["artifacts"]:
        lines.append(f"| `{row['path']}` | {row['category']} | {row['evidence_status']} | `{row['sha256']}` |")
    lines.extend(["", "## Interpretation", ""])
    lines.extend(f"- {note}" for note in report["limitations"])
    return "\n".join(lines) + "\n"
