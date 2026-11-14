from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from sprpcf.evidence.inventory import build_evidence_inventory, evidence_inventory_markdown


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _with_sidecar(dataset: Path, source: str, *, matching: bool = True) -> None:
    digest = hashlib.sha256(dataset.read_bytes()).hexdigest() if matching else "0" * 64
    _write(dataset.with_name(dataset.name + ".meta.json"),
           json.dumps({"schema_version": 1, "sha256": digest, "source": source}))


def test_inventory_empty_tree_fails_closed_without_fabricating_evidence(tmp_path: Path) -> None:
    report = build_evidence_inventory(tmp_path)
    assert report["counts"]["artifacts"] == 0
    assert report["scientific_submission_ready"] is False
    assert all(row["status"] == "blocked" for row in report["requirements"])
    assert report["registry"]["integrity_valid"] is False


def test_synthetic_data_and_mock_benchmark_are_software_only(tmp_path: Path) -> None:
    dataset = _write(tmp_path / "data" / "processed" / "synthetic.csv", "sample_id\n1\n")
    _with_sidecar(dataset, "synthetic")
    _write(tmp_path / "reports" / "phase4_hil_benchmark.json",
           json.dumps({"protocol": "mock", "latency_ms_p95": 0.1}))
    report = build_evidence_inventory(tmp_path)
    by_type = {item["category"]: item for item in report["artifacts"]}
    assert by_type["dataset"]["evidence_status"] == "software_only"
    assert by_type["device_benchmark_candidate"]["evidence_status"] == "software_only"
    assert report["requirements"][0]["status"] == "present"
    assert report["requirements"][2]["status"] == "blocked"
    assert report["scientific_submission_ready"] is False
    assert "mock" not in report["registry"]["evidence_classes_with_valid_hashes"]


def test_dataset_hash_mismatch_downgrades_provenance(tmp_path: Path) -> None:
    dataset = _write(tmp_path / "data" / "processed" / "sweep.csv", "RI,loss\n1.33,1\n")
    _with_sidecar(dataset, "comsol")
    dataset.write_text("RI,loss\n1.33,9\n", encoding="utf-8")
    report = build_evidence_inventory(tmp_path)
    record = report["artifacts"][0]
    assert record["evidence_status"] == "invalid_provenance"
    assert any("SHA-256" in item for item in record["warnings"])
    assert report["requirements"][0]["status"] == "blocked"


def test_claimed_comsol_source_is_not_independent_physics(tmp_path: Path) -> None:
    dataset = _write(tmp_path / "data" / "processed" / "solver.csv", "sample_id\n1\n")
    _with_sidecar(dataset, "comsol")
    _write(tmp_path / "physics" / "design.mph", "dummy bytes")
    _write(tmp_path / "outputs" / "closed_loop" / "iteration_manifest.json",
           json.dumps({"backend": "comsol", "evidence_class": "comsol_physics"}))
    report = build_evidence_inventory(tmp_path)
    assert any(item["evidence_status"] == "declared_physical_unqualified" for item in report["artifacts"])
    assert report["requirements"][2]["status"] == "blocked"
    assert report["scientific_submission_ready"] is False


def test_missing_dataset_sidecar_and_external_root_inventory(tmp_path: Path) -> None:
    _write(tmp_path / "data" / "dataset.parquet", "fixture")
    external = tmp_path.parent / (tmp_path.name + "_archive")
    _write(external / "raw" / "session.jsonl", '{"signal": 1}\n')
    report = build_evidence_inventory(tmp_path, external_roots=[external])
    assert report["external_roots_supplied"] == 1
    assert report["counts"]["raw_measurement_candidates"] == 1
    assert report["requirements"][0]["status"] == "blocked"
    assert report["artifacts"][0]["path"].startswith(("data/", "external_1/"))
    assert str(external) not in json.dumps(report)


def test_corrupt_registry_does_not_grant_physical_class(tmp_path: Path) -> None:
    _write(tmp_path / "outputs" / "evidence" / "evidence_registry.json",
           json.dumps({"schema_version": 1, "records": [{"evidence_class": "comsol_physics"}]}))
    report = build_evidence_inventory(tmp_path)
    assert report["registry"]["integrity_valid"] is False
    assert report["registry"]["evidence_classes_with_valid_hashes"] == []
    assert report["requirements"][2]["status"] == "blocked"


def test_cli_writes_reports_and_strict_mode_blocks_unverified_paper(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    json_out = tmp_path / "inventory.json"
    md_out = tmp_path / "inventory.md"
    process = subprocess.run(
        [
            sys.executable, str(root / "scripts" / "evidence_inventory.py"),
            "--repo-root", str(tmp_path),
            "--json-out", str(json_out),
            "--markdown-out", str(md_out),
            "--fail-on-blockers",
        ],
        capture_output=True, text=True, check=False,
    )
    assert process.returncode == 2, process.stderr
    assert json.loads(json_out.read_text(encoding="utf-8"))["scientific_submission_ready"] is False
    assert "Requirement gaps" in md_out.read_text(encoding="utf-8")
    report = build_evidence_inventory(tmp_path)
    assert evidence_inventory_markdown(report).startswith("# Research Evidence Inventory\n")
