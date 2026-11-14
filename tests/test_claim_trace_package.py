from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from sprpcf.publication.claim_trace import (
    build_claim_traceable_package,
    validate_claim_traceable_package,
)
from sprpcf.utils.reproducibility import sha256_file


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _benchmark(tmp_path: Path) -> Path:
    """Minimal synthetic software benchmark metadata; no real COMSOL data."""
    root = tmp_path / "benchmark"
    root.mkdir()
    requests = pd.DataFrame([
        {"request_id": "fake-1", "design_seed": 7, "method": "tandem_latent",
         "target_id": 0, "target_group_id": "g1"},
        {"request_id": "fake-2", "design_seed": 7, "method": "uniform_random",
         "target_id": 0, "target_group_id": "g1"},
    ])
    requests.to_csv(root / "PHYSICS_VERIFICATION_REQUESTS.csv", index=False)
    for name in ("selected_designs.csv", "validation_targets.csv", "split_membership.csv"):
        _write(root / name, "row_id,source\n0,synthetic\n")
    manifest = {
        "experiment": "matched_surrogate_query_inverse_design",
        "scientific_submission_ready": False,
        "evidence_class": "software_only",
        "outputs_sha256": {
            name: sha256_file(root / name)
            for name in (
                "PHYSICS_VERIFICATION_REQUESTS.csv",
                "selected_designs.csv", "validation_targets.csv",
                "split_membership.csv",
            )
        },
    }
    _write(root / "INVERSE_BENCHMARK_MANIFEST.json",
           json.dumps(manifest, indent=2))
    return root


def _stage5_snapshot(tmp_path: Path, benchmark: Path, *, complete: bool = False) -> Path:
    stage5 = tmp_path / "stage5"
    stage5.mkdir()
    data = {
        "audit_type": "independent_comsol_request_to_spectrum_comparison",
        "scientific_submission_ready": False,
        "numerical_archive_consistent": complete,
        "all_requested_sweeps_successful": complete,
        "errors": [] if complete else ["Real solver archives unavailable"],
        "source_checksums": {
            "benchmark_manifest_sha256": sha256_file(
                benchmark / "INVERSE_BENCHMARK_MANIFEST.json"),
            "requests_sha256": sha256_file(
                benchmark / "PHYSICS_VERIFICATION_REQUESTS.csv"),
        },
        "counts": {
            "total_requests": 2, "completed_requests": 2 if complete else 0,
            "solver_failed_requests": 0,
        },
        "output_hashes": {},
    }
    if complete:
        rows = pd.DataFrame([
            {"request_id": request_id, "status": "spectrum_consistent_unqualified",
             "method": method, "design_seed": 7, "target_id": 0,
             "target_group_id": "g1", "actual_sensitivity_nm_per_riu": 100.0,
             "actual_fom_per_riu": 5.0, "actual_lambda_res_nm": 600.0}
            for request_id, method in (
                ("fake-1", "tandem_latent"), ("fake-2", "uniform_random")
            )
        ])
        rows.to_csv(stage5 / "request_outcomes.csv", index=False)
        for name in ("method_completion_summary.csv", "paired_comsol_error_differences.csv"):
            _write(stage5 / name, "method,requests_total,complete_sweep_requests\n"
                   "tandem_latent,1,1\n")
        data["output_hashes"] = {
            name: sha256_file(stage5 / name)
            for name in ("request_outcomes.csv", "method_completion_summary.csv",
                         "paired_comsol_error_differences.csv")
        }
    _write(stage5 / "PHYSICS_VERIFICATION_AUDIT.json", json.dumps(data))
    return stage5


def test_blocker_only_draft_contains_no_unsupported_optical_numbers(tmp_path: Path) -> None:
    benchmark = _benchmark(tmp_path)
    report = build_claim_traceable_package(
        benchmark, tmp_path / "publication",
        repo_root=Path(__file__).resolve().parents[1],
    )
    out = tmp_path / "publication"
    assert report["scientific_submission_ready"] is False
    assert report["numerical_candidates_available"] is False
    assert report["requests_total"] == 2
    assert report["candidate_number_count"] == 0
    candidates = pd.read_csv(out / "CANDIDATE_NUMERICAL_VALUES_FOR_REVIEW.csv")
    assert candidates.empty
    claims = pd.read_csv(out / "CLAIMS_TO_EVIDENCE.csv")
    for name in ("C03", "C04", "C06"):
        assert claims.loc[claims["claim_id"] == name, "status"].iloc[0] == "blocked"
    assert not (out / "figures").exists()
    assert validate_claim_traceable_package(out)["ok"]


def test_fake_successful_stage5_json_without_source_is_withheld(tmp_path: Path) -> None:
    benchmark = _benchmark(tmp_path)
    stage5 = _stage5_snapshot(tmp_path, benchmark, complete=True)
    report = build_claim_traceable_package(
        benchmark, tmp_path / "publication", stage5_dir=stage5,
    )
    assert report["numerical_candidates_available"] is False
    assert report["scientific_submission_ready"] is False
    assert any("archives not both supplied" in value for value in report["blockers"])
    assert pd.read_csv(
        tmp_path / "publication" / "CANDIDATE_NUMERICAL_VALUES_FOR_REVIEW.csv"
    ).empty


def test_modified_frozen_benchmark_is_rejected(tmp_path: Path) -> None:
    benchmark = _benchmark(tmp_path)
    _write(benchmark / "PHYSICS_VERIFICATION_REQUESTS.csv",
           (benchmark / "PHYSICS_VERIFICATION_REQUESTS.csv").read_text() + "\n")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        build_claim_traceable_package(benchmark, tmp_path / "publication")
    assert not (tmp_path / "publication").exists()


def test_output_tampering_fails_checksum_validator(tmp_path: Path) -> None:
    benchmark = _benchmark(tmp_path)
    out = tmp_path / "publication"
    build_claim_traceable_package(benchmark, out)
    assert validate_claim_traceable_package(out)["ok"]
    with (out / "CLAIMS_TO_EVIDENCE.csv").open("a", encoding="utf-8") as writer:
        writer.write("C07,Fabricated result,approved,comsol_physics,,,should fail\n")
    assert not validate_claim_traceable_package(out)["ok"]


def test_recomputed_fixture_candidates_are_for_review_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Injected fixture ONLY tests the guard; it is not scientific evidence."""
    import sprpcf.publication.claim_trace as module

    benchmark = _benchmark(tmp_path)
    stage5 = _stage5_snapshot(tmp_path, benchmark, complete=True)
    snapshot = json.loads((stage5 / "PHYSICS_VERIFICATION_AUDIT.json").read_text())
    fake_registry = _write(tmp_path / "registry.json",
                           '{"schema_version":1,"records":[]}')
    monkeypatch.setattr(module, "validate_evidence_registry",
                        lambda *args, **kwargs: {
                            "ok": True, "evidence_classes": ["comsol_physics"],
                            "records": 1, "errors": [],
                        })

    def fake_reaudit(benchmark_dir, archive_dir, refinement_dir, output_dir, **kwargs):
        for name in (
            "request_outcomes.csv", "method_completion_summary.csv",
            "paired_comsol_error_differences.csv",
        ):
            source = stage5 / name
            _write(Path(output_dir) / name, source.read_text())
        return {
            "numerical_archive_consistent": True,
            "all_requested_sweeps_successful": True,
            "errors": [],
            "source_checksums": snapshot["source_checksums"],
            "output_hashes": snapshot["output_hashes"],
            "counts": snapshot["counts"],
            "method_summary": [
                {"method": "tandem_latent", "requests_total": 1,
                 "complete_sweep_requests": 1, "solver_failed_requests": 0},
                {"method": "uniform_random", "requests_total": 1,
                 "complete_sweep_requests": 1, "solver_failed_requests": 0},
            ],
        }

    monkeypatch.setattr(module, "audit_independent_comsol_comparison", fake_reaudit)
    report = build_claim_traceable_package(
        benchmark, tmp_path / "publication", stage5_dir=stage5,
        solver_archive=tmp_path, refinement_archive=tmp_path,
        evidence_registry=fake_registry,
    )
    assert report["numerical_candidates_available"] is True
    assert report["candidate_number_count"] == 6  # 2 requests x 3 metrics
    assert report["scientific_submission_ready"] is False
    assert report["source_authenticity_verified"] is False
    claims = pd.read_csv(tmp_path / "publication" / "CLAIMS_TO_EVIDENCE.csv")
    assert claims.loc[claims["claim_id"] == "C02", "status"].iloc[0] == "review_candidate_unqualified"
    assert claims.loc[claims["claim_id"] == "C03", "status"].iloc[0] == "blocked"
    assert (tmp_path / "publication" / "figures" /
            "solver_request_completion_unqualified.png").exists()
    assert validate_claim_traceable_package(tmp_path / "publication")["ok"]


def test_cli_strict_mode_writes_blocker_artifacts_then_returns_two(tmp_path: Path) -> None:
    benchmark = _benchmark(tmp_path)
    script = Path(__file__).resolve().parents[1] / "scripts" / "build_claim_evidence_package.py"
    out = tmp_path / "publication"
    p = subprocess.run(
        [sys.executable, str(script),
         "--benchmark", str(benchmark), "--out", str(out),
         "--strict-numerical-candidates"],
        capture_output=True, text=True, check=False,
    )
    assert p.returncode == 2, p.stderr
    assert (out / "PUBLICATION_EVIDENCE_MANIFEST.json").is_file()
    assert validate_claim_traceable_package(out)["ok"]
