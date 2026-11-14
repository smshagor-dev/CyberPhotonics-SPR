"""Stage-6 claim-level results staging without promotion of unreviewed optical data.

This is a publication WORKING package, not a scientific-authenticity certificate.
A COMSOL claim remains blocked even when input hashes and numerical metrics agree.
"""
from __future__ import annotations

import csv
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from sprpcf.evidence.qualification import validate_evidence_registry
from sprpcf.utils.reproducibility import git_state, sha256_file
from sprpcf.validation.independent_comsol import audit_independent_comsol_comparison

_STAGE4_FILES = (
    "PHYSICS_VERIFICATION_REQUESTS.csv",
    "selected_designs.csv",
    "validation_targets.csv",
    "split_membership.csv",
)
_STAGE5_REPORT = "PHYSICS_VERIFICATION_AUDIT.json"
_STAGE5_RESULTS = (
    "request_outcomes.csv",
    "method_completion_summary.csv",
    "paired_comsol_error_differences.csv",
)
_METRICS = (
    ("sensitivity_nm_per_riu", "nm/RIU"),
    ("fom_per_riu", "1/RIU"),
    ("lambda_res_nm", "nm"),
)
_LEDGER_COLUMNS = (
    "claim_id", "claim", "status", "evidence_class",
    "supporting_file", "supporting_sha256", "restriction",
)
_NUMERICAL_COLUMNS = (
    "request_id", "method", "design_seed", "target_id", "target_group_id",
    "metric", "candidate_value", "unit", "source_table_sha256",
    "qualification_status",
)


def _object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError(f"Cannot read JSON evidence: {path.name}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON evidence must be an object: {path.name}")
    return value


def _integrity(path: Path, expected: str | None, label: str) -> str:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Missing non-symlink evidence file: {label}")
    actual = sha256_file(path)
    if not isinstance(expected, str) or expected != actual:
        raise ValueError(f"Evidence SHA-256 mismatch: {label}")
    return actual


def _stage4(root: Path) -> tuple[dict[str, Any], dict[str, str], pd.DataFrame]:
    manifest_path = root / "INVERSE_BENCHMARK_MANIFEST.json"
    manifest = _object(manifest_path)
    if (manifest.get("experiment") != "matched_surrogate_query_inverse_design"
            or manifest.get("scientific_submission_ready") is not False):
        raise ValueError("Stage 4 requires the source-bound, non-publication-ready benchmark manifest")
    stored = manifest.get("outputs_sha256")
    if not isinstance(stored, dict):
        raise ValueError("Stage 4 output checksums are missing")
    for name in _STAGE4_FILES:
        _integrity(root / name, stored.get(name), "Stage 4 " + name)
    requests = pd.read_csv(root / _STAGE4_FILES[0], dtype={"request_id": str})
    if (requests.empty or requests["request_id"].isna().any()
            or requests["request_id"].duplicated().any()):
        raise ValueError("Stage 4 requests must be nonempty and uniquely identified")
    evidence_class = str(manifest.get("evidence_class") or "")
    if evidence_class not in {"software_only", "declared_physical_unqualified"}:
        raise ValueError("Unrecognized Stage 4 scientific evidence class")
    return manifest, {
        "stage4_manifest": sha256_file(manifest_path),
        "stage4_requests": sha256_file(root / _STAGE4_FILES[0]),
    }, requests


def _snapshot(
    stage5: Path | None, checks: dict[str, str]
) -> tuple[dict[str, Any] | None, list[str]]:
    if stage5 is None or not (stage5 / _STAGE5_REPORT).is_file():
        return None, ["No Stage 5 archived COMSOL audit was provided"]
    errors: list[str] = []
    try:
        report = _object(stage5 / _STAGE5_REPORT)
        if report.get("audit_type") != "independent_comsol_request_to_spectrum_comparison":
            raise ValueError("Stage 5 audit type mismatch")
        sources = report.get("source_checksums") or {}
        if sources.get("benchmark_manifest_sha256") != checks["stage4_manifest"]:
            raise ValueError("Stage 5 benchmark manifest binding mismatch")
        if sources.get("requests_sha256") != checks["stage4_requests"]:
            raise ValueError("Stage 5 request queue binding mismatch")
        expected = report.get("output_hashes") or {}
        if not isinstance(expected, dict):
            raise ValueError("Stage 5 output hashes malformed")
        for name, digest in expected.items():
            if name not in _STAGE5_RESULTS:
                raise ValueError("Unexpected Stage 5 result artifact")
            _integrity(stage5 / name, digest, "Stage 5 " + name)
        if report.get("numerical_archive_consistent") is True:
            for name in _STAGE5_RESULTS:
                if name not in expected:
                    raise ValueError("Complete Stage 5 audit lacks result table " + name)
        if report.get("scientific_submission_ready") is not False:
            raise ValueError("Stage 5 cannot autonomously certify scientific submission readiness")
        return report, errors
    except (ValueError, KeyError, TypeError) as exc:
        errors.append(str(exc))
        return None, errors


def _claim(
    claim_id: str, title: str, status: str, evidence_class: str,
    supporting_file: str = "", digest: str = "", restriction: str = "",
) -> dict[str, str]:
    return {
        "claim_id": claim_id, "claim": title, "status": status,
        "evidence_class": evidence_class, "supporting_file": supporting_file,
        "supporting_sha256": digest, "restriction": restriction,
    }


def _csv(path: Path, fields: tuple[str, ...], values: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(values)


def _completion_figure(path: Path, summary: list[dict[str, Any]]) -> None:
    """Completion counts only, never a graph of 'validated sensor performance'."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [str(x["method"]) for x in summary]
    total = [int(x["requests_total"]) for x in summary]
    complete = [int(x["complete_sweep_requests"]) for x in summary]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    index = list(range(len(names)))
    ax.bar(index, complete, label="Consistent archived sweeps (unqualified)")
    ax.bar(index, [a - b for a, b in zip(total, complete)], bottom=complete,
           label="Failed or incomplete")
    ax.set_xticks(index, names, rotation=15, ha="right")
    ax.set_ylabel("Number of frozen design requests")
    ax.set_title("Archived solver completion — source authenticity NOT verified")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def build_claim_traceable_package(
    benchmark_dir: str | Path,
    output_dir: str | Path,
    *,
    stage5_dir: str | Path | None = None,
    solver_archive: str | Path | None = None,
    refinement_archive: str | Path | None = None,
    evidence_registry: str | Path | None = None,
    repo_root: str | Path = ".",
    bootstrap_resamples: int = 500,
) -> dict[str, Any]:
    """Produce a blocker-first reviewer draft with fully traceable candidate values.

    To show even *candidate* solver numbers, independently re-run the Stage 5
    audit against its raw archives and compare results to the frozen Stage 5
    report. This is NOT a human physics/authenticity sign-off.
    """
    benchmark = Path(benchmark_dir).resolve()
    out = Path(output_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Publication draft output must be empty: {out}")
    if bootstrap_resamples < 200:
        raise ValueError("At least 200 geometry-level bootstrap resamples required")
    prior, checks, requests = _stage4(benchmark)
    problems: list[str] = []
    stage5 = Path(stage5_dir).resolve() if stage5_dir is not None else None
    recorded, issues = _snapshot(stage5, checks)
    problems.extend(issues)
    registry_status: dict[str, Any] = {"present": evidence_registry is not None, "hash_valid": False}
    if evidence_registry is not None:
        registry_path = Path(evidence_registry)
        validation = validate_evidence_registry(registry_path, verify_files=True)
        registry_status = {
            "present": True, "hash_valid": bool(validation.get("ok")),
            "classes": validation.get("evidence_classes", []),
            "record_count": validation.get("records", 0),
        }
        if not validation.get("ok"):
            problems.append("Evidence registry integrity invalid: " + "; ".join(validation.get("errors", [])))
    else:
        problems.append("No hash-validated physical evidence registry provided")
    if solver_archive is None or refinement_archive is None:
        problems.append("Raw COMSOL solver/refinement archives not both supplied for independent re-audit")

    consistent_data: dict[str, Any] | None = None
    candidate_results: pd.DataFrame | None = None
    if (recorded is not None and solver_archive is not None and refinement_archive is not None
            and registry_status["hash_valid"]):
        try:
            with tempfile.TemporaryDirectory(prefix="sprpcf_stage6_") as fresh_directory:
                fresh = audit_independent_comsol_comparison(
                    benchmark, solver_archive, refinement_archive, fresh_directory,
                    bootstrap_resamples=bootstrap_resamples,
                )
                if (not fresh.get("numerical_archive_consistent")
                        or not fresh.get("all_requested_sweeps_successful")
                        or fresh.get("errors")):
                    problems.append("Re-audited numerical archive incomplete/inconsistent: "
                                    + "; ".join(str(e) for e in fresh.get("errors", [])))
                elif (not recorded.get("numerical_archive_consistent")
                      or not recorded.get("all_requested_sweeps_successful")
                      or recorded.get("errors")
                      or fresh.get("source_checksums") != recorded.get("source_checksums")
                      or fresh.get("output_hashes") != recorded.get("output_hashes")
                      or fresh.get("counts") != recorded.get("counts")):
                    problems.append("Frozen Stage 5 report differs from independently recomputed result hashes/counts")
                elif int(fresh["counts"].get("total_requests", 0)) != len(requests):
                    problems.append("Stage 5 total design count differs from Stage 4 request queue")
                else:
                    consistent_data = fresh
                    candidate_results = pd.read_csv(Path(fresh_directory) / "request_outcomes.csv",
                                                    dtype={"request_id": str, "target_group_id": str})
                    if (candidate_results["request_id"].duplicated().any()
                            or set(candidate_results["request_id"]) != set(requests["request_id"])
                            or not candidate_results["status"].eq("spectrum_consistent_unqualified").all()):
                        problems.append("Re-audited numerical requests are missing, duplicated or unqualified")
                        consistent_data = None
                        candidate_results = None
        except (OSError, UnicodeError, ValueError, KeyError, TypeError) as exc:
            problems.append(f"Independent numerical re-audit failed: {exc}")

    out.mkdir(parents=True, exist_ok=True)
    for label, supplied in (
        ("stage4_manifest", benchmark / "INVERSE_BENCHMARK_MANIFEST.json"),
        ("stage4_requests", benchmark / "PHYSICS_VERIFICATION_REQUESTS.csv"),
    ):
        checks[label] = sha256_file(supplied)

    source_state = str(prior["evidence_class"])
    software_status = "documented_software_only" if source_state == "software_only" else "declared_source_unqualified"
    claims = [
        _claim("C01", "Matched-query inverse-design software comparison is recorded",
               software_status, source_state, "stage4_manifest",
               checks["stage4_manifest"], "Not an independent optical comparison"),
        _claim("C02", "Archived COMSOL spectra and computed target errors are numerically consistent",
               "review_candidate_unqualified" if consistent_data else "blocked",
               "comsol_physics" if consistent_data else "not_established",
               "stage5_audit" if consistent_data else "",
               sha256_file(stage5 / _STAGE5_REPORT) if consistent_data and stage5 else "",
               "Even a consistent audit cannot verify genuine solver execution"),
        _claim("C03", "The COMSOL model and numerical physics were independently reviewed",
               "blocked", "not_established", restriction="Requires independent qualified physics reviewer"),
        _claim("C04", "Tandem outperforms inverse baselines on a new untouched physical test",
               "blocked", "not_established",
               restriction="Needs independent test cohorts, real solver budget and multiple trained seeds"),
        _claim("C05", "Fabricated PCF-SPR sensor performance is experimentally measured",
               "not_claimed", "not_established",
               restriction="Do not include this claim without calibrated real-sensor raw measurements"),
        _claim("C06", "Numerical manuscript is cleared for journal submission",
               "blocked", "not_established",
               restriction="Independent physics review and paper-level statistical/claims audit pending"),
    ]
    candidates: list[dict[str, Any]] = []
    completion_rows: list[dict[str, Any]] = []
    figure_path: Path | None = None
    if consistent_data and candidate_results is not None:
        source_digest = consistent_data["output_hashes"]["request_outcomes.csv"]
        for _, record in candidate_results.sort_values(["request_id"]).iterrows():
            for metric, unit in _METRICS:
                candidates.append({
                    "request_id": str(record["request_id"]),
                    "method": str(record["method"]), "design_seed": int(record["design_seed"]),
                    "target_id": int(record["target_id"]),
                    "target_group_id": str(record["target_group_id"]),
                    "metric": metric, "candidate_value": float(record["actual_" + metric]),
                    "unit": unit, "source_table_sha256": source_digest,
                    "qualification_status": "numerically_consistent_source_authenticity_unverified",
                })
        completion_rows = [
            {"method": str(row["method"]), "requests_total": int(row["requests_total"]),
             "completed": int(row["complete_sweep_requests"]),
             "failed": int(row["solver_failed_requests"])}
            for row in consistent_data["method_summary"]
        ]
        figures = out / "figures"
        figures.mkdir(exist_ok=True)
        figure_path = figures / "solver_request_completion_unqualified.png"
        _completion_figure(figure_path, consistent_data["method_summary"])
    _csv(out / "CLAIMS_TO_EVIDENCE.csv", _LEDGER_COLUMNS, claims)
    _csv(out / "CANDIDATE_NUMERICAL_VALUES_FOR_REVIEW.csv", _NUMERICAL_COLUMNS, candidates)
    _csv(out / "SOLVER_COMPLETION_FOR_REVIEW.csv",
         ("method", "requests_total", "completed", "failed"), completion_rows)

    report_lines = [
        "# Paper results — reviewer-only evidence draft",
        "",
        "**NOT APPROVED FOR JOURNAL SUBMISSION.** This package has no independent",
        "scientific physics sign-off, even if recorded numeric consistency passes.",
        "",
        f"Stage 4 evidence class: **{source_state}**",
        f"Frozen requests: **{len(requests)}**",
        f"Numerical solver candidate values released for review: **{len(candidates)}**",
        "",
        "## Evidence blockers",
        "",
    ]
    report_lines.extend("- " + issue for issue in problems) if problems else report_lines.append(
        "- No machine-readable archive blockers, but independent scientific review is still required."
    )
    report_lines.extend([
        "",
        "## Claim-level matrix",
        "",
        "| Claim | Evidence status |",
        "|---|---|",
    ])
    for claim in claims:
        report_lines.append(f"| {claim['claim_id']}: {claim['claim']} | {claim['status']} |")
    report_lines.extend([
        "",
        "## Publication interpretation",
        "",
        "CANDIDATE_NUMERICAL_VALUES_FOR_REVIEW.csv is NOT a verified physics table.",
        "It is empty unless the authentic-source claim remains blocked but all",
        "archived numerical checks independently reproduce and both source snapshots match.",
        "Only an independent COMSOL model/scientific review can qualify those values.",
        "Do not quote software-only constructed sensitivities, invent results,",
        "or claim the tandem method is physically superior.",
        "Original solver logs/model and raw spectra are not copied into this draft;",
        "retain their immutable external archive alongside the reviewer submission.",
        "",
    ])
    (out / "MANUSCRIPT_RESULTS_DRAFT.md").write_text("\n".join(report_lines), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "package_type": "claim_traceable_review_draft",
        "scientific_submission_ready": False,
        "independent_physics_review_completed": False,
        "source_authenticity_verified": False,
        "numerical_candidates_available": bool(candidates),
        "candidate_number_count": len(candidates),
        "requests_total": len(requests),
        "source_evidence_class": source_state,
        "stage4": checks,
        "stage5_report_sha256": sha256_file(stage5 / _STAGE5_REPORT) if recorded is not None and stage5 else None,
        "evidence_registry": registry_status,
        "blockers": problems + [
            "No independent human review of electromagnetic model, materials, PML, modes, mesh or raw source authenticity",
            "No untouched external test, multi-training-seed statistical validation or matched end-to-end cost",
        ],
        "claim_status": {item["claim_id"]: item["status"] for item in claims},
        "git": git_state(Path(repo_root)),
        "files_sha256": {},
    }
    tracked = [out / "CLAIMS_TO_EVIDENCE.csv",
               out / "CANDIDATE_NUMERICAL_VALUES_FOR_REVIEW.csv",
               out / "SOLVER_COMPLETION_FOR_REVIEW.csv",
               out / "MANUSCRIPT_RESULTS_DRAFT.md"]
    if figure_path is not None:
        tracked.append(figure_path)
    manifest["files_sha256"] = {
        path.relative_to(out).as_posix(): sha256_file(path) for path in sorted(tracked)
    }
    manifest_path = out / "PUBLICATION_EVIDENCE_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
                             encoding="utf-8")
    all_files = sorted([*tracked, manifest_path])
    (out / "checksums.sha256").write_text(
        "\n".join(f"{sha256_file(path)}  {path.relative_to(out).as_posix()}" for path in all_files) + "\n",
        encoding="utf-8",
    )
    return manifest


def validate_claim_traceable_package(package_dir: str | Path) -> dict[str, Any]:
    """Re-verify produced package hashes and the permanent scientific gate."""
    root = Path(package_dir).resolve()
    errors: list[str] = []
    try:
        manifest_path = root / "PUBLICATION_EVIDENCE_MANIFEST.json"
        manifest = _object(manifest_path)
        if (manifest.get("schema_version") != 1
                or manifest.get("package_type") != "claim_traceable_review_draft"
                or manifest.get("scientific_submission_ready") is not False
                or manifest.get("independent_physics_review_completed") is not False
                or manifest.get("source_authenticity_verified") is not False):
            errors.append("Package scientific/readiness flags have been modified")
        file_map = manifest.get("files_sha256", {})
        if not isinstance(file_map, dict):
            raise ValueError("Manifest file digests are invalid")
        for name, expected in file_map.items():
            path = root / name
            resolved = path.resolve()
            if resolved == root or root not in resolved.parents:
                errors.append("Package file path escapes archive")
                continue
            _integrity(path, expected, name)
        claims = pd.read_csv(root / "CLAIMS_TO_EVIDENCE.csv")
        if (claims.empty or
                claims.loc[claims["claim_id"].isin(["C03", "C04", "C06"]), "status"]
                .ne("blocked").any()):
            errors.append("Physical/submission blocked claims have been promoted")
        checksums_path = root / "checksums.sha256"
        raw_lines = checksums_path.read_text(encoding="utf-8").splitlines()
        expected_lines = [
            f"{sha256_file(root / name)}  {name}"
            for name in sorted([*file_map, "PUBLICATION_EVIDENCE_MANIFEST.json"])
        ]
        if raw_lines != expected_lines:
            errors.append("Package checksum ledger mismatch")
    except (OSError, UnicodeError, ValueError, KeyError, TypeError) as exc:
        errors.append(str(exc))
    return {"ok": not errors, "errors": errors}
