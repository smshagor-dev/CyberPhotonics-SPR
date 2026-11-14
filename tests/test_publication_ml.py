from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sprpcf.simulation.synthetic import build_synthetic_dataset
from sprpcf.validation.publication_ml import (
    _bootstrap_group_mean, run_grouped_ml_study, summarize_paired_seed_ablation,
)


def _dataset(tmp_path: Path, *, source: str = "synthetic", geometries: int = 12) -> Path:
    frame = build_synthetic_dataset(samples=geometries, wavelengths=48, seed=13)
    path = tmp_path / "input.csv"
    frame.to_csv(path, index=False)
    sidecar = path.with_name(path.name + ".meta.json")
    sidecar.write_text(json.dumps({
        "schema_version": 1, "source": source,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }), encoding="utf-8")
    return path


def _study(tmp_path: Path, *, source: str = "synthetic", seeds=(7, 17)) -> tuple[dict, Path]:
    path = _dataset(tmp_path, source=source)
    output = tmp_path / "result"
    report = run_grouped_ml_study(
        path, output, seeds=seeds, trees=8,
        bootstrap_resamples=200, allow_software_only=(source == "synthetic"),
    )
    return report, output


def test_same_geometry_never_leaks_and_all_methods_share_split(tmp_path: Path) -> None:
    report, output = _study(tmp_path)
    assignments = pd.read_csv(output / "splits.csv")
    evaluations = pd.read_csv(output / "per_seed_metrics.csv")
    assert report["observations"]["unique_geometry_groups"] == 12
    assert report["source"]["evidence_class"] == "software_only"
    assert report["publication_ready"] is False
    for seed, group in assignments.groupby("seed"):
        train = set(group.loc[group["partition"] == "train", "geometry_id"])
        test = set(group.loc[group["partition"] == "test", "geometry_id"])
        assert train.isdisjoint(test)
        assert len(test) == 3
        assert group["split_sha256"].nunique() == 1
        evaluation = evaluations[evaluations["seed"] == seed]
        assert len(evaluation) == 12  # 4 methods x 3 physical-unit targets
        assert evaluation["split_sha256"].nunique() == 1
    assert report["statistical_interpretation"]


def test_paired_uncertainty_uses_geometry_count_not_ri_rows(tmp_path: Path) -> None:
    _, output = _study(tmp_path)
    paired = pd.read_csv(output / "paired_group_bootstrap.csv")
    assert paired["geometry_groups"].eq(3).all()
    assert paired["interval_status"].eq("descriptive_group_bootstrap").all()
    assert np.isfinite(paired["ci_low"].to_numpy(dtype=float)).all()
    assert (paired["ci_low"] <= paired["ci_high"]).all()


def test_rejects_synthetic_without_explicit_opt_in(tmp_path: Path) -> None:
    path = _dataset(tmp_path)
    with pytest.raises(ValueError, match="allow-software-only"):
        run_grouped_ml_study(path, tmp_path / "out", seeds=(7, 17), trees=8,
                             bootstrap_resamples=200)
    assert not (tmp_path / "out").exists()


def test_rejects_tampered_dataset_hash_and_missing_provenance(tmp_path: Path) -> None:
    path = _dataset(tmp_path)
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        run_grouped_ml_study(path, tmp_path / "out", seeds=(7, 17), trees=8,
                             bootstrap_resamples=200, allow_software_only=True)
    path.with_name(path.name + ".meta.json").unlink()
    with pytest.raises(ValueError, match="sidecar"):
        run_grouped_ml_study(path, tmp_path / "out", seeds=(7, 17), trees=8,
                             bootstrap_resamples=200, allow_software_only=True)


def test_declared_comsol_source_remains_unqualified(tmp_path: Path) -> None:
    report, _ = _study(tmp_path, source="comsol")
    assert report["source"]["evidence_class"] == "declared_physical_unqualified"
    assert report["source"]["independently_verified_physics"] is False
    assert not report["physics_performance_claim_ready"]


def test_rejects_duplicate_ri_and_nonfinite_rows_even_with_matching_hash(tmp_path: Path) -> None:
    for mode in ("duplicate", "nonfinite"):
        path = _dataset(tmp_path)
        frame = pd.read_csv(path)
        if mode == "duplicate":
            frame.iloc[1, frame.columns.get_loc("analyte_ri")] = frame.iloc[0]["analyte_ri"]
        else:
            frame.iloc[1, frame.columns.get_loc("lambda_res_nm")] = np.inf
        frame.to_csv(path, index=False)
        sidecar = path.with_name(path.name + ".meta.json")
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        data["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        sidecar.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(ValueError, match="Duplicate|NaN/Inf"):
            run_grouped_ml_study(path, tmp_path / "out", seeds=(7, 17),
                                 trees=8, bootstrap_resamples=200, allow_software_only=True)


def test_group_bootstrap_is_deterministic_and_small_sample_is_flagged() -> None:
    data = np.array([-1.0, 1.0, 2.0])
    first = _bootstrap_group_mean(data, resamples=200, seed=7)
    assert first == _bootstrap_group_mean(data, resamples=200, seed=7)
    assert first["geometry_groups"] == 3
    short = _bootstrap_group_mean(np.array([0.0, 1.0]), resamples=200, seed=7)
    assert short["ci_low"] is None
    assert short["interval_status"] == "insufficient_independent_geometry_groups"


def test_ablation_seed_pairing_and_invalid_pairs(tmp_path: Path) -> None:
    records = []
    for seed in (3, 5, 7, 11, 13):
        for variant, adjustment in (("physics_informed", 0.1), ("no_physics_penalty", 0.0)):
            records.append({
                "seed": seed, "variant": variant,
                "forward_r2": 0.6 + adjustment,
                "inverse_target_r2": 0.5 + adjustment,
                "constraint_valid_rate": 0.7 + adjustment,
                "constraint_violation_rate": 0.3 - adjustment,
            })
    path = tmp_path / "ablation_runs.csv"
    pd.DataFrame(records).to_csv(path, index=False)
    summary = summarize_paired_seed_ablation(path, bootstrap_resamples=200)
    assert summary["paired_seeds"] == [3, 5, 7, 11, 13]
    assert summary["split_identity_verified"] is False
    assert np.isclose(summary["contrasts_physics_minus_no_penalty"]["constraint_valid_rate"]["mean"], 0.1)
    records.pop()
    pd.DataFrame(records).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Missing matched"):
        summarize_paired_seed_ablation(path, bootstrap_resamples=200)


def test_cli_creates_integrity_bound_reports_and_refuses_overwrite(tmp_path: Path) -> None:
    data = _dataset(tmp_path)
    out = tmp_path / "reports"
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_grouped_ml_study.py"
    command = [
        sys.executable, str(script), "--data", str(data), "--out", str(out),
        "--seeds", "7,17", "--trees", "8", "--bootstrap-resamples", "200",
        "--allow-software-only",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    report = json.loads((out / "STUDY_MANIFEST.json").read_text(encoding="utf-8"))
    for filename, digest in report["outputs_sha256"].items():
        assert hashlib.sha256((out / filename).read_bytes()).hexdigest() == digest
    assert (out / "STUDY_REPORT.md").is_file()
    second = subprocess.run(command, capture_output=True, text=True, check=False)
    assert second.returncode != 0
    assert "Output directory must be empty" in second.stderr
