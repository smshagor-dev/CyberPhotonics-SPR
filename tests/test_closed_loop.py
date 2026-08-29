from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sprpcf.ml.dataset import GEOMETRY_COLUMNS
from sprpcf.validation.closed_loop import (
    AcceptanceThresholds,
    append_accepted_simulation_rows,
    build_validation_ri_values,
    evaluate_closed_loop_results,
    run_closed_loop_iteration,
)


def _linear_runner(geometries):
    rows = []
    for sample_id, geometry in enumerate(geometries):
        wavelength = 500.0 + 800.0 * (geometry.analyte_ri - 1.33)
        rows.append(
            {
                "sample_id": sample_id,
                "status": "ok",
                "pitch_um": geometry.pitch_um,
                "d_over_lambda": geometry.d_over_lambda,
                "metal_thickness_nm": geometry.metal_thickness_nm,
                "channel_radius_um": geometry.channel_radius_um,
                "analyte_ri": geometry.analyte_ri,
                "lambda_res_nm": wavelength,
                "fwhm_nm": 40.0,
                "sensitivity_nm_per_riu": 800.0,
                "fom_per_riu": 20.0,
            }
        )
    return pd.DataFrame(rows)


def _fixed_designer(targets: pd.DataFrame) -> pd.DataFrame:
    designed = targets.copy()
    designed["pitch_um"] = 2.0
    designed["d_over_lambda"] = 0.5
    designed["metal_thickness_nm"] = 45.0
    designed["channel_radius_um"] = 0.6
    designed["uncertainty"] = 0.02
    return designed


def test_validation_ri_sweep_is_centered_and_unique() -> None:
    values = build_validation_ri_values(1.33, span=0.04, points=5)
    assert len(values) == 5
    assert values[2] == 1.33
    assert len(set(values)) == 5

    try:
        build_validation_ri_values(1.33, points=4)
    except ValueError as exc:
        assert "odd integer" in str(exc)
    else:
        raise AssertionError("Even RI point count should be rejected.")


def test_closed_loop_acceptance_matches_linear_physics() -> None:
    ri_values = build_validation_ri_values(1.33, span=0.04, points=5)
    rows = []
    for sample_id, ri in enumerate(ri_values):
        rows.append(
            {
                "sample_id": sample_id,
                "target_id": 0,
                "source_target_id": 0,
                "target_sensitivity_nm_per_riu": 800.0,
                "target_fom_per_riu": 20.0,
                "target_lambda_res_nm": 500.0,
                "target_analyte_ri": 1.33,
                "uncertainty": 0.01,
                "status": "ok",
                "pitch_um": 2.0,
                "d_over_lambda": 0.5,
                "metal_thickness_nm": 45.0,
                "channel_radius_um": 0.6,
                "analyte_ri": ri,
                "lambda_res_nm": 500.0 + 800.0 * (ri - 1.33),
                "fwhm_nm": 40.0,
            }
        )
    verification = evaluate_closed_loop_results(
        pd.DataFrame(rows),
        AcceptanceThresholds(
            max_sensitivity_error_nm_per_riu=1e-6,
            max_fom_error_per_riu=1e-6,
            max_lambda_error_nm=1e-6,
            min_linearity_r2=0.999999,
        ),
        expected_ri_points=5,
    )
    assert len(verification) == 1
    assert bool(verification.iloc[0]["accepted"])
    assert np.isclose(verification.iloc[0]["actual_sensitivity_nm_per_riu"], 800.0)
    assert np.isclose(verification.iloc[0]["actual_fom_per_riu"], 20.0)
    assert np.isclose(verification.iloc[0]["actual_lambda_res_nm"], 500.0)


def test_failed_ri_point_rejects_target() -> None:
    ri_values = build_validation_ri_values(1.33, span=0.04, points=5)
    rows = []
    for sample_id, ri in enumerate(ri_values):
        status = "failed: solver" if sample_id == 0 else "ok"
        rows.append(
            {
                "sample_id": sample_id,
                "target_id": 0,
                "source_target_id": 0,
                "target_sensitivity_nm_per_riu": 800.0,
                "target_fom_per_riu": 20.0,
                "target_lambda_res_nm": 500.0,
                "target_analyte_ri": 1.33,
                "uncertainty": 0.01,
                "status": status,
                "pitch_um": 2.0,
                "d_over_lambda": 0.5,
                "metal_thickness_nm": 45.0,
                "channel_radius_um": 0.6,
                "analyte_ri": ri,
                "lambda_res_nm": 500.0 + 800.0 * (ri - 1.33) if status == "ok" else np.nan,
                "fwhm_nm": 40.0 if status == "ok" else np.nan,
            }
        )
    verification = evaluate_closed_loop_results(
        pd.DataFrame(rows),
        AcceptanceThresholds(),
        expected_ri_points=5,
    )
    assert not bool(verification.iloc[0]["accepted"])
    assert "expected 5 successful RI points" in verification.iloc[0]["reason"]


def test_append_accepted_rows_deduplicates_geometry_and_ri() -> None:
    base = pd.DataFrame(
        [
            {
                "pitch_um": 2.0,
                "d_over_lambda": 0.5,
                "metal_thickness_nm": 45.0,
                "channel_radius_um": 0.6,
                "analyte_ri": 1.33,
                "sensitivity_nm_per_riu": 700.0,
                "fom_per_riu": 17.5,
                "lambda_res_nm": 490.0,
            }
        ]
    )
    simulation = pd.DataFrame(
        [
            {
                "target_id": 0,
                "status": "ok",
                "pitch_um": 2.0,
                "d_over_lambda": 0.5,
                "metal_thickness_nm": 45.0,
                "channel_radius_um": 0.6,
                "analyte_ri": 1.33,
                "sensitivity_nm_per_riu": 800.0,
                "fom_per_riu": 20.0,
                "lambda_res_nm": 500.0,
            },
            {
                "target_id": 0,
                "status": "ok",
                "pitch_um": 2.0,
                "d_over_lambda": 0.5,
                "metal_thickness_nm": 45.0,
                "channel_radius_um": 0.6,
                "analyte_ri": 1.35,
                "sensitivity_nm_per_riu": 800.0,
                "fom_per_riu": 20.0,
                "lambda_res_nm": 516.0,
            },
        ]
    )
    verification = pd.DataFrame([{"target_id": 0, "accepted": True}])
    augmented, appended = append_accepted_simulation_rows(base, simulation, verification)
    assert len(augmented) == 2
    assert appended == 1
    duplicate = augmented.loc[np.isclose(augmented["analyte_ri"], 1.33)].iloc[0]
    assert np.isclose(duplicate["lambda_res_nm"], 500.0)


def test_closed_loop_iteration_writes_auditable_artifacts(tmp_path: Path) -> None:
    targets_path = tmp_path / "targets.csv"
    base_path = tmp_path / "base.csv"
    output_dir = tmp_path / "closed_loop"

    pd.DataFrame(
        [
            {
                "sensitivity_nm_per_riu": 800.0,
                "fom_per_riu": 20.0,
                "lambda_res_nm": 500.0,
                "analyte_ri": 1.33,
            }
        ]
    ).to_csv(targets_path, index=False)

    pd.DataFrame(
        [
            {
                "pitch_um": 1.5,
                "d_over_lambda": 0.4,
                "metal_thickness_nm": 35.0,
                "channel_radius_um": 0.5,
                "analyte_ri": 1.30,
                "sensitivity_nm_per_riu": 600.0,
                "fom_per_riu": 15.0,
                "lambda_res_nm": 450.0,
            }
        ]
    ).to_csv(base_path, index=False)

    artifacts = run_closed_loop_iteration(
        checkpoint_path=tmp_path / "unused-checkpoint.pt",
        target_path=targets_path,
        base_dataset_path=base_path,
        output_dir=output_dir,
        backend="synthetic",
        ri_span=0.04,
        ri_points=5,
        thresholds=AcceptanceThresholds(
            max_sensitivity_error_nm_per_riu=1e-6,
            max_fom_error_per_riu=1e-6,
            max_lambda_error_nm=1e-6,
            min_linearity_r2=0.999999,
        ),
        designer=_fixed_designer,
        runner=_linear_runner,
    )

    assert artifacts.selected_targets == 1
    assert artifacts.accepted_targets == 1
    assert artifacts.appended_rows == 5
    assert artifacts.manifest.exists()
    assert artifacts.augmented_dataset.exists()
    assert artifacts.verification_results.exists()

    manifest = json.loads(artifacts.manifest.read_text(encoding="utf-8"))
    assert manifest["backend"] == "synthetic"
    assert manifest["evidence_class"] == "software_only"
    assert manifest["accepted_targets"] == 1
    assert manifest["outputs"]["augmented_dataset_sha256"]

    augmented = pd.read_csv(artifacts.augmented_dataset)
    assert len(augmented) == 6
    assert all(column in augmented.columns for column in GEOMETRY_COLUMNS)
