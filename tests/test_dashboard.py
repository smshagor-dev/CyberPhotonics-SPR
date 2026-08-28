from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sprpcf.dashboard.core import (
    geometry_figure,
    parse_spectrum_row,
    research_report_markdown,
    selected_design_summary,
    target_frame,
    xai_feature_summary,
)


def _selected() -> pd.Series:
    return pd.Series(
        {
            "pitch_um": 2.0,
            "d_over_lambda": 0.5,
            "metal_thickness_nm": 40.0,
            "channel_radius_um": 0.6,
            "analyte_ri": 1.37,
            "sensitivity_nm_per_riu": 800.0,
            "fom_per_riu": 20.0,
            "lambda_res_nm": 750.0,
            "predicted_sensitivity_nm_per_riu": 790.0,
            "predicted_fom_per_riu": 19.5,
            "predicted_lambda_res_nm": 748.0,
            "confidence_score": 0.82,
            "ood_score": 0.6,
            "in_calibration_domain": True,
            "pareto_rank": 0,
            "fabrication_projection_distance": 0.0,
        }
    )


def test_target_frame_validates_refractive_index() -> None:
    frame = target_frame(800.0, 20.0, 750.0, 1.37)
    assert list(frame.columns) == [
        "sensitivity_nm_per_riu",
        "fom_per_riu",
        "lambda_res_nm",
        "analyte_ri",
    ]
    with pytest.raises(ValueError, match="physically plausible"):
        target_frame(800.0, 20.0, 750.0, 2.1)


def test_selected_design_and_geometry_figure() -> None:
    selected = _selected()
    summary = selected_design_summary(selected)
    assert summary["in_calibration_domain"] is True
    assert summary["geometry"]["pitch_um"] == pytest.approx(2.0)
    figure = geometry_figure(selected)
    assert figure.axes
    figure.clear()


def test_parse_spectrum_row_rejects_length_mismatch() -> None:
    wavelength, loss = parse_spectrum_row(
        {"wavelength_nm": "500,600,700", "loss_db_per_cm": "1,2,3"}
    )
    np.testing.assert_allclose(wavelength, [500.0, 600.0, 700.0])
    np.testing.assert_allclose(loss, [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="same length"):
        parse_spectrum_row({"wavelength_nm": "500,600", "loss_db_per_cm": "1,2,3"})


def test_xai_summary_ranks_absolute_attribution() -> None:
    frame = pd.DataFrame(
        {
            "pitch_um": [1.0, -1.0],
            "d_over_lambda": [0.2, -0.2],
            "metal_thickness_nm": [3.0, -3.0],
            "channel_radius_um": [0.5, -0.5],
            "analyte_ri": [2.0, -2.0],
        }
    )
    summary = xai_feature_summary(frame)
    assert summary.iloc[0]["feature"] == "metal_thickness_nm"
    assert summary.iloc[0]["mean_absolute_attribution"] == pytest.approx(3.0)


def test_report_keeps_synthetic_evidence_boundary() -> None:
    target = target_frame(800.0, 20.0, 750.0, 1.37).iloc[0]
    verification = pd.Series(
        {
            "accepted": True,
            "reason": "",
            "actual_sensitivity_nm_per_riu": 805.0,
            "actual_fom_per_riu": 20.2,
            "actual_lambda_res_nm": 751.0,
            "linearity_r2": 0.99,
        }
    )
    report = research_report_markdown(
        target,
        _selected(),
        verification,
        backend="synthetic",
        evidence={"manifest": "abc123"},
    )
    assert "Synthetic backend results validate software flow only" in report
    assert "COMSOL results may support physical simulation claims" in report
    assert "`abc123`" in report
