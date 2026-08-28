from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from sprpcf.ml.dataset import GEOMETRY_COLUMNS
from sprpcf.ml.losses import GEOMETRY_MAX, GEOMETRY_MIN


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required validation columns: {missing}")


def bootstrap_mean_ci(
    values: np.ndarray | pd.Series | list[float],
    confidence: float = 0.95,
    resamples: int = 2000,
    seed: int = 7,
) -> dict[str, float]:
    """Return a deterministic percentile bootstrap confidence interval for the mean."""
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1.")
    if resamples < 100:
        raise ValueError("resamples must be >= 100.")

    array = np.asarray(values, dtype=float).reshape(-1)
    array = array[np.isfinite(array)]
    if array.size == 0:
        raise ValueError("At least one finite value is required.")
    if array.size == 1:
        value = float(array[0])
        return {"mean": value, "ci_low": value, "ci_high": value, "confidence": float(confidence)}

    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(0, array.size, size=(resamples, array.size))
    means = array[sample_indices].mean(axis=1)
    alpha = 0.5 * (1.0 - confidence)
    low, high = np.quantile(means, [alpha, 1.0 - alpha])
    return {
        "mean": float(array.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "confidence": float(confidence),
    }


def fabrication_constraint_report(frame: pd.DataFrame) -> dict[str, float]:
    """Measure fabrication-bound and air-hole-overlap violations in physical units."""
    _require_columns(frame, GEOMETRY_COLUMNS)
    geometry = frame[GEOMETRY_COLUMNS].to_numpy(dtype=float)
    finite = np.all(np.isfinite(geometry), axis=1)

    lower = np.asarray(GEOMETRY_MIN, dtype=float)
    upper = np.asarray(GEOMETRY_MAX, dtype=float)
    in_bounds = np.all((geometry >= lower) & (geometry <= upper), axis=1)

    pitch = geometry[:, 0]
    d_over_lambda = geometry[:, 1]
    air_diameter = pitch * d_over_lambda
    no_overlap = air_diameter < pitch
    valid = finite & in_bounds & no_overlap

    rows = int(geometry.shape[0])
    if rows == 0:
        raise ValueError("At least one geometry row is required.")

    report: dict[str, float] = {
        "rows": float(rows),
        "valid_rate": float(valid.mean()),
        "violation_rate": float(1.0 - valid.mean()),
        "non_finite_rate": float((~finite).mean()),
        "overlap_violation_rate": float((~no_overlap).mean()),
    }
    for index, column in enumerate(GEOMETRY_COLUMNS):
        below = geometry[:, index] < lower[index]
        above = geometry[:, index] > upper[index]
        report[f"{column}_below_min_rate"] = float(below.mean())
        report[f"{column}_above_max_rate"] = float(above.mean())
    return report


def _linearity_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    residual = float(np.sum(np.square(y_true - y_pred)))
    centered = y_true - float(np.mean(y_true))
    total = float(np.sum(np.square(centered)))
    if total <= 1e-12:
        return 1.0 if residual <= 1e-12 else 0.0
    return float(1.0 - residual / total)


def fixed_geometry_sweep_report(
    frame: pd.DataFrame,
    group_columns: Sequence[str] = GEOMETRY_COLUMNS,
    ri_column: str = "analyte_ri",
    lambda_column: str = "lambda_res_nm",
    fwhm_column: str = "fwhm_nm",
    sensitivity_column: str = "sensitivity_nm_per_riu",
) -> pd.DataFrame:
    """Summarize each fixed-geometry RI sweep using a global linear sensitivity fit.

    The fitted slope is intentionally separate from the local finite-difference
    sensitivity labels used for learning. This gives the validation layer an
    independent, reviewer-friendly estimate of wavelength sensitivity.
    """
    required = [*group_columns, ri_column, lambda_column]
    _require_columns(frame, required)

    rows: list[dict[str, float]] = []
    for key, group in frame.groupby(list(group_columns), sort=False, dropna=False):
        valid = group.dropna(subset=[ri_column, lambda_column]).copy()
        valid = valid[np.isfinite(valid[ri_column].to_numpy(dtype=float))]
        valid = valid[np.isfinite(valid[lambda_column].to_numpy(dtype=float))]
        if valid.shape[0] < 2:
            continue

        ri = valid[ri_column].to_numpy(dtype=float)
        wavelength = valid[lambda_column].to_numpy(dtype=float)
        order = np.argsort(ri)
        ri = ri[order]
        wavelength = wavelength[order]
        if np.any(np.diff(ri) <= 0):
            raise ValueError("Each fixed-geometry validation sweep must contain unique analyte RI values.")

        slope, intercept = np.polyfit(ri, wavelength, 1)
        fitted = slope * ri + intercept
        values = key if isinstance(key, tuple) else (key,)
        record = {column: float(value) for column, value in zip(group_columns, values, strict=True)}
        record.update(
            {
                "ri_points": float(ri.size),
                "ri_min": float(ri.min()),
                "ri_max": float(ri.max()),
                "lambda_min_nm": float(wavelength.min()),
                "lambda_max_nm": float(wavelength.max()),
                "fitted_sensitivity_nm_per_riu": float(slope),
                "linearity_r2": _linearity_r2(wavelength, fitted),
            }
        )

        if fwhm_column in valid.columns:
            fwhm_values = valid[fwhm_column].to_numpy(dtype=float)
            fwhm_values = fwhm_values[np.isfinite(fwhm_values) & (fwhm_values > 0)]
            mean_fwhm = float(fwhm_values.mean()) if fwhm_values.size else float("nan")
        else:
            mean_fwhm = float("nan")
        record["mean_fwhm_nm"] = mean_fwhm
        record["fitted_fom_per_riu"] = (
            float(abs(slope) / mean_fwhm) if np.isfinite(mean_fwhm) and mean_fwhm > 0 else float("nan")
        )

        if sensitivity_column in valid.columns:
            stored = valid[sensitivity_column].to_numpy(dtype=float)
            stored = stored[np.isfinite(stored)]
            mean_stored = float(stored.mean()) if stored.size else float("nan")
            record["mean_local_sensitivity_nm_per_riu"] = mean_stored
            record["sensitivity_fit_bias_nm_per_riu"] = (
                float(mean_stored - slope) if np.isfinite(mean_stored) else float("nan")
            )
        rows.append(record)

    columns = [
        *group_columns,
        "ri_points",
        "ri_min",
        "ri_max",
        "lambda_min_nm",
        "lambda_max_nm",
        "fitted_sensitivity_nm_per_riu",
        "linearity_r2",
        "mean_fwhm_nm",
        "fitted_fom_per_riu",
        "mean_local_sensitivity_nm_per_riu",
        "sensitivity_fit_bias_nm_per_riu",
    ]
    return pd.DataFrame(rows, columns=columns)


def summarize_sweep_report(
    sweep_report: pd.DataFrame,
    bootstrap_resamples: int = 2000,
    seed: int = 7,
) -> dict[str, float | dict[str, float]]:
    """Aggregate fixed-geometry validation into manuscript-ready summary statistics."""
    if sweep_report.empty:
        raise ValueError("No valid fixed-geometry RI sweeps were found.")

    sensitivity = sweep_report["fitted_sensitivity_nm_per_riu"].to_numpy(dtype=float)
    linearity = sweep_report["linearity_r2"].to_numpy(dtype=float)
    fom = sweep_report["fitted_fom_per_riu"].to_numpy(dtype=float)
    finite_fom = fom[np.isfinite(fom)]

    result: dict[str, float | dict[str, float]] = {
        "geometry_sweeps": float(len(sweep_report)),
        "sensitivity_nm_per_riu": bootstrap_mean_ci(
            sensitivity, resamples=bootstrap_resamples, seed=seed
        ),
        "median_linearity_r2": float(np.median(linearity)),
        "minimum_linearity_r2": float(np.min(linearity)),
    }
    if finite_fom.size:
        result["fom_per_riu"] = bootstrap_mean_ci(
            finite_fom,
            resamples=bootstrap_resamples,
            seed=seed + 1,
        )

    if "sensitivity_fit_bias_nm_per_riu" in sweep_report.columns:
        bias = sweep_report["sensitivity_fit_bias_nm_per_riu"].to_numpy(dtype=float)
        bias = bias[np.isfinite(bias)]
        if bias.size:
            result["mean_sensitivity_fit_bias_nm_per_riu"] = float(bias.mean())
    return result
