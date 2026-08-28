from __future__ import annotations

import numpy as np
import pandas as pd

from sprpcf.simulation.metrics import extract_metrics, grouped_finite_difference_sensitivity
from sprpcf.simulation.schema import Geometry


SENSITIVITY_GROUP_COLUMNS = ["d_over_lambda", "pitch_um", "metal_thickness_nm", "channel_radius_um"]
DEFAULT_ANALYTE_RI = (1.33, 1.35, 1.37, 1.39, 1.41)


def synthetic_loss_spectrum(
    geometry: Geometry,
    wavelength_nm: np.ndarray,
    rng: np.random.Generator,
    noise_std: float = 0.02,
) -> np.ndarray:
    """Generate a plausible PCF-SPR loss spectrum for pipeline validation."""
    geometry.validate()
    lambda_res = (
        520.0
        + 820.0 * (geometry.analyte_ri - 1.33)
        + 165.0 * (geometry.d_over_lambda - 0.45)
        + 28.0 * (geometry.pitch_um - 2.0)
        + 1.7 * (geometry.metal_thickness_nm - 45.0)
        + 18.0 * (geometry.channel_radius_um - 0.6)
    )
    width = 28.0 + 70.0 * abs(geometry.d_over_lambda - 0.55) + 0.45 * geometry.metal_thickness_nm
    amplitude = 18.0 + 24.0 * geometry.d_over_lambda + 0.25 * geometry.metal_thickness_nm
    baseline = 0.25 + 0.03 * (wavelength_nm - wavelength_nm.min()) / np.ptp(wavelength_nm)
    lorentzian = amplitude / (1.0 + ((wavelength_nm - lambda_res) / (0.5 * width)) ** 2)
    drift = rng.normal(0.0, noise_std) + 0.015 * np.sin(wavelength_nm / 31.0)
    return baseline + lorentzian + drift + rng.normal(0.0, noise_std, wavelength_nm.size)


def sample_geometries(
    samples: int,
    rng: np.random.Generator,
    analyte_ri_values: tuple[float, ...] = DEFAULT_ANALYTE_RI,
) -> list[Geometry]:
    """Draw base geometries and pair each with an RI sweep for valid sensitivity labels."""
    if samples < 1:
        raise ValueError("samples must be >= 1.")
    if len(analyte_ri_values) < 2 or len(set(analyte_ri_values)) != len(analyte_ri_values):
        raise ValueError("analyte_ri_values must contain at least two unique values.")

    geometries: list[Geometry] = []
    for _ in range(samples):
        d_over_lambda = float(rng.uniform(0.25, 0.85))
        pitch_um = float(rng.uniform(1.0, 3.2))
        metal_thickness_nm = float(rng.uniform(20.0, 70.0))
        channel_radius_um = float(rng.uniform(0.35, 0.9))
        for analyte_ri in analyte_ri_values:
            geometries.append(
                Geometry(
                    d_over_lambda=d_over_lambda,
                    pitch_um=pitch_um,
                    metal_thickness_nm=metal_thickness_nm,
                    analyte_ri=float(analyte_ri),
                    channel_radius_um=channel_radius_um,
                )
            )
    return geometries


def build_synthetic_dataset(
    samples: int,
    wavelengths: int = 256,
    seed: int = 7,
    analyte_ri_values: tuple[float, ...] = DEFAULT_ANALYTE_RI,
) -> pd.DataFrame:
    """Create a training-ready synthetic PCF-SPR dataset with fixed-geometry RI sweeps.

    ``samples`` is the number of base sensor geometries. The number of output rows is
    ``samples * len(analyte_ri_values)``.
    """
    if wavelengths < 16:
        raise ValueError("wavelengths must be >= 16 for reliable spectral metrics.")
    rng = np.random.default_rng(seed)
    wavelength_nm = np.linspace(350.0, 950.0, wavelengths)
    rows: list[dict[str, float | int | str]] = []

    for index, geometry in enumerate(sample_geometries(samples, rng, analyte_ri_values)):
        loss = synthetic_loss_spectrum(geometry, wavelength_nm, rng)
        metrics = extract_metrics(wavelength_nm, loss)
        rows.append(
            {
                "sample_id": index,
                "d_over_lambda": geometry.d_over_lambda,
                "pitch_um": geometry.pitch_um,
                "metal_thickness_nm": geometry.metal_thickness_nm,
                "analyte_ri": geometry.analyte_ri,
                "channel_radius_um": geometry.channel_radius_um,
                "lambda_res_nm": metrics.lambda_res_nm,
                "peak_loss_db_per_cm": metrics.peak_loss_db_per_cm,
                "fwhm_nm": metrics.fwhm_nm,
                "wavelength_nm": ",".join(f"{value:.6f}" for value in wavelength_nm),
                "loss_db_per_cm": ",".join(f"{value:.6f}" for value in loss),
            }
        )

    frame = pd.DataFrame(rows)
    frame["sensitivity_nm_per_riu"] = grouped_finite_difference_sensitivity(
        frame,
        SENSITIVITY_GROUP_COLUMNS,
    )
    frame["fom_per_riu"] = frame["sensitivity_nm_per_riu"].abs() / frame["fwhm_nm"]
    return frame
