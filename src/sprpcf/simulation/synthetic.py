from __future__ import annotations

import numpy as np
import pandas as pd

from sprpcf.simulation.metrics import extract_metrics, finite_difference_sensitivity
from sprpcf.simulation.schema import Geometry


def synthetic_loss_spectrum(
    geometry: Geometry,
    wavelength_nm: np.ndarray,
    rng: np.random.Generator,
    noise_std: float = 0.02,
) -> np.ndarray:
    """Generate a plausible PCF-SPR loss spectrum for pipeline validation."""
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


def sample_geometries(samples: int, rng: np.random.Generator) -> list[Geometry]:
    """Draw random fabrication-feasible geometries."""
    return [
        Geometry(
            d_over_lambda=float(rng.uniform(0.25, 0.85)),
            pitch_um=float(rng.uniform(1.0, 3.2)),
            metal_thickness_nm=float(rng.uniform(20.0, 70.0)),
            analyte_ri=float(rng.uniform(1.33, 1.41)),
            channel_radius_um=float(rng.uniform(0.35, 0.9)),
        )
        for _ in range(samples)
    ]


def build_synthetic_dataset(
    samples: int,
    wavelengths: int = 256,
    seed: int = 7,
) -> pd.DataFrame:
    """Create a training-ready synthetic PCF-SPR dataset."""
    rng = np.random.default_rng(seed)
    wavelength_nm = np.linspace(450.0, 900.0, wavelengths)
    rows: list[dict[str, float | str]] = []

    for index, geometry in enumerate(sample_geometries(samples, rng)):
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
    frame["sensitivity_nm_per_riu"] = finite_difference_sensitivity(
        frame["lambda_res_nm"].to_numpy(),
        frame["analyte_ri"].to_numpy(),
    )
    frame["fom_per_riu"] = frame["sensitivity_nm_per_riu"] / frame["fwhm_nm"]
    return frame
