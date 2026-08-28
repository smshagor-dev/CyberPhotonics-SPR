from __future__ import annotations

from dataclasses import replace
import numpy as np
import pandas as pd
from sprpcf.simulation.metrics import assign_grouped_sensitivity, extract_metrics
from sprpcf.simulation.schema import Geometry


def synthetic_loss_spectrum(geometry: Geometry, wavelength_nm: np.ndarray, rng: np.random.Generator, noise_std: float = 0.02) -> np.ndarray:
    lambda_res = 520.0 + 820.0 * (geometry.analyte_ri - 1.33) + 165.0 * (geometry.d_over_lambda - 0.45) + 28.0 * (geometry.pitch_um - 2.0) + 1.7 * (geometry.metal_thickness_nm - 45.0) + 18.0 * (geometry.channel_radius_um - 0.6)
    width = 28.0 + 70.0 * abs(geometry.d_over_lambda - 0.55) + 0.45 * geometry.metal_thickness_nm
    amplitude = 18.0 + 24.0 * geometry.d_over_lambda + 0.25 * geometry.metal_thickness_nm
    baseline = 0.25 + 0.03 * (wavelength_nm - wavelength_nm.min()) / np.ptp(wavelength_nm)
    lorentzian = amplitude / (1.0 + ((wavelength_nm - lambda_res) / (0.5 * width)) ** 2)
    drift = rng.normal(0.0, noise_std) + 0.015 * np.sin(wavelength_nm / 31.0)
    return baseline + lorentzian + drift + rng.normal(0.0, noise_std, wavelength_nm.size)


def sample_base_geometries(samples: int, rng: np.random.Generator) -> list[Geometry]:
    return [Geometry(float(rng.uniform(0.25, 0.85)), float(rng.uniform(1.0, 3.2)), float(rng.uniform(20.0, 70.0)), 1.33, float(rng.uniform(0.35, 0.9))) for _ in range(samples)]


def build_synthetic_dataset(samples: int, wavelengths: int = 256, seed: int = 7, analyte_ri_values: tuple[float, ...] = (1.33, 1.35, 1.37, 1.39, 1.41)) -> pd.DataFrame:
    if samples < 1:
        raise ValueError("samples must be >= 1")
    if wavelengths < 8:
        raise ValueError("wavelengths must be >= 8")
    if len(set(analyte_ri_values)) < 2:
        raise ValueError("At least two unique analyte RI values are required.")
    rng = np.random.default_rng(seed)
    wavelength_nm = np.linspace(450.0, 900.0, wavelengths)
    rows = []
    group_count = int(np.ceil(samples / len(analyte_ri_values)))
    for geometry_id, base_geometry in enumerate(sample_base_geometries(group_count, rng)):
        for analyte_ri in analyte_ri_values:
            if len(rows) >= samples:
                break
            geometry = replace(base_geometry, analyte_ri=float(analyte_ri))
            loss = synthetic_loss_spectrum(geometry, wavelength_nm, rng)
            metrics = extract_metrics(wavelength_nm, loss)
            rows.append({"sample_id": len(rows), "geometry_id": geometry_id, "d_over_lambda": geometry.d_over_lambda, "pitch_um": geometry.pitch_um, "metal_thickness_nm": geometry.metal_thickness_nm, "analyte_ri": geometry.analyte_ri, "channel_radius_um": geometry.channel_radius_um, "lambda_res_nm": metrics.lambda_res_nm, "peak_loss_db_per_cm": metrics.peak_loss_db_per_cm, "fwhm_nm": metrics.fwhm_nm, "wavelength_nm": ",".join(f"{v:.6f}" for v in wavelength_nm), "loss_db_per_cm": ",".join(f"{v:.6f}" for v in loss), "source": "synthetic"})
    return assign_grouped_sensitivity(pd.DataFrame(rows))
