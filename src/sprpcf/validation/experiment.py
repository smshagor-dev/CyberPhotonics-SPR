from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from sprpcf.simulation.metrics import fwhm, resonance_wavelength
from sprpcf.utils.reproducibility import sha256_file

EXPERIMENT_MANIFEST_SCHEMA_VERSION = 1
_REQUIRED_SPECTRUM_COLUMNS = ("wavelength_nm", "loss_db_per_cm")


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"Invalid experimental measurement manifest {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("Experimental measurement manifest must contain a YAML mapping.")
    return dict(payload)


def validate_experiment_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    if int(payload.get("schema_version", 0)) != EXPERIMENT_MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {EXPERIMENT_MANIFEST_SCHEMA_VERSION}.")
    spectra = payload.get("spectra")
    if not isinstance(spectra, list) or len(spectra) < 2:
        raise ValueError("spectra must contain at least two RI-labelled calibrated spectrum entries.")

    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(spectra):
        if not isinstance(row, Mapping):
            raise ValueError(f"spectra[{index}] must be a mapping.")
        path = str(row.get("path") or "").strip()
        if not path:
            raise ValueError(f"spectra[{index}].path is required.")
        try:
            analyte_ri = float(row["analyte_ri"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"spectra[{index}].analyte_ri must be numeric.") from exc
        if not np.isfinite(analyte_ri) or not 1.0 < analyte_ri < 2.0:
            raise ValueError(f"spectra[{index}].analyte_ri must be within the physical project range (1, 2).")
        replicate = str(row.get("replicate") or index + 1).strip()
        normalized.append({"path": path, "analyte_ri": analyte_ri, "replicate": replicate})

    unique_ri = sorted({row["analyte_ri"] for row in normalized})
    if len(unique_ri) < 2:
        raise ValueError("At least two unique analyte RI values are required to estimate experimental sensitivity.")
    return {
        "schema_version": EXPERIMENT_MANIFEST_SCHEMA_VERSION,
        "experiment_id": str(payload.get("experiment_id") or "experimental-sensor-validation"),
        "spectra": normalized,
    }


def load_experiment_manifest(path: str | Path) -> dict[str, Any]:
    return validate_experiment_manifest(_read_manifest(Path(path)))


def _read_spectrum(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(path)
    elif suffix in {".parquet", ".pq"}:
        frame = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported calibrated spectrum format {path.suffix!r}; use CSV or Parquet.")
    missing = [column for column in _REQUIRED_SPECTRUM_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Calibrated spectrum {path} is missing columns: {missing}")
    frame = frame.loc[:, list(_REQUIRED_SPECTRUM_COLUMNS)].dropna()
    if frame.shape[0] < 4:
        raise ValueError(f"Calibrated spectrum {path} must contain at least four finite samples.")
    return frame


def _resolve(manifest_path: Path, stored: str) -> Path:
    candidate = Path(stored)
    return candidate if candidate.is_absolute() else (manifest_path.resolve().parent / candidate).resolve()


def _r2(observed: np.ndarray, predicted: np.ndarray) -> float:
    residual = float(np.sum((observed - predicted) ** 2))
    total = float(np.sum((observed - np.mean(observed)) ** 2))
    if total <= 0:
        return 1.0 if residual <= 1e-15 else float("nan")
    return float(1.0 - residual / total)


def _sample_std(values: pd.Series) -> float:
    return float(values.std(ddof=1)) if values.size > 1 else float("nan")


def _plot_resonance(summary: pd.DataFrame, slope: float, intercept: float, path: Path) -> None:
    x = summary["analyte_ri"].to_numpy(dtype=float)
    y = summary["lambda_res_mean_nm"].to_numpy(dtype=float)
    yerr = summary["lambda_res_std_nm"].fillna(0.0).to_numpy(dtype=float)
    order = np.argsort(x)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.errorbar(x, y, yerr=yerr, fmt="o", capsize=3, label="Measured mean ± SD")
    ax.plot(x[order], (slope * x + intercept)[order], label="OLS fit")
    ax.set_xlabel("Analyte refractive index")
    ax.set_ylabel("Resonance wavelength (nm)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _plot_repeatability(summary: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.bar(summary["analyte_ri"].astype(str), summary["lambda_res_std_nm"].fillna(0.0))
    ax.set_xlabel("Analyte refractive index")
    ax.set_ylabel("Resonance SD across replicates (nm)")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def analyze_experimental_measurements(
    manifest_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Analyze RI-labelled calibrated spectra without qualifying them as physical evidence.

    Qualification remains a separate hash-bound registry action. This function only derives
    measured-spectrum statistics from files explicitly listed in the experimental manifest.
    """
    manifest_file = Path(manifest_path)
    manifest = load_experiment_manifest(manifest_file)
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Experimental analysis output must be empty or absent: {out}")
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    source_hashes: list[dict[str, str]] = []
    for item in manifest["spectra"]:
        spectrum_path = _resolve(manifest_file, item["path"])
        if not spectrum_path.is_file():
            raise FileNotFoundError(f"Calibrated experimental spectrum not found: {spectrum_path}")
        frame = _read_spectrum(spectrum_path)
        wavelength = frame["wavelength_nm"].to_numpy(dtype=float)
        loss = frame["loss_db_per_cm"].to_numpy(dtype=float)
        lambda_res, peak_loss = resonance_wavelength(wavelength, loss)
        width = fwhm(wavelength, loss)
        rows.append(
            {
                "analyte_ri": float(item["analyte_ri"]),
                "replicate": item["replicate"],
                "spectrum": str(spectrum_path),
                "lambda_res_nm": lambda_res,
                "peak_loss_db_per_cm": peak_loss,
                "fwhm_nm": width,
            }
        )
        source_hashes.append({"path": str(spectrum_path), "sha256": sha256_file(spectrum_path)})

    replicate_frame = pd.DataFrame(rows).sort_values(["analyte_ri", "replicate"]).reset_index(drop=True)
    grouped = replicate_frame.groupby("analyte_ri", sort=True)
    ri_summary = grouped.agg(
        replicate_count=("lambda_res_nm", "size"),
        lambda_res_mean_nm=("lambda_res_nm", "mean"),
        lambda_res_std_nm=("lambda_res_nm", _sample_std),
        fwhm_mean_nm=("fwhm_nm", "mean"),
    ).reset_index()

    ri = ri_summary["analyte_ri"].to_numpy(dtype=float)
    mean_lambda = ri_summary["lambda_res_mean_nm"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(ri, mean_lambda, 1)
    fitted = slope * ri + intercept
    r2 = _r2(mean_lambda, fitted)
    finite_widths = replicate_frame["fwhm_nm"].to_numpy(dtype=float)
    finite_widths = finite_widths[np.isfinite(finite_widths) & (finite_widths > 0)]
    mean_fwhm = float(np.mean(finite_widths)) if finite_widths.size else None
    fom = float(abs(slope) / mean_fwhm) if mean_fwhm else None
    repeatability = ri_summary["lambda_res_std_nm"].dropna().to_numpy(dtype=float)

    replicate_path = out / "replicate_metrics.csv"
    summary_path = out / "ri_summary.csv"
    replicate_frame.to_csv(replicate_path, index=False)
    ri_summary.to_csv(summary_path, index=False)
    resonance_plot = out / "resonance_vs_ri.png"
    repeatability_plot = out / "repeatability_by_ri.png"
    _plot_resonance(ri_summary, float(slope), float(intercept), resonance_plot)
    _plot_repeatability(ri_summary, repeatability_plot)

    summary = {
        "schema_version": 1,
        "experiment_id": manifest["experiment_id"],
        "qualification_status": "unqualified_candidate_analysis",
        "source_manifest": str(manifest_file.resolve()),
        "source_manifest_sha256": sha256_file(manifest_file),
        "source_spectra": source_hashes,
        "replicate_count": int(replicate_frame.shape[0]),
        "unique_ri_count": int(ri_summary.shape[0]),
        "sensitivity_nm_per_riu": float(slope),
        "fit_intercept_nm": float(intercept),
        "fit_r2": r2,
        "mean_fwhm_nm": mean_fwhm,
        "fom_per_riu": fom,
        "repeatability_mean_sd_nm": float(np.mean(repeatability)) if repeatability.size else None,
        "repeatability_max_sd_nm": float(np.max(repeatability)) if repeatability.size else None,
        "outputs": {
            "replicate_metrics": str(replicate_path),
            "ri_summary": str(summary_path),
            "resonance_vs_ri": str(resonance_plot),
            "repeatability_by_ri": str(repeatability_plot),
        },
        "scientific_boundary": (
            "These statistics are derived from the listed calibrated spectra. They become evidence for experimental claims "
            "only when the underlying raw measurements, protocol, and calibration are separately qualified in the evidence registry."
        ),
    }
    (out / "experimental_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
