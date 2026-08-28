from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from sprpcf.simulation.metrics import extract_metrics, grouped_finite_difference_sensitivity
from sprpcf.simulation.schema import Geometry

LOGGER = logging.getLogger(__name__)
SENSITIVITY_GROUP_COLUMNS = ["d_over_lambda", "pitch_um", "metal_thickness_nm", "channel_radius_um"]


def _load_mph() -> Any:
    try:
        import mph
    except ImportError as exc:
        raise RuntimeError("The 'mph' package is required for COMSOL automation.") from exc
    return mph


def _load_config(config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("COMSOL sweep configuration must be a YAML mapping.")
    return config


def _grid(values: dict[str, list[float]]) -> list[Geometry]:
    required = ["d_over_lambda", "pitch_um", "metal_thickness_nm", "analyte_ri"]
    missing = [name for name in required if name not in values]
    if missing:
        raise ValueError(f"Sweep configuration is missing keys: {missing}")

    geometries: list[Geometry] = []
    for d_over_lambda in values["d_over_lambda"]:
        for pitch_um in values["pitch_um"]:
            for metal_thickness_nm in values["metal_thickness_nm"]:
                for analyte_ri in values["analyte_ri"]:
                    for channel_radius_um in values.get("channel_radius_um", [0.6]):
                        geometry = Geometry(
                            d_over_lambda=float(d_over_lambda),
                            pitch_um=float(pitch_um),
                            metal_thickness_nm=float(metal_thickness_nm),
                            analyte_ri=float(analyte_ri),
                            channel_radius_um=float(channel_radius_um),
                        )
                        geometry.validate()
                        geometries.append(geometry)
    return geometries


def _normalize_evaluated_spectrum(
    wavelength: np.ndarray,
    loss: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    wavelength_nm = np.asarray(wavelength, dtype=float).ravel() * float(config.get("wavelength_scale_to_nm", 1.0))
    loss_db_per_cm = np.asarray(loss, dtype=float).ravel() * float(config.get("loss_scale_to_db_per_cm", 1.0))
    if wavelength_nm.size != loss_db_per_cm.size or wavelength_nm.size < 3:
        raise ValueError("COMSOL wavelength and loss arrays must have equal length >= 3.")
    if not np.all(np.isfinite(wavelength_nm)) or not np.all(np.isfinite(loss_db_per_cm)):
        raise ValueError("COMSOL returned non-finite wavelength or loss values.")

    delta = np.diff(wavelength_nm)
    if np.all(delta < 0):
        wavelength_nm = wavelength_nm[::-1]
        loss_db_per_cm = loss_db_per_cm[::-1]
    elif not np.all(delta > 0):
        raise ValueError("COMSOL wavelength samples must be strictly monotonic.")

    expected = config.get("expected_wavelength_nm", [300.0, 2500.0])
    if not isinstance(expected, (list, tuple)) or len(expected) != 2:
        raise ValueError("expected_wavelength_nm must be [min_nm, max_nm].")
    lower, upper = float(expected[0]), float(expected[1])
    if wavelength_nm.min() < lower or wavelength_nm.max() > upper:
        raise ValueError(
            "COMSOL wavelength values are outside expected_wavelength_nm. "
            "Check wavelength_scale_to_nm and the model export expression."
        )
    return wavelength_nm, loss_db_per_cm


def _run_geometries(model_path: Path, config: dict[str, Any], geometries: Sequence[Geometry]) -> pd.DataFrame:
    mph = _load_mph()
    wavelength_expression = config.get("wavelength_expression", "lambda")
    loss_expression = config.get("loss_expression", "loss")
    study = config.get("study", "std1")

    client = mph.start()
    model = client.load(str(model_path))
    rows: list[dict[str, float | int | str]] = []

    for sample_id, geometry in enumerate(geometries):
        geometry.validate()
        LOGGER.info("Running sample %s: %s", sample_id, geometry)
        try:
            model.parameter("d_over_lambda", geometry.d_over_lambda)
            model.parameter("pitch_um", f"{geometry.pitch_um}[um]")
            model.parameter("metal_thickness_nm", f"{geometry.metal_thickness_nm}[nm]")
            model.parameter("analyte_ri", geometry.analyte_ri)
            model.parameter("channel_radius_um", f"{geometry.channel_radius_um}[um]")
            model.solve(study)
            wavelength_nm, loss_db_per_cm = _normalize_evaluated_spectrum(
                model.evaluate(wavelength_expression),
                model.evaluate(loss_expression),
                config,
            )
            metrics = extract_metrics(wavelength_nm, loss_db_per_cm)
            rows.append(
                {
                    "sample_id": sample_id,
                    "status": "ok",
                    **geometry.__dict__,
                    **metrics.__dict__,
                    "wavelength_nm": ",".join(f"{value:.6f}" for value in wavelength_nm),
                    "loss_db_per_cm": ",".join(f"{value:.6f}" for value in loss_db_per_cm),
                }
            )
        except Exception as exc:
            LOGGER.exception("COMSOL sample %s failed.", sample_id)
            rows.append({"sample_id": sample_id, "status": f"failed: {exc}", **geometry.__dict__})

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    ok = frame["status"].eq("ok")
    if ok.any():
        successful = frame.loc[ok].copy()
        sensitivity = grouped_finite_difference_sensitivity(successful, SENSITIVITY_GROUP_COLUMNS)
        frame.loc[successful.index, "sensitivity_nm_per_riu"] = sensitivity
        frame.loc[successful.index, "fom_per_riu"] = (
            frame.loc[successful.index, "sensitivity_nm_per_riu"].abs()
            / frame.loc[successful.index, "fwhm_nm"]
        )
    return frame


def run_comsol_sweep(model_path: Path, config_path: Path) -> pd.DataFrame:
    """Run the configured COMSOL grid and return a structured dataframe."""
    config = _load_config(config_path)
    if "sweep" not in config:
        raise ValueError("COMSOL sweep configuration requires a 'sweep' mapping.")
    return _run_geometries(model_path, config, _grid(config["sweep"]))


def run_comsol_geometries(
    model_path: Path,
    config_path: Path,
    geometries: Sequence[Geometry],
) -> pd.DataFrame:
    """Run only explicitly selected geometries, used by the active-learning loop."""
    config = _load_config(config_path)
    return _run_geometries(model_path, config, geometries)


def write_dataset(frame: pd.DataFrame, output: Path, metadata: dict[str, Any] | None = None) -> None:
    """Write a dataset plus a provenance sidecar containing a content hash."""
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".parquet":
        frame.to_parquet(output, index=False)
    else:
        frame.to_csv(output, index=False)

    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    sidecar = {
        "schema_version": 1,
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "sha256": digest,
        **(metadata or {}),
    }
    metadata_path = output.with_suffix(output.suffix + ".meta.json")
    metadata_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run COMSOL PCF-SPR parametric sweeps.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)
    frame = run_comsol_sweep(args.model, args.config)
    write_dataset(
        frame,
        args.out,
        metadata={"source": "comsol", "model": str(args.model), "config": str(args.config)},
    )
    LOGGER.info("Wrote %s rows to %s", len(frame), args.out)


if __name__ == "__main__":
    main()
