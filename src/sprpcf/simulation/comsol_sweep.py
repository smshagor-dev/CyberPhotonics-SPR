from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from sprpcf.simulation.metrics import extract_metrics, finite_difference_sensitivity
from sprpcf.simulation.schema import Geometry

LOGGER = logging.getLogger(__name__)


def _load_mph() -> Any:
    try:
        import mph
    except ImportError as exc:
        raise RuntimeError("The 'mph' package is required for COMSOL automation.") from exc
    return mph


def _grid(values: dict[str, list[float]]) -> list[Geometry]:
    geometries: list[Geometry] = []
    for d_over_lambda in values["d_over_lambda"]:
        for pitch_um in values["pitch_um"]:
            for metal_thickness_nm in values["metal_thickness_nm"]:
                for analyte_ri in values["analyte_ri"]:
                    for channel_radius_um in values.get("channel_radius_um", [0.6]):
                        geometries.append(
                            Geometry(
                                d_over_lambda=float(d_over_lambda),
                                pitch_um=float(pitch_um),
                                metal_thickness_nm=float(metal_thickness_nm),
                                analyte_ri=float(analyte_ri),
                                channel_radius_um=float(channel_radius_um),
                            )
                        )
    return geometries


def run_comsol_sweep(model_path: Path, config_path: Path) -> pd.DataFrame:
    """Run a COMSOL parametric sweep and return a structured dataframe.

    The COMSOL model is expected to expose parameters matching the field names in
    `Geometry`, and numerical export expressions configured in the YAML file.
    """
    mph = _load_mph()
    config = yaml.safe_load(config_path.read_text())
    wavelength_expression = config.get("wavelength_expression", "lambda")
    loss_expression = config.get("loss_expression", "loss")
    study = config.get("study", "std1")
    geometries = _grid(config["sweep"])

    client = mph.start()
    model = client.load(str(model_path))
    rows: list[dict[str, float | str]] = []

    for sample_id, geometry in enumerate(geometries):
        LOGGER.info("Running sample %s: %s", sample_id, geometry)
        try:
            model.parameter("d_over_lambda", geometry.d_over_lambda)
            model.parameter("pitch_um", f"{geometry.pitch_um}[um]")
            model.parameter("metal_thickness_nm", f"{geometry.metal_thickness_nm}[nm]")
            model.parameter("analyte_ri", geometry.analyte_ri)
            model.parameter("channel_radius_um", f"{geometry.channel_radius_um}[um]")
            model.solve(study)
            wavelength_nm = np.asarray(model.evaluate(wavelength_expression), dtype=float).ravel()
            loss = np.asarray(model.evaluate(loss_expression), dtype=float).ravel()
            metrics = extract_metrics(wavelength_nm, loss)
            rows.append(
                {
                    "sample_id": sample_id,
                    "status": "ok",
                    **geometry.__dict__,
                    **metrics.__dict__,
                    "wavelength_nm": ",".join(f"{value:.6f}" for value in wavelength_nm),
                    "loss_db_per_cm": ",".join(f"{value:.6f}" for value in loss),
                }
            )
        except Exception as exc:
            LOGGER.exception("COMSOL sample %s failed.", sample_id)
            rows.append({"sample_id": sample_id, "status": f"failed: {exc}", **geometry.__dict__})

    frame = pd.DataFrame(rows)
    ok = frame["status"].eq("ok") if "status" in frame else pd.Series(dtype=bool)
    if ok.any():
        frame.loc[ok, "sensitivity_nm_per_riu"] = finite_difference_sensitivity(
            frame.loc[ok, "lambda_res_nm"].to_numpy(dtype=float),
            frame.loc[ok, "analyte_ri"].to_numpy(dtype=float),
        )
        frame.loc[ok, "fom_per_riu"] = frame.loc[ok, "sensitivity_nm_per_riu"] / frame.loc[ok, "fwhm_nm"]
    return frame


def write_dataset(frame: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".parquet":
        frame.to_parquet(output, index=False)
    else:
        frame.to_csv(output, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run COMSOL PCF-SPR parametric sweeps.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)
    frame = run_comsol_sweep(args.model, args.config)
    write_dataset(frame, args.out)
    LOGGER.info("Wrote %s rows to %s", len(frame), args.out)


if __name__ == "__main__":
    main()
