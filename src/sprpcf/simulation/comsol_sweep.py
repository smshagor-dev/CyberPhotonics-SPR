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

from sprpcf.simulation.metrics import finite_difference_sensitivity
from sprpcf.simulation.schema import Geometry
from sprpcf.simulation.spectrum_quality import normalize_comsol_export, validate_optical_spectrum

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
    """Normalize one real modal spectrum without flattening multiple solutions."""
    return normalize_comsol_export(wavelength, loss, config)


def _optional_grid_derivatives(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive diagnostics ONLY for a uniquely sampled complete explicit RI grid.

    Arbitrary COMSOL batch/request subsets do not carry frozen request
    identities. Computing finite differences across their rows can silently
    combine unrelated inverse-design candidates with identical geometry.
    Duplicated RI points previously raised after COMSOL had already solved
    the entire expensive batch; now they produce an explicit unqualified
    diagnostic instead, without discarding the original spectra.
    """
    frame = frame.copy()
    frame["sensitivity_nm_per_riu"] = float("nan")
    frame["fom_per_riu"] = float("nan")
    frame["sensitivity_diagnostic_status"] = "solver_failed"
    for _, group in frame.groupby(
        SENSITIVITY_GROUP_COLUMNS, sort=False, dropna=False
    ):
        if not group["status"].eq("ok").all():
            indices = group.index[group["status"].eq("ok")]
            frame.loc[indices, "sensitivity_diagnostic_status"] = (
                "incomplete_grid_solver_failure"
            )
            continue
        if len(group) < 2:
            frame.loc[group.index, "sensitivity_diagnostic_status"] = (
                "insufficient_unique_ri_points"
            )
            continue
        ri = group["analyte_ri"].to_numpy(dtype=float)
        if not np.isfinite(ri).all() or len(np.unique(ri)) != len(ri):
            frame.loc[group.index, "sensitivity_diagnostic_status"] = (
                "duplicate_ri_points_no_derivative"
            )
            LOGGER.warning(
                "Skipping COMSOL grid sensitivity for repeated RI in geometry %s",
                tuple(group.iloc[0][SENSITIVITY_GROUP_COLUMNS]),
            )
            continue
        if not np.isfinite(group["lambda_res_nm"].to_numpy(dtype=float)).all():
            frame.loc[group.index, "sensitivity_diagnostic_status"] = (
                "nonfinite_resonance_no_derivative"
            )
            continue
        estimates = finite_difference_sensitivity(
            group["lambda_res_nm"].to_numpy(dtype=float), ri,
        )
        fwhm = group["fwhm_nm"].to_numpy(dtype=float)
        if not np.isfinite(fwhm).all() or np.any(fwhm <= 0):
            frame.loc[group.index, "sensitivity_diagnostic_status"] = (
                "invalid_fwhm_no_derivative"
            )
            continue
        frame.loc[group.index, "sensitivity_nm_per_riu"] = estimates
        frame.loc[group.index, "fom_per_riu"] = np.abs(estimates) / fwhm
        frame.loc[group.index, "sensitivity_diagnostic_status"] = (
            "grid_diagnostic_only_unqualified"
        )
    return frame


class ComsolModelSession:
    """Own one loaded .mph model for a bounded checkpoint execution session.

    MPh is process-singleton. Reuse its in-memory model within one invocation,
    release *only that model* on exit, and leave the shared client/server alone.
    No automatic reconnect, fallback or fabricated result is permitted.
    """

    def __init__(self, model_path: Path, config_path: Path):
        self.model_path = Path(model_path).resolve()
        self.config_path = Path(config_path).resolve()
        self.config = _load_config(self.config_path)
        self.client: Any | None = None
        self.model: Any | None = None

    def __enter__(self) -> "ComsolModelSession":
        client = _load_mph().start()
        loaded = client.load(str(self.model_path))
        self.client, self.model = client, loaded
        return self

    def run(
        self, model_path: Path, config_path: Path, geometries: Sequence[Geometry],
    ) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("No active loaded COMSOL model session")
        if (Path(model_path).resolve() != self.model_path
                or Path(config_path).resolve() != self.config_path):
            raise ValueError("Loaded COMSOL model session cannot switch model/config paths")
        return _run_geometries(
            self.model_path, self.config, geometries,
            derive_grid_diagnostics=False, loaded_model=self.model,
        )

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        try:
            if self.model is not None and self.client is not None:
                # MPh Client.remove(model) is the documented ownership cleanup.
                # Never call clear() or disconnect() on a shared MPh singleton.
                remove = getattr(self.client, "remove", None)
                if callable(remove):
                    remove(self.model)
        finally:
            self.model = None
            self.client = None
        return False


def _run_geometries(
    model_path: Path, config: dict[str, Any], geometries: Sequence[Geometry], *,
    derive_grid_diagnostics: bool = False, loaded_model: Any | None = None,
) -> pd.DataFrame:
    wavelength_expression = config.get("wavelength_expression", "lambda")
    loss_expression = config.get("loss_expression", "loss")
    study = config.get("study", "std1")

    # Legacy direct calls retain their existing independent load behavior.
    # Checkpoint mode passes a loaded model for all batches in one invocation.
    if loaded_model is None:
        client = _load_mph().start()
        model = client.load(str(model_path))
    else:
        model = loaded_model
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
            metrics = validate_optical_spectrum(wavelength_nm, loss_db_per_cm)
            rows.append(
                {
                    "sample_id": sample_id,
                    "status": "ok",
                    **geometry.__dict__,
                    "lambda_res_nm": metrics.lambda_res_nm,
                    "peak_loss_db_per_cm": metrics.peak_loss_db_per_cm,
                    "fwhm_nm": metrics.fwhm_nm,
                    "wavelength_nm": ",".join(format(float(value), ".17g") for value in wavelength_nm),
                    "loss_db_per_cm": ",".join(format(float(value), ".17g") for value in loss_db_per_cm),
                }
            )
        except Exception as exc:
            LOGGER.exception("COMSOL sample %s failed.", sample_id)
            rows.append({"sample_id": sample_id, "status": f"failed: {exc}", **geometry.__dict__})

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    # Finite differences require one deliberately constructed RI grid.
    # Source-request identity is absent in arbitrary batch calls, so never
    # calculate sensitivity or FOM across those independent candidates.
    return _optional_grid_derivatives(frame) if derive_grid_diagnostics else frame


def run_comsol_sweep(model_path: Path, config_path: Path) -> pd.DataFrame:
    """Run the configured COMSOL grid and return a structured dataframe."""
    config = _load_config(config_path)
    if "sweep" not in config:
        raise ValueError("COMSOL sweep configuration requires a 'sweep' mapping.")
    return _run_geometries(
        model_path, config, _grid(config["sweep"]),
        derive_grid_diagnostics=True,
    )


def run_comsol_geometries(
    model_path: Path,
    config_path: Path,
    geometries: Sequence[Geometry],
) -> pd.DataFrame:
    """Run raw requested geometries; never infer derivatives between requests.

    This method is used for inverse candidates, live API calls and COMSOL
    verification batches, where a batch boundary does NOT define a complete
    independent RI sweep. Qualified sensitivities are computed later from
    fixed-geometry, request-bound complete RI sweeps with true source review.
    """
    config = _load_config(config_path)
    return _run_geometries(
        model_path, config, geometries, derive_grid_diagnostics=False
    )


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
