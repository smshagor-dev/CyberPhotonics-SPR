from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sprpcf.ml.dataset import CONDITION_COLUMNS, GEOMETRY_COLUMNS, METRIC_COLUMNS
from sprpcf.simulation.schema import Geometry


def target_frame(
    sensitivity_nm_per_riu: float,
    fom_per_riu: float,
    lambda_res_nm: float,
    analyte_ri: float,
) -> pd.DataFrame:
    values = [
        float(sensitivity_nm_per_riu),
        float(fom_per_riu),
        float(lambda_res_nm),
        float(analyte_ri),
    ]
    if not np.all(np.isfinite(values)):
        raise ValueError("Dashboard target values must be finite.")
    if not 1.0 < analyte_ri < 2.0:
        raise ValueError("analyte_ri must be in the physically plausible interval (1, 2).")
    return pd.DataFrame(
        [
            {
                "sensitivity_nm_per_riu": values[0],
                "fom_per_riu": values[1],
                "lambda_res_nm": values[2],
                "analyte_ri": values[3],
            }
        ]
    )


def selected_design_summary(row: Mapping[str, Any]) -> dict[str, Any]:
    missing = [column for column in [*GEOMETRY_COLUMNS, *METRIC_COLUMNS, *CONDITION_COLUMNS] if column not in row]
    if missing:
        raise ValueError(f"Selected design is missing required columns: {missing}")
    geometry = Geometry(
        pitch_um=float(row["pitch_um"]),
        d_over_lambda=float(row["d_over_lambda"]),
        metal_thickness_nm=float(row["metal_thickness_nm"]),
        channel_radius_um=float(row["channel_radius_um"]),
        analyte_ri=float(row["analyte_ri"]),
    )
    geometry.validate()
    return {
        "geometry": {column: float(row[column]) for column in GEOMETRY_COLUMNS},
        "target": {column: float(row[column]) for column in [*METRIC_COLUMNS, *CONDITION_COLUMNS]},
        "predicted": {
            column: float(row.get(f"predicted_{column}", np.nan))
            for column in METRIC_COLUMNS
        },
        "confidence_score": float(row.get("confidence_score", np.nan)),
        "ood_score": float(row.get("ood_score", np.nan)),
        "in_calibration_domain": bool(row.get("in_calibration_domain", False)),
        "pareto_rank": int(row.get("pareto_rank", -1)),
        "fabrication_projection_distance": float(row.get("fabrication_projection_distance", np.nan)),
    }


def geometry_figure(row: Mapping[str, Any]):
    summary = selected_design_summary(row)
    geometry = summary["geometry"]
    pitch = geometry["pitch_um"]
    diameter = pitch * geometry["d_over_lambda"]
    channel = geometry["channel_radius_um"]

    fig = plt.figure(figsize=(7.2, 5.8))
    ax = fig.add_subplot(111, projection="3d")
    z = np.linspace(0.0, max(1.5, pitch * 0.9), 24)
    theta = np.linspace(0.0, 2.0 * np.pi, 72)

    def cylinder(cx: float, cy: float, radius: float, linewidth: float = 0.8) -> None:
        for zz in (z[0], z[-1]):
            ax.plot(
                cx + radius * np.cos(theta),
                cy + radius * np.sin(theta),
                np.full_like(theta, zz),
                linewidth=linewidth,
            )
        for angle in np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False):
            ax.plot(
                [cx + radius * np.cos(angle), cx + radius * np.cos(angle)],
                [cy + radius * np.sin(angle), cy + radius * np.sin(angle)],
                [z[0], z[-1]],
                linewidth=linewidth,
            )

    hole_radius = diameter / 2.0
    for ring in (1, 2):
        count = 6 * ring
        radius = ring * pitch
        for index in range(count):
            angle = 2.0 * np.pi * index / count
            cylinder(radius * np.cos(angle), radius * np.sin(angle), hole_radius)

    cylinder(0.0, 0.0, channel, linewidth=1.2)
    boundary = 2.65 * pitch
    cylinder(0.0, 0.0, boundary, linewidth=1.3)

    lim = boundary * 1.15
    ax.set(xlim=(-lim, lim), ylim=(-lim, lim), zlim=(z[0], z[-1]))
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.set_zlabel("schematic depth")
    ax.set_title("PCF-SPR Geometry — schematic, not COMSOL mesh")
    ax.view_init(elev=24, azim=38)
    fig.tight_layout()
    return fig


def parse_spectrum_row(row: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    def parse(value: Any, label: str) -> np.ndarray:
        if isinstance(value, str):
            array = np.fromstring(value, sep=",", dtype=float)
        else:
            array = np.asarray(value, dtype=float)
        if array.ndim != 1 or array.size < 2 or not np.all(np.isfinite(array)):
            raise ValueError(f"{label} must contain at least two finite values.")
        return array

    wavelength = parse(row["wavelength_nm"], "wavelength_nm")
    loss = parse(row["loss_db_per_cm"], "loss_db_per_cm")
    if wavelength.size != loss.size:
        raise ValueError("Wavelength and loss arrays must have the same length.")
    if not np.all(np.diff(wavelength) > 0):
        raise ValueError("wavelength_nm must be strictly increasing.")
    return wavelength, loss


def spectrum_figure(simulation: pd.DataFrame, target_id: int | None = None):
    frame = simulation.copy()
    if target_id is not None and "target_id" in frame.columns:
        frame = frame.loc[frame["target_id"].astype(int).eq(int(target_id))]
    frame = frame.loc[frame.get("status", "ok").eq("ok")] if "status" in frame.columns else frame
    if frame.empty:
        raise ValueError("No successful spectrum rows are available.")

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for _, row in frame.iterrows():
        wavelength, loss = parse_spectrum_row(row)
        label = f"RI={float(row['analyte_ri']):.5f}" if "analyte_ri" in row else None
        ax.plot(wavelength, loss, label=label)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Loss (dB/cm)")
    ax.set_title("Physics/Sensor Spectrum")
    if len(frame) <= 8:
        ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def xai_feature_summary(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in [*GEOMETRY_COLUMNS, *CONDITION_COLUMNS] if column not in frame.columns]
    if missing:
        raise ValueError(f"XAI attribution table is missing columns: {missing}")
    values = frame[[*GEOMETRY_COLUMNS, *CONDITION_COLUMNS]].astype(float)
    summary = (
        values.abs()
        .mean(axis=0)
        .sort_values(ascending=False)
        .rename("mean_absolute_attribution")
        .reset_index()
    )
    summary.columns = ["feature", "mean_absolute_attribution"]
    return summary


def evidence_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def research_report_markdown(
    target: Mapping[str, Any],
    selected: Mapping[str, Any] | None = None,
    verification: Mapping[str, Any] | None = None,
    *,
    backend: str | None = None,
    evidence: Mapping[str, str] | None = None,
) -> str:
    lines = [
        "# CyberPhotonics-SPR Dashboard Evidence Report",
        "",
        "## Design target",
        "",
        f"- Sensitivity: {float(target['sensitivity_nm_per_riu']):.6g} nm/RIU",
        f"- FOM: {float(target['fom_per_riu']):.6g} /RIU",
        f"- Resonance wavelength: {float(target['lambda_res_nm']):.6g} nm",
        f"- Analyte RI: {float(target['analyte_ri']):.6g}",
    ]
    if selected is not None:
        summary = selected_design_summary(selected)
        lines.extend(["", "## Selected inverse design", ""])
        for name, value in summary["geometry"].items():
            lines.append(f"- {name}: {value:.6g}")
        lines.extend(
            [
                f"- Pareto rank: {summary['pareto_rank']}",
                f"- Confidence score: {summary['confidence_score']:.6g}",
                f"- OOD score: {summary['ood_score']:.6g}",
                f"- In calibration domain: {summary['in_calibration_domain']}",
                f"- Fabrication projection distance: {summary['fabrication_projection_distance']:.6g}",
            ]
        )
    if verification is not None:
        lines.extend(["", "## Physics verification", ""])
        lines.extend(
            [
                f"- Backend: {backend or 'unknown'}",
                f"- Accepted: {bool(verification.get('accepted', False))}",
                f"- Reason: {verification.get('reason', '') or 'accepted'}",
                f"- Actual sensitivity: {float(verification.get('actual_sensitivity_nm_per_riu', np.nan)):.6g} nm/RIU",
                f"- Actual FOM: {float(verification.get('actual_fom_per_riu', np.nan)):.6g} /RIU",
                f"- Actual resonance: {float(verification.get('actual_lambda_res_nm', np.nan)):.6g} nm",
                f"- RI linearity R²: {float(verification.get('linearity_r2', np.nan)):.6g}",
            ]
        )
    if evidence:
        lines.extend(["", "## Evidence hashes", ""])
        for name, digest in sorted(evidence.items()):
            lines.append(f"- {name}: `{digest}`")
    lines.extend(
        [
            "",
            "## Evidence interpretation",
            "",
            (
                "COMSOL results may support physical simulation claims only when the backend above is `comsol` "
                "and the model/configuration are independently verified. Synthetic backend results validate "
                "software flow only and are not physical performance evidence."
            ),
            "",
        ]
    )
    return "\n".join(lines)
