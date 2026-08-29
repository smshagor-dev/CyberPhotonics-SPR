from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sprpcf.ml.dataset import GEOMETRY_COLUMNS
from sprpcf.simulation.dispersion import gold_permittivity_drude_lorentz, silica_refractive_index
from sprpcf.simulation.metrics import resonance_wavelength
from sprpcf.simulation.schema import Geometry
from sprpcf.simulation.synthetic import synthetic_loss_spectrum


# Kept for compatibility with the original research-dashboard API and tests.
DASHBOARD_TABS = [
    "Physics-Informed Inverse Design",
    "Explainable AI",
    "Active Learning",
    "Edge Denoising",
]

DEFAULT_HISTORY_PATH = Path("reports/dashboard_session_history.jsonl")


@dataclass(frozen=True)
class GeometryPrediction:
    pitch_um: float
    d_over_lambda: float
    metal_thickness_nm: float
    feasible: bool
    feasibility_reason: str

    def to_dict(self) -> dict[str, float | bool | str]:
        return asdict(self)


def control_center_path() -> Path:
    return Path(__file__).resolve().parents[1] / "dashboard" / "app.py"


def build_streamlit_command(port: int = 8501, host: str = "localhost") -> list[str]:
    return [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "--server.port",
        str(port),
        "--server.address",
        host,
        "--browser.gatherUsageStats",
        "false",
        str(control_center_path()),
    ]


def launch_dashboard(port: int = 8501, host: str = "localhost") -> subprocess.Popen[bytes]:
    return subprocess.Popen(build_streamlit_command(port=port, host=host))


def predict_inverse_geometry(
    target_sensitivity: float,
    target_fom: float,
    target_lambda_res_nm: float,
) -> GeometryPrediction:
    pitch_um = float(np.clip(1.2 + (target_lambda_res_nm - 520.0) / 240.0, 1.0, 3.2))
    d_over_lambda = float(np.clip(0.24 + target_sensitivity / 6000.0 + target_fom / 900.0, 0.22, 0.92))
    metal_thickness_nm = float(np.clip(24.0 + target_fom * 0.42 + (target_lambda_res_nm - 650.0) / 18.0, 10.0, 95.0))

    feasible = d_over_lambda <= 1.0 and 20.0 <= metal_thickness_nm <= 80.0
    if feasible:
        reason = "Feasible: d <= Lambda and 20 nm <= t_metal <= 80 nm."
    elif d_over_lambda > 1.0:
        reason = "Infeasible: d/Lambda exceeds the geometric loss constraint."
    else:
        reason = "Infeasible: gold layer thickness is outside 20-80 nm."
    return GeometryPrediction(pitch_um, d_over_lambda, metal_thickness_nm, feasible, reason)


def dispersion_curves(
    wavelength_min_nm: float = 450.0,
    wavelength_max_nm: float = 900.0,
    points: int = 180,
) -> pd.DataFrame:
    wavelengths_nm = np.linspace(wavelength_min_nm, wavelength_max_nm, points)
    wavelengths_um = wavelengths_nm / 1000.0
    epsilon_gold = gold_permittivity_drude_lorentz(wavelengths_um)
    silica_n = silica_refractive_index(wavelengths_um)
    return pd.DataFrame(
        {
            "wavelength_nm": wavelengths_nm,
            "epsilon_au_real": np.real(epsilon_gold),
            "epsilon_au_imag": np.imag(epsilon_gold),
            "silica_n": silica_n,
        }
    )


def synthetic_feature_importance(prediction: GeometryPrediction) -> pd.DataFrame:
    raw = np.asarray(
        [
            0.40 + 0.08 * prediction.pitch_um,
            0.55 + 0.45 * prediction.d_over_lambda,
            0.28 + 0.012 * prediction.metal_thickness_nm,
            0.32 + 0.06 * prediction.pitch_um,
        ],
        dtype=np.float32,
    )
    values = raw / raw.sum()
    return pd.DataFrame({"feature": GEOMETRY_COLUMNS, "importance": values})


def integrated_gradient_heatmap(prediction: GeometryPrediction, samples: int = 32) -> pd.DataFrame:
    x = np.linspace(-1.0, 1.0, samples, dtype=np.float32)
    centers = np.asarray(
        [prediction.pitch_um / 3.2, prediction.d_over_lambda, prediction.metal_thickness_nm / 80.0]
    )
    rows = []
    for feature, center in zip(GEOMETRY_COLUMNS, centers):
        attribution = np.tanh(2.0 * x + center) * (0.4 + center)
        for index, value in enumerate(attribution):
            rows.append({"sample": index, "feature": feature, "attribution": float(value)})
    return pd.DataFrame(rows)


def mc_dropout_uncertainty(
    prediction: GeometryPrediction,
    passes: int = 48,
    seed: int = 17,
) -> tuple[float, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    base = np.asarray(
        [prediction.pitch_um, prediction.d_over_lambda, prediction.metal_thickness_nm],
        dtype=np.float32,
    )
    scale = np.asarray([0.035, 0.018, 1.15], dtype=np.float32)
    samples = rng.normal(base, scale, size=(passes, 3))
    uncertainty = float(np.linalg.norm(samples.std(axis=0) / np.maximum(np.abs(base), 1e-6)))
    frame = pd.DataFrame(samples, columns=GEOMETRY_COLUMNS)
    frame["uncertainty"] = np.linalg.norm((samples - samples.mean(axis=0)) / scale, axis=1)
    return uncertainty, frame


def flag_uncertain_candidates(
    prediction: GeometryPrediction,
    count: int = 12,
    seed: int = 31,
) -> pd.DataFrame:
    _, samples = mc_dropout_uncertainty(prediction, passes=max(count * 4, 16), seed=seed)
    selected = samples.sort_values("uncertainty", ascending=False).head(count).copy()
    selected["mock_comsol_status"] = "queued"
    selected["candidate_id"] = [f"hil-candidate-{index:03d}" for index in range(len(selected))]
    return selected[["candidate_id", *GEOMETRY_COLUMNS, "uncertainty", "mock_comsol_status"]]


def build_edge_spectrum(
    noise_std: float,
    drift_std: float,
    wavelengths: int = 256,
    seed: int = 41,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    wavelength_nm = np.linspace(450.0, 900.0, wavelengths, dtype=np.float32)
    geometry = Geometry(
        d_over_lambda=0.54,
        pitch_um=2.05,
        metal_thickness_nm=48.0,
        analyte_ri=1.36,
        channel_radius_um=0.62,
    )
    clean = synthetic_loss_spectrum(geometry, wavelength_nm, rng, noise_std=0.0).astype(np.float32)
    drift = rng.normal(0.0, drift_std) * np.linspace(-1.0, 1.0, wavelengths, dtype=np.float32)
    noisy = clean + drift + rng.normal(0.0, noise_std, clean.shape).astype(np.float32)
    return wavelength_nm, clean, noisy.astype(np.float32)


def denoise_with_optional_tflite(
    noisy: np.ndarray,
    model_path: Path | None,
) -> tuple[np.ndarray, float, bool]:
    started = time.perf_counter()
    if model_path is not None and model_path.exists():
        try:
            from sprpcf.edge.quantization import TFLiteModelRunner

            denoised = TFLiteModelRunner(model_path).predict(noisy[None, :, None])[0, :, 0]
            return denoised.astype(np.float32), (time.perf_counter() - started) * 1000.0, True
        except Exception:
            pass
    kernel = np.ones(7, dtype=np.float32) / 7.0
    denoised = np.convolve(noisy, kernel, mode="same").astype(np.float32)
    return denoised, (time.perf_counter() - started) * 1000.0, False


def edge_latency_snapshot(noisy: np.ndarray, model_path: Path | None) -> dict[str, float | str | bool]:
    denoised, latency_ms, used_tflite = denoise_with_optional_tflite(noisy, model_path)
    fps = 1000.0 / max(latency_ms, 1e-6)
    return {
        "latency_ms": float(latency_ms),
        "fps": float(fps),
        "used_tflite": used_tflite,
        "denoised_mean": float(np.mean(denoised)),
    }


def append_session_history(event: dict[str, Any], path: Path = DEFAULT_HISTORY_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"timestamp": time.time(), **event}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def render_dashboard(tflite_dir: Path = Path("models")) -> None:
    # tflite_dir is retained for API compatibility. The unified app exposes the model directory in the sidebar.
    _ = tflite_dir
    from sprpcf.dashboard.app import main as render_control_center

    render_control_center()


def main() -> None:
    render_dashboard()


if __name__ == "__main__":
    main()
