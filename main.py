from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if SRC_ROOT.exists() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

DEFAULT_DATASET = Path("data/processed/synthetic.parquet")
DEFAULT_MODEL_DIR = Path("models")

BACKEND_COMMAND_MODULES: dict[str, tuple[str, ...]] = {
    "generate-data": ("sprpcf", "numpy", "pandas", "pyarrow"),
    "train-inverse": ("sprpcf", "numpy", "pandas", "pyarrow", "torch"),
    "train-edge": ("sprpcf", "numpy", "pandas", "pyarrow", "tensorflow"),
    "run-pipeline": ("sprpcf", "numpy", "pandas", "pyarrow", "torch", "tensorflow"),
    "simulate-stream": ("sprpcf", "numpy", "pandas", "pyarrow", "tensorflow"),
    "hil-benchmark": ("sprpcf", "numpy", "tensorflow"),
    "design-sensor": ("sprpcf", "numpy", "pandas", "torch"),
    "verify-physics": ("sprpcf", "numpy", "pandas", "torch"),
    "generate-report": ("sprpcf", "numpy", "pandas"),
    "web-dashboard": ("sprpcf", "streamlit"),
}


def _project_env() -> dict[str, str]:
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(SRC_ROOT) if not pythonpath else f"{SRC_ROOT}{os.pathsep}{pythonpath}"
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return env


def _python_can_import(python_executable: Path | str, modules: tuple[str, ...]) -> bool:
    statements = "; ".join(f"import {module}" for module in modules)
    return (
        subprocess.call(
            [str(python_executable), "-c", statements],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=_project_env(),
        )
        == 0
    )


def _preferred_backend_python(modules: tuple[str, ...]) -> str:
    venv311_python = PROJECT_ROOT / ".venv311" / "Scripts" / "python.exe"
    if venv311_python.exists() and _python_can_import(venv311_python, modules):
        return str(venv311_python)
    return sys.executable


def _maybe_reexec_backend(argv: list[str] | None) -> None:
    if argv is not None:
        return
    if "-h" in sys.argv[1:] or "--help" in sys.argv[1:]:
        return
    command = next((arg for arg in sys.argv[1:] if not arg.startswith("-")), None)
    modules = BACKEND_COMMAND_MODULES.get(command or "")
    if modules is None or _python_can_import(sys.executable, modules):
        return
    preferred = _preferred_backend_python(modules)
    if Path(preferred).resolve() == Path(sys.executable).resolve():
        return
    completed = subprocess.run([preferred, *sys.argv], env=_project_env())
    raise SystemExit(completed.returncode)


def generate_data(samples: int, output: Path, wavelengths: int, seed: int) -> None:
    from sprpcf.simulation.comsol_sweep import write_dataset
    from sprpcf.simulation.synthetic import DEFAULT_ANALYTE_RI, build_synthetic_dataset

    frame = build_synthetic_dataset(samples=samples, wavelengths=wavelengths, seed=seed)
    write_dataset(
        frame,
        output,
        metadata={
            "source": "synthetic",
            "seed": seed,
            "base_geometries": samples,
            "wavelength_samples": wavelengths,
            "analyte_ri_values": list(DEFAULT_ANALYTE_RI),
        },
    )
    print(f"Wrote {len(frame)} rows ({samples} base geometries) to {output}")


def train_inverse(args: argparse.Namespace) -> None:
    from sprpcf.ml.train_tandem import train_tandem_pipeline

    metrics = train_tandem_pipeline(
        data_path=args.data,
        checkpoint_out=args.checkpoint,
        onnx_out=args.export_onnx,
        epochs=args.epochs,
        forward_epochs=args.forward_epochs,
        inverse_epochs=args.inverse_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device_name=args.device,
        alpha=args.alpha,
        beta=args.beta,
        dispersion_weight=args.dispersion_weight,
        seed=args.seed,
    )
    print(json.dumps(metrics, indent=2))


def train_edge(args: argparse.Namespace) -> None:
    from sprpcf.edge.train_denoiser import train_edge_models

    export_dir = args.export_dir
    metrics = train_edge_models(
        data_path=args.data,
        denoiser_out=export_dir / "edge_denoiser.keras",
        predictor_out=export_dir / "edge_ri_predictor.keras",
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        quantize=args.quantize,
        denoiser_tflite_out=export_dir / "edge_denoiser_quantized.tflite",
        predictor_tflite_out=export_dir / "edge_ri_predictor_quantized.tflite",
        seed=args.seed,
    )
    print(json.dumps(metrics, indent=2))


def simulate_stream(
    data_path: Path,
    tflite_dir: Path,
    duration_sec: float,
    noise_std: float,
    drift_std: float,
) -> dict[str, float]:
    import numpy as np

    from sprpcf.edge.quantization import TFLiteModelRunner
    from sprpcf.edge.train_denoiser import add_sensor_noise, normalize_spectra, parse_spectra, psnr
    from sprpcf.ml.dataset import read_table

    denoiser_path = tflite_dir / "edge_denoiser_quantized.tflite"
    predictor_path = tflite_dir / "edge_ri_predictor_quantized.tflite"
    if not denoiser_path.exists():
        raise FileNotFoundError(f"Missing denoiser TFLite model: {denoiser_path}")
    if not predictor_path.exists():
        raise FileNotFoundError(f"Missing RI predictor TFLite model: {predictor_path}")

    frame = read_table(data_path).dropna(subset=["loss_db_per_cm", "analyte_ri", "lambda_res_nm"])
    clean, _, _ = normalize_spectra(parse_spectra(frame))
    targets = frame[["analyte_ri", "lambda_res_nm"]].to_numpy(np.float32)
    noisy = add_sensor_noise(clean, noise_std=noise_std, drift_std=drift_std, seed=19)

    denoiser = TFLiteModelRunner(denoiser_path)
    predictor = TFLiteModelRunner(predictor_path)
    deadline = time.perf_counter() + duration_sec
    latencies: list[float] = []
    psnr_values: list[float] = []
    ri_errors: list[float] = []
    frames = 0
    started_all = time.perf_counter()

    while time.perf_counter() < deadline or frames == 0:
        index = frames % clean.shape[0]
        started = time.perf_counter()
        denoised = denoiser.predict(noisy[index : index + 1, :, None])
        prediction = predictor.predict(denoised)[0]
        latencies.append((time.perf_counter() - started) * 1000.0)
        psnr_values.append(psnr(clean[index : index + 1, :, None], denoised))
        ri_errors.append(float(abs(prediction[0] - targets[index, 0])))
        frames += 1

    elapsed = max(time.perf_counter() - started_all, 1e-9)
    stats = {
        "frames": float(frames),
        "average_denoising_psnr": float(np.mean(psnr_values)),
        "predicted_ri_mae": float(np.mean(ri_errors)),
        "average_latency_ms": float(np.mean(latencies)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "fps": float(frames / elapsed),
    }
    print(json.dumps(stats, indent=2))
    return stats


def run_pipeline(args: argparse.Namespace) -> None:
    from sprpcf.edge.train_denoiser import train_edge_models
    from sprpcf.ml.train_tandem import train_tandem_pipeline

    args.export_dir.mkdir(parents=True, exist_ok=True)
    generate_data(args.samples, args.data, args.wavelengths, args.seed)
    inverse_metrics = train_tandem_pipeline(
        data_path=args.data,
        checkpoint_out=args.export_dir / "tandem.pt",
        onnx_out=args.export_dir / "inverse_pcf_spr.onnx",
        epochs=args.inverse_epochs,
        batch_size=args.batch_size,
        device_name=args.device,
        dispersion_weight=args.dispersion_weight,
        seed=args.seed,
    )
    edge_metrics = train_edge_models(
        data_path=args.data,
        denoiser_out=args.export_dir / "edge_denoiser.keras",
        predictor_out=args.export_dir / "edge_ri_predictor.keras",
        epochs=args.edge_epochs,
        batch_size=args.batch_size,
        device=args.edge_device,
        quantize=True,
        denoiser_tflite_out=args.export_dir / "edge_denoiser_quantized.tflite",
        predictor_tflite_out=args.export_dir / "edge_ri_predictor_quantized.tflite",
        seed=args.seed,
    )
    stream_metrics = simulate_stream(args.data, args.export_dir, args.duration_sec, args.noise_std, args.drift_std)
    print(json.dumps({"inverse": inverse_metrics, "edge": edge_metrics, "stream": stream_metrics}, indent=2))


def run_hil_benchmark(args: argparse.Namespace) -> None:
    from sprpcf.edge.hil_engine import HILBenchmarkEngine
    from sprpcf.edge.train_denoiser import normalize_spectra, parse_spectra
    from sprpcf.simulation.synthetic import build_synthetic_dataset

    frame = build_synthetic_dataset(samples=args.samples, wavelengths=args.wavelengths, seed=args.seed)
    spectra, _, _ = normalize_spectra(parse_spectra(frame))
    engine = HILBenchmarkEngine(
        clean_spectra=spectra,
        protocol=args.protocol,
        buffer_size=args.buffer_size,
        seed=args.seed,
        serial_port=args.serial_port,
        baudrate=args.baudrate,
        socket_host=args.socket_host,
        socket_port=args.socket_port,
    )
    report = engine.benchmark(
        tflite_dir=args.tflite_dir,
        duration_sec=args.duration,
        inject_thermal_drift=args.inject_thermal_drift,
        fps=args.fps,
        report_path=args.report,
    )
    print(engine.format_ascii_summary(report))
    print(f"\nWrote benchmark report to {args.report}")


def design_sensor(args: argparse.Namespace) -> None:
    from sprpcf.dashboard.core import target_frame
    from sprpcf.ml.multiobjective import optimize_target_table

    target = target_frame(args.sensitivity, args.fom, args.lambda_res, args.analyte_ri)
    result = optimize_target_table(
        args.checkpoint,
        target,
        args.data,
        candidates_per_target=args.candidates,
        confidence=args.confidence,
        latent_scale=args.latent_scale,
        seed=args.seed,
        device=args.device,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    target.to_csv(args.out / "design_target.csv", index=False)
    result.candidates.to_csv(args.out / "pareto_candidates.csv", index=False)
    result.selected.to_csv(args.out / "pareto_selected_designs.csv", index=False)
    (args.out / "design_calibration.json").write_text(
        json.dumps(result.calibration, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"Generated {len(result.candidates)} candidates; selected {len(result.selected)} design(s).")
    print(f"Wrote design evidence to {args.out}")


def verify_physics(args: argparse.Namespace) -> None:
    import pandas as pd

    from sprpcf.ml.dataset import CONDITION_COLUMNS, METRIC_COLUMNS
    from sprpcf.validation.closed_loop import AcceptanceThresholds, run_closed_loop_iteration

    selected = pd.read_csv(args.selected)
    if selected.empty:
        raise ValueError("Selected design CSV is empty.")
    required = [*METRIC_COLUMNS, *CONDITION_COLUMNS]
    missing = [column for column in required if column not in selected.columns]
    if missing:
        raise ValueError(f"Selected design is missing target columns: {missing}")
    target = selected[required].head(1).copy()
    args.out.mkdir(parents=True, exist_ok=True)
    target_path = args.out / "verification_target.csv"
    target.to_csv(target_path, index=False)

    fixed = selected.head(1).copy()

    def fixed_designer(_target: pd.DataFrame) -> pd.DataFrame:
        return fixed.copy()

    model_path = Path(args.model) if args.model else None
    config_path = Path(args.config) if args.config else None
    artifacts = run_closed_loop_iteration(
        checkpoint_path=args.checkpoint,
        target_path=target_path,
        base_dataset_path=args.data,
        output_dir=args.out,
        backend=args.backend,
        model_path=model_path,
        config_path=config_path,
        ri_span=args.ri_span,
        ri_points=args.ri_points,
        thresholds=AcceptanceThresholds(),
        designer=fixed_designer,
        retrain=False,
        seed=args.seed,
    )
    print(json.dumps({
        "backend": artifacts.backend,
        "selected_targets": artifacts.selected_targets,
        "accepted_targets": artifacts.accepted_targets,
        "verification": str(artifacts.verification_results),
        "simulation": str(artifacts.simulation_results),
    }, indent=2))


def generate_report(args: argparse.Namespace) -> None:
    import pandas as pd

    from sprpcf.dashboard.core import research_report_markdown, target_frame

    selected_frame = pd.read_csv(args.selected)
    if selected_frame.empty:
        raise ValueError("Selected design CSV is empty.")
    selected = selected_frame.iloc[0]
    target = target_frame(
        float(selected["sensitivity_nm_per_riu"]),
        float(selected["fom_per_riu"]),
        float(selected["lambda_res_nm"]),
        float(selected["analyte_ri"]),
    ).iloc[0]
    verification = None
    if args.verification.exists():
        verification_frame = pd.read_csv(args.verification)
        if not verification_frame.empty:
            verification = verification_frame.iloc[0]
    backend = "synthetic"
    report = research_report_markdown(target, selected, verification, backend=backend, evidence={})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print(f"Wrote research evidence report to {args.out}")


def launch_desktop(_args: argparse.Namespace | None = None) -> None:
    from sprpcf.desktop import launch_desktop as run_desktop

    raise SystemExit(run_desktop())


def launch_web_dashboard(args: argparse.Namespace) -> None:
    from sprpcf.ui.dashboard import build_streamlit_command

    command = build_streamlit_command(port=args.port, host=args.host)
    print(f"Launching legacy web dashboard at http://{args.host}:{args.port}")
    os = __import__("os")
    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    subprocess = __import__("subprocess")
    subprocess.run(command, check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CyberPhotonics-SPR orchestrator. Run without a subcommand to open the native desktop GUI."
    )
    subparsers = parser.add_subparsers(dest="command")

    generate = subparsers.add_parser("generate-data", help="Generate fixed-geometry RI-sweep synthetic data.")
    generate.add_argument("--samples", type=int, default=100, help="Number of base geometries; each gets a five-point RI sweep.")
    generate.add_argument("--wavelengths", type=int, default=256)
    generate.add_argument("--seed", type=int, default=7)
    generate.add_argument("--out", type=Path, default=DEFAULT_DATASET)
    generate.set_defaults(func=lambda args: generate_data(args.samples, args.out, args.wavelengths, args.seed))

    inverse = subparsers.add_parser("train-inverse", help="Train conditioned tandem inverse model and export ONNX.")
    inverse.add_argument("--data", type=Path, default=DEFAULT_DATASET)
    inverse.add_argument("--epochs", type=int, default=100)
    inverse.add_argument("--forward-epochs", type=int, default=None)
    inverse.add_argument("--inverse-epochs", type=int, default=None)
    inverse.add_argument("--batch-size", type=int, default=64)
    inverse.add_argument("--lr", type=float, default=1e-3)
    inverse.add_argument("--device", default="auto")
    inverse.add_argument("--alpha", type=float, default=1.0)
    inverse.add_argument("--beta", type=float, default=1.0)
    inverse.add_argument("--dispersion-weight", type=float, default=0.0)
    inverse.add_argument("--seed", type=int, default=7)
    inverse.add_argument("--checkpoint", type=Path, default=DEFAULT_MODEL_DIR / "tandem.pt")
    inverse.add_argument("--export-onnx", type=Path, default=DEFAULT_MODEL_DIR / "inverse_pcf_spr.onnx")
    inverse.set_defaults(func=train_inverse)

    edge = subparsers.add_parser("train-edge", help="Train edge denoiser and RI predictor.")
    edge.add_argument("--data", type=Path, default=DEFAULT_DATASET)
    edge.add_argument("--epochs", type=int, default=50)
    edge.add_argument("--batch-size", type=int, default=64)
    edge.add_argument("--device", default="auto")
    edge.add_argument("--quantize", action="store_true")
    edge.add_argument("--seed", type=int, default=7)
    edge.add_argument("--export-dir", type=Path, default=DEFAULT_MODEL_DIR)
    edge.set_defaults(func=train_edge)

    pipeline = subparsers.add_parser("run-pipeline", help="Run synthetic data -> inverse model -> edge model -> benchmark.")
    pipeline.add_argument("--full", action="store_true")
    pipeline.add_argument("--samples", type=int, default=100)
    pipeline.add_argument("--wavelengths", type=int, default=256)
    pipeline.add_argument("--seed", type=int, default=7)
    pipeline.add_argument("--data", type=Path, default=DEFAULT_DATASET)
    pipeline.add_argument("--export-dir", type=Path, default=DEFAULT_MODEL_DIR)
    pipeline.add_argument("--inverse-epochs", type=int, default=100)
    pipeline.add_argument("--edge-epochs", type=int, default=50)
    pipeline.add_argument("--batch-size", type=int, default=64)
    pipeline.add_argument("--device", default="auto")
    pipeline.add_argument("--dispersion-weight", type=float, default=0.0)
    pipeline.add_argument("--edge-device", default="auto")
    pipeline.add_argument("--duration-sec", type=float, default=10.0)
    pipeline.add_argument("--noise-std", type=float, default=0.08)
    pipeline.add_argument("--drift-std", type=float, default=0.03)
    pipeline.set_defaults(func=run_pipeline)

    stream = subparsers.add_parser("simulate-stream", help="Benchmark quantized TFLite models on stored spectra.")
    stream.add_argument("--data", type=Path, default=DEFAULT_DATASET)
    stream.add_argument("--tflite-dir", type=Path, default=DEFAULT_MODEL_DIR)
    stream.add_argument("--duration-sec", type=float, default=10.0)
    stream.add_argument("--noise-std", type=float, default=0.08)
    stream.add_argument("--drift-std", type=float, default=0.03)
    stream.set_defaults(func=lambda args: simulate_stream(args.data, args.tflite_dir, args.duration_sec, args.noise_std, args.drift_std))

    hil = subparsers.add_parser("hil-benchmark", help="Run Phase 4 HIL and edge hardware benchmarking.")
    hil.add_argument("--tflite-dir", type=Path, default=DEFAULT_MODEL_DIR)
    hil.add_argument("--duration", type=float, default=30.0)
    hil.add_argument("--inject-thermal-drift", action="store_true")
    hil.add_argument("--report", type=Path, default=Path("reports/phase4_hil_benchmark.json"))
    hil.add_argument("--protocol", choices=["mock", "serial", "socket"], default="mock")
    hil.add_argument("--fps", type=float, default=30.0)
    hil.add_argument("--samples", type=int, default=256)
    hil.add_argument("--wavelengths", type=int, default=256)
    hil.add_argument("--seed", type=int, default=23)
    hil.add_argument("--buffer-size", type=int, default=256)
    hil.add_argument("--serial-port", default=None)
    hil.add_argument("--baudrate", type=int, default=115200)
    hil.add_argument("--socket-host", default="127.0.0.1")
    hil.add_argument("--socket-port", type=int, default=9000)
    hil.set_defaults(func=run_hil_benchmark)

    design = subparsers.add_parser("design-sensor", help=argparse.SUPPRESS)
    design.add_argument("--checkpoint", type=Path, default=DEFAULT_MODEL_DIR / "tandem.pt")
    design.add_argument("--data", type=Path, default=DEFAULT_DATASET)
    design.add_argument("--sensitivity", type=float, default=800.0)
    design.add_argument("--fom", type=float, default=20.0)
    design.add_argument("--lambda-res", type=float, default=750.0)
    design.add_argument("--analyte-ri", type=float, default=1.37)
    design.add_argument("--candidates", type=int, default=128)
    design.add_argument("--confidence", type=float, default=0.95)
    design.add_argument("--latent-scale", type=float, default=0.10)
    design.add_argument("--seed", type=int, default=7)
    design.add_argument("--device", default="cpu")
    design.add_argument("--out", type=Path, default=Path("outputs/dashboard/design"))
    design.set_defaults(func=design_sensor)

    verify = subparsers.add_parser("verify-physics", help=argparse.SUPPRESS)
    verify.add_argument("--checkpoint", type=Path, default=DEFAULT_MODEL_DIR / "tandem.pt")
    verify.add_argument("--data", type=Path, default=DEFAULT_DATASET)
    verify.add_argument("--selected", type=Path, default=Path("outputs/dashboard/design/pareto_selected_designs.csv"))
    verify.add_argument("--backend", choices=["synthetic", "comsol"], default="synthetic")
    verify.add_argument("--ri-span", type=float, default=0.04)
    verify.add_argument("--ri-points", type=int, default=5)
    verify.add_argument("--seed", type=int, default=7)
    verify.add_argument("--out", type=Path, default=Path("outputs/dashboard/physics"))
    verify.add_argument("--model", default="")
    verify.add_argument("--config", default="")
    verify.set_defaults(func=verify_physics)

    report = subparsers.add_parser("generate-report", help=argparse.SUPPRESS)
    report.add_argument("--selected", type=Path, default=Path("outputs/dashboard/design/pareto_selected_designs.csv"))
    report.add_argument("--verification", type=Path, default=Path("outputs/dashboard/physics/verification.csv"))
    report.add_argument("--out", type=Path, default=Path("outputs/dashboard/dashboard_evidence_report.md"))
    report.set_defaults(func=generate_report)

    desktop = subparsers.add_parser("gui", help="Launch the native desktop control center.")
    desktop.set_defaults(func=launch_desktop)
    dashboard = subparsers.add_parser("dashboard", help="Launch the native desktop control center.")
    dashboard.set_defaults(func=launch_desktop)

    web = subparsers.add_parser("web-dashboard", help="Launch the legacy Streamlit web dashboard.")
    web.add_argument("--port", type=int, default=8501)
    web.add_argument("--host", default="localhost")
    web.set_defaults(func=launch_web_dashboard)
    return parser


def main(argv: list[str] | None = None) -> None:
    _maybe_reexec_backend(argv)
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        launch_desktop()
        return
    args.func(args)


if __name__ == "__main__":
    main()
