from __future__ import annotations

import argparse
import json
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

DEFAULT_DATASET = Path("data/dataset.parquet")
DEFAULT_MODEL_DIR = Path("models")


def generate_data(samples: int, output: Path, wavelengths: int, seed: int) -> None:
    from sprpcf.simulation.comsol_sweep import write_dataset
    from sprpcf.simulation.synthetic import build_synthetic_dataset

    frame = build_synthetic_dataset(samples=samples, wavelengths=wavelengths, seed=seed)
    write_dataset(frame, output)
    print(f"Wrote {len(frame)} samples to {output}")


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
        latency_ms = (time.perf_counter() - started) * 1000.0
        latencies.append(latency_ms)
        psnr_values.append(psnr(clean[index : index + 1, :, None], denoised))
        ri_errors.append(float(abs(prediction[0] - targets[index, 0])))
        frames += 1

    elapsed = max(time.perf_counter() - started_all, 1e-9)
    stats = {
        "frames": float(frames),
        "average_denoising_psnr": float(np.mean(psnr_values)),
        "predicted_ri_mae": float(np.mean(ri_errors)),
        "average_latency_ms": float(np.mean(latencies)),
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


def launch_dashboard(args: argparse.Namespace) -> None:
    from sprpcf.ui.dashboard import build_streamlit_command

    command = build_streamlit_command(port=args.port, host=args.host)
    print(f"Launching dashboard at http://{args.host}:{args.port}")
    os = __import__("os")
    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    subprocess = __import__("subprocess")
    subprocess.run(command, check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PCF-SPR inverse design and edge deployment orchestrator.")
    subparsers = parser.add_subparsers(dest="command")

    generate = subparsers.add_parser("generate-data", help="Generate synthetic PCF-SPR training data.")
    generate.add_argument("--samples", type=int, default=1000)
    generate.add_argument("--wavelengths", type=int, default=256)
    generate.add_argument("--seed", type=int, default=7)
    generate.add_argument("--out", type=Path, default=DEFAULT_DATASET)
    generate.set_defaults(func=lambda args: generate_data(args.samples, args.out, args.wavelengths, args.seed))

    inverse = subparsers.add_parser("train-inverse", help="Train tandem inverse model and export ONNX.")
    inverse.add_argument("--data", type=Path, default=DEFAULT_DATASET)
    inverse.add_argument("--epochs", type=int, default=100)
    inverse.add_argument("--forward-epochs", type=int, default=None)
    inverse.add_argument("--inverse-epochs", type=int, default=None)
    inverse.add_argument("--batch-size", type=int, default=64)
    inverse.add_argument("--lr", type=float, default=1e-3)
    inverse.add_argument("--device", default="auto")
    inverse.add_argument("--alpha", type=float, default=1.0)
    inverse.add_argument("--beta", type=float, default=1e-3)
    inverse.add_argument("--dispersion-weight", type=float, default=0.0)
    inverse.add_argument("--checkpoint", type=Path, default=DEFAULT_MODEL_DIR / "tandem.pt")
    inverse.add_argument("--export-onnx", type=Path, default=DEFAULT_MODEL_DIR / "inverse_pcf_spr.onnx")
    inverse.set_defaults(func=train_inverse)

    edge = subparsers.add_parser("train-edge", help="Train edge denoiser and RI predictor.")
    edge.add_argument("--data", type=Path, default=DEFAULT_DATASET)
    edge.add_argument("--epochs", type=int, default=50)
    edge.add_argument("--batch-size", type=int, default=64)
    edge.add_argument("--device", default="auto")
    edge.add_argument("--quantize", action="store_true")
    edge.add_argument("--export-dir", type=Path, default=DEFAULT_MODEL_DIR)
    edge.set_defaults(func=train_edge)

    pipeline = subparsers.add_parser("run-pipeline", help="Run A -> B -> C end to end.")
    pipeline.add_argument("--full", action="store_true")
    pipeline.add_argument("--samples", type=int, default=1000)
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

    stream = subparsers.add_parser("simulate-stream", help="Benchmark quantized TFLite models on streaming spectra.")
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

    dashboard = subparsers.add_parser("dashboard", help="Launch the Streamlit web dashboard.")
    dashboard.add_argument("--port", type=int, default=8501)
    dashboard.add_argument("--host", default="localhost")
    dashboard.set_defaults(func=launch_dashboard)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
