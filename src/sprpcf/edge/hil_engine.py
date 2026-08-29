from __future__ import annotations

import json
import socket
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Literal

import numpy as np

from sprpcf.edge.profiler import BenchmarkMetrics, EdgeHardwareProfiler


ProtocolName = Literal["mock", "serial", "socket"]


@dataclass
class HILFrame:
    """One frame received through the HIL bridge."""

    sequence_id: int
    spectrum: np.ndarray
    temperature_c: float
    baseline_drift: float
    received_at: float


@dataclass
class ThermalDriftReport:
    """Quantifies how strongly the stream changed under thermal drift."""

    enabled: bool
    temperature_start_c: float
    temperature_end_c: float
    max_baseline_drift: float
    mean_absolute_signal_shift: float
    tolerance_score: float


@dataclass
class HILBenchmarkReport:
    """Structured Phase 4 benchmark report."""

    duration_sec: float
    protocol: str
    frames_received: int
    frames_lost: int
    buffer_capacity: int
    buffer_occupancy: int
    thermal_drift: ThermalDriftReport
    models: dict[str, dict[str, float | int | str]]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["thermal_drift"] = asdict(self.thermal_drift)
        return payload


class HILBenchmarkEngine:
    """Hardware-in-the-loop stream bridge and benchmark harness.

    The engine can consume deterministic mock spectral feeds for CI, plus
    simple line-delimited Serial or Socket feeds from hardware. Hardware lines
    are expected as CSV or JSON with a sequence id and spectral values.
    """

    def __init__(
        self,
        clean_spectra: np.ndarray,
        protocol: ProtocolName = "mock",
        buffer_size: int = 256,
        seed: int = 23,
        serial_port: str | None = None,
        baudrate: int = 115200,
        socket_host: str = "127.0.0.1",
        socket_port: int = 9000,
    ) -> None:
        self.clean_spectra = np.asarray(clean_spectra, dtype=np.float32)
        if self.clean_spectra.ndim != 2:
            raise ValueError("clean_spectra must have shape (frames, wavelengths).")
        self.protocol = protocol
        self.buffer: deque[HILFrame] = deque(maxlen=buffer_size)
        self.rng = np.random.default_rng(seed)
        self.serial_port = serial_port
        self.baudrate = baudrate
        self.socket_host = socket_host
        self.socket_port = socket_port
        self.frames_received = 0
        self.frames_lost = 0
        self.overwritten_frames = 0
        self._last_sequence_id: int | None = None

    @property
    def buffer_capacity(self) -> int:
        return self.buffer.maxlen or 0

    def stream_frames(
        self,
        duration_sec: float,
        fps: float = 30.0,
        inject_thermal_drift: bool = False,
    ) -> Iterator[HILFrame]:
        source = self._hardware_source() if self.protocol != "mock" else self._mock_source(duration_sec, fps)
        deadline = time.perf_counter() + max(duration_sec, 0.0)
        for raw_frame in source:
            if self.protocol != "mock" and time.perf_counter() > deadline:
                break
            frame = self._with_environment(raw_frame, inject_thermal_drift)
            self._record_frame(frame)
            yield frame

    def collect_stream(
        self,
        duration_sec: float,
        fps: float = 30.0,
        inject_thermal_drift: bool = False,
    ) -> list[HILFrame]:
        return list(self.stream_frames(duration_sec, fps=fps, inject_thermal_drift=inject_thermal_drift))

    def benchmark(
        self,
        tflite_dir: Path,
        duration_sec: float,
        inject_thermal_drift: bool = False,
        fps: float = 30.0,
        report_path: Path = Path("reports/phase4_hil_benchmark.json"),
    ) -> HILBenchmarkReport:
        frames = self.collect_stream(duration_sec, fps=fps, inject_thermal_drift=inject_thermal_drift)
        spectra = np.asarray([frame.spectrum for frame in frames], dtype=np.float32)
        thermal = self.evaluate_thermal_drift(frames)
        profiler = EdgeHardwareProfiler()
        models = self._benchmark_available_models(profiler, tflite_dir, spectra, len(frames))
        report = HILBenchmarkReport(
            duration_sec=float(duration_sec),
            protocol=self.protocol,
            frames_received=self.frames_received,
            frames_lost=self.frames_lost,
            buffer_capacity=self.buffer_capacity,
            buffer_occupancy=len(self.buffer),
            thermal_drift=thermal,
            models=models,
        )
        self.write_report(report, report_path)
        return report

    def evaluate_thermal_drift(self, frames: list[HILFrame]) -> ThermalDriftReport:
        if not frames:
            return ThermalDriftReport(False, 0.0, 0.0, 0.0, 0.0, 1.0)

        temperatures = np.asarray([frame.temperature_c for frame in frames], dtype=np.float32)
        baselines = np.asarray([frame.baseline_drift for frame in frames], dtype=np.float32)
        spectra = np.asarray([frame.spectrum for frame in frames], dtype=np.float32)
        references = self.clean_spectra[[frame.sequence_id % self.clean_spectra.shape[0] for frame in frames]]
        shift = float(np.mean(np.abs(spectra - references)))
        dynamic_range = float(np.ptp(references) + 1e-6)
        tolerance_score = float(np.clip(1.0 - shift / dynamic_range, 0.0, 1.0))
        return ThermalDriftReport(
            enabled=bool(np.ptp(temperatures) > 1e-3 or np.max(np.abs(baselines)) > 1e-9),
            temperature_start_c=float(temperatures[0]),
            temperature_end_c=float(temperatures[-1]),
            max_baseline_drift=float(np.max(np.abs(baselines))),
            mean_absolute_signal_shift=shift,
            tolerance_score=tolerance_score,
        )

    def format_ascii_summary(self, report: HILBenchmarkReport) -> str:
        lines = [
            "Phase 4 HIL Benchmark Summary",
            "+----------------------+----------------------+",
            f"| Protocol             | {report.protocol:<20} |",
            f"| Frames Received      | {report.frames_received:<20} |",
            f"| Frames Lost          | {report.frames_lost:<20} |",
            f"| Buffer Occupancy     | {report.buffer_occupancy}/{report.buffer_capacity:<18} |",
            f"| Thermal Tolerance    | {report.thermal_drift.tolerance_score:<20.4f} |",
            "+----------------------+----------------------+",
            "",
            "+----------+--------+------------+----------+----------+----------+",
            "| Model    | Frames | Lat ms avg | P95 ms   | FPS      | RAM MB   |",
            "+----------+--------+------------+----------+----------+----------+",
        ]
        for label, metrics in report.models.items():
            lines.append(
                f"| {label:<8} | {int(metrics['frames_processed']):<6} | "
                f"{float(metrics['average_latency_ms']):<10.3f} | "
                f"{float(metrics['p95_latency_ms']):<8.3f} | "
                f"{float(metrics['throughput_fps']):<8.2f} | "
                f"{float(metrics['peak_ram_mb']):<8.2f} |"
            )
        lines.append("+----------+--------+------------+----------+----------+----------+")
        return "\n".join(lines)

    def write_report(self, report: HILBenchmarkReport, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

    def _benchmark_available_models(
        self,
        profiler: EdgeHardwareProfiler,
        tflite_dir: Path,
        spectra: np.ndarray,
        expected_frames: int,
    ) -> dict[str, dict[str, float | int | str]]:
        models: dict[str, dict[str, float | int | str]] = {}
        candidates = {
            "int8": tflite_dir / "edge_denoiser_quantized.tflite",
            "fp32": tflite_dir / "edge_denoiser_fp32.tflite",
        }
        for label, path in candidates.items():
            if not path.exists():
                continue
            metrics, _ = profiler.benchmark_model(path, spectra, expected_frames=expected_frames)
            models[label] = metrics.to_dict()
        if not models:
            raise FileNotFoundError(
                f"No denoiser TFLite model found in {tflite_dir}. "
                "Expected edge_denoiser_quantized.tflite or edge_denoiser_fp32.tflite."
            )
        return models

    def _record_frame(self, frame: HILFrame) -> None:
        if self._last_sequence_id is not None and frame.sequence_id > self._last_sequence_id + 1:
            self.frames_lost += frame.sequence_id - self._last_sequence_id - 1
        self._last_sequence_id = frame.sequence_id
        self.frames_received += 1
        if len(self.buffer) == self.buffer_capacity:
            self.overwritten_frames += 1
        self.buffer.append(frame)

    def _with_environment(self, frame: HILFrame, inject_thermal_drift: bool) -> HILFrame:
        progress = self._thermal_progress(frame.sequence_id)
        temperature = 20.0 + 30.0 * progress if inject_thermal_drift else 20.0
        baseline_drift = 0.08 * progress if inject_thermal_drift else 0.0
        thermal_noise = self.rng.normal(0.0, 0.003 + 0.0008 * (temperature - 20.0), frame.spectrum.shape)
        slope = np.linspace(-0.5, 0.5, frame.spectrum.size, dtype=np.float32)
        drifted = frame.spectrum + baseline_drift + slope * baseline_drift * 0.35 + thermal_noise
        return HILFrame(
            sequence_id=frame.sequence_id,
            spectrum=drifted.astype(np.float32),
            temperature_c=float(temperature),
            baseline_drift=float(baseline_drift),
            received_at=frame.received_at,
        )

    def _thermal_progress(self, sequence_id: int) -> float:
        denominator = max(self.clean_spectra.shape[0] - 1, 1)
        return float(np.clip((sequence_id % self.clean_spectra.shape[0]) / denominator, 0.0, 1.0))

    def _mock_source(self, duration_sec: float, fps: float) -> Iterator[HILFrame]:
        frame_count = max(int(round(duration_sec * fps)), 1)
        for sequence_id in range(frame_count):
            clean = self.clean_spectra[sequence_id % self.clean_spectra.shape[0]]
            yield HILFrame(
                sequence_id=sequence_id,
                spectrum=clean.copy(),
                temperature_c=20.0,
                baseline_drift=0.0,
                received_at=time.perf_counter(),
            )

    def _hardware_source(self) -> Iterator[HILFrame]:
        if self.protocol == "serial":
            yield from self._serial_source()
        elif self.protocol == "socket":
            yield from self._socket_source()
        else:
            raise ValueError(f"Unsupported HIL protocol: {self.protocol}")

    def _serial_source(self) -> Iterator[HILFrame]:
        try:
            import serial
        except ImportError as exc:  # pragma: no cover - optional hardware path
            raise RuntimeError("Serial HIL requires pyserial to be installed.") from exc
        if self.serial_port is None:
            raise ValueError("serial_port is required for serial HIL.")
        with serial.Serial(self.serial_port, self.baudrate, timeout=1.0) as device:
            while True:
                line = device.readline().decode("utf-8", errors="ignore").strip()
                if line:
                    yield self._parse_hardware_line(line)

    def _socket_source(self) -> Iterator[HILFrame]:
        with socket.create_connection((self.socket_host, self.socket_port), timeout=5.0) as sock:
            buffer = ""
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                buffer += chunk.decode("utf-8", errors="ignore")
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if line:
                        yield self._parse_hardware_line(line)

    def _parse_hardware_line(self, line: str) -> HILFrame:
        if line.startswith("{"):
            payload = json.loads(line)
            sequence_id = int(payload.get("sequence_id", payload.get("index", 0)))
            spectrum = np.asarray(payload["spectrum"], dtype=np.float32)
        else:
            values = [float(part) for part in line.split(",") if part.strip()]
            if len(values) < 2:
                raise ValueError(f"Hardware frame must include sequence id and spectrum: {line!r}")
            sequence_id = int(values[0])
            spectrum = np.asarray(values[1:], dtype=np.float32)
        return HILFrame(
            sequence_id=sequence_id,
            spectrum=spectrum,
            temperature_c=20.0,
            baseline_drift=0.0,
            received_at=time.perf_counter(),
        )
