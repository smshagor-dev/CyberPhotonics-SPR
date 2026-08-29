from __future__ import annotations

import os
import sys
import time
import tracemalloc
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from sprpcf.edge.quantization import TFLiteModelRunner


try:
    import psutil
except ImportError:  # pragma: no cover - exercised only on minimal environments
    psutil = None


if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes


@dataclass
class FrameProfile:
    """Per-frame edge inference timing and host resource snapshot."""

    frame_index: int
    latency_ms: float
    ram_mb: float
    cpu_percent: float


@dataclass
class BenchmarkMetrics:
    """Aggregate model profile for streaming HIL runs."""

    model_name: str
    frames_processed: int
    frames_lost: int
    average_latency_ms: float
    p95_latency_ms: float
    max_latency_ms: float
    throughput_fps: float
    peak_ram_mb: float
    average_cpu_percent: float
    estimated_power_w: float

    def to_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


class EdgeHardwareProfiler:
    """Profile TFLite models under streaming real-time input conditions."""

    def __init__(self, power_base_w: float = 0.65, power_cpu_scale_w: float = 1.35) -> None:
        self.power_base_w = power_base_w
        self.power_cpu_scale_w = power_cpu_scale_w
        self.process = psutil.Process(os.getpid()) if psutil is not None else None

    def benchmark_runner(
        self,
        runner: TFLiteModelRunner,
        frames: Iterable[np.ndarray],
        model_name: str,
        expected_frames: int | None = None,
        frame_loss_provider: Callable[[], int] | None = None,
    ) -> tuple[BenchmarkMetrics, list[FrameProfile]]:
        profiles: list[FrameProfile] = []
        latencies: list[float] = []
        start = time.perf_counter()

        tracemalloc.start()
        if self.process is not None:
            self.process.cpu_percent(interval=None)

        for frame_index, frame in enumerate(frames):
            input_frame = self._as_tflite_input(frame)
            started = time.perf_counter()
            runner.predict(input_frame)
            latency_ms = (time.perf_counter() - started) * 1000.0

            _, peak_bytes = tracemalloc.get_traced_memory()
            ram_mb = self._resident_ram_mb(peak_bytes)
            cpu_percent = self._cpu_percent()
            latencies.append(latency_ms)
            profiles.append(
                FrameProfile(
                    frame_index=frame_index,
                    latency_ms=float(latency_ms),
                    ram_mb=float(ram_mb),
                    cpu_percent=float(cpu_percent),
                )
            )

        elapsed = max(time.perf_counter() - start, 1e-9)
        tracemalloc.stop()
        processed = len(profiles)
        expected = expected_frames if expected_frames is not None else processed
        source_loss = frame_loss_provider() if frame_loss_provider is not None else 0
        frames_lost = max(int(expected) - processed, 0) + int(source_loss)
        latency_array = np.asarray(latencies, dtype=np.float32)
        avg_cpu = float(np.mean([profile.cpu_percent for profile in profiles])) if profiles else 0.0

        metrics = BenchmarkMetrics(
            model_name=model_name,
            frames_processed=processed,
            frames_lost=frames_lost,
            average_latency_ms=float(np.mean(latency_array)) if processed else 0.0,
            p95_latency_ms=float(np.percentile(latency_array, 95)) if processed else 0.0,
            max_latency_ms=float(np.max(latency_array)) if processed else 0.0,
            throughput_fps=float(processed / elapsed),
            peak_ram_mb=float(max((profile.ram_mb for profile in profiles), default=0.0)),
            average_cpu_percent=avg_cpu,
            estimated_power_w=self._estimate_power(avg_cpu),
        )
        return metrics, profiles

    def benchmark_model(
        self,
        model_path: Path,
        frames: Iterable[np.ndarray],
        expected_frames: int | None = None,
        frame_loss_provider: Callable[[], int] | None = None,
    ) -> tuple[BenchmarkMetrics, list[FrameProfile]]:
        runner = TFLiteModelRunner(model_path)
        return self.benchmark_runner(
            runner=runner,
            frames=frames,
            model_name=model_path.name,
            expected_frames=expected_frames,
            frame_loss_provider=frame_loss_provider,
        )

    def compare_models(
        self,
        int8_model_path: Path,
        frames: np.ndarray,
        fp32_model_path: Path | None = None,
    ) -> dict[str, dict[str, float | int | str]]:
        results: dict[str, dict[str, float | int | str]] = {}
        int8_metrics, _ = self.benchmark_model(int8_model_path, frames, expected_frames=len(frames))
        results["int8"] = int8_metrics.to_dict()
        if fp32_model_path is not None and fp32_model_path.exists():
            fp32_metrics, _ = self.benchmark_model(fp32_model_path, frames, expected_frames=len(frames))
            results["fp32"] = fp32_metrics.to_dict()
        return results

    def _resident_ram_mb(self, tracemalloc_peak_bytes: int) -> float:
        if self.process is not None:
            return self.process.memory_info().rss / (1024.0 * 1024.0)
        windows_rss = self._windows_rss_mb()
        if windows_rss is not None:
            return windows_rss
        return tracemalloc_peak_bytes / (1024.0 * 1024.0)

    def _cpu_percent(self) -> float:
        if self.process is None:
            return 0.0
        return float(self.process.cpu_percent(interval=None))

    def _estimate_power(self, cpu_percent: float) -> float:
        return float(self.power_base_w + self.power_cpu_scale_w * max(cpu_percent, 0.0) / 100.0)

    @staticmethod
    def _windows_rss_mb() -> float | None:
        if sys.platform != "win32":
            return None

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(ProcessMemoryCounters)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        ok = ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
        if not ok:
            return None
        return float(counters.WorkingSetSize / (1024.0 * 1024.0))

    @staticmethod
    def _as_tflite_input(frame: np.ndarray) -> np.ndarray:
        array = np.asarray(frame, dtype=np.float32)
        if array.ndim == 1:
            return array[None, :, None]
        if array.ndim == 2:
            return array[None, :, :] if array.shape[-1] == 1 else array[:, :, None]
        return array
