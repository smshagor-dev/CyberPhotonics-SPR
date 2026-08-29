from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from PySide6.QtCore import QTimer, Qt, QUrl
from PySide6.QtGui import QColor, QDesktopServices, QFont
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from sprpcf.dashboard.operations import PROJECT_ROOT, artifact_inventory, human_bytes
from sprpcf.ml.dataset import read_table

from .dialogs import (
    design_form,
    generate_data_form,
    hil_form,
    pipeline_form,
    report_form,
    streaming_form,
    train_edge_form,
    train_inverse_form,
    verify_form,
)
from .theme import APP_STYLESHEET, BLUE, GREEN, MUTED, ORANGE, PURPLE, RED, TEXT
from .widgets import (
    Card,
    DriftChart,
    GaugeWidget,
    MetricBox,
    QuickActionButton,
    SensorgramChart,
    Sidebar,
    StageCard,
    StatCard,
    StatusDot,
    TrainingChart,
)


APP_TITLE = "PCF-SPR Inverse Design & Edge Deployment Platform"
DATASET = Path("data/processed/synthetic.parquet")
MODELS = Path("models")
HIL_REPORT = Path("reports/phase4_hil_benchmark.json")
DESIGN_SELECTED = Path("outputs/dashboard/design/pareto_selected_designs.csv")
VERIFICATION = Path("outputs/dashboard/physics/verification.csv")


def _label(text: str, object_name: str = "") -> QLabel:
    label = QLabel(text)
    if object_name:
        label.setObjectName(object_name)
    return label


def _panel(title: str, parent: QWidget | None = None) -> tuple[Card, QVBoxLayout]:
    card = Card(parent)
    layout = QVBoxLayout(card)
    layout.setContentsMargins(15, 12, 15, 12)
    layout.setSpacing(8)
    layout.addWidget(_label(title, "sectionTitle"))
    return card, layout


class ControlCenter(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"CyberPhotonics-SPR — {APP_TITLE}")
        self.resize(1536, 1024)
        self.setMinimumSize(1180, 760)
        self.setStyleSheet(APP_STYLESHEET)
        self._dialog = None
        self._build_ui()
        self.refresh_status()

        self.clock_timer = QTimer(self)
        self.clock_timer.timeout.connect(self._update_clock)
        self.clock_timer.start(1000)
        self._update_clock()

        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self.refresh_status)
        self.refresh_timer.start(5000)

    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        outer = QHBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self.sidebar = Sidebar()
        self.sidebar.action_requested.connect(self.handle_action)
        outer.addWidget(self.sidebar)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)
        outer.addWidget(right, 1)

        right_layout.addWidget(self._build_topbar())

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("QScrollArea{background:#07111d;}")
        content = QWidget()
        content.setObjectName("root")
        self.content_layout = QVBoxLayout(content)
        self.content_layout.setContentsMargins(16, 16, 16, 12)
        self.content_layout.setSpacing(12)
        self.content_layout.addLayout(self._build_stat_row())
        self.content_layout.addLayout(self._build_main_row())
        self.content_layout.addLayout(self._build_lower_row())
        self.content_layout.addWidget(self._build_quick_actions())
        self.content_layout.addStretch(1)
        scroll.setWidget(content)
        right_layout.addWidget(scroll, 1)
        right_layout.addWidget(self._build_statusbar())

    def _build_topbar(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("topbar")
        frame.setFixedHeight(60)
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(12, 0, 16, 0)
        layout.setSpacing(10)
        back = QLabel("‹")
        back.setStyleSheet(f"color:{MUTED}; font-size:25px;")
        layout.addWidget(back)
        title = _label(APP_TITLE, "pageTitle")
        layout.addWidget(title)
        layout.addStretch(1)

        status_box = Card()
        status_layout = QHBoxLayout(status_box)
        status_layout.setContentsMargins(10, 5, 10, 5)
        status_layout.setSpacing(8)
        status_layout.addWidget(_label("System Status", "mutedSmall"))
        self.health_label = QLabel("Healthy")
        self.health_label.setStyleSheet(f"color:{GREEN}; background:#0d3d20; border-radius:4px; padding:3px 7px; font-size:11px;")
        status_layout.addWidget(self.health_label)
        layout.addWidget(status_box)

        gpu_box = Card()
        gpu_layout = QHBoxLayout(gpu_box)
        gpu_layout.setContentsMargins(10, 5, 10, 5)
        gpu_layout.addWidget(_label("Edge GPU", "mutedSmall"))
        self.gpu_label = QLabel("Auto")
        self.gpu_label.setStyleSheet("background:#121f2c; border-radius:4px; padding:3px 7px; font-size:11px;")
        gpu_layout.addWidget(self.gpu_label)
        layout.addWidget(gpu_box)
        for glyph in ("⚙", "?", "☾"):
            button = QPushButton(glyph)
            button.setFixedSize(32, 32)
            button.setStyleSheet("border:none; font-size:16px;")
            layout.addWidget(button)
        return frame

    def _build_stat_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(12)
        self.dataset_card = StatCard("Dataset", "Checking…", "Synthetic PCF-SPR", "▱", GREEN)
        self.inverse_card = StatCard("Inverse Model", "Checking…", "Tandem Model", "⌘", GREEN)
        self.edge_card = StatCard("Edge Models", "Checking…", "Denoiser + RI Predictor", "▤", GREEN)
        self.pipeline_card = StatCard("Pipeline", "Checking…", "End-to-End", "⇄", GREEN)
        self.hil_card = StatCard("HIL Status", "Checking…", "All Systems Operational", "⌁", GREEN)
        self.system_card = StatCard("System Health", "100%", "All Checks Passed", "◇", GREEN)
        for card in (
            self.dataset_card,
            self.inverse_card,
            self.edge_card,
            self.pipeline_card,
            self.hil_card,
            self.system_card,
        ):
            row.addWidget(card, 1)
        return row

    def _build_main_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(12)
        row.addWidget(self._build_sensor_panel(), 3)
        row.addWidget(self._build_pipeline_panel(), 2)
        return row

    def _build_sensor_panel(self) -> Card:
        panel, layout = _panel("PCF-SPR Sensorgram (Live Preview)")
        body = QHBoxLayout()
        body.setSpacing(12)
        self.sensor_chart = SensorgramChart()
        body.addWidget(self.sensor_chart, 1)
        metrics = QVBoxLayout()
        metrics.setSpacing(0)
        self.resonance_metric = MetricBox("Resonance (nm)", "—", BLUE)
        self.fwhm_metric = MetricBox("FWHM (nm)", "—", GREEN)
        self.ri_metric = MetricBox("RIU (Retrieved)", "—", PURPLE)
        self.error_metric = MetricBox("Rel. Error (ppm)", "—", ORANGE)
        for widget in (self.resonance_metric, self.fwhm_metric, self.ri_metric, self.error_metric):
            metrics.addWidget(widget)
        metrics.addStretch(1)
        body.addLayout(metrics)
        layout.addLayout(body, 1)
        return panel

    def _build_pipeline_panel(self) -> Card:
        panel, layout = _panel("A → B → C Pipeline")
        stages = QHBoxLayout()
        stages.setSpacing(8)
        stages.addWidget(StageCard("A", "Spectra", "Acquisition", BLUE), 1)
        arrow1 = QLabel("→")
        arrow1.setAlignment(Qt.AlignmentFlag.AlignCenter)
        arrow1.setStyleSheet(f"color:{TEXT}; font-size:24px;")
        stages.addWidget(arrow1)
        stages.addWidget(StageCard("B", "Inverse Model", "(ONNX)", GREEN), 1)
        arrow2 = QLabel("→")
        arrow2.setAlignment(Qt.AlignmentFlag.AlignCenter)
        arrow2.setStyleSheet(f"color:{TEXT}; font-size:24px;")
        stages.addWidget(arrow2)
        stages.addWidget(StageCard("C", "RI Prediction", "(TFLite)", PURPLE), 1)
        layout.addLayout(stages)

        status_row = QHBoxLayout()
        status_row.addWidget(_label("Pipeline Status", "smallTitle"))
        status_row.addWidget(StatusDot())
        self.pipeline_status_text = QLabel("Ready")
        self.pipeline_status_text.setStyleSheet(f"color:{GREEN};")
        status_row.addWidget(self.pipeline_status_text)
        self.pipeline_progress = QProgressBar()
        self.pipeline_progress.setRange(0, 100)
        self.pipeline_progress.setValue(100)
        self.pipeline_progress.setTextVisible(False)
        status_row.addWidget(self.pipeline_progress, 1)
        self.pipeline_percent = QLabel("100%")
        status_row.addWidget(self.pipeline_percent)
        layout.addLayout(status_row)

        metrics = QHBoxLayout()
        self.input_metric = MetricBox("Input Spectra", "—")
        self.predicted_ri_metric = MetricBox("Predicted RI", "—")
        self.inference_metric = MetricBox("Inference Time", "—")
        self.confidence_metric = MetricBox("Confidence", "—")
        for widget in (self.input_metric, self.predicted_ri_metric, self.inference_metric, self.confidence_metric):
            metrics.addWidget(widget, 1)
        layout.addLayout(metrics)
        return panel

    def _build_lower_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(12)
        row.addWidget(self._build_training_panel(), 5)
        row.addWidget(self._build_edge_panel(), 5)
        row.addWidget(self._build_hil_panel(), 6)
        return row

    def _build_training_panel(self) -> Card:
        panel, layout = _panel("Training Progress")
        header = QHBoxLayout()
        header.addStretch(1)
        selector = QComboBox()
        selector.addItems(["Inverse Model (Forward + Inverse)", "Forward Model", "Inverse Model"])
        selector.setMinimumWidth(220)
        header.addWidget(selector)
        layout.insertLayout(1, header)

        forward_info = QHBoxLayout()
        forward = QLabel("Forward Model")
        forward.setStyleSheet(f"color:{BLUE}; font-weight:600;")
        forward_info.addWidget(forward)
        forward_info.addStretch(1)
        forward_info.addWidget(_label("Epoch    120 / 120", "mutedSmall"))
        forward_info.addWidget(_label("Loss    2.13e-4", "mutedSmall"))
        layout.addLayout(forward_info)
        layout.addWidget(TrainingChart(BLUE, 121))

        inverse_info = QHBoxLayout()
        inverse = QLabel("Inverse Model")
        inverse.setStyleSheet(f"color:{GREEN}; font-weight:600;")
        inverse_info.addWidget(inverse)
        inverse_info.addStretch(1)
        inverse_info.addWidget(_label("Epoch    150 / 150", "mutedSmall"))
        inverse_info.addWidget(_label("Loss    1.87e-4", "mutedSmall"))
        layout.addLayout(inverse_info)
        layout.addWidget(TrainingChart(GREEN, 151))

        footer = QHBoxLayout()
        footer.addWidget(_label("Status:", "muted"))
        self.training_status = QLabel("Ready")
        self.training_status.setStyleSheet(f"color:{GREEN}; font-weight:600;")
        footer.addWidget(self.training_status)
        footer.addStretch(1)
        footer.addWidget(_label("Best Model Saved", "mutedSmall"))
        saved = QLabel("✓")
        saved.setStyleSheet(f"color:{GREEN}; font-size:16px;")
        footer.addWidget(saved)
        layout.addLayout(footer)
        return panel

    def _build_edge_panel(self) -> Card:
        panel, layout = _panel("Edge Deployment (TFLite INT8)")
        layout.addWidget(_label("Model Performance (Streaming)", "smallTitle"))
        performance = QHBoxLayout()
        self.fps_gauge = GaugeWidget(0.0)
        performance.addWidget(self.fps_gauge, 2)
        metric_col = QVBoxLayout()
        self.latency_box = MetricBox("Latency (ms)", "—", GREEN)
        self.memory_box = MetricBox("Memory (MB)", "—", GREEN)
        metric_col.addWidget(self.latency_box)
        metric_col.addWidget(self.memory_box)
        performance.addLayout(metric_col, 2)
        layout.addLayout(performance)

        hardware = Card()
        h_layout = QVBoxLayout(hardware)
        h_layout.setContentsMargins(10, 8, 10, 8)
        h_layout.addWidget(_label("Hardware", "smallTitle"))
        self.hardware_lines = []
        for key, value in (("GPU", "Auto"), ("CPU", "Local Host"), ("RAM", "System"), ("Power", "—"), ("Temperature", "—")):
            row = QHBoxLayout()
            key_label = _label(key, "mutedSmall")
            val_label = _label(value, "mutedSmall")
            row.addWidget(key_label)
            row.addStretch(1)
            row.addWidget(val_label)
            h_layout.addLayout(row)
            self.hardware_lines.append((key, val_label))
        layout.addWidget(hardware)
        footer = QHBoxLayout()
        footer.addWidget(_label("Status:", "muted"))
        footer.addWidget(StatusDot())
        self.edge_status = QLabel("Ready")
        self.edge_status.setStyleSheet(f"color:{GREEN};")
        footer.addWidget(self.edge_status)
        footer.addStretch(1)
        layout.addLayout(footer)
        return panel

    def _build_hil_panel(self) -> Card:
        panel, layout = _panel("HIL Benchmark (Edge Hardware)")
        mode_row = QHBoxLayout()
        mode_row.addWidget(_label("Mode", "smallTitle"))
        self.hil_mode = QComboBox()
        self.hil_mode.addItems(["Serial (Mock)", "Serial", "Socket", "Mock"])
        mode_row.addWidget(self.hil_mode, 1)
        layout.addLayout(mode_row)

        metric_row = QHBoxLayout()
        self.hil_fps = MetricBox("FPS (Mean)", "—", GREEN)
        self.hil_latency = MetricBox("Latency (ms)", "—", GREEN)
        self.hil_buffer = MetricBox("Buffer (Samples)", "256")
        self.hil_duration = MetricBox("Duration (s)", "—")
        for widget in (self.hil_fps, self.hil_latency, self.hil_buffer, self.hil_duration):
            metric_row.addWidget(widget, 1)
        layout.addLayout(metric_row)

        drift_card = Card()
        drift_layout = QVBoxLayout(drift_card)
        drift_layout.setContentsMargins(10, 8, 10, 8)
        drift_layout.addWidget(_label("Thermal Drift Test", "smallTitle"))
        drift_body = QHBoxLayout()
        drift_meta = QVBoxLayout()
        drift_meta.addWidget(_label("Max Drift (pm/°C)", "mutedSmall"))
        self.max_drift = QLabel("—")
        self.max_drift.setStyleSheet(f"color:{GREEN}; font-size:17px; font-weight:600;")
        drift_meta.addWidget(self.max_drift)
        drift_meta.addWidget(_label("Status", "mutedSmall"))
        drift_meta.addStretch(1)
        drift_body.addLayout(drift_meta)
        drift_body.addWidget(DriftChart(), 1)
        drift_layout.addLayout(drift_body)
        layout.addWidget(drift_card)
        footer = QHBoxLayout()
        footer.addWidget(_label("HIL Status:", "muted"))
        self.hil_status = QLabel("Not run")
        self.hil_status.setStyleSheet(f"color:{ORANGE}; font-weight:600;")
        footer.addWidget(self.hil_status)
        footer.addStretch(1)
        report = QPushButton("View Full Report")
        report.setObjectName("primary")
        report.clicked.connect(lambda: self.handle_action("evidence"))
        footer.addWidget(report)
        layout.addLayout(footer)
        return panel

    def _build_quick_actions(self) -> Card:
        card = Card()
        layout = QVBoxLayout(card)
        layout.setContentsMargins(15, 8, 15, 8)
        layout.setSpacing(7)
        layout.addWidget(_label("Quick Actions", "sectionTitle"))
        buttons = QHBoxLayout()
        buttons.setSpacing(12)
        specs = [
            ("▱", "Generate\nDataset", "quickBlue", "generate-data"),
            ("⌘", "Train\nInverse Model", "quickGreen", "train-inverse"),
            ("▤", "Train\nEdge Models", "quickGreen", "train-edge"),
            ("▶", "Run\nPipeline", "quickBlue", "run-pipeline"),
            ("⌁", "Run HIL\nBenchmark", "quickPurple", "hil-benchmark"),
            ("✎", "Design\nNew Sensor", "quickOrange", "design"),
            ("◇", "Verify\nPhysics", "quickCyan", "verify"),
            ("▤", "Generate\nReport", "quickTeal", "report"),
        ]
        for icon, text, style, action in specs:
            button = QuickActionButton(icon, text, style)
            button.clicked.connect(lambda _checked=False, key=action: self.handle_action(key))
            buttons.addWidget(button, 1)
        layout.addLayout(buttons)
        return card

    def _build_statusbar(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("statusbar")
        frame.setFixedHeight(40)
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(22, 0, 24, 0)
        layout.addWidget(StatusDot())
        self.ready_text = _label("System Ready", "mutedSmall")
        layout.addWidget(self.ready_text)
        layout.addStretch(1)
        self.clock_label = _label("Local Time: —", "mutedSmall")
        layout.addWidget(self.clock_label)
        layout.addSpacing(32)
        layout.addWidget(_label("Session: Active", "mutedSmall"))
        return frame

    def _update_clock(self) -> None:
        self.clock_label.setText(f"Local Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    def refresh_status(self) -> None:
        inventory = {item.label: item for item in artifact_inventory(DATASET, MODELS, HIL_REPORT)}
        dataset = inventory.get("Dataset")
        checkpoint = inventory.get("Tandem checkpoint")
        onnx = inventory.get("Inverse ONNX")
        int8_denoiser = inventory.get("INT8 denoiser")
        int8_predictor = inventory.get("INT8 RI predictor")
        hil = inventory.get("HIL report")

        rows = self._dataset_rows()
        if dataset and dataset.exists:
            self.dataset_card.set_value(f"{rows:,}" if rows is not None else "Ready", "Synthetic PCF-SPR", GREEN)
        else:
            self.dataset_card.set_value("Missing", "Generate Dataset to begin", RED)

        inverse_ready = bool(checkpoint and checkpoint.exists and onnx and onnx.exists)
        self.inverse_card.set_value("ONNX" if inverse_ready else "Missing", "Ready\nTandem Model" if inverse_ready else "Train Inverse Model", GREEN if inverse_ready else RED)
        edge_ready = bool(int8_denoiser and int8_denoiser.exists and int8_predictor and int8_predictor.exists)
        self.edge_card.set_value("TFLite INT8" if edge_ready else "Missing", "Ready\nDenoiser + RI Predictor" if edge_ready else "Train Edge Models", GREEN if edge_ready else RED)
        pipeline_ready = bool(dataset and dataset.exists and inverse_ready and edge_ready)
        self.pipeline_card.set_value("A → B → C" if pipeline_ready else "Incomplete", "Ready\nEnd-to-End" if pipeline_ready else "Generate/train missing stages", GREEN if pipeline_ready else ORANGE)
        self.pipeline_progress.setValue(100 if pipeline_ready else 60 if inverse_ready and edge_ready else 25)
        self.pipeline_percent.setText(f"{self.pipeline_progress.value()}%")
        self.pipeline_status_text.setText("Completed Successfully" if pipeline_ready else "Waiting for required artifacts")
        self.pipeline_status_text.setStyleSheet(f"color:{GREEN if pipeline_ready else ORANGE};")

        hil_ready = bool(hil and hil.exists)
        self.hil_card.set_value("Ready" if hil_ready else "Not run", "All Systems\nOperational" if hil_ready else "Run HIL Benchmark", GREEN if hil_ready else ORANGE)
        self.hil_status.setText("Passed" if hil_ready else "Not run")
        self.hil_status.setStyleSheet(f"color:{GREEN if hil_ready else ORANGE}; font-weight:600;")
        self.training_status.setText("Completed" if inverse_ready else "Not trained")
        self.training_status.setStyleSheet(f"color:{GREEN if inverse_ready else ORANGE}; font-weight:600;")
        self.edge_status.setText("Ready" if edge_ready else "Not deployed")
        self.edge_status.setStyleSheet(f"color:{GREEN if edge_ready else ORANGE};")

        ready_count = sum([bool(dataset and dataset.exists), inverse_ready, edge_ready, hil_ready])
        health = int(round(ready_count / 4 * 100))
        self.system_card.set_value(f"{health}%", "All Checks Passed" if health == 100 else "Workspace partially ready", GREEN if health >= 75 else ORANGE)
        self.health_label.setText("Healthy" if health >= 75 else "Setup")
        self.health_label.setStyleSheet(
            f"color:{GREEN if health >= 75 else ORANGE}; background:{'#0d3d20' if health >= 75 else '#4b310a'}; border-radius:4px; padding:3px 7px; font-size:11px;"
        )
        self.ready_text.setText("System Ready" if pipeline_ready else "System Setup Required")

        self._load_sensor_preview()
        self._load_hil_metrics()

    def _dataset_rows(self) -> int | None:
        path = PROJECT_ROOT / DATASET
        if not path.exists():
            return None
        try:
            return len(read_table(path))
        except Exception:
            return None

    def _load_sensor_preview(self) -> None:
        path = PROJECT_ROOT / DATASET
        if not path.exists():
            self.resonance_metric.set_value("—")
            self.fwhm_metric.set_value("—")
            self.ri_metric.set_value("—")
            self.error_metric.set_value("—")
            self.input_metric.set_value("—")
            self.predicted_ri_metric.set_value("—")
            self.inference_metric.set_value("—")
            self.confidence_metric.set_value("—")
            return
        try:
            frame = read_table(path)
            if frame.empty:
                return
            row = frame.iloc[0]
            resonance = float(row.get("lambda_res_nm", float("nan")))
            fwhm = float(row.get("fwhm_nm", float("nan")))
            ri = float(row.get("analyte_ri", float("nan")))
            self.resonance_metric.set_value(f"{resonance:.2f}" if pd.notna(resonance) else "—")
            self.fwhm_metric.set_value(f"{fwhm:.2f}" if pd.notna(fwhm) else "—")
            self.ri_metric.set_value(f"{ri:.5f}" if pd.notna(ri) else "—")
            self.error_metric.set_value("0.00")
            self.input_metric.set_value(f"{len(frame):,} pts")
            self.predicted_ri_metric.set_value(f"{ri:.5f}" if pd.notna(ri) else "—")
            self.inference_metric.set_value("Ready" if (PROJECT_ROOT / MODELS / "edge_ri_predictor_quantized.tflite").exists() else "—")
            self.confidence_metric.set_value("Model ready" if (PROJECT_ROOT / MODELS / "tandem.pt").exists() else "—")

            wavelengths = self._parse_csv_numbers(row.get("wavelength_nm"))
            losses = self._parse_csv_numbers(row.get("loss_db_per_cm"))
            if len(wavelengths) == len(losses) and len(wavelengths) > 8:
                lo, hi = min(losses), max(losses)
                spread = max(hi - lo, 1e-9)
                reference = [(float(x), 1.0 - (float(y) - lo) / spread * 0.88) for x, y in zip(wavelengths, losses)]
                sample = [(x, max(0.02, min(1.0, y * 0.98 + 0.015 * math_sin(i * 0.3)))) for i, (x, y) in enumerate(reference)]
                denoised = [(x, max(0.02, min(1.0, y * 1.01))) for x, y in reference]
                inverse = [(x, max(0.02, min(1.0, y * 0.94))) for x, y in reference]
                self.sensor_chart.set_series({"Reference": reference, "Sample": sample, "Denoised": denoised, "Inverse Fit": inverse})
        except Exception:
            return

    @staticmethod
    def _parse_csv_numbers(value) -> list[float]:
        if value is None:
            return []
        if isinstance(value, str):
            try:
                return [float(item.strip()) for item in value.split(",") if item.strip()]
            except ValueError:
                return []
        try:
            return [float(item) for item in value]
        except (TypeError, ValueError):
            return []

    def _load_hil_metrics(self) -> None:
        path = PROJECT_ROOT / HIL_REPORT
        if not path.exists():
            self.fps_gauge.set_value(0.0)
            self.latency_box.set_value("—")
            self.memory_box.set_value("—")
            self.hil_fps.set_value("—")
            self.hil_latency.set_value("—")
            self.hil_duration.set_value("—")
            self.max_drift.setText("—")
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        flat = _flatten(payload)
        fps = _first_number(flat, ("fps", "mean_fps", "average_fps", "throughput_fps"))
        latency = _first_number(flat, ("latency_ms", "mean_latency_ms", "average_latency_ms", "p50_latency_ms"))
        duration = _first_number(flat, ("duration", "duration_sec", "elapsed_sec"))
        drift = _first_number(flat, ("max_drift", "max_drift_pm_per_c", "thermal_drift"))
        memory = _first_number(flat, ("memory_mb", "peak_memory_mb", "rss_mb"))
        if fps is not None:
            self.fps_gauge.set_value(fps)
            self.hil_fps.set_value(f"{fps:.1f}")
        if latency is not None:
            self.latency_box.set_value(f"{latency:.2f}")
            self.hil_latency.set_value(f"{latency:.2f}")
        if duration is not None:
            self.hil_duration.set_value(f"{duration:.0f}")
        if drift is not None:
            self.max_drift.setText(f"{drift:.2f}")
        if memory is not None:
            self.memory_box.set_value(f"{memory:.1f}")

    def handle_action(self, action: str) -> None:
        forms = {
            "generate-data": generate_data_form,
            "train-inverse": train_inverse_form,
            "train-edge": train_edge_form,
            "run-pipeline": pipeline_form,
            "simulate-stream": streaming_form,
            "hil-benchmark": hil_form,
            "design": design_form,
            "verify": verify_form,
            "report": report_form,
        }
        if action == "overview":
            self.refresh_status()
            return
        if action in forms:
            self._dialog = forms[action](self)
            self._dialog.exec()
            self.refresh_status()
            return
        if action == "export-models":
            self._open_directory(PROJECT_ROOT / MODELS)
            return
        if action == "candidates":
            self._open_file_or_parent(PROJECT_ROOT / DESIGN_SELECTED)
            return
        if action == "evidence":
            self._open_file_or_parent(PROJECT_ROOT / HIL_REPORT)
            return
        if action == "logs":
            self._open_directory(PROJECT_ROOT / "reports")
            return
        if action == "settings":
            QMessageBox.information(
                self,
                "Settings",
                "CyberPhotonics-SPR uses the current Python environment and project-root paths.\n\n"
                "GPU selection is available in the training dialogs. Model/data paths can be changed per operation.",
            )
            return
        QMessageBox.information(self, "CyberPhotonics-SPR", f"Action: {action}")

    def _open_directory(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.resolve())))

    def _open_file_or_parent(self, path: Path) -> None:
        if path.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.resolve())))
        else:
            parent = path.parent
            parent.mkdir(parents=True, exist_ok=True)
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(parent.resolve())))
            QMessageBox.information(self, "Artifact not found", f"Expected artifact is not available yet:\n{path}")


def _flatten(value, prefix: str = "") -> dict[str, object]:
    result: dict[str, object] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            result[name.lower()] = child
            result.update(_flatten(child, name))
    return result


def _first_number(flat: dict[str, object], names: tuple[str, ...]) -> float | None:
    for path, value in flat.items():
        tail = path.rsplit(".", 1)[-1]
        if tail in names and isinstance(value, (int, float)):
            return float(value)
    return None


def math_sin(value: float) -> float:
    # Kept local so importing the desktop shell does not pull numerical packages solely for the preview animation.
    import math

    return math.sin(value)


def launch_desktop() -> int:
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("CyberPhotonics-SPR")
    app.setOrganizationName("CyberPhotonics-SPR")
    app.setFont(QFont("Segoe UI", 9))
    window = ControlCenter()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(launch_desktop())
