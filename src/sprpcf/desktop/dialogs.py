from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QProcess, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .theme import GREEN, MUTED, ORANGE, RED


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MAIN_SCRIPT = PROJECT_ROOT / "main.py"


@dataclass(frozen=True)
class FieldSpec:
    key: str
    label: str
    kind: str = "text"
    default: object = ""
    minimum: float | int | None = None
    maximum: float | int | None = None
    step: float | int | None = None
    choices: tuple[str, ...] = ()
    browse: str | None = None


class ProcessConsole(QDialog):
    completed = Signal(bool)

    def __init__(self, title: str, command: list[str], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(900, 560)
        self.setMinimumSize(720, 430)
        self._command = command
        self._process = QProcess(self)
        self._process.setWorkingDirectory(str(PROJECT_ROOT))
        self._process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self._process.readyReadStandardOutput.connect(self._read_output)
        self._process.finished.connect(self._finished)
        self._process.errorOccurred.connect(self._process_error)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(18, 18, 18, 18)
        outer.setSpacing(10)
        header = QLabel(title)
        header.setObjectName("sectionTitle")
        outer.addWidget(header)
        self.status = QLabel("Starting…")
        self.status.setStyleSheet(f"color:{ORANGE}; font-weight:600;")
        outer.addWidget(self.status)
        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.console.setStyleSheet("font-family:Consolas, 'Cascadia Mono', monospace; font-size:11px;")
        outer.addWidget(self.console, 1)

        footer = QHBoxLayout()
        self.cancel_button = QPushButton("Stop")
        self.cancel_button.clicked.connect(self._stop)
        footer.addWidget(self.cancel_button)
        footer.addStretch(1)
        self.close_button = QPushButton("Close")
        self.close_button.setObjectName("primary")
        self.close_button.setEnabled(False)
        self.close_button.clicked.connect(self.accept)
        footer.addWidget(self.close_button)
        outer.addLayout(footer)

    def start(self) -> None:
        if not self._command:
            return
        program, *args = self._command
        self.console.appendPlainText(f"[{self.windowTitle()}]\n")
        self.status.setText("Running")
        self.status.setStyleSheet(f"color:{GREEN}; font-weight:600;")
        self._process.start(program, args)

    def showEvent(self, event) -> None:  # type: ignore[override]
        super().showEvent(event)
        if self._process.state() == QProcess.ProcessState.NotRunning and not self.close_button.isEnabled():
            self.start()

    def _read_output(self) -> None:
        chunk = bytes(self._process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if chunk:
            self.console.moveCursor(self.console.textCursor().MoveOperation.End)
            self.console.insertPlainText(chunk)
            self.console.ensureCursorVisible()

    def _finished(self, exit_code: int, _status) -> None:
        self._read_output()
        success = exit_code == 0
        self.status.setText("Completed successfully" if success else f"Failed · exit code {exit_code}")
        self.status.setStyleSheet(f"color:{GREEN if success else RED}; font-weight:600;")
        self.cancel_button.setEnabled(False)
        self.close_button.setEnabled(True)
        self.completed.emit(success)

    def _process_error(self, error) -> None:
        if self._process.state() != QProcess.ProcessState.NotRunning:
            return
        self.status.setText(f"Process error: {error}")
        self.status.setStyleSheet(f"color:{RED}; font-weight:600;")
        self.cancel_button.setEnabled(False)
        self.close_button.setEnabled(True)
        self.completed.emit(False)

    def _stop(self) -> None:
        if self._process.state() != QProcess.ProcessState.NotRunning:
            self.status.setText("Stopping…")
            self._process.terminate()
            if not self._process.waitForFinished(1600):
                self._process.kill()

    def reject(self) -> None:
        if self._process.state() != QProcess.ProcessState.NotRunning:
            answer = QMessageBox.question(self, "Stop operation?", "This operation is still running. Stop it and close?")
            if answer != QMessageBox.StandardButton.Yes:
                return
            self._stop()
        super().reject()


class OperationForm(QDialog):
    def __init__(
        self,
        title: str,
        fields: list[FieldSpec],
        command_builder: Callable[[dict[str, object]], list[str]],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(580)
        self._title = title
        self._command_builder = command_builder
        self._widgets: dict[str, QWidget] = {}
        self.process_dialog: ProcessConsole | None = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(20, 20, 20, 20)
        title_label = QLabel(title)
        title_label.setObjectName("sectionTitle")
        outer.addWidget(title_label)
        description = QLabel("Configure the operation below. Execution uses the current virtual environment and the project backend.")
        description.setWordWrap(True)
        description.setStyleSheet(f"color:{MUTED};")
        outer.addWidget(description)

        form = QFormLayout()
        form.setHorizontalSpacing(16)
        form.setVerticalSpacing(10)
        for field in fields:
            widget = self._create_widget(field)
            self._widgets[field.key] = widget
            if field.browse and isinstance(widget, QLineEdit):
                row = QWidget()
                row_layout = QHBoxLayout(row)
                row_layout.setContentsMargins(0, 0, 0, 0)
                row_layout.setSpacing(6)
                row_layout.addWidget(widget, 1)
                browse = QPushButton("Browse")
                browse.clicked.connect(lambda _checked=False, line=widget, mode=field.browse: self._browse(line, mode))
                row_layout.addWidget(browse)
                form.addRow(field.label, row)
            else:
                form.addRow(field.label, widget)
        outer.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        run = QPushButton("Run")
        run.setObjectName("primary")
        run.clicked.connect(self._run)
        buttons.addButton(run, QDialogButtonBox.ButtonRole.AcceptRole)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

    def _create_widget(self, field: FieldSpec) -> QWidget:
        if field.kind == "int":
            widget = QSpinBox()
            widget.setRange(int(field.minimum if field.minimum is not None else -2_147_483_648), int(field.maximum if field.maximum is not None else 2_147_483_647))
            widget.setValue(int(field.default))
            if field.step is not None:
                widget.setSingleStep(int(field.step))
            return widget
        if field.kind == "float":
            widget = QDoubleSpinBox()
            widget.setDecimals(6)
            widget.setRange(float(field.minimum if field.minimum is not None else -1e12), float(field.maximum if field.maximum is not None else 1e12))
            widget.setValue(float(field.default))
            if field.step is not None:
                widget.setSingleStep(float(field.step))
            return widget
        if field.kind == "choice":
            widget = QComboBox()
            widget.addItems(list(field.choices))
            if str(field.default) in field.choices:
                widget.setCurrentText(str(field.default))
            return widget
        if field.kind == "bool":
            widget = QCheckBox()
            widget.setChecked(bool(field.default))
            return widget
        widget = QLineEdit(str(field.default))
        return widget

    def _values(self) -> dict[str, object]:
        values: dict[str, object] = {}
        for key, widget in self._widgets.items():
            if isinstance(widget, QSpinBox):
                values[key] = widget.value()
            elif isinstance(widget, QDoubleSpinBox):
                values[key] = widget.value()
            elif isinstance(widget, QComboBox):
                values[key] = widget.currentText()
            elif isinstance(widget, QCheckBox):
                values[key] = widget.isChecked()
            elif isinstance(widget, QLineEdit):
                values[key] = widget.text().strip()
        return values

    def _run(self) -> None:
        try:
            command = self._command_builder(self._values())
        except Exception as exc:
            QMessageBox.critical(self, "Invalid operation", str(exc))
            return
        self.process_dialog = ProcessConsole(self._title, command, self)
        self.process_dialog.completed.connect(lambda _ok: None)
        self.process_dialog.exec()
        if self.process_dialog is not None and self.process_dialog.close_button.isEnabled():
            self.accept()

    def _browse(self, line: QLineEdit, mode: str | None) -> None:
        if mode == "dir":
            value = QFileDialog.getExistingDirectory(self, "Select directory", line.text() or str(PROJECT_ROOT))
        elif mode == "save":
            value, _ = QFileDialog.getSaveFileName(self, "Select output", line.text() or str(PROJECT_ROOT))
        else:
            value, _ = QFileDialog.getOpenFileName(self, "Select file", line.text() or str(PROJECT_ROOT))
        if value:
            line.setText(value)


def _command(subcommand: str, args: list[object]) -> list[str]:
    return [sys.executable, "-u", str(MAIN_SCRIPT), subcommand, *[str(value) for value in args]]


def generate_data_form(parent: QWidget | None = None) -> OperationForm:
    fields = [
        FieldSpec("samples", "Base geometries", "int", 100, 1, 100000, 10),
        FieldSpec("wavelengths", "Wavelength samples", "int", 256, 32, 8192, 32),
        FieldSpec("seed", "Random seed", "int", 7, 0, 2_147_483_647, 1),
        FieldSpec("out", "Dataset output", "text", "data/processed/synthetic.parquet", browse="save"),
    ]
    return OperationForm(
        "Generate Dataset",
        fields,
        lambda v: _command("generate-data", ["--samples", v["samples"], "--wavelengths", v["wavelengths"], "--seed", v["seed"], "--out", v["out"]]),
        parent,
    )


def train_inverse_form(parent: QWidget | None = None) -> OperationForm:
    fields = [
        FieldSpec("data", "Training dataset", "text", "data/processed/synthetic.parquet", browse="file"),
        FieldSpec("epochs", "Epochs", "int", 100, 1, 10000, 10),
        FieldSpec("batch", "Batch size", "int", 64, 1, 4096, 16),
        FieldSpec("lr", "Learning rate", "float", 0.001, 0.000001, 1.0, 0.0001),
        FieldSpec("device", "Device", "choice", "auto", choices=("auto", "cpu", "cuda")),
        FieldSpec("seed", "Seed", "int", 7, 0, 2_147_483_647, 1),
        FieldSpec("checkpoint", "Checkpoint", "text", "models/tandem.pt", browse="save"),
        FieldSpec("onnx", "ONNX export", "text", "models/inverse_pcf_spr.onnx", browse="save"),
    ]
    return OperationForm(
        "Train Inverse Model",
        fields,
        lambda v: _command("train-inverse", ["--data", v["data"], "--epochs", v["epochs"], "--batch-size", v["batch"], "--lr", v["lr"], "--device", v["device"], "--seed", v["seed"], "--checkpoint", v["checkpoint"], "--export-onnx", v["onnx"]]),
        parent,
    )


def train_edge_form(parent: QWidget | None = None) -> OperationForm:
    fields = [
        FieldSpec("data", "Training dataset", "text", "data/processed/synthetic.parquet", browse="file"),
        FieldSpec("epochs", "Epochs", "int", 50, 1, 10000, 10),
        FieldSpec("batch", "Batch size", "int", 64, 1, 4096, 16),
        FieldSpec("device", "Device", "choice", "auto", choices=("auto", "cpu", "/GPU:0")),
        FieldSpec("quantize", "Export INT8 TFLite", "bool", True),
        FieldSpec("seed", "Seed", "int", 7, 0, 2_147_483_647, 1),
        FieldSpec("out", "Model directory", "text", "models", browse="dir"),
    ]

    def build(v: dict[str, object]) -> list[str]:
        args: list[object] = ["--data", v["data"], "--epochs", v["epochs"], "--batch-size", v["batch"], "--device", v["device"], "--seed", v["seed"], "--export-dir", v["out"]]
        if v["quantize"]:
            args.append("--quantize")
        return _command("train-edge", args)

    return OperationForm("Train Edge Models", fields, build, parent)


def pipeline_form(parent: QWidget | None = None) -> OperationForm:
    fields = [
        FieldSpec("samples", "Base geometries", "int", 100, 1, 100000, 10),
        FieldSpec("wavelengths", "Wavelength samples", "int", 256, 32, 8192, 32),
        FieldSpec("inverse_epochs", "Inverse epochs", "int", 100, 1, 10000, 10),
        FieldSpec("edge_epochs", "Edge epochs", "int", 50, 1, 10000, 10),
        FieldSpec("batch", "Batch size", "int", 64, 1, 4096, 16),
        FieldSpec("device", "Inverse device", "choice", "auto", choices=("auto", "cpu", "cuda")),
        FieldSpec("edge_device", "Edge device", "choice", "auto", choices=("auto", "cpu", "/GPU:0")),
        FieldSpec("duration", "Streaming seconds", "float", 10.0, 0.1, 3600.0, 1.0),
        FieldSpec("data", "Dataset output", "text", "data/processed/synthetic.parquet", browse="save"),
        FieldSpec("out", "Model directory", "text", "models", browse="dir"),
    ]
    return OperationForm(
        "Run A → B → C Pipeline",
        fields,
        lambda v: _command("run-pipeline", ["--samples", v["samples"], "--wavelengths", v["wavelengths"], "--data", v["data"], "--export-dir", v["out"], "--inverse-epochs", v["inverse_epochs"], "--edge-epochs", v["edge_epochs"], "--batch-size", v["batch"], "--device", v["device"], "--edge-device", v["edge_device"], "--duration-sec", v["duration"]]),
        parent,
    )


def streaming_form(parent: QWidget | None = None) -> OperationForm:
    fields = [
        FieldSpec("data", "Dataset", "text", "data/processed/synthetic.parquet", browse="file"),
        FieldSpec("models", "TFLite directory", "text", "models", browse="dir"),
        FieldSpec("duration", "Duration (s)", "float", 10.0, 0.1, 3600.0, 1.0),
        FieldSpec("noise", "Noise std", "float", 0.08, 0.0, 2.0, 0.01),
        FieldSpec("drift", "Drift std", "float", 0.03, 0.0, 2.0, 0.01),
    ]
    return OperationForm(
        "Streaming Benchmark",
        fields,
        lambda v: _command("simulate-stream", ["--data", v["data"], "--tflite-dir", v["models"], "--duration-sec", v["duration"], "--noise-std", v["noise"], "--drift-std", v["drift"]]),
        parent,
    )


def hil_form(parent: QWidget | None = None) -> OperationForm:
    fields = [
        FieldSpec("models", "TFLite directory", "text", "models", browse="dir"),
        FieldSpec("protocol", "Transport", "choice", "mock", choices=("mock", "serial", "socket")),
        FieldSpec("duration", "Duration (s)", "float", 30.0, 0.1, 86400.0, 5.0),
        FieldSpec("fps", "Target FPS", "float", 30.0, 0.1, 10000.0, 1.0),
        FieldSpec("buffer", "Buffer size", "int", 256, 1, 65536, 32),
        FieldSpec("drift", "Inject thermal drift", "bool", False),
        FieldSpec("report", "Report", "text", "reports/phase4_hil_benchmark.json", browse="save"),
        FieldSpec("serial", "Serial port", "text", "COM3"),
        FieldSpec("host", "Socket host", "text", "127.0.0.1"),
        FieldSpec("port", "Socket port", "int", 9000, 1, 65535, 1),
    ]

    def build(v: dict[str, object]) -> list[str]:
        args: list[object] = ["--tflite-dir", v["models"], "--duration", v["duration"], "--protocol", v["protocol"], "--fps", v["fps"], "--buffer-size", v["buffer"], "--report", v["report"]]
        if v["drift"]:
            args.append("--inject-thermal-drift")
        if v["protocol"] == "serial":
            args.extend(["--serial-port", v["serial"]])
        elif v["protocol"] == "socket":
            args.extend(["--socket-host", v["host"], "--socket-port", v["port"]])
        return _command("hil-benchmark", args)

    return OperationForm("HIL Benchmark", fields, build, parent)


def design_form(parent: QWidget | None = None) -> OperationForm:
    fields = [
        FieldSpec("checkpoint", "Tandem checkpoint", "text", "models/tandem.pt", browse="file"),
        FieldSpec("data", "Reference dataset", "text", "data/processed/synthetic.parquet", browse="file"),
        FieldSpec("sensitivity", "Target sensitivity", "float", 800.0, 1.0, 100000.0, 10.0),
        FieldSpec("fom", "Target FOM", "float", 20.0, 0.01, 10000.0, 0.5),
        FieldSpec("lambda", "Target resonance λ (nm)", "float", 750.0, 100.0, 5000.0, 1.0),
        FieldSpec("ri", "Analyte RI", "float", 1.37, 1.000001, 1.999999, 0.001),
        FieldSpec("candidates", "Candidates", "int", 128, 4, 2048, 4),
        FieldSpec("confidence", "Calibration confidence", "float", 0.95, 0.5, 0.99, 0.01),
        FieldSpec("out", "Output directory", "text", "outputs/dashboard/design", browse="dir"),
    ]
    return OperationForm(
        "Design New Sensor",
        fields,
        lambda v: _command("design-sensor", ["--checkpoint", v["checkpoint"], "--data", v["data"], "--sensitivity", v["sensitivity"], "--fom", v["fom"], "--lambda-res", v["lambda"], "--analyte-ri", v["ri"], "--candidates", v["candidates"], "--confidence", v["confidence"], "--out", v["out"]]),
        parent,
    )


def verify_form(parent: QWidget | None = None) -> OperationForm:
    fields = [
        FieldSpec("checkpoint", "Tandem checkpoint", "text", "models/tandem.pt", browse="file"),
        FieldSpec("data", "Base dataset", "text", "data/processed/synthetic.parquet", browse="file"),
        FieldSpec("selected", "Selected design CSV", "text", "outputs/dashboard/design/pareto_selected_designs.csv", browse="file"),
        FieldSpec("backend", "Physics backend", "choice", "synthetic", choices=("synthetic", "comsol")),
        FieldSpec("ri_span", "RI sweep span", "float", 0.04, 0.001, 0.5, 0.005),
        FieldSpec("ri_points", "RI points", "int", 5, 3, 21, 2),
        FieldSpec("out", "Output directory", "text", "outputs/dashboard/physics", browse="dir"),
        FieldSpec("model", "COMSOL model (optional)", "text", ""),
        FieldSpec("config", "COMSOL config (optional)", "text", ""),
    ]
    return OperationForm(
        "Verify Physics",
        fields,
        lambda v: _command("verify-physics", ["--checkpoint", v["checkpoint"], "--data", v["data"], "--selected", v["selected"], "--backend", v["backend"], "--ri-span", v["ri_span"], "--ri-points", v["ri_points"], "--out", v["out"], "--model", v["model"], "--config", v["config"]]),
        parent,
    )


def report_form(parent: QWidget | None = None) -> OperationForm:
    fields = [
        FieldSpec("selected", "Selected design CSV", "text", "outputs/dashboard/design/pareto_selected_designs.csv", browse="file"),
        FieldSpec("verification", "Verification CSV", "text", "outputs/dashboard/physics/verification_results.csv", browse="file"),
        FieldSpec("out", "Markdown report", "text", "outputs/dashboard/dashboard_evidence_report.md", browse="save"),
    ]
    return OperationForm(
        "Generate Research Report",
        fields,
        lambda v: _command("generate-report", ["--selected", v["selected"], "--verification", v["verification"], "--out", v["out"]]),
        parent,
    )
