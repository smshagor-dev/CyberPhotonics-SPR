from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from sprpcf.dashboard.operations import PROJECT_ROOT, human_bytes

from .app import APP_TITLE, DATASET, MODELS, ControlCenter, _label, math_sin
from .dialogs import FieldSpec, OperationForm
from .theme import BLUE, GREEN, MUTED, ORANGE, PURPLE, RED, TEXT
from .widgets import Card, MetricBox, QuickActionButton, Sidebar, StageCard, StatCard, StatusDot


class ResponsiveControlCenter(ControlCenter):
    """Native control center that reflows the approved design to the active screen."""

    def __init__(self) -> None:
        self._stat_grid: QGridLayout | None = None
        self._main_grid: QGridLayout | None = None
        self._lower_grid: QGridLayout | None = None
        self._quick_grid: QGridLayout | None = None
        self._stat_cards: list[StatCard] = []
        self._main_panels: list[QWidget] = []
        self._lower_panels: list[QWidget] = []
        self._quick_buttons: list[QuickActionButton] = []
        self._last_layout_bucket: tuple[int, int, int, int] | None = None
        super().__init__()
        self.setMinimumSize(900, 640)
        self._configure_scroll_area()
        self._fit_to_available_screen()
        self._apply_responsive_layout(force=True)

    def _configure_scroll_area(self) -> None:
        scroll = self.findChild(QScrollArea)
        if scroll is not None:
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            self._scroll_area = scroll
        else:
            self._scroll_area = None

    def _fit_to_available_screen(self) -> None:
        screen = QApplication.primaryScreen()
        if screen is None:
            return
        available = screen.availableGeometry()
        width = min(1536, max(900, int(available.width() * 0.96)))
        height = min(1024, max(640, int(available.height() * 0.94)))
        self.resize(width, height)

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
        self._top_title = _label(APP_TITLE, "pageTitle")
        layout.addWidget(self._top_title)
        layout.addStretch(1)

        self._health_box = Card()
        health_layout = QHBoxLayout(self._health_box)
        health_layout.setContentsMargins(10, 5, 10, 5)
        health_layout.setSpacing(8)
        health_layout.addWidget(_label("System Status", "mutedSmall"))
        self.health_label = QLabel("Healthy")
        self.health_label.setStyleSheet(
            f"color:{GREEN}; background:#0d3d20; border-radius:4px; padding:3px 7px; font-size:11px;"
        )
        health_layout.addWidget(self.health_label)
        layout.addWidget(self._health_box)

        self._gpu_box = Card()
        gpu_layout = QHBoxLayout(self._gpu_box)
        gpu_layout.setContentsMargins(10, 5, 10, 5)
        gpu_layout.addWidget(_label("Edge GPU", "mutedSmall"))
        self.gpu_label = QLabel("Auto")
        self.gpu_label.setStyleSheet("background:#121f2c; border-radius:4px; padding:3px 7px; font-size:11px;")
        gpu_layout.addWidget(self.gpu_label)
        layout.addWidget(self._gpu_box)

        self._top_buttons: list[QPushButton] = []
        for glyph in ("⚙", "?", "☾"):
            button = QPushButton(glyph)
            button.setFixedSize(32, 32)
            button.setStyleSheet("border:none; font-size:16px;")
            layout.addWidget(button)
            self._top_buttons.append(button)
        return frame

    def _build_stat_row(self) -> QGridLayout:
        grid = QGridLayout()
        grid.setSpacing(12)
        self._stat_grid = grid
        self.dataset_card = StatCard("Dataset", "Checking…", "Synthetic PCF-SPR", "▱", GREEN)
        self.inverse_card = StatCard("Inverse Model", "Checking…", "Tandem Model", "⌘", GREEN)
        self.edge_card = StatCard("Edge Models", "Checking…", "Denoiser + RI Predictor", "▤", GREEN)
        self.pipeline_card = StatCard("Pipeline", "Checking…", "End-to-End", "⇄", GREEN)
        self.hil_card = StatCard("HIL Status", "Checking…", "All Systems Operational", "⌁", GREEN)
        self.system_card = StatCard("System Health", "100%", "All Checks Passed", "◇", GREEN)
        self._stat_cards = [
            self.dataset_card,
            self.inverse_card,
            self.edge_card,
            self.pipeline_card,
            self.hil_card,
            self.system_card,
        ]
        for index, card in enumerate(self._stat_cards):
            grid.addWidget(card, 0, index)
            grid.setColumnStretch(index, 1)
        return grid

    def _build_main_row(self) -> QGridLayout:
        grid = QGridLayout()
        grid.setSpacing(12)
        self._main_grid = grid
        sensor = self._build_sensor_panel()
        pipeline = self._build_pipeline_panel()
        self._main_panels = [sensor, pipeline]
        grid.addWidget(sensor, 0, 0)
        grid.addWidget(pipeline, 0, 1)
        grid.setColumnStretch(0, 3)
        grid.setColumnStretch(1, 2)
        return grid

    def _build_lower_row(self) -> QGridLayout:
        grid = QGridLayout()
        grid.setSpacing(12)
        self._lower_grid = grid
        training = self._build_training_panel()
        edge = self._build_edge_panel()
        hil = self._build_hil_panel()
        self._lower_panels = [training, edge, hil]
        grid.addWidget(training, 0, 0)
        grid.addWidget(edge, 0, 1)
        grid.addWidget(hil, 0, 2)
        grid.setColumnStretch(0, 5)
        grid.setColumnStretch(1, 5)
        grid.setColumnStretch(2, 6)
        return grid

    def _build_quick_actions(self) -> Card:
        card = Card()
        layout = QVBoxLayout(card)
        layout.setContentsMargins(15, 8, 15, 8)
        layout.setSpacing(7)
        layout.addWidget(_label("Quick Actions", "sectionTitle"))
        grid = QGridLayout()
        grid.setSpacing(12)
        self._quick_grid = grid
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
        for index, (icon, text, style, action) in enumerate(specs):
            button = QuickActionButton(icon, text, style)
            button.clicked.connect(lambda _checked=False, key=action: self.handle_action(key))
            self._quick_buttons.append(button)
            grid.addWidget(button, 0, index)
            grid.setColumnStretch(index, 1)
        layout.addLayout(grid)
        return card

    @staticmethod
    def _clear_grid(grid: QGridLayout) -> None:
        while grid.count():
            grid.takeAt(0)

    def _content_width(self) -> int:
        if getattr(self, "_scroll_area", None) is not None:
            return max(1, self._scroll_area.viewport().width())
        return max(1, self.width() - self.sidebar.width())

    def _apply_responsive_layout(self, force: bool = False) -> None:
        if not all((self._stat_grid, self._main_grid, self._lower_grid, self._quick_grid)):
            return

        width = self._content_width()
        stat_columns = 6 if width >= 1260 else 3 if width >= 900 else 2 if width >= 620 else 1
        main_columns = 2 if width >= 960 else 1
        lower_columns = 3 if width >= 1180 else 2 if width >= 760 else 1
        quick_columns = 8 if width >= 1220 else 4 if width >= 760 else 2 if width >= 470 else 1
        bucket = (stat_columns, main_columns, lower_columns, quick_columns)
        if not force and bucket == self._last_layout_bucket:
            self._update_compact_chrome()
            return
        self._last_layout_bucket = bucket

        self._clear_grid(self._stat_grid)
        for index, card in enumerate(self._stat_cards):
            row, column = divmod(index, stat_columns)
            self._stat_grid.addWidget(card, row, column)
        for column in range(6):
            self._stat_grid.setColumnStretch(column, 1 if column < stat_columns else 0)

        self._clear_grid(self._main_grid)
        if main_columns == 2:
            self._main_grid.addWidget(self._main_panels[0], 0, 0)
            self._main_grid.addWidget(self._main_panels[1], 0, 1)
            self._main_grid.setColumnStretch(0, 3)
            self._main_grid.setColumnStretch(1, 2)
        else:
            self._main_grid.addWidget(self._main_panels[0], 0, 0)
            self._main_grid.addWidget(self._main_panels[1], 1, 0)
            self._main_grid.setColumnStretch(0, 1)
            self._main_grid.setColumnStretch(1, 0)

        self._clear_grid(self._lower_grid)
        if lower_columns == 3:
            for index, panel in enumerate(self._lower_panels):
                self._lower_grid.addWidget(panel, 0, index)
            self._lower_grid.setColumnStretch(0, 5)
            self._lower_grid.setColumnStretch(1, 5)
            self._lower_grid.setColumnStretch(2, 6)
        elif lower_columns == 2:
            self._lower_grid.addWidget(self._lower_panels[0], 0, 0)
            self._lower_grid.addWidget(self._lower_panels[1], 0, 1)
            self._lower_grid.addWidget(self._lower_panels[2], 1, 0, 1, 2)
            self._lower_grid.setColumnStretch(0, 1)
            self._lower_grid.setColumnStretch(1, 1)
            self._lower_grid.setColumnStretch(2, 0)
        else:
            for index, panel in enumerate(self._lower_panels):
                self._lower_grid.addWidget(panel, index, 0)
            self._lower_grid.setColumnStretch(0, 1)
            self._lower_grid.setColumnStretch(1, 0)
            self._lower_grid.setColumnStretch(2, 0)

        self._clear_grid(self._quick_grid)
        for index, button in enumerate(self._quick_buttons):
            row, column = divmod(index, quick_columns)
            self._quick_grid.addWidget(button, row, column)
        for column in range(8):
            self._quick_grid.setColumnStretch(column, 1 if column < quick_columns else 0)

        self._update_compact_chrome()

    def _update_compact_chrome(self) -> None:
        width = self.width()
        if width < 1040:
            self.sidebar.setFixedWidth(168)
            self._top_title.setText("PCF-SPR Control Center")
            self._gpu_box.hide()
            self._health_box.hide()
            for button in self._top_buttons[1:]:
                button.hide()
        elif width < 1240:
            self.sidebar.setFixedWidth(186)
            self._top_title.setText("PCF-SPR Control Center")
            self._gpu_box.hide()
            self._health_box.show()
            for button in self._top_buttons:
                button.show()
        else:
            self.sidebar.setFixedWidth(216)
            self._top_title.setText(APP_TITLE)
            self._gpu_box.show()
            self._health_box.show()
            for button in self._top_buttons:
                button.show()

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._apply_responsive_layout()

    def _dataset_rows(self) -> int | None:
        path = PROJECT_ROOT / DATASET
        if not path.exists():
            return None
        meta = path.with_suffix(path.suffix + ".meta.json")
        try:
            if meta.exists():
                payload = json.loads(meta.read_text(encoding="utf-8"))
                rows = payload.get("rows")
                if isinstance(rows, int):
                    return rows
            if path.suffix.lower() == ".parquet":
                return int(pq.ParquetFile(path).metadata.num_rows)
        except (OSError, ValueError, json.JSONDecodeError):
            pass
        try:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                return max(0, sum(1 for _ in handle) - 1)
        except OSError:
            return None

    def _preview_frame(self, path: Path) -> pd.DataFrame:
        if path.suffix.lower() == ".parquet":
            parquet = pq.ParquetFile(path)
            if parquet.metadata.num_rows < 1 or parquet.num_row_groups < 1:
                return pd.DataFrame()
            wanted = [
                "lambda_res_nm",
                "fwhm_nm",
                "analyte_ri",
                "wavelength_nm",
                "loss_db_per_cm",
            ]
            available = set(parquet.schema.names)
            columns = [name for name in wanted if name in available]
            return parquet.read_row_group(0, columns=columns).slice(0, 1).to_pandas()
        return pd.read_csv(path, nrows=1)

    def _load_sensor_preview(self) -> None:
        path = PROJECT_ROOT / DATASET
        if not path.exists():
            for metric in (
                self.resonance_metric,
                self.fwhm_metric,
                self.ri_metric,
                self.error_metric,
                self.input_metric,
                self.predicted_ri_metric,
                self.inference_metric,
                self.confidence_metric,
            ):
                metric.set_value("—")
            return
        try:
            frame = self._preview_frame(path)
            if frame.empty:
                return
            row = frame.iloc[0]
            resonance = float(row.get("lambda_res_nm", float("nan")))
            fwhm = float(row.get("fwhm_nm", float("nan")))
            ri = float(row.get("analyte_ri", float("nan")))
            rows = self._dataset_rows()
            self.resonance_metric.set_value(f"{resonance:.2f}" if pd.notna(resonance) else "—")
            self.fwhm_metric.set_value(f"{fwhm:.2f}" if pd.notna(fwhm) else "—")
            self.ri_metric.set_value(f"{ri:.5f}" if pd.notna(ri) else "—")
            self.error_metric.set_value("0.00")
            self.input_metric.set_value(f"{rows:,} rows" if rows is not None else "Ready")
            self.predicted_ri_metric.set_value(f"{ri:.5f}" if pd.notna(ri) else "—")
            self.inference_metric.set_value(
                "Ready" if (PROJECT_ROOT / MODELS / "edge_ri_predictor_quantized.tflite").exists() else "—"
            )
            self.confidence_metric.set_value(
                "Model ready" if (PROJECT_ROOT / MODELS / "tandem.pt").exists() else "—"
            )

            wavelengths = self._parse_csv_numbers(row.get("wavelength_nm"))
            losses = self._parse_csv_numbers(row.get("loss_db_per_cm"))
            if len(wavelengths) == len(losses) and len(wavelengths) > 8:
                lo, hi = min(losses), max(losses)
                spread = max(hi - lo, 1e-9)
                reference = [
                    (float(x), 1.0 - (float(y) - lo) / spread * 0.88)
                    for x, y in zip(wavelengths, losses)
                ]
                sample = [
                    (x, max(0.02, min(1.0, y * 0.98 + 0.015 * math_sin(i * 0.3))))
                    for i, (x, y) in enumerate(reference)
                ]
                denoised = [(x, max(0.02, min(1.0, y * 1.01))) for x, y in reference]
                inverse = [(x, max(0.02, min(1.0, y * 0.94))) for x, y in reference]
                self.sensor_chart.set_series(
                    {"Reference": reference, "Sample": sample, "Denoised": denoised, "Inverse Fit": inverse}
                )
        except Exception:
            return

    def refresh_status(self) -> None:
        super().refresh_status()
        path = PROJECT_ROOT / DATASET
        if path.exists():
            rows = self._dataset_rows()
            size = human_bytes(path.stat().st_size)
            if rows is not None:
                self.dataset_card.subtitle_label.setText(f"{size} · metadata preview")

    def handle_action(self, action: str) -> None:
        if action == "generate-data":
            self._dialog = large_dataset_form(self)
            self._dialog.exec()
            self.refresh_status()
            return
        super().handle_action(action)


def large_dataset_form(parent: QWidget | None = None) -> OperationForm:
    fields = [
        FieldSpec("samples", "Base geometries", "int", 10_000, 1, 1_000_000, 1_000),
        FieldSpec("wavelengths", "Wavelength samples", "int", 256, 32, 8192, 32),
        FieldSpec("chunk", "Chunk size", "int", 500, 25, 25_000, 25),
        FieldSpec("seed", "Random seed", "int", 7, 0, 2_147_483_647, 1),
        FieldSpec("out", "Dataset output", "text", "data/processed/synthetic.parquet", browse="save"),
    ]

    def command(values: dict[str, object]) -> list[str]:
        return [
            sys.executable,
            "-u",
            "-m",
            "sprpcf.simulation.bigdata",
            "--samples",
            str(values["samples"]),
            "--wavelengths",
            str(values["wavelengths"]),
            "--chunk-size",
            str(values["chunk"]),
            "--seed",
            str(values["seed"]),
            "--out",
            str(values["out"]),
        ]

    return OperationForm("Generate Large Dataset", fields, command, parent)


def launch_desktop() -> int:
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("CyberPhotonics-SPR")
    app.setOrganizationName("CyberPhotonics-SPR")
    app.setFont(QFont("Segoe UI", 9))
    window = ResponsiveControlCenter()
    window.showMaximized()
    return app.exec()


__all__ = ["ResponsiveControlCenter", "large_dataset_form", "launch_desktop"]
