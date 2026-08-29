from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .theme import (
    BG_DEEP,
    BLUE,
    BORDER,
    GREEN,
    MUTED,
    ORANGE,
    PANEL,
    PURPLE,
    TEXT,
)


def _label(text: str, object_name: str = "", alignment: Qt.AlignmentFlag | None = None) -> QLabel:
    label = QLabel(text)
    if object_name:
        label.setObjectName(object_name)
    if alignment is not None:
        label.setAlignment(alignment)
    return label


class Card(QFrame):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("card")
        self.setProperty("class", "card")


class StatCard(Card):
    def __init__(
        self,
        title: str,
        value: str,
        subtitle: str,
        icon: str,
        color: str = GREEN,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setMinimumHeight(106)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(17, 13, 15, 13)
        layout.setSpacing(10)

        text_col = QVBoxLayout()
        text_col.setSpacing(4)
        title_label = _label(title, "smallTitle")
        self.value_label = _label(value)
        self.value_label.setStyleSheet(f"color:{color}; font-size:20px; font-weight:700;")
        self.subtitle_label = _label(subtitle, "mutedSmall")
        self.subtitle_label.setWordWrap(True)
        text_col.addWidget(title_label)
        text_col.addWidget(self.value_label)
        text_col.addWidget(self.subtitle_label)
        text_col.addStretch(1)
        layout.addLayout(text_col, 1)

        icon_label = _label(icon, alignment=Qt.AlignmentFlag.AlignCenter)
        icon_label.setFixedSize(48, 48)
        icon_label.setStyleSheet(f"color:{color}; font-size:28px; font-weight:600;")
        layout.addWidget(icon_label)

    def set_value(self, value: str, subtitle: str | None = None, color: str | None = None) -> None:
        self.value_label.setText(value)
        if subtitle is not None:
            self.subtitle_label.setText(subtitle)
        if color is not None:
            self.value_label.setStyleSheet(f"color:{color}; font-size:20px; font-weight:700;")


class PlotBase(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumHeight(180)

    @staticmethod
    def _map(value: float, source_min: float, source_max: float, target_min: float, target_max: float) -> float:
        if math.isclose(source_max, source_min):
            return target_min
        ratio = (value - source_min) / (source_max - source_min)
        return target_min + ratio * (target_max - target_min)

    def _draw_grid(
        self,
        painter: QPainter,
        plot: QRectF,
        x_ticks: Iterable[tuple[float, str]],
        y_ticks: Iterable[tuple[float, str]],
        x_range: tuple[float, float],
        y_range: tuple[float, float],
    ) -> None:
        grid_pen = QPen(QColor("#20303e"), 1)
        painter.setFont(QFont("Segoe UI", 8))
        for value, text in x_ticks:
            x = self._map(value, x_range[0], x_range[1], plot.left(), plot.right())
            painter.setPen(grid_pen)
            painter.drawLine(QPointF(x, plot.top()), QPointF(x, plot.bottom()))
            painter.setPen(QColor(MUTED))
            painter.drawText(QRectF(x - 35, plot.bottom() + 5, 70, 20), Qt.AlignmentFlag.AlignHCenter, text)
        for value, text in y_ticks:
            y = self._map(value, y_range[0], y_range[1], plot.bottom(), plot.top())
            painter.setPen(grid_pen)
            painter.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
            painter.setPen(QColor(MUTED))
            painter.drawText(QRectF(0, y - 8, plot.left() - 8, 16), Qt.AlignmentFlag.AlignRight, text)


class SensorgramChart(PlotBase):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(252)
        self.series = self._default_series()

    @staticmethod
    def _default_series() -> dict[str, list[tuple[float, float]]]:
        xs = [1500 + i * 2 for i in range(51)]

        def curve(center1: float, depth1: float, center2: float, depth2: float, width1: float, width2: float) -> list[tuple[float, float]]:
            values = []
            for x in xs:
                baseline = 0.985 - 0.0006 * (x - 1500)
                dip1 = depth1 * math.exp(-((x - center1) / width1) ** 2)
                dip2 = depth2 * math.exp(-((x - center2) / width2) ** 2)
                values.append((x, max(0.03, baseline - dip1 - dip2)))
            return values

        return {
            "Reference": curve(1540, 0.60, 1564, 0.25, 6.2, 9.5),
            "Sample": curve(1540, 0.52, 1567, 0.35, 5.8, 8.4),
            "Denoised": curve(1539, 0.66, 1564, 0.43, 5.8, 7.4),
            "Inverse Fit": curve(1540, 0.82, 1563, 0.47, 4.8, 7.0),
        }

    def set_series(self, series: dict[str, list[tuple[float, float]]]) -> None:
        if series:
            self.series = series
            self.update()

    def paintEvent(self, event) -> None:  # type: ignore[override]
        _ = event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(PANEL))
        plot = QRectF(51, 42, max(50, self.width() - 69), max(50, self.height() - 80))
        self._draw_grid(
            painter,
            plot,
            [(1500, "1500"), (1520, "1520"), (1540, "1540"), (1560, "1560"), (1580, "1580"), (1600, "1600")],
            [(0.0, "0.00"), (0.25, "0.25"), (0.5, "0.50"), (0.75, "0.75"), (1.0, "1.00")],
            (1500, 1600),
            (0.0, 1.0),
        )
        colors = {
            "Reference": BLUE,
            "Sample": GREEN,
            "Denoised": ORANGE,
            "Inverse Fit": PURPLE,
        }
        legend_x = plot.left() + 12
        painter.setFont(QFont("Segoe UI", 8))
        for name in ("Reference", "Sample", "Denoised", "Inverse Fit"):
            painter.setPen(QPen(QColor(colors[name]), 2))
            painter.drawLine(QPointF(legend_x, 17), QPointF(legend_x + 15, 17))
            painter.setPen(QColor(TEXT))
            painter.drawText(QRectF(legend_x + 20, 8, 72, 20), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, name)
            legend_x += 98

        for name, points in self.series.items():
            if not points:
                continue
            path = QPainterPath()
            for index, (x_value, y_value) in enumerate(points):
                x = self._map(x_value, 1500, 1600, plot.left(), plot.right())
                y = self._map(y_value, 0.0, 1.0, plot.bottom(), plot.top())
                if index == 0:
                    path.moveTo(x, y)
                else:
                    path.lineTo(x, y)
            painter.setPen(QPen(QColor(colors.get(name, BLUE)), 1.5))
            painter.drawPath(path)

        painter.setPen(QColor(TEXT))
        painter.setFont(QFont("Segoe UI", 8))
        painter.drawText(QRectF(plot.left(), self.height() - 24, plot.width(), 18), Qt.AlignmentFlag.AlignCenter, "Wavelength (nm)")
        painter.save()
        painter.translate(13, plot.center().y())
        painter.rotate(-90)
        painter.drawText(QRectF(-70, -10, 140, 20), Qt.AlignmentFlag.AlignCenter, "Intensity (a.u.)")
        painter.restore()


class TrainingChart(PlotBase):
    def __init__(self, color: str = BLUE, points: int = 121, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.color = color
        self.points = points
        self.setMinimumHeight(86)

    def paintEvent(self, event) -> None:  # type: ignore[override]
        _ = event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(PANEL))
        plot = QRectF(36, 8, max(40, self.width() - 43), max(32, self.height() - 24))
        painter.setPen(QPen(QColor("#1d2b38"), 1))
        for ratio in (0.0, 0.33, 0.66, 1.0):
            y = plot.top() + ratio * plot.height()
            painter.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
        path = QPainterPath()
        for index in range(self.points):
            x = plot.left() + (index / max(1, self.points - 1)) * plot.width()
            raw = math.exp(-index / 20.0) * 0.84 + 0.018 + 0.045 * math.sin(index * 1.8) / (1 + index * 0.05)
            y = plot.bottom() - max(0.0, min(1.0, raw)) * plot.height()
            if index == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)
        painter.setPen(QPen(QColor(self.color), 1.4))
        painter.drawPath(path)
        painter.setPen(QColor(MUTED))
        painter.setFont(QFont("Segoe UI", 7))
        painter.drawText(QRectF(plot.left() - 3, plot.bottom() + 1, 30, 14), "0")
        painter.drawText(QRectF(plot.right() - 24, plot.bottom() + 1, 24, 14), Qt.AlignmentFlag.AlignRight, str(self.points - 1))


class GaugeWidget(QWidget):
    def __init__(self, value: float = 412.7, maximum: float = 1000.0, color: str = GREEN, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.value = value
        self.maximum = maximum
        self.color = color
        self.setMinimumSize(150, 112)

    def set_value(self, value: float) -> None:
        self.value = max(0.0, value)
        self.update()

    def paintEvent(self, event) -> None:  # type: ignore[override]
        _ = event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(24, 10, self.width() - 48, self.height() - 12)
        start_angle = 210 * 16
        span = -240 * 16
        painter.setPen(QPen(QColor("#263746"), 11, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawArc(rect, start_angle, span)
        ratio = min(1.0, self.value / max(self.maximum, 1e-9))
        painter.setPen(QPen(QColor(self.color), 11, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawArc(rect, start_angle, int(span * ratio))
        painter.setPen(QColor(TEXT))
        painter.setFont(QFont("Segoe UI", 8))
        painter.drawText(QRectF(0, 44, self.width(), 18), Qt.AlignmentFlag.AlignCenter, "FPS")
        painter.setPen(QColor(self.color))
        painter.setFont(QFont("Segoe UI", 17, QFont.Weight.Bold))
        painter.drawText(QRectF(0, 59, self.width(), 28), Qt.AlignmentFlag.AlignCenter, f"{self.value:.1f}")
        painter.setPen(QColor(MUTED))
        painter.setFont(QFont("Segoe UI", 7))
        painter.drawText(QRectF(10, self.height() - 17, 30, 14), Qt.AlignmentFlag.AlignLeft, "0")
        painter.drawText(QRectF(self.width() - 44, self.height() - 17, 34, 14), Qt.AlignmentFlag.AlignRight, f"{int(self.maximum)}")


class DriftChart(PlotBase):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(100)

    def paintEvent(self, event) -> None:  # type: ignore[override]
        _ = event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(PANEL))
        plot = QRectF(35, 8, max(50, self.width() - 42), max(40, self.height() - 25))
        painter.setPen(QPen(QColor("#1d2b38"), 1))
        for ratio in (0.0, 0.5, 1.0):
            y = plot.top() + ratio * plot.height()
            painter.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
        path = QPainterPath()
        drift = 0.0
        for index in range(121):
            drift += math.sin(index * 1.7) * 0.09 + math.sin(index * 0.19) * 0.05
            drift *= 0.94
            normalized = 0.5 + max(-0.46, min(0.46, drift * 0.7))
            x = plot.left() + index / 120 * plot.width()
            y = plot.bottom() - normalized * plot.height()
            if index == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)
        painter.setPen(QPen(QColor(GREEN), 1.5))
        painter.drawPath(path)
        painter.setPen(QColor(MUTED))
        painter.setFont(QFont("Segoe UI", 7))
        painter.drawText(QRectF(0, plot.top() - 4, 30, 14), Qt.AlignmentFlag.AlignRight, "20")
        painter.drawText(QRectF(0, plot.center().y() - 7, 30, 14), Qt.AlignmentFlag.AlignRight, "0")
        painter.drawText(QRectF(0, plot.bottom() - 10, 30, 14), Qt.AlignmentFlag.AlignRight, "-20")
        painter.drawText(QRectF(plot.left(), plot.bottom() + 1, 30, 14), "0")
        painter.drawText(QRectF(plot.right() - 30, plot.bottom() + 1, 30, 14), Qt.AlignmentFlag.AlignRight, "120")


class MetricBox(Card):
    def __init__(self, title: str, value: str, color: str = TEXT, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(11, 9, 11, 9)
        layout.setSpacing(2)
        layout.addWidget(_label(title, "mutedSmall"))
        self.value_label = _label(value)
        self.value_label.setStyleSheet(f"color:{color}; font-size:16px; font-weight:600;")
        layout.addWidget(self.value_label)

    def set_value(self, value: str, color: str | None = None) -> None:
        self.value_label.setText(value)
        if color is not None:
            self.value_label.setStyleSheet(f"color:{color}; font-size:16px; font-weight:600;")


class QuickActionButton(QPushButton):
    def __init__(self, icon: str, text: str, style_name: str, parent: QWidget | None = None) -> None:
        super().__init__(f"{icon}   {text}", parent)
        self.setObjectName(style_name)
        self.setMinimumHeight(72)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setStyleSheet(self.styleSheet() + "font-size:12px; font-weight:600; text-align:left; padding-left:17px;")


@dataclass(frozen=True)
class NavItem:
    label: str
    icon: str
    action: str


class Sidebar(QFrame):
    action_requested = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("sidebar")
        self.setFixedWidth(216)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 13, 10, 12)
        outer.setSpacing(3)

        brand_row = QHBoxLayout()
        logo = QLabel("◉")
        logo.setStyleSheet(f"color:{BLUE}; font-size:30px; font-weight:700;")
        brand_row.addWidget(logo)
        brand_text = QVBoxLayout()
        brand_text.setSpacing(0)
        brand_text.addWidget(_label("CyberPhotonics-SPR", "brandTitle"))
        brand_text.addWidget(_label("Control Center", "brandSub"))
        brand_row.addLayout(brand_text, 1)
        outer.addLayout(brand_row)
        outer.addSpacing(9)

        self._nav_buttons: dict[str, QPushButton] = {}
        overview = self._add_nav(outer, NavItem("Overview", "⌂", "overview"), checkable=True)
        overview.setChecked(True)

        groups: list[tuple[str, list[NavItem]]] = [
            (
                "DATA & TRAINING",
                [
                    NavItem("Generate Dataset", "▣", "generate-data"),
                    NavItem("Train Inverse Model", "⌘", "train-inverse"),
                    NavItem("Train Edge Models", "▤", "train-edge"),
                    NavItem("Export Models", "⇥", "export-models"),
                ],
            ),
            (
                "PIPELINE & STREAMING",
                [
                    NavItem("Run Pipeline (A→B→C)", "≋", "run-pipeline"),
                    NavItem("Streaming Benchmark", "◌", "simulate-stream"),
                ],
            ),
            ("HIL LAB", [NavItem("HIL Benchmark", "▧", "hil-benchmark")]),
            (
                "RESEARCH DESIGN",
                [
                    NavItem("Inverse Design Studio", "⌁", "design"),
                    NavItem("Design Candidates", "▱", "candidates"),
                ],
            ),
            ("PHYSICS GATE", [NavItem("Physics Verification", "◈", "verify")]),
            (
                "EVIDENCE & REPORT",
                [
                    NavItem("Evidence Center", "▣", "evidence"),
                    NavItem("Generate Report", "▤", "report"),
                ],
            ),
        ]
        for title, items in groups:
            outer.addSpacing(9)
            section = _label(title)
            section.setStyleSheet(f"color:{MUTED}; font-size:10px; font-weight:600; letter-spacing:.3px;")
            outer.addWidget(section)
            for item in items:
                self._add_nav(outer, item)

        outer.addStretch(1)
        system_label = _label("SYSTEM")
        system_label.setStyleSheet(f"color:{MUTED}; font-size:10px; font-weight:600;")
        outer.addWidget(system_label)
        self._add_nav(outer, NavItem("Settings", "⚙", "settings"))
        self._add_nav(outer, NavItem("Logs", "▦", "logs"))
        outer.addSpacing(8)
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet(f"color:{BORDER};")
        outer.addWidget(line)
        footer = QHBoxLayout()
        footer_logo = QLabel("◎")
        footer_logo.setStyleSheet(f"color:{BLUE}; font-size:20px;")
        footer.addWidget(footer_logo)
        footer_text = QVBoxLayout()
        footer_text.setSpacing(0)
        footer_text.addWidget(_label("CyberPhotonics-SPR", "mutedSmall"))
        footer_text.addWidget(_label("v1.0.0", "mutedSmall"))
        footer.addLayout(footer_text, 1)
        outer.addLayout(footer)

    def _add_nav(self, layout: QVBoxLayout, item: NavItem, checkable: bool = False) -> QPushButton:
        button = QPushButton(f"{item.icon}   {item.label}")
        button.setObjectName("navButton")
        button.setCheckable(checkable)
        button.clicked.connect(lambda _checked=False, action=item.action: self.action_requested.emit(action))
        layout.addWidget(button)
        self._nav_buttons[item.action] = button
        return button


class StageCard(Card):
    def __init__(self, letter: str, title: str, subtitle: str, color: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setStyleSheet(
            f"QFrame#card{{background:{QColor(color).darker(420).name()}; border:1px solid {QColor(color).darker(170).name()}; border-radius:10px;}}"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(13, 14, 13, 14)
        layout.setSpacing(5)
        letter_label = _label(letter, alignment=Qt.AlignmentFlag.AlignCenter)
        letter_label.setStyleSheet(f"color:{color}; font-size:27px; font-weight:700;")
        title_label = _label(title, alignment=Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("font-size:12px; font-weight:600;")
        subtitle_label = _label(subtitle, "mutedSmall", Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(letter_label)
        layout.addWidget(title_label)
        layout.addWidget(subtitle_label)


class StatusDot(QWidget):
    def __init__(self, color: str = GREEN, diameter: int = 8, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.color = QColor(color)
        self.setFixedSize(diameter, diameter)

    def paintEvent(self, event) -> None:  # type: ignore[override]
        _ = event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(self.color)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(self.rect())
