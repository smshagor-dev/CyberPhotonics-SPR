from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QApplication

from .theme import BLUE, GREEN, ORANGE, PURPLE, palette_for
from .widgets import DriftChart, GaugeWidget, SensorgramChart, StageCard, TrainingChart


def _mode() -> str:
    app = QApplication.instance()
    if app is None:
        return "dark"
    value = app.property("sprpcfTheme")
    return str(value or "dark")


def _theme():
    return palette_for(_mode())


def _blend(first: str, second: str, second_weight: float) -> str:
    a = QColor(first)
    b = QColor(second)
    weight = max(0.0, min(1.0, second_weight))
    red = round(a.red() * (1.0 - weight) + b.red() * weight)
    green = round(a.green() * (1.0 - weight) + b.green() * weight)
    blue = round(a.blue() * (1.0 - weight) + b.blue() * weight)
    return QColor(red, green, blue).name()


class ThemedSensorgramChart(SensorgramChart):
    def _draw_theme_grid(
        self,
        painter: QPainter,
        plot: QRectF,
        x_ticks: list[tuple[float, str]],
        y_ticks: list[tuple[float, str]],
        x_range: tuple[float, float],
        y_range: tuple[float, float],
    ) -> None:
        theme = _theme()
        grid_pen = QPen(QColor(theme.grid), 1)
        painter.setFont(QFont("Segoe UI", 8))
        for value, text in x_ticks:
            x = self._map(value, x_range[0], x_range[1], plot.left(), plot.right())
            painter.setPen(grid_pen)
            painter.drawLine(QPointF(x, plot.top()), QPointF(x, plot.bottom()))
            painter.setPen(QColor(theme.muted))
            painter.drawText(QRectF(x - 35, plot.bottom() + 5, 70, 20), Qt.AlignmentFlag.AlignHCenter, text)
        for value, text in y_ticks:
            y = self._map(value, y_range[0], y_range[1], plot.bottom(), plot.top())
            painter.setPen(grid_pen)
            painter.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
            painter.setPen(QColor(theme.muted))
            painter.drawText(QRectF(0, y - 8, plot.left() - 8, 16), Qt.AlignmentFlag.AlignRight, text)

    def paintEvent(self, event) -> None:  # type: ignore[override]
        _ = event
        theme = _theme()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(theme.plot_bg))
        plot = QRectF(51, 42, max(50, self.width() - 69), max(50, self.height() - 80))
        self._draw_theme_grid(
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
            painter.setPen(QColor(theme.text))
            painter.drawText(
                QRectF(legend_x + 20, 8, 72, 20),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                name,
            )
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
            painter.setPen(QPen(QColor(colors.get(name, BLUE)), 1.6))
            painter.drawPath(path)

        painter.setPen(QColor(theme.text))
        painter.setFont(QFont("Segoe UI", 8))
        painter.drawText(
            QRectF(plot.left(), self.height() - 24, plot.width(), 18),
            Qt.AlignmentFlag.AlignCenter,
            "Wavelength (nm)",
        )
        painter.save()
        painter.translate(13, plot.center().y())
        painter.rotate(-90)
        painter.drawText(QRectF(-70, -10, 140, 20), Qt.AlignmentFlag.AlignCenter, "Intensity (a.u.)")
        painter.restore()


class ThemedTrainingChart(TrainingChart):
    def paintEvent(self, event) -> None:  # type: ignore[override]
        _ = event
        theme = _theme()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(theme.plot_bg))
        plot = QRectF(36, 8, max(40, self.width() - 43), max(32, self.height() - 24))
        painter.setPen(QPen(QColor(theme.grid), 1))
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
        painter.setPen(QPen(QColor(self.color), 1.5))
        painter.drawPath(path)
        painter.setPen(QColor(theme.muted))
        painter.setFont(QFont("Segoe UI", 7))
        painter.drawText(QRectF(plot.left() - 3, plot.bottom() + 1, 30, 14), "0")
        painter.drawText(
            QRectF(plot.right() - 24, plot.bottom() + 1, 24, 14),
            Qt.AlignmentFlag.AlignRight,
            str(self.points - 1),
        )


class ThemedGaugeWidget(GaugeWidget):
    def paintEvent(self, event) -> None:  # type: ignore[override]
        _ = event
        theme = _theme()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(24, 10, self.width() - 48, self.height() - 12)
        start_angle = 210 * 16
        span = -240 * 16
        painter.setPen(QPen(QColor(theme.border), 11, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawArc(rect, start_angle, span)
        ratio = min(1.0, self.value / max(self.maximum, 1e-9))
        painter.setPen(QPen(QColor(self.color), 11, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawArc(rect, start_angle, int(span * ratio))
        painter.setPen(QColor(theme.text))
        painter.setFont(QFont("Segoe UI", 8))
        painter.drawText(QRectF(0, 44, self.width(), 18), Qt.AlignmentFlag.AlignCenter, "FPS")
        painter.setPen(QColor(self.color))
        painter.setFont(QFont("Segoe UI", 17, QFont.Weight.Bold))
        painter.drawText(QRectF(0, 59, self.width(), 28), Qt.AlignmentFlag.AlignCenter, f"{self.value:.1f}")
        painter.setPen(QColor(theme.muted))
        painter.setFont(QFont("Segoe UI", 7))
        painter.drawText(QRectF(10, self.height() - 17, 30, 14), Qt.AlignmentFlag.AlignLeft, "0")
        painter.drawText(
            QRectF(self.width() - 44, self.height() - 17, 34, 14),
            Qt.AlignmentFlag.AlignRight,
            f"{int(self.maximum)}",
        )


class ThemedDriftChart(DriftChart):
    def paintEvent(self, event) -> None:  # type: ignore[override]
        _ = event
        theme = _theme()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(theme.plot_bg))
        plot = QRectF(35, 8, max(50, self.width() - 42), max(40, self.height() - 25))
        painter.setPen(QPen(QColor(theme.grid), 1))
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
        painter.setPen(QPen(QColor(GREEN), 1.6))
        painter.drawPath(path)
        painter.setPen(QColor(theme.muted))
        painter.setFont(QFont("Segoe UI", 7))
        painter.drawText(QRectF(0, plot.top() - 4, 30, 14), Qt.AlignmentFlag.AlignRight, "20")
        painter.drawText(QRectF(0, plot.center().y() - 7, 30, 14), Qt.AlignmentFlag.AlignRight, "0")
        painter.drawText(QRectF(0, plot.bottom() - 10, 30, 14), Qt.AlignmentFlag.AlignRight, "-20")
        painter.drawText(QRectF(plot.left(), plot.bottom() + 1, 30, 14), "0")
        painter.drawText(
            QRectF(plot.right() - 30, plot.bottom() + 1, 30, 14),
            Qt.AlignmentFlag.AlignRight,
            "120",
        )


class ThemedStageCard(StageCard):
    def __init__(self, letter: str, title: str, subtitle: str, color: str, parent=None) -> None:
        self._accent = color
        super().__init__(letter, title, subtitle, color, parent)
        self.apply_theme()

    def apply_theme(self) -> None:
        theme = _theme()
        if theme.name == "light":
            background = _blend("#ffffff", self._accent, 0.055)
            border = _blend("#ffffff", self._accent, 0.34)
        else:
            background = QColor(self._accent).darker(420).name()
            border = QColor(self._accent).darker(170).name()
        self.setStyleSheet(
            f"QFrame#card{{background:{background}; border:1px solid {border}; border-radius:10px;}}"
        )
