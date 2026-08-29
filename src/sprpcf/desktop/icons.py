from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPainterPath, QPen, QPixmap, QPolygonF

from .theme import TEXT


_ACTION_ICON_NAMES = {
    "overview": "home",
    "generate-data": "database",
    "train-inverse": "model",
    "train-edge": "chip",
    "export-models": "export",
    "run-pipeline": "pipeline",
    "simulate-stream": "stream",
    "hil-benchmark": "hardware",
    "design": "design",
    "candidates": "candidates",
    "verify": "verify",
    "evidence": "evidence",
    "report": "report",
    "settings": "settings",
    "logs": "logs",
    "back": "back",
    "help": "help",
    "theme": "theme",
    "refresh": "refresh",
    "folder": "folder",
    "brand": "brand",
}


def action_icon(action: str, color: str = TEXT, size: int = 48) -> QIcon:
    return vector_icon(_ACTION_ICON_NAMES.get(action, action), color=color, size=size)


def vector_icon(name: str, color: str = TEXT, size: int = 48) -> QIcon:
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    scale = size / 24.0
    painter.scale(scale, scale)
    pen = QPen(QColor(color), 1.8)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)

    if name == "home":
        painter.drawPolyline(QPolygonF([QPointF(3, 11), QPointF(12, 4), QPointF(21, 11)]))
        painter.drawRoundedRect(QRectF(5, 10, 14, 10), 1.5, 1.5)
        painter.drawRect(QRectF(10, 14, 4, 6))
    elif name == "database":
        painter.drawEllipse(QRectF(4, 4, 16, 5))
        painter.drawArc(QRectF(4, 8, 16, 5), 180 * 16, 180 * 16)
        painter.drawArc(QRectF(4, 13, 16, 5), 180 * 16, 180 * 16)
        painter.drawArc(QRectF(4, 15, 16, 5), 0, -180 * 16)
        painter.drawLine(QPointF(4, 6.5), QPointF(4, 17.5))
        painter.drawLine(QPointF(20, 6.5), QPointF(20, 17.5))
    elif name == "model":
        nodes = [(6, 6), (18, 6), (12, 12), (6, 18), (18, 18)]
        for a, b in ((0, 2), (1, 2), (2, 3), (2, 4), (0, 1), (3, 4)):
            painter.drawLine(QPointF(*nodes[a]), QPointF(*nodes[b]))
        for x, y in nodes:
            painter.drawEllipse(QRectF(x - 1.8, y - 1.8, 3.6, 3.6))
    elif name in {"chip", "hardware"}:
        painter.drawRoundedRect(QRectF(6, 6, 12, 12), 2, 2)
        for value in (8, 12, 16):
            painter.drawLine(QPointF(value, 3), QPointF(value, 6))
            painter.drawLine(QPointF(value, 18), QPointF(value, 21))
            painter.drawLine(QPointF(3, value), QPointF(6, value))
            painter.drawLine(QPointF(18, value), QPointF(21, value))
        if name == "hardware":
            path = QPainterPath(QPointF(8, 13))
            path.lineTo(10, 10)
            path.lineTo(12, 14)
            path.lineTo(14, 9)
            path.lineTo(16, 12)
            painter.drawPath(path)
        else:
            painter.drawRoundedRect(QRectF(9, 9, 6, 6), 1, 1)
    elif name == "export":
        painter.drawRoundedRect(QRectF(4, 8, 12, 12), 1.5, 1.5)
        painter.drawLine(QPointF(11, 13), QPointF(20, 4))
        painter.drawPolyline(QPolygonF([QPointF(14, 4), QPointF(20, 4), QPointF(20, 10)]))
    elif name == "pipeline":
        for x in (5, 12, 19):
            painter.drawEllipse(QRectF(x - 2.2, 9.8, 4.4, 4.4))
        painter.drawLine(QPointF(7.2, 12), QPointF(9.8, 12))
        painter.drawLine(QPointF(14.2, 12), QPointF(16.8, 12))
        painter.drawLine(QPointF(9, 10.8), QPointF(10.2, 12))
        painter.drawLine(QPointF(9, 13.2), QPointF(10.2, 12))
        painter.drawLine(QPointF(16, 10.8), QPointF(17.2, 12))
        painter.drawLine(QPointF(16, 13.2), QPointF(17.2, 12))
    elif name == "stream":
        path = QPainterPath(QPointF(3, 12))
        path.cubicTo(6, 4, 9, 20, 12, 12)
        path.cubicTo(15, 4, 18, 20, 21, 12)
        painter.drawPath(path)
        painter.drawLine(QPointF(3, 18.5), QPointF(21, 18.5))
    elif name == "design":
        painter.drawLine(QPointF(5, 19), QPointF(16.8, 7.2))
        painter.drawPolyline(QPolygonF([QPointF(16.8, 7.2), QPointF(19.5, 4.5), QPointF(21, 6), QPointF(18.2, 8.7)]))
        painter.drawPolyline(QPolygonF([QPointF(5, 19), QPointF(4, 21), QPointF(6, 20)]))
        painter.drawArc(QRectF(6, 5, 9, 9), 30 * 16, 235 * 16)
    elif name == "candidates":
        painter.drawRoundedRect(QRectF(5, 5, 13, 14), 1.5, 1.5)
        painter.drawRoundedRect(QRectF(8, 3, 11, 14), 1.5, 1.5)
        painter.drawLine(QPointF(11, 8), QPointF(16, 8))
        painter.drawLine(QPointF(11, 11), QPointF(16, 11))
        painter.drawLine(QPointF(11, 14), QPointF(14, 14))
    elif name == "verify":
        shield = QPainterPath(QPointF(12, 3))
        shield.lineTo(19, 6)
        shield.lineTo(18, 13)
        shield.cubicTo(17.5, 17, 14.8, 19.6, 12, 21)
        shield.cubicTo(9.2, 19.6, 6.5, 17, 6, 13)
        shield.lineTo(5, 6)
        shield.closeSubpath()
        painter.drawPath(shield)
        painter.drawPolyline(QPolygonF([QPointF(8.5, 12), QPointF(11, 14.5), QPointF(15.8, 9.5)]))
    elif name == "evidence":
        painter.drawRoundedRect(QRectF(5, 5, 14, 16), 1.5, 1.5)
        painter.drawRoundedRect(QRectF(9, 3, 6, 4), 1, 1)
        painter.drawPolyline(QPolygonF([QPointF(8, 12), QPointF(10, 14), QPointF(13, 10)]))
        painter.drawLine(QPointF(14, 12), QPointF(17, 12))
        painter.drawLine(QPointF(8, 17), QPointF(17, 17))
    elif name == "report":
        page = QPainterPath(QPointF(6, 3))
        page.lineTo(15, 3)
        page.lineTo(19, 7)
        page.lineTo(19, 21)
        page.lineTo(6, 21)
        page.closeSubpath()
        painter.drawPath(page)
        painter.drawPolyline(QPolygonF([QPointF(15, 3), QPointF(15, 7), QPointF(19, 7)]))
        painter.drawLine(QPointF(9, 11), QPointF(16, 11))
        painter.drawLine(QPointF(9, 14), QPointF(16, 14))
        painter.drawLine(QPointF(9, 17), QPointF(14, 17))
    elif name == "settings":
        painter.drawEllipse(QRectF(8.5, 8.5, 7, 7))
        painter.drawEllipse(QRectF(11, 11, 2, 2))
        for x1, y1, x2, y2 in (
            (12, 3, 12, 7), (12, 17, 12, 21), (3, 12, 7, 12), (17, 12, 21, 12),
            (5.6, 5.6, 8.3, 8.3), (15.7, 15.7, 18.4, 18.4),
            (18.4, 5.6, 15.7, 8.3), (8.3, 15.7, 5.6, 18.4),
        ):
            painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))
    elif name == "logs":
        painter.drawRoundedRect(QRectF(3.5, 5, 17, 14), 1.5, 1.5)
        painter.drawPolyline(QPolygonF([QPointF(7, 9), QPointF(10, 12), QPointF(7, 15)]))
        painter.drawLine(QPointF(12, 15), QPointF(17, 15))
    elif name == "back":
        painter.drawLine(QPointF(20, 12), QPointF(5, 12))
        painter.drawPolyline(QPolygonF([QPointF(10, 6), QPointF(4, 12), QPointF(10, 18)]))
    elif name == "help":
        painter.drawEllipse(QRectF(3, 3, 18, 18))
        painter.setFont(QFont("Arial", 13, QFont.Weight.DemiBold))
        painter.drawText(QRectF(3, 3, 18, 18), Qt.AlignmentFlag.AlignCenter, "?")
    elif name == "theme":
        moon = QPainterPath()
        moon.addEllipse(QRectF(5, 4, 14, 16))
        cut = QPainterPath()
        cut.addEllipse(QRectF(10, 2.5, 12, 14))
        painter.setBrush(QColor(color))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawPath(moon.subtracted(cut))
    elif name == "refresh":
        painter.drawArc(QRectF(4, 4, 16, 16), 35 * 16, 285 * 16)
        painter.drawPolyline(QPolygonF([QPointF(17.5, 4.3), QPointF(20.5, 4.5), QPointF(20.1, 7.5)]))
    elif name == "folder":
        path = QPainterPath(QPointF(3, 7))
        path.lineTo(9, 7)
        path.lineTo(11, 9)
        path.lineTo(21, 9)
        path.lineTo(19, 19)
        path.lineTo(3, 19)
        path.closeSubpath()
        painter.drawPath(path)
    elif name == "brand":
        painter.drawEllipse(QRectF(5, 5, 14, 14))
        painter.drawEllipse(QRectF(9, 9, 6, 6))
        for angle in range(0, 360, 45):
            import math
            radians = math.radians(angle)
            painter.drawLine(
                QPointF(12 + math.cos(radians) * 9, 12 + math.sin(radians) * 9),
                QPointF(12 + math.cos(radians) * 11, 12 + math.sin(radians) * 11),
            )
    else:
        painter.drawRoundedRect(QRectF(4, 4, 16, 16), 2, 2)

    painter.end()
    return QIcon(pixmap)
