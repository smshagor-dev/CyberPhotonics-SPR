from __future__ import annotations

import os
import sys

from PySide6.QtCore import QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication, QSizePolicy

from .responsive import ResponsiveControlCenter as _ResponsiveControlCenter
from .responsive import large_dataset_form


class ResponsiveControlCenter(_ResponsiveControlCenter):
    """Windows-safe responsive shell with zero horizontal dashboard overflow."""

    def _configure_scroll_area(self) -> None:
        super()._configure_scroll_area()
        scroll = getattr(self, "_scroll_area", None)
        if scroll is None:
            self._responsive_content = None
            return

        content = scroll.widget()
        self._responsive_content = content
        if content is None:
            return

        # Windows/Segoe UI reports wider minimum size hints than Linux Qt.
        # Ignore horizontal size hints, but preserve the normal vertical size
        # hint so the dashboard can still scroll down on laptop-height screens.
        content.setMinimumWidth(0)
        policy = content.sizePolicy()
        policy.setHorizontalPolicy(QSizePolicy.Policy.Ignored)
        policy.setVerticalPolicy(QSizePolicy.Policy.Preferred)
        content.setSizePolicy(policy)

    def _content_width(self) -> int:
        width = super()._content_width()
        # At laptop widths reserve extra room for Windows font/DPI differences.
        # This forces lower panels and quick actions to stack sooner, while the
        # approved 1536px desktop composition remains unchanged.
        if width < 1000:
            return max(1, width - 180)
        return width

    def _apply_responsive_layout(self, force: bool = False) -> None:
        super()._apply_responsive_layout(force=force)
        self._sync_content_width()
        # Sidebar/topbar compaction can change the viewport a moment after the
        # main resize event. Re-pin after Qt has processed that geometry change.
        QTimer.singleShot(0, self._sync_content_width)

    def _sync_content_width(self) -> None:
        scroll = getattr(self, "_scroll_area", None)
        content = getattr(self, "_responsive_content", None)
        if scroll is None or content is None:
            return
        viewport_width = max(1, scroll.viewport().width())
        content.setFixedWidth(viewport_width)


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
