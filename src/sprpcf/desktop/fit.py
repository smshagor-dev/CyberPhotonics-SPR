from __future__ import annotations

import os
import sys

from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication, QLayout, QSizePolicy

from .responsive import ResponsiveControlCenter as _ResponsiveControlCenter
from .responsive import large_dataset_form


class ResponsiveControlCenter(_ResponsiveControlCenter):
    """Windows-safe responsive shell with zero horizontal dashboard overflow."""

    def _configure_scroll_area(self) -> None:
        super()._configure_scroll_area()
        scroll = getattr(self, "_scroll_area", None)
        if scroll is None:
            return

        content = scroll.widget()
        if content is None:
            return

        # Windows/Segoe UI reports wider minimum size hints than Linux Qt.
        # Let the scroll viewport own horizontal sizing and allow responsive
        # grids to reflow instead of letting child size hints widen the canvas.
        content.setMinimumWidth(0)
        policy = content.sizePolicy()
        policy.setHorizontalPolicy(QSizePolicy.Policy.Ignored)
        policy.setVerticalPolicy(QSizePolicy.Policy.Preferred)
        content.setSizePolicy(policy)
        layout = content.layout()
        if layout is not None:
            layout.setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)

    def _content_width(self) -> int:
        width = super()._content_width()
        # At laptop widths reserve extra room for Windows font/DPI differences.
        # This forces lower panels and quick actions to stack sooner, while the
        # approved 1536px desktop composition remains unchanged.
        if width < 1000:
            return max(1, width - 180)
        return width


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
