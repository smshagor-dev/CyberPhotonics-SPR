from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from sprpcf.desktop import ResponsiveControlCenter


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_responsive_control_center_reflows_without_horizontal_scroll() -> None:
    app = _app()
    window = ResponsiveControlCenter()
    try:
        window.resize(1024, 720)
        window.show()
        app.processEvents()
        window._apply_responsive_layout(force=True)
        app.processEvents()

        assert window._last_layout_bucket is not None
        stat_columns, main_columns, lower_columns, quick_columns = window._last_layout_bucket
        assert stat_columns <= 3
        assert main_columns == 1
        assert lower_columns <= 2
        assert quick_columns <= 4
        assert window.sidebar.width() <= 186

        scroll = window._scroll_area
        assert scroll is not None
        assert scroll.horizontalScrollBar().maximum() == 0

        window.resize(1536, 1024)
        app.processEvents()
        window._apply_responsive_layout(force=True)
        app.processEvents()
        assert window._last_layout_bucket[0] >= 3
        assert window._last_layout_bucket[1] == 2
        assert window._last_layout_bucket[2] >= 2
        assert scroll.horizontalScrollBar().maximum() == 0
    finally:
        window.close()


def test_large_dataset_form_defaults_to_research_scale() -> None:
    from sprpcf.desktop.responsive import large_dataset_form

    _app()
    form = large_dataset_form()
    try:
        assert form.windowTitle() == "Generate Large Dataset"
        assert form._widgets["samples"].value() == 10_000
        assert form._widgets["chunk"].value() == 500
        assert form._widgets["wavelengths"].value() == 256
    finally:
        form.close()
