from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QPushButton

from sprpcf.desktop import ResponsiveControlCenter
from sprpcf.desktop.app import APP_TITLE, ControlCenter
from sprpcf.desktop.dialogs import generate_data_form, pipeline_form
from sprpcf.desktop.widgets import StatCard


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_native_control_center_renders_without_web_runtime() -> None:
    app = _app()
    window = ControlCenter()
    try:
        window.resize(1536, 1024)
        window.show()
        app.processEvents()

        assert APP_TITLE in window.windowTitle()
        assert window.minimumWidth() >= 1100
        assert window.sidebar is not None
        cards = window.findChildren(StatCard)
        assert len(cards) == 6
        assert window.dataset_card.value_label.text()
        assert window.inverse_card.value_label.text()
        assert window.pipeline_card.value_label.text()
        labels = [button.text() for button in window.findChildren(QPushButton)]
        assert any("Generate" in value and "Dataset" in value for value in labels)
        assert any("Run" in value and "Pipeline" in value for value in labels)
        assert any("Verify" in value and "Physics" in value for value in labels)
        assert any("Generate" in value and "Report" in value for value in labels)

        rendered = window.grab()
        assert not rendered.isNull()
        assert rendered.width() == 1536
        assert rendered.height() == 1024
    finally:
        window.close()
        app.processEvents()


def test_responsive_control_center_renders_at_laptop_size() -> None:
    app = _app()
    window = ResponsiveControlCenter()
    try:
        window.resize(1024, 720)
        window.show()
        app.processEvents()
        window._apply_responsive_layout(force=True)
        app.processEvents()

        assert window.minimumWidth() == 900
        assert window._last_layout_bucket is not None
        assert window._last_layout_bucket[1] == 1
        assert window.sidebar.width() <= 186
        assert window._scroll_area is not None
        assert window._scroll_area.horizontalScrollBar().maximum() == 0

        rendered = window.grab()
        assert not rendered.isNull()
        assert rendered.width() == 1024
        assert rendered.height() == 720
    finally:
        window.close()
        app.processEvents()


def test_operation_forms_are_native_qt_dialogs() -> None:
    _app()
    generate = generate_data_form()
    pipeline = pipeline_form()
    try:
        assert generate.windowTitle() == "Generate Dataset"
        assert pipeline.windowTitle() == "Run A → B → C Pipeline"
        assert "samples" in generate._widgets
        assert "inverse_epochs" in pipeline._widgets
    finally:
        generate.close()
        pipeline.close()
