from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QPalette
from PySide6.QtWidgets import QApplication, QPushButton

from sprpcf.desktop import ResponsiveControlCenter
from sprpcf.desktop.app import APP_TITLE, ControlCenter
from sprpcf.desktop.dialogs import generate_data_form, pipeline_form
from sprpcf.desktop.fit import ACTION_META, QUICK_ACTIONS
from sprpcf.desktop.icons import action_icon
from sprpcf.desktop.theme import DARK_THEME, LIGHT_THEME, palette_for, stylesheet_for
from sprpcf.desktop.themed_widgets import ThemedSensorgramChart
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
        assert window._scroll_area.verticalScrollBar().maximum() > 0

        rendered = window.grab()
        assert not rendered.isNull()
        assert rendered.width() == 1024
        assert rendered.height() == 720
    finally:
        window.close()
        app.processEvents()


def test_every_sidebar_menu_has_semantic_icon_action_and_tooltip() -> None:
    app = _app()
    window = ResponsiveControlCenter()
    try:
        window.show()
        app.processEvents()
        buttons = window.sidebar._nav_buttons
        assert set(buttons) == set(ACTION_META)
        for action, button in buttons.items():
            label, tooltip = ACTION_META[action]
            assert button.text() == label
            assert button.property("action") == action
            assert button.toolTip() == tooltip
            assert button.isCheckable()
            assert button.autoExclusive()
            assert not button.icon().isNull()
            assert button.accessibleName() == label
            assert button.accessibleDescription() == tooltip
    finally:
        window.close()
        app.processEvents()


def test_quick_actions_and_topbar_icons_are_actionable() -> None:
    app = _app()
    window = ResponsiveControlCenter()
    try:
        window.show()
        app.processEvents()
        expected_quick = [action for action, _label_text in QUICK_ACTIONS]
        assert [button.property("action") for button in window._quick_buttons] == expected_quick
        assert all(not button.icon().isNull() for button in window._quick_buttons)
        assert all(button.toolTip() for button in window._quick_buttons)

        top_actions = [button.property("action") for button in window._top_buttons]
        assert top_actions == ["settings", "help", "toggle-theme"]
        assert window._back_button.property("action") == "overview"
        assert not window._back_button.icon().isNull()
        assert all(not button.icon().isNull() for button in window._top_buttons)
        assert all(button.toolTip() for button in window._top_buttons)
    finally:
        window.close()
        app.processEvents()


def test_back_button_returns_to_overview_and_theme_toggle_persists() -> None:
    app = _app()
    window = ResponsiveControlCenter()
    original_theme = window._theme_mode()
    try:
        window.show()
        app.processEvents()
        window._set_active_action("report")
        assert window._current_action == "report"
        window._back_button.click()
        app.processEvents()
        assert window._current_action == "overview"
        assert window.sidebar._nav_buttons["overview"].isChecked()

        window._toggle_theme()
        assert window._theme_mode() != original_theme
        assert window._theme_mode() in {"dark", "light"}
        assert "Switch to" in window._top_theme_button.toolTip()
    finally:
        window._ui_settings.setValue("theme", original_theme)
        window._ui_settings.sync()
        window.close()
        app.processEvents()


def test_light_mode_is_true_white_and_recolors_custom_charts() -> None:
    app = _app()
    window = ResponsiveControlCenter()
    original_theme = window._theme_mode()
    try:
        window._ui_settings.setValue("theme", "light")
        window._ui_settings.sync()
        window._apply_preferences()
        window.resize(1200, 800)
        window.show()
        app.processEvents()

        assert window._theme_mode() == "light"
        assert window.property("theme") == "light"
        assert palette_for("light").window == "#ffffff"
        assert app.palette().color(QPalette.ColorRole.Window).name() == "#ffffff"
        assert isinstance(window.sensor_chart, ThemedSensorgramChart)
        chart = window.sensor_chart.grab().toImage()
        assert chart.pixelColor(2, 2).name() == LIGHT_THEME.plot_bg
        assert LIGHT_THEME.primary == "#0b2a4a"
        assert "background: #ffffff" in stylesheet_for("light")
    finally:
        window._ui_settings.setValue("theme", original_theme)
        window._ui_settings.sync()
        window.close()
        app.processEvents()


def test_dark_mode_is_deep_navy_and_not_light_blue() -> None:
    app = _app()
    window = ResponsiveControlCenter()
    original_theme = window._theme_mode()
    try:
        window._ui_settings.setValue("theme", "dark")
        window._ui_settings.sync()
        window._apply_preferences()
        window.show()
        app.processEvents()

        assert window._theme_mode() == "dark"
        assert DARK_THEME.window == "#050c15"
        assert DARK_THEME.panel == "#0b1622"
        assert app.palette().color(QPalette.ColorRole.Window).name() == DARK_THEME.window
        chart = window.sensor_chart.grab().toImage()
        assert chart.pixelColor(2, 2).name() == DARK_THEME.plot_bg
    finally:
        window._ui_settings.setValue("theme", original_theme)
        window._ui_settings.sync()
        window.close()
        app.processEvents()


def test_legacy_high_contrast_setting_migrates_to_supported_dark_mode() -> None:
    app = _app()
    window = ResponsiveControlCenter()
    original_theme = window._theme_mode()
    try:
        window._ui_settings.setValue("theme", "high-contrast")
        window._ui_settings.sync()
        assert window._theme_mode() == "dark"
        window._migrate_theme_setting()
        assert str(window._ui_settings.value("theme")) == "dark"
    finally:
        window._ui_settings.setValue("theme", original_theme)
        window._ui_settings.sync()
        window.close()
        app.processEvents()


def test_vector_icon_factory_has_non_null_icons_for_all_actions() -> None:
    _app()
    for action in (*ACTION_META, "back", "help", "theme", "refresh", "folder", "brand"):
        icon = action_icon(action)
        assert not icon.isNull()
        assert not icon.pixmap(24, 24).isNull()


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
