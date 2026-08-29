from __future__ import annotations

import os
import sys
from pathlib import Path

from PySide6.QtCore import QSettings, QSize
from PySide6.QtGui import QDesktopServices, QFont, QIcon, QUrl
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from sprpcf.dashboard.operations import PROJECT_ROOT

from . import app as desktop_app
from .fit import ACTION_META, QUICK_ACTIONS, REPOSITORY_URL
from .fit import ResponsiveControlCenter as _ActionableControlCenter
from .fit import large_dataset_form
from .icons import action_icon
from .theme import GREEN, ORANGE, normalize_theme, palette_for, qt_palette_for, stylesheet_for
from .themed_widgets import (
    ThemedDriftChart,
    ThemedGaugeWidget,
    ThemedSensorgramChart,
    ThemedStageCard,
    ThemedTrainingChart,
)


_SIDEBAR_SECTION_LABELS = {
    "DATA & TRAINING",
    "PIPELINE & STREAMING",
    "HIL LAB",
    "RESEARCH DESIGN",
    "PHYSICS GATE",
    "EVIDENCE & REPORT",
    "SYSTEM",
}


def _install_theme_aware_visuals() -> None:
    # ControlCenter builder methods resolve these globals at runtime. Replacing
    # only the custom-painted widgets keeps all existing backend/action logic
    # intact while allowing their canvases to follow the active palette.
    desktop_app.SensorgramChart = ThemedSensorgramChart
    desktop_app.TrainingChart = ThemedTrainingChart
    desktop_app.GaugeWidget = ThemedGaugeWidget
    desktop_app.DriftChart = ThemedDriftChart
    desktop_app.StageCard = ThemedStageCard


def _stateful_icon(action: str, normal_color: str, selected_color: str, size: int = 48) -> QIcon:
    icon = QIcon()
    normal = action_icon(action, color=normal_color, size=size).pixmap(size, size)
    selected = action_icon(action, color=selected_color, size=size).pixmap(size, size)
    icon.addPixmap(normal, QIcon.Mode.Normal, QIcon.State.Off)
    icon.addPixmap(selected, QIcon.Mode.Normal, QIcon.State.On)
    icon.addPixmap(normal, QIcon.Mode.Active, QIcon.State.Off)
    icon.addPixmap(selected, QIcon.Mode.Active, QIcon.State.On)
    return icon


class ResponsiveControlCenter(_ActionableControlCenter):
    """Production native shell with true white/light and deep-navy/dark modes."""

    def __init__(self) -> None:
        _install_theme_aware_visuals()
        super().__init__()
        self._migrate_theme_setting()
        self._apply_preferences()

    def _configure_scroll_area(self) -> None:
        super()._configure_scroll_area()
        scroll = getattr(self, "_scroll_area", None)
        if scroll is not None:
            # The original dashboard embedded a dark background directly on the
            # QScrollArea. Remove it so the runtime theme controls the canvas.
            scroll.setStyleSheet("")

    def _theme_mode(self) -> str:
        return normalize_theme(str(self._ui_settings.value("theme", "dark")))

    def _migrate_theme_setting(self) -> None:
        raw = str(self._ui_settings.value("theme", "dark"))
        normalized = normalize_theme(raw)
        if raw != normalized:
            self._ui_settings.setValue("theme", normalized)
            self._ui_settings.sync()

    def _icon_button(self, icon_name: str, tooltip: str, action: str) -> QPushButton:
        theme = palette_for(self._theme_mode())
        button = QPushButton()
        button.setObjectName("topIconButton")
        button.setFixedSize(34, 34)
        button.setIcon(action_icon(icon_name, color=theme.text))
        button.setIconSize(QSize(19, 19))
        button.setToolTip(tooltip)
        button.setAccessibleName(tooltip)
        button.setProperty("action", action)
        button.clicked.connect(lambda _checked=False, key=action: self.handle_action(key))
        return button

    def _refresh_action_icons(self) -> None:
        theme = palette_for(self._theme_mode())
        normal_nav = theme.nav_text
        selected_nav = "#ffffff"
        nav_buttons = getattr(self.sidebar, "_nav_buttons", {})
        for action, button in nav_buttons.items():
            button.setIcon(_stateful_icon(action, normal_nav, selected_nav))
            button.setIconSize(QSize(18, 18))

        quick_icon_color = theme.primary if theme.name == "light" else theme.text
        for button in getattr(self, "_quick_buttons", []):
            action = str(button.property("action") or "")
            if action:
                button.setIcon(action_icon(action, color=quick_icon_color))
                button.setIconSize(QSize(23, 23))

        if hasattr(self, "_back_button"):
            self._back_button.setIcon(action_icon("back", color=theme.text))
        for button in getattr(self, "_top_buttons", []):
            action = str(button.property("action") or "")
            icon_name = "theme" if action == "toggle-theme" else action
            button.setIcon(action_icon(icon_name, color=theme.text))

        # Brand marks are pixmaps, so rebuild them as the palette changes.
        for label in self.sidebar.findChildren(QLabel):
            if label.property("brandMark"):
                size = int(label.property("brandSize") or 20)
                label.setPixmap(action_icon("brand", color=theme.primary).pixmap(size, size))

    def _tag_brand_marks(self) -> None:
        # The actionable-icon layer already replaced legacy glyphs with pixmaps.
        # Tag those pixmap-only labels once so light/dark switching can recolor them.
        for label in self.sidebar.findChildren(QLabel):
            pixmap = label.pixmap()
            if not label.text() and pixmap is not None and not pixmap.isNull():
                width = label.width()
                if width <= 34:
                    label.setProperty("brandMark", True)
                    label.setProperty("brandSize", 28 if width >= 28 else 20)

    def _restyle_sidebar_sections(self) -> None:
        theme = palette_for(self._theme_mode())
        for label in self.sidebar.findChildren(QLabel):
            if label.text() in _SIDEBAR_SECTION_LABELS:
                label.setStyleSheet(
                    f"color:{theme.muted}; font-size:10px; font-weight:600; letter-spacing:.3px;"
                )

    def _restyle_dynamic_labels(self) -> None:
        theme = palette_for(self._theme_mode())
        healthy = getattr(self, "health_label", None) is not None and self.health_label.text() == "Healthy"
        if hasattr(self, "health_label"):
            color = GREEN if healthy else ORANGE
            background = theme.status_good_bg if healthy else theme.status_warn_bg
            self.health_label.setStyleSheet(
                f"color:{color}; background:{background}; border-radius:4px; padding:3px 7px; font-size:11px;"
            )
        if hasattr(self, "gpu_label"):
            self.gpu_label.setStyleSheet(
                f"color:{theme.text}; background:{theme.panel_alt}; border:1px solid {theme.border_soft}; "
                "border-radius:4px; padding:3px 7px; font-size:11px;"
            )

        # A few legacy labels use inline dark-mode text colors. Convert only
        # neutral/chrome labels; semantic green/orange status colors remain.
        for label in self.findChildren(QLabel):
            if label.text() == "→":
                label.setStyleSheet(f"color:{theme.text}; font-size:24px;")

    def _apply_preferences(self) -> None:
        mode = self._theme_mode()
        app = QApplication.instance()
        if app is not None:
            app.setProperty("sprpcfTheme", mode)
            app.setPalette(qt_palette_for(mode))

        self.setProperty("theme", mode)
        self.setStyleSheet(stylesheet_for(mode))
        self._tag_brand_marks()
        self._refresh_action_icons()
        self._restyle_sidebar_sections()
        self._restyle_dynamic_labels()

        for stage in self.findChildren(ThemedStageCard):
            stage.apply_theme()
        for widget in self.findChildren(QWidget):
            widget.update()

        if self._auto_refresh_enabled():
            self.refresh_timer.start(self._refresh_seconds() * 1000)
        else:
            self.refresh_timer.stop()

        if hasattr(self, "_top_theme_button"):
            target = "Light" if mode == "dark" else "Dark"
            self._top_theme_button.setToolTip(f"Switch to {target} mode")
            self._top_theme_button.setAccessibleName(f"Switch to {target} mode")

    def _toggle_theme(self) -> None:
        target = "light" if self._theme_mode() == "dark" else "dark"
        self._ui_settings.setValue("theme", target)
        self._ui_settings.sync()
        self._apply_preferences()

    def refresh_status(self) -> None:
        super().refresh_status()
        self._restyle_dynamic_labels()

    def _show_settings_dialog(self) -> None:
        theme_colors = palette_for(self._theme_mode())
        dialog = QDialog(self)
        dialog.setWindowTitle("CyberPhotonics-SPR Settings")
        dialog.setMinimumWidth(520)
        outer = QVBoxLayout(dialog)
        outer.setContentsMargins(20, 20, 20, 20)
        outer.setSpacing(12)
        title = QLabel("Control Center Settings")
        title.setObjectName("sectionTitle")
        outer.addWidget(title)
        description = QLabel("Preferences are stored locally for this desktop control center.")
        description.setWordWrap(True)
        description.setStyleSheet(f"color:{theme_colors.muted};")
        outer.addWidget(description)

        form = QFormLayout()
        appearance = QComboBox()
        appearance.addItem("Dark — Deep Navy", "dark")
        appearance.addItem("Light — White", "light")
        index = appearance.findData(self._theme_mode())
        appearance.setCurrentIndex(max(0, index))
        form.addRow("Appearance", appearance)

        auto_refresh = QCheckBox("Refresh dashboard automatically")
        auto_refresh.setChecked(self._auto_refresh_enabled())
        form.addRow("Live status", auto_refresh)

        refresh_seconds = QSpinBox()
        refresh_seconds.setRange(1, 300)
        refresh_seconds.setSuffix(" s")
        refresh_seconds.setValue(self._refresh_seconds())
        form.addRow("Refresh interval", refresh_seconds)
        outer.addLayout(form)

        shortcuts = QHBoxLayout()
        for text, path in (
            ("Project", PROJECT_ROOT),
            ("Data", PROJECT_ROOT / "data"),
            ("Models", PROJECT_ROOT / "models"),
            ("Reports", PROJECT_ROOT / "reports"),
        ):
            button = QPushButton(text)
            button.setIcon(action_icon("folder", color=theme_colors.text))
            button.setIconSize(QSize(17, 17))
            button.setToolTip(f"Open {path}")
            button.clicked.connect(lambda _checked=False, target=path: self._open_directory(Path(target)))
            shortcuts.addWidget(button)
        outer.addLayout(shortcuts)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        outer.addWidget(buttons)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._ui_settings.setValue("theme", appearance.currentData())
            self._ui_settings.setValue("auto_refresh", auto_refresh.isChecked())
            self._ui_settings.setValue("refresh_seconds", refresh_seconds.value())
            self._ui_settings.sync()
            self._apply_preferences()

    def _show_help_dialog(self) -> None:
        colors = palette_for(self._theme_mode())
        dialog = QDialog(self)
        dialog.setWindowTitle("CyberPhotonics-SPR Help")
        dialog.setMinimumWidth(540)
        outer = QVBoxLayout(dialog)
        outer.setContentsMargins(20, 20, 20, 20)
        outer.setSpacing(10)
        heading = QLabel("CyberPhotonics-SPR Control Center")
        heading.setObjectName("sectionTitle")
        outer.addWidget(heading)
        text = QLabel(
            "Every navigation item runs a real local workflow or opens the corresponding project artifact. "
            "Use the links below for project documentation and source history."
        )
        text.setWordWrap(True)
        text.setStyleSheet(f"color:{colors.muted};")
        outer.addWidget(text)

        actions = QHBoxLayout()
        readme = QPushButton("Open README")
        readme.setIcon(action_icon("report", color=colors.text))
        readme.clicked.connect(lambda: self._open_local_or_parent(PROJECT_ROOT / "README.md"))
        actions.addWidget(readme)

        repository = QPushButton("Open GitHub")
        repository.setIcon(action_icon("export", color=colors.text))
        repository.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(REPOSITORY_URL)))
        actions.addWidget(repository)

        reports = QPushButton("Open Reports")
        reports.setIcon(action_icon("evidence", color=colors.text))
        reports.clicked.connect(lambda: self._open_directory(PROJECT_ROOT / "reports"))
        actions.addWidget(reports)
        outer.addLayout(actions)

        close = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        close.rejected.connect(dialog.reject)
        close.clicked.connect(dialog.accept)
        outer.addWidget(close)
        dialog.exec()


def launch_desktop() -> int:
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("CyberPhotonics-SPR")
    app.setOrganizationName("CyberPhotonics-SPR")
    app.setFont(QFont("Segoe UI", 9))

    settings = QSettings("CyberPhotonics-SPR", "ControlCenter")
    mode = normalize_theme(str(settings.value("theme", "dark")))
    app.setProperty("sprpcfTheme", mode)
    app.setPalette(qt_palette_for(mode))

    window = ResponsiveControlCenter()
    window.showMaximized()
    return app.exec()


__all__ = ["ACTION_META", "QUICK_ACTIONS", "ResponsiveControlCenter", "large_dataset_form", "launch_desktop"]
