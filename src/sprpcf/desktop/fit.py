from __future__ import annotations

import os
import sys
from pathlib import Path

from PySide6.QtCore import QSettings, QSize, QTimer, QUrl
from PySide6.QtGui import QDesktopServices, QFont
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
)

from sprpcf.dashboard.operations import PROJECT_ROOT

from .app import APP_TITLE, _label
from .icons import action_icon
from .responsive import ResponsiveControlCenter as _ResponsiveControlCenter
from .responsive import large_dataset_form
from .theme import APP_STYLESHEET, GREEN, MUTED, TEXT
from .widgets import Card


REPOSITORY_URL = "https://github.com/smshagor-dev/CyberPhotonics-SPR"

ACTION_META: dict[str, tuple[str, str]] = {
    "overview": ("Overview", "Show the live system overview and refresh status."),
    "generate-data": ("Generate Dataset", "Generate a research-scale synthetic PCF-SPR dataset."),
    "train-inverse": ("Train Inverse Model", "Train and export the tandem inverse model."),
    "train-edge": ("Train Edge Models", "Train and export edge denoiser and RI predictor models."),
    "export-models": ("Export Models", "Open the model export directory."),
    "run-pipeline": ("Run Pipeline (A→B→C)", "Run dataset, inverse-model, edge-model and streaming stages."),
    "simulate-stream": ("Streaming Benchmark", "Benchmark real-time streaming inference."),
    "hil-benchmark": ("HIL Benchmark", "Run the hardware-in-the-loop benchmark."),
    "design": ("Inverse Design Studio", "Run the inverse design workflow for a new sensor."),
    "candidates": ("Design Candidates", "Open the selected Pareto design candidates."),
    "verify": ("Physics Verification", "Verify selected designs against the physics gate."),
    "evidence": ("Evidence Center", "Open generated benchmark evidence and reports."),
    "report": ("Generate Report", "Generate the current research/dashboard report package."),
    "settings": ("Settings", "Configure dashboard refresh, appearance and workspace shortcuts."),
    "logs": ("Logs", "Open the local logs directory."),
}

QUICK_ACTIONS: tuple[tuple[str, str], ...] = (
    ("generate-data", "Generate\nDataset"),
    ("train-inverse", "Train\nInverse Model"),
    ("train-edge", "Train\nEdge Models"),
    ("run-pipeline", "Run\nPipeline"),
    ("hil-benchmark", "Run HIL\nBenchmark"),
    ("design", "Design\nNew Sensor"),
    ("verify", "Verify\nPhysics"),
    ("report", "Generate\nReport"),
)

HIGH_CONTRAST_OVERRIDES = """
QMainWindow, QWidget#root { background:#02060b; }
QFrame#sidebar, QFrame#topbar, QFrame#statusbar { background:#03080e; }
QFrame#card, QFrame.card, QFrame#panel { background:#08111b; border-color:#4b6478; }
QLabel#muted, QLabel#mutedSmall, QLabel#brandSub { color:#d4dde6; }
QPushButton#navButton { color:#f4f8fb; }
QPushButton#navButton:hover { background:#10283e; border-color:#4ba3ff; }
QPushButton#navButton:checked { background:#0c3e73; border:1px solid #57adff; color:#ffffff; }
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit, QPlainTextEdit { border-color:#526b7d; }
"""


class ResponsiveControlCenter(_ResponsiveControlCenter):
    """Windows-safe responsive shell with fully actionable icon navigation."""

    def __init__(self) -> None:
        self._ui_settings = QSettings("CyberPhotonics-SPR", "ControlCenter")
        self._current_action = "overview"
        super().__init__()
        self._upgrade_action_surfaces()
        self._apply_preferences()

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
        if width < 1000:
            return max(1, width - 180)
        return width

    def _apply_responsive_layout(self, force: bool = False) -> None:
        super()._apply_responsive_layout(force=force)
        self._sync_content_width()
        QTimer.singleShot(0, self._sync_content_width)

    def _sync_content_width(self) -> None:
        scroll = getattr(self, "_scroll_area", None)
        content = getattr(self, "_responsive_content", None)
        if scroll is None or content is None:
            return
        content.setFixedWidth(max(1, scroll.viewport().width()))

    def _build_topbar(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("topbar")
        frame.setFixedHeight(60)
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(12, 0, 16, 0)
        layout.setSpacing(8)

        self._back_button = self._icon_button("back", "Return to Overview", "overview")
        layout.addWidget(self._back_button)

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
        gpu_layout.setSpacing(8)
        gpu_layout.addWidget(_label("Edge GPU", "mutedSmall"))
        self.gpu_label = QLabel("Auto")
        self.gpu_label.setStyleSheet("background:#121f2c; border-radius:4px; padding:3px 7px; font-size:11px;")
        gpu_layout.addWidget(self.gpu_label)
        layout.addWidget(self._gpu_box)

        settings = self._icon_button("settings", "Settings", "settings")
        help_button = self._icon_button("help", "Help and project links", "help")
        self._top_theme_button = self._icon_button("theme", "Toggle appearance", "toggle-theme")
        self._top_buttons = [settings, help_button, self._top_theme_button]
        for button in self._top_buttons:
            layout.addWidget(button)
        return frame

    def _icon_button(self, icon_name: str, tooltip: str, action: str) -> QPushButton:
        button = QPushButton()
        button.setObjectName("topIconButton")
        button.setFixedSize(34, 34)
        button.setIcon(action_icon(icon_name, color=TEXT))
        button.setIconSize(QSize(19, 19))
        button.setToolTip(tooltip)
        button.setAccessibleName(tooltip)
        button.setProperty("action", action)
        button.setStyleSheet("QPushButton{border:none; padding:5px;} QPushButton:hover{background:#102438; border-radius:7px;}")
        button.clicked.connect(lambda _checked=False, key=action: self.handle_action(key))
        return button

    def _upgrade_action_surfaces(self) -> None:
        nav_buttons = getattr(self.sidebar, "_nav_buttons", {})
        for action, button in nav_buttons.items():
            label, tooltip = ACTION_META.get(action, (button.text(), action))
            button.setText(label)
            button.setIcon(action_icon(action, color="#dbe6ef"))
            button.setIconSize(QSize(18, 18))
            button.setToolTip(tooltip)
            button.setAccessibleName(label)
            button.setAccessibleDescription(tooltip)
            button.setProperty("action", action)
            button.setCheckable(True)
            button.setAutoExclusive(True)

        if "overview" in nav_buttons:
            nav_buttons["overview"].setChecked(True)

        for button, (action, label) in zip(self._quick_buttons, QUICK_ACTIONS):
            tooltip = ACTION_META[action][1]
            button.setText(label)
            button.setIcon(action_icon(action, color="#f4f8fb"))
            button.setIconSize(QSize(23, 23))
            button.setToolTip(tooltip)
            button.setAccessibleName(ACTION_META[action][0])
            button.setAccessibleDescription(tooltip)
            button.setProperty("action", action)

        # Replace the two decorative font glyphs in the sidebar brand/footer.
        for label in self.sidebar.findChildren(QLabel):
            if label.text() in {"◉", "◎"}:
                size = 28 if label.text() == "◉" else 20
                label.setText("")
                label.setPixmap(action_icon("brand", color="#2493ff").pixmap(size, size))
                label.setFixedSize(size + 2, size + 2)

    def _set_active_action(self, action: str) -> None:
        button = getattr(self.sidebar, "_nav_buttons", {}).get(action)
        if button is not None:
            button.setChecked(True)
            self._current_action = action

    def handle_action(self, action: str) -> None:
        if action in ACTION_META:
            self._set_active_action(action)

        if action == "overview":
            super().handle_action(action)
            scroll = getattr(self, "_scroll_area", None)
            if scroll is not None:
                scroll.verticalScrollBar().setValue(0)
            return
        if action == "settings":
            self._show_settings_dialog()
            return
        if action == "help":
            self._show_help_dialog()
            return
        if action == "toggle-theme":
            self._toggle_theme()
            return
        if action == "logs":
            self._open_directory(PROJECT_ROOT / "logs")
            return
        super().handle_action(action)

    def _show_settings_dialog(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("CyberPhotonics-SPR Settings")
        dialog.setMinimumWidth(520)
        outer = QVBoxLayout(dialog)
        outer.setContentsMargins(20, 20, 20, 20)
        outer.setSpacing(12)
        outer.addWidget(_label("Control Center Settings", "sectionTitle"))
        description = QLabel("Preferences are stored locally for this desktop control center.")
        description.setWordWrap(True)
        description.setStyleSheet(f"color:{MUTED};")
        outer.addWidget(description)

        form = QFormLayout()
        theme = QComboBox()
        theme.addItem("Dark", "dark")
        theme.addItem("High Contrast", "high-contrast")
        current_theme = self._theme_mode()
        index = theme.findData(current_theme)
        theme.setCurrentIndex(max(0, index))
        form.addRow("Appearance", theme)

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
            button.setIcon(action_icon("folder", color=TEXT))
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
            self._ui_settings.setValue("theme", theme.currentData())
            self._ui_settings.setValue("auto_refresh", auto_refresh.isChecked())
            self._ui_settings.setValue("refresh_seconds", refresh_seconds.value())
            self._ui_settings.sync()
            self._apply_preferences()

    def _show_help_dialog(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("CyberPhotonics-SPR Help")
        dialog.setMinimumWidth(540)
        outer = QVBoxLayout(dialog)
        outer.setContentsMargins(20, 20, 20, 20)
        outer.setSpacing(10)
        outer.addWidget(_label("CyberPhotonics-SPR Control Center", "sectionTitle"))
        text = QLabel(
            "Every navigation item runs a real local workflow or opens the corresponding project artifact. "
            "Use the links below for project documentation and source history."
        )
        text.setWordWrap(True)
        text.setStyleSheet(f"color:{MUTED};")
        outer.addWidget(text)

        actions = QHBoxLayout()
        readme = QPushButton("Open README")
        readme.setIcon(action_icon("report", color=TEXT))
        readme.clicked.connect(lambda: self._open_local_or_parent(PROJECT_ROOT / "README.md"))
        actions.addWidget(readme)

        repository = QPushButton("Open GitHub")
        repository.setIcon(action_icon("export", color=TEXT))
        repository.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(REPOSITORY_URL)))
        actions.addWidget(repository)

        reports = QPushButton("Open Reports")
        reports.setIcon(action_icon("evidence", color=TEXT))
        reports.clicked.connect(lambda: self._open_directory(PROJECT_ROOT / "reports"))
        actions.addWidget(reports)
        outer.addLayout(actions)

        close = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        close.rejected.connect(dialog.reject)
        close.clicked.connect(dialog.accept)
        outer.addWidget(close)
        dialog.exec()

    def _open_local_or_parent(self, path: Path) -> None:
        target = path if path.exists() else path.parent
        target.mkdir(parents=True, exist_ok=True) if target.suffix == "" else None
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(target.resolve())))

    def _theme_mode(self) -> str:
        value = str(self._ui_settings.value("theme", "dark"))
        return value if value in {"dark", "high-contrast"} else "dark"

    def _auto_refresh_enabled(self) -> bool:
        value = self._ui_settings.value("auto_refresh", True)
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() not in {"0", "false", "no", "off"}

    def _refresh_seconds(self) -> int:
        try:
            return max(1, min(300, int(self._ui_settings.value("refresh_seconds", 5))))
        except (TypeError, ValueError):
            return 5

    def _apply_preferences(self) -> None:
        mode = self._theme_mode()
        self.setStyleSheet(APP_STYLESHEET + (HIGH_CONTRAST_OVERRIDES if mode == "high-contrast" else ""))
        if self._auto_refresh_enabled():
            self.refresh_timer.start(self._refresh_seconds() * 1000)
        else:
            self.refresh_timer.stop()
        self._update_theme_button()

    def _toggle_theme(self) -> None:
        new_mode = "high-contrast" if self._theme_mode() == "dark" else "dark"
        self._ui_settings.setValue("theme", new_mode)
        self._ui_settings.sync()
        self._apply_preferences()

    def _update_theme_button(self) -> None:
        button = getattr(self, "_top_theme_button", None)
        if button is None:
            return
        target = "High Contrast" if self._theme_mode() == "dark" else "Dark"
        button.setToolTip(f"Switch to {target} appearance")
        button.setAccessibleName(button.toolTip())


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
