# SYNAPSE - A Gateway of Intelligent Perception for Traffic Management
# Copyright (C) 2026 Noxfort Systems
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# File: ui/main_window.py
# Author: Gabriel Moraes
# Date: 2026-03-01

from PyQt6.QtCore import QEvent, QSettings
from PyQt6.QtWidgets import QMainWindow, QApplication, QSystemTrayIcon, QMenu
from PyQt6.QtGui import QIcon, QAction

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.main_controller import MainController

from ui.tabs.central_tabs import CentralTabs
from ui.components.main_menu import MainMenu
from ui.components.dock_manager import DockManager
from ui.handlers.dialog_handler import DialogHandler
from ui.handlers.signal_router import SignalRouter
from ui.utilities.translation_manager import TranslationManager
from ui.styles.theme_manager import ThemeManager

class MainWindow(QMainWindow):
    """
    The Main Application Window (View Layer).
    
    Refactored V4.0 (Ultimate SOLID/Orchestrator):
    - Delegates UI construction to CentralTabs, MainMenu, and DockManager.
    - Delegates dialog logic to DialogHandler.
    - Delegates signal routing to SignalRouter.
    - Acts purely as the structural foundation tying components together.
    """

    def __init__(self, controller: 'MainController'):
        super().__init__()
        
        self.controller = controller
        self.app_state = self.controller.app_state
        
        # Configuration (Persistent Settings)
        self.settings = QSettings("Noxfort Systems", "SYNAPSE")
        self.current_language = self.settings.value("General/language", "pt_BR", type=str)
        self.current_theme = self.settings.value("General/theme", 0, type=int)
        self.current_log_level = self.settings.value("General/log_level", "INFO", type=str)
        
        self.monitor_enabled = self.settings.value("Monitor/enabled", False, type=bool)
        self.monitor_host = self.settings.value("Monitor/host", "localhost", type=str)
        self.monitor_port = self.settings.value("Monitor/port", 1883, type=int)
        
        # Theme System Boot Configuration
        theme_map = {0: "dark", 1: "light", 2: "dark"} # Default translates to Dark
        ThemeManager.set_theme(theme_map.get(self.current_theme, "dark"))
        
        # Translation Setup
        self.translation_manager = TranslationManager(QApplication.instance())
        self.translation_manager.load_language(self.current_language)
        
        self.resize(1280, 720)

        # --- 1. Handlers & Managers ---
        self.dialog_handler = DialogHandler(self, self.app_state, self.current_language)
        self.dock_manager = DockManager(self)
        
        # --- 2. Central Component (Tabs) ---
        self.tabs = CentralTabs(self)
        self.setCentralWidget(self.tabs)

        # --- 3. Menu Component ---
        self.main_menu = MainMenu(self)
        self.setMenuBar(self.main_menu)
        
        # Inject View actions from Docks into the Menu
        for act in self.dock_manager.get_view_actions():
            self.main_menu.add_view_action(act)

        # --- 4. Status Bar ---
        self.status_bar = self.statusBar()
        
        self.controller.init_telemetry(self.monitor_enabled, self.monitor_host, self.monitor_port)


        # --- 6. Signal Routing ---
        self.signal_router = SignalRouter(self, self.controller)
        self.signal_router.setup_routing()
        
        # --- 6b. Restore Persisted Sources (emit signals AFTER wiring) ---
        self.app_state.emit_restored_sources()
        
        # --- 7. Apply Initial Settings ---
        ThemeManager.apply_theme(self.current_theme)
        from src.utils.logging_setup import set_global_level
        set_global_level(self.current_log_level)

        # --- 8. Apply Initial Translations ---
        self._retranslate_ui()

        # --- 9. System Tray Integration ---
        self._setup_system_tray()

    def log(self, message: str):
        """Convenience method to add a log message to the control dock (Law of Demeter fix)."""
        self.dock_manager.control_dock.add_log_message(message)

    def _retranslate_ui(self):
        """Updates main window texts when language changes."""
        self.setWindowTitle(self.tr("SYNAPSE | Intelligent Perception Gateway"))
        self.status_bar.showMessage(self.tr("System Ready (Waiting for Controller)"))

    def _setup_system_tray(self):
        """Initializes the System Tray Icon for background execution."""
        self._is_quitting = False
        
        self.tray_icon = QSystemTrayIcon(self)
        self.tray_icon.setIcon(QIcon("ui/assets/images/logo.png"))
        
        self.tray_menu = QMenu()
        
        self.show_action = QAction(self.tr("Open SYNAPSE"), self)
        self.show_action.triggered.connect(self.showNormal)
        self.show_action.triggered.connect(self.activateWindow)
        
        self.quit_action = QAction(self.tr("Quit SYNAPSE"), self)
        self.quit_action.triggered.connect(self._quit_application)
        
        self.tray_menu.addAction(self.show_action)
        self.tray_menu.addSeparator()
        self.tray_menu.addAction(self.quit_action)
        
        self.tray_icon.setContextMenu(self.tray_menu)
        self.tray_icon.activated.connect(self._on_tray_activated)
        self.tray_icon.show()

    def _on_tray_activated(self, reason):
        if reason == QSystemTrayIcon.ActivationReason.Trigger or reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            self.showNormal()
            self.activateWindow()

    def _quit_application(self):
        """Forces the application to quit by bypassing the hide-on-close behavior."""
        self._is_quitting = True
        self.close()

    def closeEvent(self, event):
        """Intercepts the application exit. Minimizes to tray unless explicitly quitting."""
        if not hasattr(self, '_is_quitting') or not self._is_quitting:
            event.ignore()
            self.hide()
            self.tray_icon.showMessage(
                self.tr("SYNAPSE"),
                self.tr("Running in background."),
                QSystemTrayIcon.MessageIcon.Information,
                2000
            )
        else:
            self.controller.report_shutdown()
            super().closeEvent(event)

    def changeEvent(self, event):
        """Catches the LanguageChange event to trigger UI retranslation down the tree."""
        if event.type() == QEvent.Type.LanguageChange:
            self._retranslate_ui()
            self.tabs.retranslate_ui()
            self.main_menu.retranslate_ui()
        super().changeEvent(event)