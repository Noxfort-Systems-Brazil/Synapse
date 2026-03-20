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
# File: ui/dialogs/settings_dialog.py
# Author: Gabriel Moraes
# Date: 2026-03-01

from typing import Optional
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QTabWidget, QDialogButtonBox

# Import Backend Managers
from src.infrastructure.postgres_manager import PostgresManager

# Import UI Tabs (Adhering to SRP)
from ui.widgets.settings_database_tab import SettingsDatabaseTab
from ui.tabs.settings_monitor_tab import SettingsMonitorTab
from ui.tabs.settings_general_tab import SettingsGeneralTab

class SettingsDialog(QDialog):
    """
    Main configuration window for SYNAPSE.
    Refactored to act purely as an Orchestrator/Container for independent tab components.
    Safely handles QThread destruction for the background PostgresManager.
    """

    def __init__(self, current_language: str = "pt_BR", current_theme: int = 0, current_log_level: str = "INFO", 
                 monitor_enabled: bool = False, monitor_host: str = "localhost", monitor_port: int = 1883, 
                 db_config: Optional[dict] = None, parent=None):
        super().__init__(parent) # type: ignore
        self.setWindowTitle(self.tr("SYNAPSE Settings"))
        self.resize(700, 550)
        self.setModal(True)
        
        self.current_language = current_language
        self.current_theme = current_theme
        self.current_log_level = current_log_level
        self.monitor_enabled = monitor_enabled
        self.monitor_host = monitor_host
        self.monitor_port = monitor_port

        # --- Backend Managers ---
        self.pg_manager = PostgresManager()

        # --- Main Layout ---
        layout = QVBoxLayout(self)

        # --- Tab Widget ---
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # --- Instantiate and Add Tabs ---
        self.tab_database = SettingsDatabaseTab()
        if db_config:
            self.tab_database.set_config_data(db_config)
            

        self.tab_monitor = SettingsMonitorTab(
            monitor_enabled=self.monitor_enabled,
            monitor_host=self.monitor_host,
            monitor_port=self.monitor_port
        )
        self.tab_general = SettingsGeneralTab(
            current_language=self.current_language,
            current_theme=self.current_theme,
            current_log_level=self.current_log_level
        )

        self.tabs.addTab(self.tab_database, self.tr("Database (PostgreSQL)"))

        self.tabs.addTab(self.tab_monitor, self.tr("Monitor (Telemetry)"))
        self.tabs.addTab(self.tab_general, self.tr("General"))

        # --- Wiring Signals (MVC Pattern for DB Tab) ---
        self.tab_database.request_initialize.connect(self.pg_manager.initialize_database)
        self.tab_database.request_check_connection.connect(self.pg_manager.check_connection)
        self.tab_database.request_reset_db.connect(self.pg_manager.initialize_database)
        
        self.pg_manager.setup_complete.connect(self.tab_database.on_task_finished)
        self.pg_manager.status_received.connect(self.tab_database.on_task_finished)

        # --- Action Buttons (Save/Cancel) ---
        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | 
            QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

    def _stop_background_tasks(self):
        """Safely stops and waits for the background thread to finish before destroying the object."""
        if hasattr(self, 'pg_manager') and self.pg_manager is not None:
            self.pg_manager.stop()
            if hasattr(self.pg_manager, 'wait'):
                self.pg_manager.wait()

    def closeEvent(self, event):
        """Ensure background threads are stopped when dialog is closed via the X button."""
        self._stop_background_tasks()
        super().closeEvent(event)

    def accept(self):
        """
        Triggered when the Save button is clicked. 
        Safely stops threads before passing values up.
        """
        self._stop_background_tasks()
        super().accept()

    def reject(self):
        """Triggered when the Cancel button is clicked. Safely stops threads."""
        self._stop_background_tasks()
        super().reject()