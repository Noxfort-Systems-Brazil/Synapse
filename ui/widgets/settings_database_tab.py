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
# File: ui/widgets/settings_database_tab.py
# Author: Gabriel Moraes
# Date: 2026-02-28

from PyQt6.QtCore import pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QStackedWidget, QMessageBox
)

from ui.panels.database_setup_panel import DatabaseSetupPanel
from ui.panels.database_management_panel import DatabaseManagementPanel

class SettingsDatabaseTab(QWidget):
    """
    The dedicated 'Database' tab for the Settings Dialog.
    
    Now acting as an Orchestrator:
    1. Manages the state between Setup Mode and Management Mode.
    2. Routes signals from child panels to the global controller.
    """
    
    # Signals to request actions from the backend (via SettingsDialog -> Controller)
    # Payload: (root_password, config_dict)
    request_initialize = pyqtSignal(str, dict)
    request_check_connection = pyqtSignal(dict)
    request_reset_db = pyqtSignal(str, dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(10, 10, 10, 10)
        self._layout.setSpacing(10)
        
        # State Machine Visualizer
        self.stack = QStackedWidget()
        self._layout.addWidget(self.stack)
        
        # Child Panels
        self.panel_setup = DatabaseSetupPanel()
        self.panel_manage = DatabaseManagementPanel()
        
        self.stack.addWidget(self.panel_setup)
        self.stack.addWidget(self.panel_manage)
        
        self._connect_signals()
        
        # Default to Setup until explicitly checked
        self.stack.setCurrentIndex(0)

    def _connect_signals(self):
        """Binds child panel signals to the orchestrator's signals."""
        
        # Setup Panel routing
        self.panel_setup.request_initialize.connect(self._on_setup_requested)
        
        # Management Panel routing
        self.panel_manage.request_check_connection.connect(self.request_check_connection)
        self.panel_manage.request_reset_db.connect(self.request_reset_db)

    @pyqtSlot(str)
    def _on_setup_requested(self, root_pass: str):
        """
        Intercepts the setup request to append the default configuration dictionary 
        before emitting to the backend controller.
        """
        config = self.panel_manage._get_current_config()
        self.request_initialize.emit(root_pass, config)

    def get_config_data(self) -> dict:
        return self.panel_manage._get_current_config()

    def set_config_data(self, config: dict):
        self.panel_manage.set_config(config)

    def set_setup_mode(self, is_setup_needed: bool):
        """Switches the visible page."""
        index = 0 if is_setup_needed else 1
        self.stack.setCurrentIndex(index)

    def on_task_finished(self, success: bool, message: str):
        """Receives task completion events and updates the active panel."""
        
        # Always reset the setup panel UI state if it was processing
        self.panel_setup.set_ui_state(is_processing=False)
        
        if success:
            QMessageBox.information(self, "Success", message)
            # Switch to Manage mode automatically upon successful setup
            if self.stack.currentIndex() == 0:
                self.set_setup_mode(False)
                self.panel_setup.clear_password()
        else:
            QMessageBox.critical(self, "Error", message)