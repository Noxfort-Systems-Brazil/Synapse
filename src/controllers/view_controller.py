# SYNAPSE - A Gateway of Intelligent Perception for Traffic Management
# Copyright (C) 2025 Noxfort Systems
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
# File: src/controllers/view_controller.py
# Author: Gabriel Moraes
# Date: 2025-12-27

from typing import Optional
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

# --- Utils ---
from src.utils.logging_setup import logger

class ViewController(QObject):
    """
    The Presentation Manager (View Controller).
    
    Responsibilities (SRP):
    - Manages UI State (Active Tab, Dock Visibility).
    - Aggregates Logs and Status messages from all subsystems.
    - Dictates UI reactions to System State changes (e.g., Auto-switch tabs).
    """

    # --- UI COMMAND SIGNALS (Consumed by MainWindow) ---
    request_tab_change = pyqtSignal(int)        # Index of the tab to show
    update_dock_visibility = pyqtSignal(str, bool)
    
    # --- UI DATA SIGNALS (Consumed by Status Bar / Log Console) ---
    append_log = pyqtSignal(str)
    update_status = pyqtSignal(str)
    update_progress = pyqtSignal(int)
    
    # --- UI STATE SIGNALS ---
    view_mode_changed = pyqtSignal(str)         # e.g., "2D", "3D", "Heatmap"

    def __init__(self):
        super().__init__()
        # Tab Indices (Configuration)
        self.TAB_DASHBOARD = 0
        self.TAB_MAP = 1
        self.TAB_XAI = 2
        self.TAB_SETTINGS = 3

    # =========================================================================
    # LOG & STATUS AGGREGATION
    # =========================================================================

    @pyqtSlot(str)
    def log(self, message: str):
        """Central hub for logging messages to the UI Console."""
        # We can add filtering or formatting here if needed
        self.append_log.emit(message)

    @pyqtSlot(str)
    def status(self, message: str):
        """Updates the main window status bar."""
        self.update_status.emit(message)

    @pyqtSlot(str)
    def error_alert(self, message: str):
        """Special handler for errors (could trigger a popup in future)."""
        logger.error(f"[ViewController] Alert: {message}")
        self.append_log.emit(f"❌ ERROR: {message}")
        self.update_status.emit(f"Error: {message}")

    # =========================================================================
    # STATE REACTION LOGIC (Presentation Logic)
    # =========================================================================

    @pyqtSlot()
    def on_optimization_started(self):
        """
        When Phase 0 starts, force view to Settings/Logs 
        so user sees the progress.
        """
        self.log(">>> Switching View to Optimization Console...")
        self.request_tab_change.emit(self.TAB_SETTINGS)
        self.update_dock_visibility.emit("console_dock", True)

    @pyqtSlot()
    def on_online_system_started(self):
        """
        When Phase 2 starts, force view to Dashboard 
        to visualize real-time data.
        """
        self.log(">>> System Online. Switching to Dashboard...")
        self.request_tab_change.emit(self.TAB_DASHBOARD)
        # Ensure critical docks are open
        self.update_dock_visibility.emit("control_dock", True)

    @pyqtSlot(str)
    def on_project_loaded(self, name: str):
        self.status(f"Project '{name}' loaded ready.")
        self.request_tab_change.emit(self.TAB_MAP)

    # =========================================================================
    # MANUAL UI ACTIONS
    # =========================================================================

    @pyqtSlot(str)
    def set_map_view_mode(self, mode: str):
        """User clicked a '2D/3D' toggle button."""
        self.log(f"Map View Mode changed to: {mode}")
        self.view_mode_changed.emit(mode)