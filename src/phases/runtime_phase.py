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
# File: src/phases/runtime_phase.py
# Author: Gabriel Moraes
# Date: 2026-02-13

from typing import TYPE_CHECKING
from PyQt6.QtCore import QObject, pyqtSignal

from src.domain.app_state import AppState
from src.domain.entities import SourceType

# --- Internal Components (The "Split" Logic) ---
from src.phases.runtime_connector import RuntimeConnector
from src.phases.runtime_launcher import RuntimeLauncher

if TYPE_CHECKING:
    from src.services.fenix_service import FenixService

class RuntimePhase(QObject):
    """
    Orchestrator for Phase 2: Online Operation.
    
    Location: src/phases/runtime_phase.py
    
    Refactored V37 (Component Split):
    - Acts as a FACADE for the Runtime subsystem.
    - Coordinates 'RuntimeConnector' (Network) and 'RuntimeLauncher' (Compute).
    - Bridges signals between Network, AI, and the Main Controller.
    - Does NOT contain complex logic itself; only wiring.
    """

    # --- PUBLIC SIGNALS (Consumed by SystemController) ---
    log_message = pyqtSignal(str)
    status_message = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    
    system_started = pyqtSignal()
    system_stopped = pyqtSignal()
    
    # Data Streams
    engine_data_processed = pyqtSignal(dict)
    engine_global_results = pyqtSignal(dict)
    kinetic_data_ready = pyqtSignal(dict)
    
    # Advanced Streams
    audit_update = pyqtSignal(bool, float, float, list)
    linguist_update = pyqtSignal(str, str, float)
    drift_update = pyqtSignal(str, dict)
    xai_result_received = pyqtSignal(dict)

    def __init__(self, app_state: AppState, fenix_service: 'FenixService'):
        super().__init__()
        
        self.app_state = app_state
        
        # 1. Initialize Sub-Components
        self.connector = RuntimeConnector(app_state)
        self.launcher = RuntimeLauncher(app_state, fenix_service)
        
        # 2. Wire Internal Logic (The "Glue")
        self._wire_internals()
        
        # 3. Wire External Outputs (Bubbling up signals)
        self._wire_outputs()

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def start(self) -> bool:
        """Entry point: Validates live sources, then starts the connection sequence."""
        # Mandatory Check: at least 1 LOCAL + 1 GLOBAL live source
        all_sources = self.app_state.get_all_data_sources()
        
        live_sources = [
            s for s in all_sources
            if s.source_type != SourceType.SUMO_NET_XML
            and not (isinstance(s.connection_string, str) and s.connection_string.endswith(".parquet"))
            and "Historical Base" not in s.name
        ]
        
        has_local = any(s.is_local for s in live_sources)
        has_global = any(not s.is_local for s in live_sources)
        
        if not has_local or not has_global:
            missing = []
            if not has_local:
                missing.append("LOCAL (sensor/camera/radar)")
            if not has_global:
                missing.append("GLOBAL (Waze/TomTom API)")
            error_msg = f"❌ Runtime Aborted: Missing LIVE data sources: {', '.join(missing)}. Both LOCAL and GLOBAL sources are MANDATORY."
            self.log_message.emit(error_msg)
            self.error_occurred.emit(error_msg)
            return False
        
        # Validation passed — start the connector
        self.connector.start_connection()
        return True

    def stop(self):
        """Stops both network and compute layers."""
        self.connector.stop_connection()
        self.launcher.stop_engines()
        self.system_stopped.emit()

    def handle_command(self, command_type: str, payload: object = None):
        """Delegates commands to the appropriate component (mostly Launcher)."""
        self.launcher.handle_command(command_type, payload)

    # =========================================================================
    # WIRING LOGIC
    # =========================================================================

    def _wire_internals(self):
        """Connects the Connector and Launcher to each other."""
        
        # A. Start Sequence: Connector Success -> Start Engines
        self.connector.connection_ready_to_start.connect(self.launcher.start_engines)
        
        # B. Data Pipeline: Physics Packet Ready -> Send via Network
        self.launcher.packet_ready_to_send.connect(self.connector.send_packet)
        
        # C. Lifecycle Synchronization
        # If connection fails hard, we might want to stop engines (optional, simpler to keep separate for now)
        pass

    def _wire_outputs(self):
        """Connects sub-components signals to this Facade's signals."""
        
        # --- From Connector ---
        self.connector.log_message.connect(self.log_message)
        self.connector.status_message.connect(self.status_message)
        self.connector.connection_failed.connect(self.error_occurred)
        
        # --- From Launcher ---
        self.launcher.log_message.connect(self.log_message)
        self.launcher.error_occurred.connect(self.error_occurred)
        self.launcher.engines_started.connect(self.system_started)
        
        # Data Streams
        self.launcher.engine_data_processed.connect(self.engine_data_processed)
        self.launcher.engine_global_results.connect(self.engine_global_results)
        self.launcher.kinetic_data_ready.connect(self.kinetic_data_ready)
        
        # Advanced Streams
        self.launcher.audit_update.connect(self.audit_update)
        self.launcher.linguist_update.connect(self.linguist_update)
        self.launcher.drift_update.connect(self.drift_update)
        self.launcher.xai_result_received.connect(self.xai_result_received)