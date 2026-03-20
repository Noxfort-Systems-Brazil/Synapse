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
# File: src/phases/optimization_phase.py
# Author: Gabriel Moraes
# Date: 2026-02-16

import os
import threading
import traceback
from typing import Optional, TYPE_CHECKING

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from src.domain.app_state import AppState
from src.domain.entities import SourceType

if TYPE_CHECKING:
    from src.services.optimizer_service import OptimizerService

class OptimizationPhase(QObject):
    """
    Handles Phase 0: Optimization (AutoML).
    
    Location: src/phases/optimization_phase.py
    
    Responsibilities:
    - Runs the OptimizerService in a background thread.
    - Validates mandatory prerequisites (Map Source for GATv2).
    - Signals completion so the system can move to Bootstrap or Runtime.
    """

    # --- SIGNALS ---
    log_message = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    optimization_finished = pyqtSignal()

    def __init__(self, app_state: AppState):
        super().__init__()
        self.app_state = app_state
        
        # Lazy Import to avoid circular dependencies
        from src.services.optimizer_service import OptimizerService
        self._OptimizerService = OptimizerService

        self.optimizer_service: Optional['OptimizerService'] = None
        self.opt_thread: Optional[threading.Thread] = None

    def start(self) -> bool:
        """Starts the optimization process in a separate thread. Returns False if validation fails."""
        if self.opt_thread and self.opt_thread.is_alive(): 
            self.log_message.emit("Optimization already running.")
            return False

        self.log_message.emit(">>> Starting Phase 0: Optimization...")
        
        # 1. Mandatory Check: Map Source
        # GATv2 Lite requires graph topology for calibration.
        map_path = self.app_state.get_map_source_path()
        
        if not map_path or not os.path.exists(map_path):
            error_msg = "❌ Optimization Aborted: A Map Source is MANDATORY. GATv2 Lite requires topology for calibration."
            self.log_message.emit(error_msg)
            self.error_occurred.emit(error_msg)
            return False

        # 1b. Mandatory Check: LIVE Data Sources (at least 1 local + 1 global)
        # Historical/Parquet bases and SUMO network files are NOT live sensors.
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
            error_msg = f"❌ Optimization Aborted: Missing LIVE data sources: {', '.join(missing)}. Both LOCAL and GLOBAL sources are MANDATORY."
            self.log_message.emit(error_msg)
            self.error_occurred.emit(error_msg)
            return False

        # 2. Resolve Historical Data (Optional but recommended)
        parquet_path = None
        for src in self.app_state.get_all_data_sources():
            # FIXED: Defensive check to prevent AttributeError if connection_string is bool
            conn_str = src.connection_string
            if isinstance(conn_str, str) and conn_str.endswith(".parquet"):
                if os.path.exists(conn_str):
                    parquet_path = conn_str
                    break
        
        if not parquet_path:
            self.log_message.emit("⚠️ Warning: No historical Parquet data found. Optimization will be limited.")

        # 3. Initialize Service
        self.optimizer_service = self._OptimizerService()
        self.optimizer_service.optimization_finished.connect(self._on_optimization_finished)
        
        # 4. Start Thread
        self.opt_thread = threading.Thread(
            target=self._run_optimizer_adapter,
            args=(map_path, parquet_path),
            daemon=True
        )
        self.opt_thread.start()
        return True

    def stop(self):
        """Stops the optimization service gracefully."""
        self.optimizer_service = None
        self.log_message.emit("<<< Optimization Stop Requested.")

    def _run_optimizer_adapter(self, map_path, parquet_path):
        """Thread worker function."""
        try:
            if self.optimizer_service:
                self.optimizer_service.run(map_file_path=map_path, history_file_path=parquet_path)
            self.log_message.emit("✅ AutoML Cycle Finished successfully.")
        except Exception as e:
            self.log_message.emit(f"❌ Optimization Failed: {str(e)}")
            self.error_occurred.emit(str(e))
            traceback.print_exc()

    @pyqtSlot()
    def _on_optimization_finished(self):
        """Callback when the service finishes its work."""
        self.log_message.emit(">>> Optimization Phase Complete.")
        self.optimization_finished.emit()
        self.optimizer_service = None