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
# File: src/controllers/system_controller.py
# Author: Gabriel Moraes
# Date: 2026-02-16

import os
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

# --- Domain ---
from src.domain.app_state import AppState

# --- Managers ---
from src.managers.storage_manager import StorageManager

# --- Services (Abstracted interface) ---
from typing import Dict, Any

class SystemController(QObject):
    """
    The Master Operations Controller.
    
    Refactored V39 (SOLID Architecture):
    - [SRP] Removed OS folder scaffolding (Moved to StorageManager).
    - [DIP] Receives initialized `FenixService` and `phases` dictionary via Injection.
    - [OCP] Uses a flexible State Machine ruleset instead of hardcoded if/else rules.
    """

    # --- TELEMETRY SIGNALS (Aggregated) ---
    log_message = pyqtSignal(str)
    status_message = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    # --- LIFECYCLE SIGNALS ---
    optimization_finished = pyqtSignal()
    bootstrap_finished = pyqtSignal()
    online_system_started = pyqtSignal()
    online_system_stopped = pyqtSignal()
    
    # --- DATA STREAMS (Proxied from Runtime) ---
    engine_data_processed = pyqtSignal(dict)
    engine_global_results = pyqtSignal(dict)
    kinetic_data_ready = pyqtSignal(dict)
    
    # --- ADVANCED STREAMS ---
    audit_update = pyqtSignal(bool, float, float, list)
    linguist_update = pyqtSignal(str, str, float)
    drift_update = pyqtSignal(str, dict)
    xai_result_received = pyqtSignal(dict)

    # --- COMMANDS ---
    cmd_process_data = pyqtSignal(str, object)
    cmd_run_cycle = pyqtSignal()
    cmd_update_model = pyqtSignal(str)
    
    cmd_explain_buffer = pyqtSignal()
    cmd_explain_local = pyqtSignal(str)
    cmd_explain_global = pyqtSignal()

    def __init__(self, app_state: AppState, storage_manager: StorageManager, phases: Dict[str, Any], fenix_service: Any):
        super().__init__()
        self.app_state = app_state
        self.storage = storage_manager
        
        # [DIP] Injected Dependencies
        self.fenix_service = fenix_service
        self.phases = phases
        
        # [OCP] Dynamic State Tracking
        self._completed_phases = set()

        # Connect Storage
        if self.storage:
            try:
                self.storage.connect()
            except Exception as e:
                print(f"[SystemController] Storage Connection Warning: {e}")

        self._connect_fenix_signals()
        
        # --- Wire Up Injected Phases ---
        if "optimization" in self.phases:
            self._connect_optimization_phase(self.phases["optimization"])
        if "bootstrap" in self.phases:
            self._connect_bootstrap_phase(self.phases["bootstrap"])
        if "runtime" in self.phases:
            self._connect_runtime_phase(self.phases["runtime"])
            self._connect_commands(self.phases["runtime"])
        
        self.log_message.emit("[SystemController] 🟢 Initialized (V39 SOLID Final).")

    def _connect_fenix_signals(self):
        """Bridges FENIX service signals to UI log/status."""
        self.fenix_service.cycle_started.connect(lambda msg: self.log_message.emit(f"[FENIX] 🔥 {msg}"))
        self.fenix_service.cycle_finished.connect(self._on_fenix_finished)
        self.fenix_service.progress_update.connect(lambda msg, pct: self.status_message.emit(f"FENIX: {msg} ({pct}%)"))

    def _on_fenix_finished(self, success, msg):
        status_icon = '✅' if success else '❌'
        self.log_message.emit(f"[FENIX] {status_icon} {msg}")

    # =========================================================================
    # PHASE WIRING
    # =========================================================================

    def _connect_optimization_phase(self, phase_obj):
        phase_obj.log_message.connect(self.log_message)
        phase_obj.error_occurred.connect(self.error_occurred)
        phase_obj.optimization_finished.connect(self._on_optimization_finished)

    def _connect_bootstrap_phase(self, phase_obj):
        phase_obj.log_message.connect(self.log_message)
        phase_obj.bootstrap_finished.connect(self._on_bootstrap_finished)

    def _connect_runtime_phase(self, phase_obj):
        # Telemetry
        phase_obj.log_message.connect(self.log_message)
        phase_obj.status_message.connect(self.status_message)
        phase_obj.error_occurred.connect(self.error_occurred)
        
        # Lifecycle
        phase_obj.system_started.connect(self.online_system_started)
        phase_obj.system_stopped.connect(self.online_system_stopped)
        
        # Data Streams
        phase_obj.engine_data_processed.connect(self.engine_data_processed)
        phase_obj.engine_global_results.connect(self.engine_global_results)
        phase_obj.kinetic_data_ready.connect(self.kinetic_data_ready)
        phase_obj.audit_update.connect(self.audit_update)
        phase_obj.linguist_update.connect(self.linguist_update)
        phase_obj.drift_update.connect(self.drift_update)
        phase_obj.xai_result_received.connect(self.xai_result_received)

    def _connect_commands(self, runtime_obj):
        """Routes external commands (e.g. from UI) to the Runtime Phase."""
        self.cmd_process_data.connect(lambda t, p: runtime_obj.handle_command("process_data", p))
        self.cmd_run_cycle.connect(lambda: runtime_obj.handle_command("run_cycle"))
        self.cmd_update_model.connect(lambda p: runtime_obj.handle_command("update_model", p))
        self.cmd_explain_buffer.connect(lambda: runtime_obj.handle_command("explain_buffer"))
        self.cmd_explain_local.connect(lambda pid: runtime_obj.handle_command("explain_local", pid))
        self.cmd_explain_global.connect(lambda: runtime_obj.handle_command("explain_global"))

    # =========================================================================
    # INTERNAL HANDLERS (Dynamic State Completion)
    # =========================================================================

    def _on_optimization_finished(self):
        self._completed_phases.add("optimization")
        self.log_message.emit("🔒 Security: Phase 0 (Optimization) Verified.")
        self.status_message.emit("Phase 0 Complete. Ready for Phase 1.")
        self.optimization_finished.emit()

    def _on_bootstrap_finished(self):
        self._completed_phases.add("bootstrap")
        self.log_message.emit("🔒 Security: Phase 1 (Bootstrap) Verified.")
        self.status_message.emit("Phase 1 Complete. Ready for Phase 2.")
        self.bootstrap_finished.emit()

    # =========================================================================
    # PUBLIC API (State Machine)
    # =========================================================================

    def _execute_phase(self, phase_name: str, prerequisites: list) -> bool:
        """
        Generic State Machine executor. Evaluates rules before starting a mapped phase.
        """
        for req in prerequisites:
            if req not in self._completed_phases:
                msg = f"⛔ Phase '{req}' must be completed before starting '{phase_name}'."
                self.log_message.emit(msg)
                self.error_occurred.emit(msg)
                return False
                
        if phase_name in self.phases:
            result = self.phases[phase_name].start()
            # If start() returns a bool, respect it; otherwise assume success
            return result if isinstance(result, bool) else True
        else:
            self.error_occurred.emit(f"⛔ Required system phase '{phase_name}' is not registered.")
            return False

    def _halt_phase(self, phase_name: str):
        if phase_name in self.phases:
            self.phases[phase_name].stop()

    # --- External Invocation API ---
    def start_optimization(self):
        self._completed_phases.clear() # Reset state
        return self._execute_phase("optimization", prerequisites=[])

    def stop_optimization(self):
        self._halt_phase("optimization")

    def start_offline_bootstrap(self):
        self._execute_phase("bootstrap", prerequisites=["optimization"])

    def stop_offline_bootstrap(self):
        self._halt_phase("bootstrap")

    def start_online_operation(self):
        return self._execute_phase("runtime", prerequisites=["optimization", "bootstrap"])

    def stop_online_operation(self):
        self._halt_phase("runtime")