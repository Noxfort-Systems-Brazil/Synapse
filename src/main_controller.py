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
# File: src/main_controller.py
# Author: Gabriel Moraes
# Date: 2025-12-27

from typing import TYPE_CHECKING
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from src.domain.app_state import AppState
from src.managers.storage_manager import StorageManager
from src.infrastructure.monitor_client import MonitorClient

# --- Lazy Imports Type Hinting ---
# Previne o ciclo de importação durante o tempo de carregamento
if TYPE_CHECKING:
    from src.controllers.system_controller import SystemController
    from src.controllers.project_controller import ProjectController
    from src.controllers.view_controller import ViewController

class MainController(QObject):
    """
    The Central Facade (Main Controller).
    
    Refactored V31 (Fix: Lazy Loading Sub-Controllers):
    - Moved System, Project, and View imports inside __init__.
    - This breaks the circular dependency chain that was causing ImportError.
    """

    # --- PROXY SIGNALS (To maintain UI Compatibility) ---
    
    # Telemetry
    log_message = pyqtSignal(str)
    status_message = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    progress_update = pyqtSignal(int)

    # Lifecycle Events
    optimization_finished = pyqtSignal()
    bootstrap_finished = pyqtSignal()
    online_system_started = pyqtSignal()
    online_system_stopped = pyqtSignal()

    # Data Streams (Fast Path)
    engine_data_processed = pyqtSignal(dict)
    engine_global_results = pyqtSignal(dict)
    kinetic_data_ready = pyqtSignal(dict)

    # Intelligence Streams
    audit_update = pyqtSignal(bool, float, float, list)
    linguist_update = pyqtSignal(str, str, float)
    drift_update = pyqtSignal(str, dict)
    xai_result_received = pyqtSignal(dict)

    # UI Control Signals
    request_tab_change = pyqtSignal(int)
    update_dock_visibility = pyqtSignal(str, bool)
    view_mode_changed = pyqtSignal(str)

    # Engine Commands
    cmd_process_data = pyqtSignal(str, object)
    cmd_run_cycle = pyqtSignal()
    cmd_update_model = pyqtSignal(str)
    
    # XAI Commands
    cmd_explain_buffer = pyqtSignal()
    cmd_explain_local = pyqtSignal(str)
    cmd_explain_global = pyqtSignal()

    def __init__(self):
        super().__init__()
        
        # --- LAZY IMPORTS (Critical Fix) ---
        # Imports happen ONLY when MainController is instantiated, 
        # ensuring the class is fully defined first.
        from src.controllers.system_controller import SystemController
        from src.controllers.project_controller import ProjectController
        from src.controllers.view_controller import ViewController

        # 1. Initialize Domain & Persistence
        self.app_state = AppState()
        self.storage = StorageManager()
        self.monitor_client = None

        # 2. Initialize Core Services & Phases (SOLID - Dependency Injection)
        from src.services.fenix_service import FenixService
        from src.phases.optimization_phase import OptimizationPhase
        from src.phases.bootstrap_phase import BootstrapPhase
        from src.phases.runtime_phase import RuntimePhase

        self.fenix_service = FenixService(self.storage, self.app_state)
        
        # Build the Phase Pipeline
        phases_pipeline = {
            "optimization": OptimizationPhase(self.app_state),
            "bootstrap": BootstrapPhase(self.app_state),
            "runtime": RuntimePhase(self.app_state, self.fenix_service)
        }

        # 3. Initialize Sub-Controllers
        self.project_ctrl = ProjectController(self.app_state, self.storage)
        self.system_ctrl = SystemController(self.app_state, self.storage, phases_pipeline, self.fenix_service)
        self.view_ctrl = ViewController()

        # 4. Wire Internal Signals
        self._wire_subsystems()

    def _wire_subsystems(self):
        """Connects the separated controllers to work as a cohesive unit."""
        
        self.system_ctrl.log_message.connect(self.view_ctrl.log)
        self.system_ctrl.status_message.connect(self.view_ctrl.status)
        self.system_ctrl.error_occurred.connect(self.view_ctrl.error_alert)
        self.system_ctrl.error_occurred.connect(lambda msg: self._report_telemetry_error(msg))
        
        self.project_ctrl.log_message.connect(self.view_ctrl.log)
        self.project_ctrl.error_occurred.connect(self.view_ctrl.error_alert)
        self.project_ctrl.error_occurred.connect(lambda msg: self._report_telemetry_error(msg))

        self.view_ctrl.append_log.connect(self.log_message)
        self.view_ctrl.update_status.connect(self.status_message)
        self.view_ctrl.update_progress.connect(self.progress_update)

        # --- B. Lifecycle & View Reaction ---
        self.system_ctrl.online_system_started.connect(self.view_ctrl.on_online_system_started)
        self.project_ctrl.project_loaded.connect(self.view_ctrl.on_project_loaded)
        
        self.system_ctrl.optimization_finished.connect(self.optimization_finished)
        self.system_ctrl.bootstrap_finished.connect(self.bootstrap_finished)
        self.system_ctrl.online_system_started.connect(self.online_system_started)
        self.system_ctrl.online_system_stopped.connect(self.online_system_stopped)

        # --- C. Data Streaming (Engine -> UI) ---
        self.system_ctrl.engine_data_processed.connect(self.engine_data_processed)
        self.system_ctrl.engine_global_results.connect(self.engine_global_results)
        self.system_ctrl.kinetic_data_ready.connect(self.kinetic_data_ready)
        
        self.system_ctrl.audit_update.connect(self.audit_update)
        self.system_ctrl.linguist_update.connect(self.linguist_update)
        self.system_ctrl.drift_update.connect(self.drift_update)
        self.system_ctrl.xai_result_received.connect(self.xai_result_received)

        # --- D. View Control (View -> UI) ---
        self.view_ctrl.request_tab_change.connect(self.request_tab_change)
        self.view_ctrl.update_dock_visibility.connect(self.update_dock_visibility)
        self.view_ctrl.view_mode_changed.connect(self.view_mode_changed)

        # --- E. Command Routing (Proxy -> System) ---
        self.cmd_process_data.connect(self.system_ctrl.cmd_process_data)
        self.cmd_run_cycle.connect(self.system_ctrl.cmd_run_cycle)
        self.cmd_update_model.connect(self.system_ctrl.cmd_update_model)
        
        self.cmd_explain_buffer.connect(self.system_ctrl.cmd_explain_buffer)
        self.cmd_explain_local.connect(self.system_ctrl.cmd_explain_local)
        self.cmd_explain_global.connect(self.system_ctrl.cmd_explain_global)

    # =========================================================================
    # TELEMETRY (Client Integration)
    # =========================================================================
    
    def init_telemetry(self, enabled: bool, host: str, port: int):
        if self.monitor_client:
            self.monitor_client.stop()
            
        self.monitor_client = MonitorClient(host=host, port=port, enabled=enabled)
        
    def reconfigure_telemetry(self, enabled: bool, host: str, port: int):
        self.init_telemetry(enabled, host, port)
        
    def _report_telemetry_error(self, message: str):
        if self.monitor_client and self.monitor_client.enabled:
            # Send Critical Error through MQTT
            self.monitor_client.report_incident(category="SOFTWARE", level="CRITICAL", message=message)

    def report_shutdown(self):
        """Called automatically when the application is cleanly exited by the user."""
        if self.monitor_client and self.monitor_client.enabled:
            from PyQt6.QtCore import QCoreApplication
            msg = QCoreApplication.translate("MainController", "System Shutdown Initiated")
            self.monitor_client.report_incident(category="SOFTWARE", level="CRITICAL", message=msg)
            
            # Blocking tiny delay to ensure MQTT queue flush before threading die-off
            import time
            time.sleep(0.5)

    # =========================================================================
    # PUBLIC API PROXIES (For UI Compatibility)
    # =========================================================================

    # --- System Operations ---
    @pyqtSlot() 
    def start_optimization(self): 
        started = self.system_ctrl.start_optimization()
        if started:
            self.view_ctrl.on_optimization_started()

    @pyqtSlot() 
    def stop_optimization(self): self.system_ctrl.stop_optimization()

    @pyqtSlot() 
    def start_offline_bootstrap(self): self.system_ctrl.start_offline_bootstrap()

    @pyqtSlot() 
    def stop_offline_bootstrap(self): self.system_ctrl.stop_offline_bootstrap()

    @pyqtSlot() 
    def start_online_operation(self): self.system_ctrl.start_online_operation()

    @pyqtSlot() 
    def stop_online_operation(self): self.system_ctrl.stop_online_operation()

    # --- Project Operations ---
    @pyqtSlot() 
    def create_new_project(self): self.project_ctrl.create_new_project()

    @pyqtSlot(str) 
    def load_project(self, path: str): self.project_ctrl.load_project(path)

    @pyqtSlot(str) 
    def save_project(self, path: str): self.project_ctrl.save_project(path)

    @pyqtSlot(str, str, str) 
    def add_data_source(self, n, t, c): self.project_ctrl.add_data_source(n, t, c)

    @pyqtSlot(str) 
    def remove_data_source(self, sid): self.project_ctrl.remove_data_source(sid)

    # --- XAI Proxies ---
    @pyqtSlot()
    def request_buffer_explanation(self): self.cmd_explain_buffer.emit()
    
    @pyqtSlot(str)
    def request_local_explanation(self, sid): self.cmd_explain_local.emit(sid)
    
    @pyqtSlot()
    def request_global_explanation(self): self.cmd_explain_global.emit()
    
    # --- View Proxies ---
    @pyqtSlot(str)
    def set_map_view_mode(self, mode): self.view_ctrl.set_map_view_mode(mode)