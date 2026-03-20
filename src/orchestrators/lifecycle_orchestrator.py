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
# File: src/orchestrators/lifecycle_orchestrator.py
# Author: Gabriel Moraes
# Date: 2025-12-25
#
# Refactored V2 (2026-03-09): SOLID Compliance
# - SRP: DI wiring extracted to ServiceContainer
# - SRP: Boot logic extracted to BootSequence
# - DIP: Depends on container abstractions, not concrete imports

import enum
from PyQt6.QtCore import QObject, pyqtSignal, QTimer, pyqtSlot

from src.domain.app_state import AppState
from src.factories.service_container import ServiceContainer
from src.phases.boot_sequence import BootSequence
from src.services.optimizer_service import OptimizerService
from src.utils.logging_setup import logger


class SystemMode(enum.Enum):
    OFFLINE = "OFFLINE"
    BOOTING = "BOOTING"
    RUNNING = "RUNNING"     # Normal Mode (Inference)
    TRAINING = "TRAINING"   # Optimization Mode (Neural Training)
    ERROR = "ERROR"


class LifecycleOrchestrator(QObject):
    """
    The Master Conductor — Pure State Machine.
    
    Refactored V2 (SOLID):
    - SRP: Only manages state transitions and timer loops.
    - DIP: Receives subsystems from ServiceContainer (not created here).
    - OCP: New agents can be added to ServiceContainer without modifying this class.
    
    Responsibilities:
    1. State Machine: Controls flow (OFFLINE -> BOOTING -> RUNNING -> TRAINING).
    2. Timer Management: Runs engine cycle (200ms) and health check (2s).
    3. Optimization Lifecycle: Start/Stop training mode.
    """
    
    # Signals for UI
    status_changed = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    training_feedback = pyqtSignal(dict)
    
    def __init__(self, main_controller=None):
        super().__init__()
        self.controller = main_controller
        self.current_mode = SystemMode.OFFLINE
        
        logger.info("[Lifecycle] Initializing Orchestrator...")
        
        # 1. State
        self.app_state = AppState()
        
        # 2. Wire all subsystems via DI Container (SRP)
        container = ServiceContainer(self.app_state)
        self._services = container.build()
        
        # 3. Extract frequently-used services for convenience
        self.inference_engine = self._services["inference_engine"]
        self.cartographer = self._services["cartographer"]
        self.fenix = self._services["fenix"]
        self.storage_manager = self._services["storage_manager"]
        self.optimizer_service = None  # Lazy: created on demand
        
        # 4. Timers (Heartbeats)
        # A. Main Cycle (200ms) - Inference Engine
        self.engine_timer = QTimer()
        self.engine_timer.timeout.connect(self._trigger_engine_cycle)
        self.cycle_interval_ms = 200 
        
        # B. Watchdog (2000ms) - Health Monitor
        self.health_timer = QTimer()
        self.health_timer.timeout.connect(self._run_health_check)
        self.health_interval_ms = 2000

    # =========================================================================
    # STATE MACHINE: BOOT & SHUTDOWN
    # =========================================================================

    def boot_system(self):
        """Power On Sequence — delegates to BootSequence (SRP)."""
        if self.current_mode != SystemMode.OFFLINE:
            logger.warning("[Lifecycle] System already active.")
            return

        try:
            self._set_mode(SystemMode.BOOTING)
            logger.info("[Lifecycle] 🔌 Booting Sequence Initiated...")
            
            # Delegate all boot logic to BootSequence (SRP)
            BootSequence.execute(self.app_state, self._services)
            
            # Start Loops
            self.engine_timer.start(self.cycle_interval_ms)
            self.health_timer.start(self.health_interval_ms)
            
            self._set_mode(SystemMode.RUNNING)
            logger.info("[Lifecycle] ✅ System ONLINE and Stable.")
            
        except Exception as e:
            logger.critical(f"[Lifecycle] 💥 BOOT FAILED: {e}")
            self._emergency_stop(str(e))

    def shutdown_system(self):
        """Graceful Shutdown."""
        logger.info("[Lifecycle] 🛑 Shutdown Sequence Initiated...")
        
        # Stop Timers
        self.engine_timer.stop()
        self.health_timer.stop()
        
        # Stop Optimization
        if self.optimizer_service:
            self.stop_optimization()
            
        # Stop Subsystems
        self.inference_engine.stop()
        self.fenix.stop_watchdog()
        
        self._set_mode(SystemMode.OFFLINE)
        logger.info("[Lifecycle] 💤 System is OFFLINE.")

    # =========================================================================
    # OPTIMIZATION MANAGEMENT
    # =========================================================================

    def start_optimization(self):
        """Transitions to Training Mode."""
        if self.current_mode == SystemMode.OFFLINE:
            logger.error("[Lifecycle] Cannot optimize while OFFLINE.")
            return

        logger.info("[Lifecycle] 🚀 Initializing Optimization...")
        self._set_mode(SystemMode.TRAINING)
        
        try:
            # 1. Locate the Model in hierarchy: Engine -> Linguist -> Brain -> LSTM
            if not hasattr(self.inference_engine, 'linguist'):
                raise AttributeError("InferenceEngine missing 'linguist'.")
            
            brain = self.inference_engine.linguist.brain
            target_model = brain.physical_model
            
            if not target_model:
                raise ValueError("Neural Model not initialized.")

            # 2. Create Optimizer
            self.optimizer_service = OptimizerService(model=target_model)
            
            # 3. Connect Feedback Signal (Optimizer -> Lifecycle -> UI)
            self.optimizer_service.training_finished.connect(self._on_optimizer_feedback)
            
            logger.info("[Lifecycle] 🏋️ Optimization Service Started.")
            
        except Exception as e:
            logger.error(f"[Lifecycle] ❌ Optimization Failed: {e}")
            self._set_mode(SystemMode.RUNNING)  # Fallback

    def stop_optimization(self):
        """Stops training and frees memory."""
        if self.optimizer_service:
            logger.info("[Lifecycle] ⏹️ Stopping Optimization...")
            try:
                self.optimizer_service.training_finished.disconnect()
            except: pass
            
            # Destroy instance
            self.optimizer_service = None
            
        if self.current_mode != SystemMode.OFFLINE:
            self._set_mode(SystemMode.RUNNING)

    @pyqtSlot(dict)
    def _on_optimizer_feedback(self, stats: dict):
        """Relays training stats to UI."""
        self.training_feedback.emit(stats)

    # =========================================================================
    # LOOPS & HEALTH
    # =========================================================================

    def _trigger_engine_cycle(self):
        """Main Loop (200ms)."""
        if self.current_mode in [SystemMode.RUNNING, SystemMode.TRAINING]:
            self.inference_engine.run_global_cycle()

    def _run_health_check(self):
        """Watchdog Loop (2s). Reports status to Fenix."""
        status = {
            "mode": self.current_mode.value,
            "alive": True,
            "sources": len(self.app_state.get_all_data_sources())
        }
        self.fenix.report_health(status)

    def _emergency_stop(self, reason: str):
        self.error_occurred.emit(reason)
        self.shutdown_system()
        self._set_mode(SystemMode.ERROR)

    def _set_mode(self, mode: SystemMode):
        self.current_mode = mode
        self.status_changed.emit(mode.value)