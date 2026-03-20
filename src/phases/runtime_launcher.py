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
# File: src/phases/runtime_launcher.py
# Author: Gabriel Moraes
# Date: 2026-02-13
#
# Refactored V3 (4-Thread Architecture):
#
# Thread 1 (Main)   → Qt UI event loop (never blocked)
# Thread 2 (Neural) → _EngineWorker: heartbeat-driven AI inference
# Thread 3 (XAI)    → Ephemeral: spawn on demand, die after analysis
# Thread 4 (Linguist) → Standby: sleeps until signal, processes, returns to standby

import traceback
from typing import Optional, TYPE_CHECKING

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot, QThread, QTimer, Qt

from src.domain.app_state import AppState

if TYPE_CHECKING:
    from src.engine.inference_engine import InferenceEngine
    from src.managers.kse_manager import KSEManager
    from src.services.fenix_service import FenixService
    from src.services.linguist_service import LinguistService
    from src.workers.xai_worker import XAIWorker


class _EngineWorker(QObject):
    """
    Thread 2 — Neural Inference Worker.
    
    Lives INSIDE the engine QThread. Handles:
    - InferenceEngine construction (loads neural models)
    - Heartbeat QTimer (1s inference cycles)
    - Cross-thread signal bridge to Linguist (Thread 4) and XAI (Thread 3)
    """

    # --- Proxy signals (bubbled up to RuntimeLauncher) ---
    log_message = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    engines_started = pyqtSignal()

    # Data streams
    engine_data_processed = pyqtSignal(dict)
    engine_global_results = pyqtSignal(dict)
    kinetic_data_ready = pyqtSignal(dict)

    # Advanced streams
    audit_update = pyqtSignal(bool, float, float, list)
    linguist_update = pyqtSignal(str, str, float)
    drift_update = pyqtSignal(str, dict)
    xai_result_received = pyqtSignal(dict)

    def __init__(self, app_state: AppState, fenix_service):
        super().__init__()
        self.app_state = app_state
        self.fenix_service = fenix_service
        self.inference_engine: Optional['InferenceEngine'] = None
        self._heartbeat: Optional[QTimer] = None

        # Thread 4 — Linguist (Standby)
        self._linguist_thread: Optional[QThread] = None
        self._linguist_service: Optional['LinguistService'] = None

        # Thread 3 — XAI (On-Demand)
        self._xai_worker: Optional['XAIWorker'] = None

    @pyqtSlot()
    def initialize(self):
        """
        Called when the QThread starts. Runs ENTIRELY on the worker thread.
        Builds the 3 subsystems: Neural (here), Linguist (Thread 4), XAI (Thread 3).
        """
        try:
            self.log_message.emit("[Launcher] ⏳ Building AI Engine on worker thread...")

            # --- Lazy imports ---
            from src.engine.inference_engine import InferenceEngine
            from src.workers.ingestion_worker import IngestionWorker
            from src.managers.graph_manager import GraphManager
            from src.services.historical_manager import HistoricalManager
            from src.services.linguist_service import LinguistService
            from src.managers.xai_manager import XAIManager
            from src.workers.xai_worker import XAIWorker
            from src.services.semantic_enricher import SemanticEnricher
            from src.factories.agent_factory import AgentFactory

            # ── 1. Shared Dependencies ──────────────────────────────────
            ingestion = IngestionWorker(self.app_state)
            graph_manager = GraphManager(self.app_state)
            historical_manager = HistoricalManager(self.app_state)
            enricher = SemanticEnricher(self.app_state)

            # ── 2. Thread 3 — XAI (On-Demand) ──────────────────────────
            # XAIWorker is now a QObject that spawns ephemeral threads
            self._xai_worker = XAIWorker(model_config={"feature_dim": 4})
            xai_manager = XAIManager(self._xai_worker, enricher)

            # Wire XAI results → launcher proxy
            self._xai_worker.result_ready.connect(self.xai_result_received)
            self.log_message.emit("[XAI] On-demand worker ready (ephemeral threads).")

            # ── 3. Thread 2 — Neural (this thread) ─────────────────────
            # InferenceEngine V4: no linguist/xai_worker, pure neural
            self.inference_engine = InferenceEngine(
                self.app_state,
                ingestion,
                graph_manager,
                historical_manager,
                xai_manager
            )

            # Wire neural output signals
            self.inference_engine.data_processed.connect(self.engine_data_processed)
            self.inference_engine.global_cycle_results.connect(self.engine_global_results)
            self.inference_engine.kinetic_data_ready.connect(self.kinetic_data_ready)
            self.inference_engine.audit_update.connect(self.audit_update)
            self.inference_engine.drift_update.connect(self.drift_update)

            # FENIX Health Check
            self.inference_engine.drift_update.connect(
                lambda t, p: self.fenix_service.check_health_metrics(p.get('loss', 0.0))
            )

            # ── 4. Thread 4 — Linguist (Standby) ──────────────────────
            config = getattr(self.app_state, 'config', {})
            agent_factory = AgentFactory(config)
            self._linguist_service = LinguistService(self.app_state, ingestion, agent_factory)
            self._linguist_thread = QThread()
            self._linguist_service.moveToThread(self._linguist_thread)
            self._linguist_thread.start()

            # Wire: InferenceEngine → Linguist (cross-thread, queued)
            self.inference_engine.linguist_check_requested.connect(
                self._linguist_service.run_check
            )
            # Wire: Linguist results → launcher proxy
            self._linguist_service.update_signal.connect(self.linguist_update)

            self.log_message.emit("[Linguist] Standby thread started (Thread 4).")

            # ── 5. Boot subsystems ─────────────────────────────────────
            self.inference_engine.initialize_system()

            # Heartbeat ON THIS THREAD (drives neural inference)
            self._heartbeat = QTimer()
            self._heartbeat.timeout.connect(self._on_tick)
            self._heartbeat.start(1000)

            self.log_message.emit("[Launcher] Inference Engine Online.")
            self.engines_started.emit()

        except Exception as e:
            self.error_occurred.emit(f"Failed to build AI Engine: {e}")
            traceback.print_exc()

    @pyqtSlot()
    def _on_tick(self):
        """Heartbeat: runs run_global_cycle on the worker thread."""
        if self.inference_engine:
            self.inference_engine.run_global_cycle()

    @pyqtSlot()
    def stop(self):
        """Graceful shutdown of this thread and child threads."""
        # Stop heartbeat
        if self._heartbeat:
            self._heartbeat.stop()

        # Stop neural engine
        if self.inference_engine:
            self.inference_engine.stop()

        # Stop Linguist thread (Thread 4)
        if self._linguist_thread:
            self._linguist_thread.quit()
            self._linguist_thread.wait()
            self._linguist_thread = None
            self._linguist_service = None

        # Cleanup XAI resources (Thread 3, ephemeral)
        if self._xai_worker:
            self._xai_worker.unload_resources()
            self._xai_worker = None


class RuntimeLauncher(QObject):
    """
    Handles the Compute Layer (Phase 2b).
    
    Refactored V3 (4-Thread Architecture):
    - Thread 2 (Neural): _EngineWorker in QThread — heartbeat AI inference
    - Thread 3 (XAI):    Ephemeral — spawns on-demand, dies after analysis
    - Thread 4 (Linguist): Standby — sleeps until signaled, processes, returns to idle
    - KSE (Physics): Dedicated thread for Kalman state estimation
    """

    # --- TELEMETRY SIGNALS ---
    log_message = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    
    # --- LIFECYCLE SIGNALS ---
    engines_started = pyqtSignal()
    engines_stopped = pyqtSignal()
    
    # --- DATA OUTPUTS ---
    packet_ready_to_send = pyqtSignal(dict) 
    
    # Visualization / Debug Streams
    engine_data_processed = pyqtSignal(dict)
    engine_global_results = pyqtSignal(dict)
    kinetic_data_ready = pyqtSignal(dict)
    
    # --- ADVANCED STREAMS ---
    audit_update = pyqtSignal(bool, float, float, list)
    linguist_update = pyqtSignal(str, str, float)
    drift_update = pyqtSignal(str, dict)
    xai_result_received = pyqtSignal(dict)

    def __init__(self, app_state: AppState, fenix_service: 'FenixService'):
        super().__init__()
        self.app_state = app_state
        self.fenix_service = fenix_service

        from src.managers.kse_manager import KSEManager
        self._KSEManager = KSEManager

        # Components
        self.engine_thread: Optional[QThread] = None
        self._engine_worker: Optional[_EngineWorker] = None
        
        self.kse_thread: Optional[QThread] = None
        self.kse_manager: Optional['KSEManager'] = None

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def start_engines(self):
        """Initializes and starts all compute threads."""
        if self._engine_worker:
            self.log_message.emit("[Launcher] Engines already running.")
            return

        self.log_message.emit("[Launcher] 🚀 Initializing Compute Engines...")
        
        try:
            # ── A. KSE Manager (Physics Layer) ──────────────────────────
            self.kse_thread = QThread()
            self.kse_manager = self._KSEManager(self.app_state)
            self.kse_manager.moveToThread(self.kse_thread)
            
            self.kse_thread.started.connect(self.kse_manager.start)
            self.kse_manager.data_ready_for_transmission.connect(self.packet_ready_to_send)
            
            self.kse_thread.start()
            self.log_message.emit("[KSE] Physics Engine Started on Dedicated Thread.")

            # ── B. AI Engine (Thread 2 + Thread 3 + Thread 4) ───────────
            self._engine_worker = _EngineWorker(self.app_state, self.fenix_service)
            self.engine_thread = QThread()
            self._engine_worker.moveToThread(self.engine_thread)

            self._wire_worker_signals()

            self.engine_thread.started.connect(self._engine_worker.initialize)
            self.engine_thread.finished.connect(self.engine_thread.deleteLater)

            self.engine_thread.start()
            
        except Exception as e:
            self.error_occurred.emit(f"Failed to start Engines: {str(e)}")
            traceback.print_exc()

    def stop_engines(self):
        """Safely terminates all compute threads."""
        self.log_message.emit("[Launcher] 🛑 Stopping Compute Engines...")
        
        # 1. Stop KSE
        if self.kse_manager:
            self.kse_manager.stop()
        if self.kse_thread:
            self.kse_thread.quit()
            self.kse_thread.wait()
            self.kse_thread = None
            self.kse_manager = None

        # 2. Stop AI Engine Worker (also stops Linguist & XAI inside)
        if self._engine_worker:
            self._engine_worker.stop()
        if self.engine_thread:
            self.engine_thread.quit()
            self.engine_thread.wait()
            self.engine_thread = None
            self._engine_worker = None
            
        self.engines_stopped.emit()

    def handle_command(self, command_type: str, payload: object = None):
        """Route commands to the appropriate thread."""
        if not self._engine_worker or not self._engine_worker.inference_engine:
            return

        engine = self._engine_worker.inference_engine

        if command_type == "process_data":
            engine.process_data_point(payload)
        elif command_type == "run_cycle":
            engine.run_global_cycle()
        elif command_type == "explain_buffer":
            engine.process_veto_buffer()
        elif command_type == "explain_local":
            engine.explain_local_agent(payload)
        elif command_type == "explain_global":
            engine.explain_global_fuser()
        elif command_type == "update_model":
            self.log_message.emit(f"[Launcher] Model update requested: {payload}")

    # =========================================================================
    # INTERNAL WIRING
    # =========================================================================

    def _wire_worker_signals(self):
        """Connects _EngineWorker signals to RuntimeLauncher proxy signals."""
        w = self._engine_worker

        # Telemetry
        w.log_message.connect(self.log_message)
        w.error_occurred.connect(self.error_occurred)

        # Lifecycle
        w.engines_started.connect(self.engines_started)

        # Data Streams
        w.engine_data_processed.connect(self.engine_data_processed)
        w.engine_global_results.connect(self.engine_global_results)
        w.kinetic_data_ready.connect(self.kinetic_data_ready)

        # Advanced Streams
        w.audit_update.connect(self.audit_update)
        w.linguist_update.connect(self.linguist_update)
        w.drift_update.connect(self.drift_update)
        w.xai_result_received.connect(self.xai_result_received)

        # Critical Link: AI → Physics (cross-thread, safe via QueuedConnection)
        w.engine_global_results.connect(self._relay_engine_to_kse)

    @pyqtSlot(dict)
    def _relay_engine_to_kse(self, results: dict):
        """Syncs the Physics Engine (KSE) with the latest AI Snapshot.
        
        V2: Always calls sync_with_reality to keep KSE transmission clock alive,
        even if sensor_snapshot is empty (allows KSE to send DB-sourced data).
        """
        if self.kse_manager:
            snapshot = results.get('sensor_snapshot', {})
            self.kse_manager.sync_with_reality(snapshot)
