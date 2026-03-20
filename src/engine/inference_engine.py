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
# File: src/engine/inference_engine.py
# Author: Gabriel Moraes
# Date: 2026-02-16
#
# Refactored V4 (4-Thread Architecture):
# - Removed XAIWorker lifecycle management (XAI is now ephemeral, managed externally).
# - Removed direct LinguistService.run_check() call (emits signal instead).
# - This engine is a PURE neural inference pipeline.

import time
import torch
import logging
from typing import Any, Dict
from datetime import datetime

# Qt Imports (CRITICAL for moveToThread)
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

# --- Domain & State ---
from src.domain.app_state import AppState

# --- Subsystems ---
from src.workers.ingestion_worker import IngestionWorker
from src.services.historical_manager import HistoricalManager
from src.engine.cycle_processor import CycleProcessor

# --- Factories & Builders ---
from src.engine.neural_factory import NeuralFactory
from src.engine.snapshot_builder import SnapshotBuilder

# --- Managers ---
from src.managers.graph_manager import GraphManager
from src.managers.xai_manager import XAIManager
from src.utils.debug_logger import perf_logger

class InferenceEngine(QObject):
    """
    The Real-Time Neural Inference Orchestrator.
    
    Refactored V4 (4-Thread Architecture):
    - PURE neural pipeline: Coordinator → Fuser → Auditor.
    - XAI and Linguist are decoupled and managed externally on their own threads.
    - Emits `linguist_check_requested` signal for the Linguist standby thread.
    """
    
    # --- CONFIGURATION ---
    LINGUIST_THROTTLE_CYCLES = 5
    
    # --- SIGNALS ---
    started = pyqtSignal()
    
    # Fast Path: Raw sensor data for UI Graphs
    data_processed = pyqtSignal(dict)
    
    # Slow Path: Full Inference Cycle Results
    global_cycle_results = pyqtSignal(dict)
    
    # Legacy Visualization
    kinetic_data_ready = pyqtSignal(dict)
    
    # Semantic & Security Signals
    audit_update = pyqtSignal(bool, float, float, list)
    drift_update = pyqtSignal(str, dict)
    
    # Cross-Thread Coordination
    linguist_check_requested = pyqtSignal()

    def __init__(self, 
                 app_state: AppState, 
                 ingestion: IngestionWorker,
                 graph_manager: GraphManager,
                 historical_manager: HistoricalManager,
                 xai_manager: XAIManager):
        super().__init__()
        self.app_state = app_state
        self.cycle_count = 0
        self.logger = logging.getLogger(__name__)
        
        # 1. Store Injected Dependencies (SOLID - DIP)
        self.ingestion = ingestion
        self.graph_manager = graph_manager
        self.historical_manager = historical_manager
        self.xai_manager = xai_manager
        
        # 2. Initialize Internal Core Builders
        self.neural_factory = NeuralFactory()
        self.device = self.neural_factory.get_device()
        self.agents = self.neural_factory.build_all(self.app_state)
        
        self.snapshot_builder = SnapshotBuilder(
            app_state=self.app_state,
            graph_manager=self.graph_manager,
            embedding_dim=32
        )
        
        self.processor = CycleProcessor(
            app_state=self.app_state,
            device=self.device,
            coordinator=self.agents.get('coordinator'),
            fuser=self.agents.get('fuser'),
            auditor=self.agents.get('auditor'),
            xai_manager=self.xai_manager,
            graph_manager=self.graph_manager
        )
        
        # 3. Signal Wiring
        self.ingestion.data_ready.connect(self._handle_data_flow)

    @pyqtSlot()
    def initialize_system(self):
        """Lifecycle Hook: Boots up ingestion and graph."""
        self.logger.info("[InferenceEngine] Booting Subsystems...")
        
        self.ingestion.start()
        self.graph_manager.rebuild_graph()
        
        if not self.historical_manager.is_ready:
            self.historical_manager.load_data()
        
        self.started.emit()
        self.logger.info("[InferenceEngine] System ONLINE (Neural Core Active).")

    @pyqtSlot()
    def stop(self):
        """Lifecycle Hook: Graceful shutdown."""
        self.logger.info("[InferenceEngine] Shutting down...")
        self.ingestion.stop()

    # =========================================================================
    # UNIFIED DATA PIPELINE
    # =========================================================================

    @pyqtSlot(str, object)
    def _handle_data_flow(self, source_id: str, payload: Any):
        """Centralized handler for incoming data."""
        if not source_id: return
        
        val = payload.get('value') if isinstance(payload, dict) else payload
        
        # Update Graph Memory
        self.graph_manager.update_node_memory(source_id, val)
        
        # Emit Fast Path
        self.data_processed.emit({"id": source_id, "raw": val})

    @pyqtSlot(str, object)
    def process_data_point(self, source_id: str, payload: Any):
        """External command (Manual Control)."""
        for src in self.app_state.get_all_data_sources():
            if src.id == source_id:
                src.latest_value = payload.get('value') if isinstance(payload, dict) else payload
                src.last_update = time.time()
                break
        self._handle_data_flow(source_id, payload)

    # =========================================================================
    # GLOBAL CYCLE ORCHESTRATION
    # =========================================================================

    @pyqtSlot()
    def run_global_cycle(self):
        """Executes the Neural Pipeline."""
        self.cycle_count += 1
        _cycle_start = time.time()
        _ts_start = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        now = time.time()
        
        # 1. Async Inputs
        self.ingestion.check_global_fetch(now)
        
        # 2. Linguist Trigger (Cross-Thread Signal → Standby Thread)
        if self.cycle_count % self.LINGUIST_THROTTLE_CYCLES == 0:
            self.linguist_check_requested.emit()
            
        # 3. Tick All Nodes (Architecture Fix: KSE Dead Reckoning)
        # Nodes that received sensor data this cycle already ran step().
        # Nodes that didn't will run ghost_step() via tick() → KSE predict.
        for node in self.graph_manager.nodes.values():
            node.tick()
        
        # 4. Build Snapshot (Now uses cached embeddings from step/ghost_step)
        snapshot = self.snapshot_builder.gather_snapshot()
        
        # 5. Global Inference
        results, _ = self.processor.run_logic(snapshot)
        
        # 5. Route Events
        if results.get("alert_event"):
            event = results["alert_event"]
            payload = event["payload"]
            self.drift_update.emit(event["title"], payload)
            
            if payload.get("status") in ["DRIFT", "ATTACK"]:
                loss = payload.get("loss", 0.0)
                self.audit_update.emit(True, loss, 0.15, [])
        
        # 6. Telemetry
        results['sensor_snapshot'] = snapshot
        self.global_cycle_results.emit(results)
        self.kinetic_data_ready.emit({}) 
        
        # --- Debug Log: GLOBAL_CYCLE ---
        _ts_end = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        perf_logger.info(
            f"GLOBAL_CYCLE | cycle={self.cycle_count} "
            f"| start={_ts_start} | end={_ts_end} "
            f"| total={(time.time() - _cycle_start)*1000:.2f}ms"
        )

    # =========================================================================
    # XAI PROXY
    # =========================================================================

    @pyqtSlot()
    def process_veto_buffer(self):
        active_nodes = [n.id for n in self.app_state.get_all_nodes()]
        self.xai_manager.process_buffer_strategy(active_nodes)

    @pyqtSlot(str)
    def explain_local_agent(self, source_id: str):
        node = self.graph_manager.get_node(source_id)
        if node:
            self.xai_manager.explain_local(source_id, node)
        else:
            self.xai_result_ready.emit({"type": "ERROR", "semantic_text": f"Node {source_id} not found."})

    @pyqtSlot()
    def explain_global_fuser(self):
        if self.agents.get('fuser') and self.graph_manager.nodes:
            self.xai_manager.explain_global(self.agents['fuser'], self.graph_manager.nodes, seq_len=60)
        else:
            self.xai_result_ready.emit({"type": "ERROR", "semantic_text": "Global Model not ready."})