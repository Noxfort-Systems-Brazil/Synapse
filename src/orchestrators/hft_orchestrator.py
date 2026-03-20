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
# File: src/orchestrators/hft_orchestrator.py
# Author: Gabriel Moraes
# Date: 2025-12-24

from typing import Optional
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot, QTimer

from src.workers.hft_bootstrapper import HFTBootstrapper
from src.domain.app_state import AppState

class HFTOrchestrator(QObject):
    """
    Manages the High-Frequency Traffic (HFT) connectivity protocol.
    
    Responsibilities:
    1. Establish gRPC connection with Carina/Simulator.
    2. Handle the 3-way Handshake (Ping -> Map Upload -> Arm System).
    3. Construct the Map Payload from AppState.
    4. Provide clear success/failure signals to the Main Orchestrator.
    
    Refactoring V2: Renamed from HFTController to HFTOrchestrator.
    """

    # --- SIGNALS ---
    log_message = pyqtSignal(str)
    status_message = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    
    connection_finished_success = pyqtSignal() # Ready to launch engine
    connection_aborted = pyqtSignal()          # Stopped by user or error

    def __init__(self, app_state: AppState):
        super().__init__()
        self.app_state = app_state
        self.hft_worker: Optional[HFTBootstrapper] = None
        self.is_connecting = False

    def start_connection_sequence(self):
        """Initiates the connection loop."""
        if self.is_connecting:
            self.log_message.emit("HFT Connection sequence already in progress.")
            return

        map_path = self.app_state.get_map_source_path()
        if not map_path:
            self.error_occurred.emit("Cannot start HFT: No map loaded.")
            return

        self.is_connecting = True
        self.log_message.emit(">>> Phase 2 Triggered. Initializing HFT Worker...")
        self.status_message.emit("Connecting to Carina...")

        # 1. Initialize Worker
        self.hft_worker = HFTBootstrapper(endpoint="localhost:50051")
        
        # 2. Connect Signals
        self.hft_worker.log_message.connect(self.log_message)
        self.hft_worker.worker_ready.connect(self._on_worker_ready)
        
        self.hft_worker.connection_success.connect(self._on_ping_success)
        self.hft_worker.connection_failed.connect(self._on_ping_failed)
        
        self.hft_worker.map_upload_success.connect(self._on_map_success)
        self.hft_worker.map_upload_failed.connect(self._on_map_failed)
        
        self.hft_worker.system_armed_success.connect(self._on_arm_success)
        self.hft_worker.system_armed_failed.connect(self._on_arm_failed)
        
        # 3. Start Thread
        self.hft_worker.start()

    def stop_connection_sequence(self):
        """Aborts any ongoing connection attempt."""
        self.log_message.emit("<<< Stopping HFT Connection...")
        
        if self.hft_worker and self.hft_worker.isRunning():
            self.hft_worker.stop_worker()
            self.hft_worker = None
        
        self.is_connecting = False
        self.connection_aborted.emit()

    # --- WORKER CALLBACKS ---

    @pyqtSlot()
    def _on_worker_ready(self):
        self.log_message.emit("[HFT] 📡 Thread Ready. Pinging Carina...")
        if self.hft_worker:
            self.hft_worker.request_ping()

    @pyqtSlot()
    def _on_ping_success(self):
        """Stage 1 OK -> Upload Map Structured Payload"""
        self.log_message.emit("[HFT] ✅ Connection Established.")
        
        raw_path = self.app_state.get_map_source_path()
        if not raw_path:
            self.error_occurred.emit("Map path lost during connection.")
            self.stop_connection_sequence()
            return
            
        self.status_message.emit(f"Uploading Map Topology...")
        self._upload_map_payload(raw_path)

    @pyqtSlot()
    def _on_ping_failed(self):
        """Loop: Retry Ping every 2 seconds"""
        self.log_message.emit("[HFT] ⚠️ Ping Failed. Retrying in 2s...")
        QTimer.singleShot(2000, lambda: self.hft_worker.request_ping() if self.hft_worker else None)

    def _upload_map_payload(self, raw_path: str):
        """Constructs the rich payload and sends it."""
        try:
            # --- Extract Geometry & Graph Topology ---
            geometry_shapes = []
            for e in self.app_state.get_all_edges():
                if e.shape and len(e.shape) > 0:
                    # Flatten tuples (x,y) -> [x, y, x, y]
                    flat_coords = [coord for point in e.shape for coord in point]
                    geometry_shapes.append({
                        "edge_id": e.id,
                        "coords": flat_coords
                    })

            # Build the rich payload Carina expects
            map_payload = {
                "map_hash": "v1.0_synapse_auto",
                "map_file_path": raw_path, 
                "graph": {
                    "nodes": [{
                        "id": n.id, 
                        "type": n.node_type, 
                        "x": n.x, 
                        "y": n.y,
                        "tl_logic_id": n.tl_logic_id 
                    } for n in self.app_state.get_all_nodes()],
                    
                    "edges": [{
                        "id": e.id, 
                        "source_node": e.from_node, 
                        "target_node": e.to_node
                    } for e in self.app_state.get_all_edges()]
                },
                "geometry": {
                    "shapes": geometry_shapes 
                }
            }
            
            if self.hft_worker:
                self.hft_worker.request_map_upload(map_payload)

        except Exception as e:
            self.error_occurred.emit(f"Map Payload Build Error: {str(e)}")
            self.stop_connection_sequence()

    @pyqtSlot()
    def _on_map_success(self):
        """Stage 2 OK -> Arm System"""
        self.log_message.emit("[HFT] ✅ Map Topology Accepted.")
        self.status_message.emit("Arming System...")
        if self.hft_worker:
            self.hft_worker.request_system_arm()

    @pyqtSlot(str)
    def _on_map_failed(self, error):
        self.error_occurred.emit(f"Map Upload Failed: {error}")
        self.stop_connection_sequence()

    @pyqtSlot()
    def _on_arm_success(self):
        """Stage 3 OK -> Handover to Lifecycle Controller"""
        self.log_message.emit("[HFT] ✅ System ARMED. Connection Complete.")
        
        # Stop the HFT worker as its job is done (unless we need heartbeats later)
        if self.hft_worker:
            self.hft_worker.stop_worker()
            self.hft_worker = None
            
        self.is_connecting = False
        self.connection_finished_success.emit()

    @pyqtSlot(str)
    def _on_arm_failed(self, error):
        self.error_occurred.emit(f"System Arm Failed: {error}")
        self.stop_connection_sequence()