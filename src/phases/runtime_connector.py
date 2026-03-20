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
# File: src/phases/runtime_connector.py
# Author: Gabriel Moraes
# Date: 2026-02-13
import os
import gzip
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, TYPE_CHECKING, List, Tuple, Dict, Any
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot, QTimer

from src.domain.app_state import AppState

if TYPE_CHECKING:
    from src.workers.hft_bootstrapper import HFTBootstrapper

class RuntimeConnector(QObject):
    """
    Handles the Network Connection State Machine (Phase 2a).
    
    Location: src/phases/runtime_connector.py
    
    Responsibilities:
    - Establishes connection with CARINA (gRPC).
    - Handles the Handshake sequence: Ping -> Map Upload -> System Arm.
    - Manages connection retries and failures.
    """

    # --- SIGNALS ---
    log_message = pyqtSignal(str)
    status_message = pyqtSignal(str)
    connection_failed = pyqtSignal(str)
    
    # Critical Success Signal: Emitted when system is fully ARMED and ready for engines
    connection_ready_to_start = pyqtSignal() 

    def __init__(self, app_state: AppState):
        super().__init__()
        self.app_state = app_state
        
        # Lazy Import
        from src.workers.hft_bootstrapper import HFTBootstrapper
        self._HFTBootstrapper = HFTBootstrapper
        
        self.hft_worker: Optional['HFTBootstrapper'] = None
        self.is_connecting = False

    def start_connection(self):
        """Begins the connection sequence."""
        if self.is_connecting:
            return

        self.is_connecting = True
        self.status_message.emit("Connecting to Carina...")
        
        # Initialize Worker
        self.hft_worker = self._HFTBootstrapper(endpoint="localhost:50051")
        
        # Wire Signals
        self.hft_worker.log_message.connect(self.log_message)
        self.hft_worker.worker_ready.connect(self._on_worker_ready)
        self.hft_worker.connection_success.connect(self._on_ping_success)
        self.hft_worker.connection_failed.connect(self._on_ping_failed)
        self.hft_worker.map_upload_success.connect(self._on_map_success)
        self.hft_worker.map_upload_failed.connect(self._on_map_failed)
        self.hft_worker.system_armed_success.connect(self._on_arm_success)
        self.hft_worker.system_armed_failed.connect(self._on_arm_failed)
        
        self.hft_worker.start()

    def stop_connection(self):
        """Terminates the network worker."""
        if self.hft_worker and self.hft_worker.isRunning():
            self.hft_worker.stop_worker()
            self.hft_worker = None
        self.is_connecting = False

    def send_packet(self, packet: dict):
        """Proxy method to send runtime data via the active worker."""
        if self.hft_worker and hasattr(self.hft_worker, 'send_runtime_command'):
            self.hft_worker.send_runtime_command(packet)

    # =========================================================================
    # STATE MACHINE HANDLERS
    # =========================================================================

    @pyqtSlot()
    def _on_worker_ready(self):
        self.log_message.emit("[HFT] 📡 Thread Ready. Pinging Carina...")
        if self.hft_worker:
            self.hft_worker.request_ping()

    @pyqtSlot()
    def _on_ping_success(self):
        self.log_message.emit("[HFT] ✅ Connection Established.")
        
        raw_path = self.app_state.get_map_source_path()
        if not raw_path:
            self.connection_failed.emit("Cannot start: No map file loaded.")
            return
            
        self.status_message.emit(f"Uploading Map Topology...")
        try:
            # Build Map Payload
            map_payload = self._build_map_payload(raw_path)
            self.hft_worker.request_map_upload(map_payload)
        except Exception as e:
            self.connection_failed.emit(f"Map Payload Build Error: {str(e)}")

    @pyqtSlot()
    def _on_map_success(self):
        self.log_message.emit("[HFT] ✅ Map Topology Accepted.")
        self.status_message.emit("Arming System...")
        
        # Synchronization Check: M.E.H must be ready
        if not getattr(self.app_state, 'is_meh_ready', False):
            self.log_message.emit("[HFT] ⏳ Waiting for M.E.H classification to finish before arming...")
            QTimer.singleShot(500, self._on_map_success)
            return

        self.hft_worker.request_system_arm()

    @pyqtSlot(str)
    def _on_map_failed(self, error):
        self.connection_failed.emit(f"Map Upload Failed: {error}")

    @pyqtSlot()
    def _on_arm_success(self):
        self.log_message.emit("[HFT] ✅ System ARMED. Network Link Ready.")
        # Handover control to the next phase (Launcher)
        self.connection_ready_to_start.emit()

    @pyqtSlot(str)
    def _on_arm_failed(self, error):
        self.connection_failed.emit(f"System Arm Failed: {error}")

    @pyqtSlot()
    def _on_ping_failed(self):
        self.log_message.emit("[HFT] ⚠️ Ping Failed. Retrying in 2s...")
        QTimer.singleShot(2000, lambda: self.hft_worker.request_ping() if self.hft_worker else None)

    # =========================================================================
    # UTILS
    # =========================================================================

    def _build_map_payload(self, raw_path: str) -> dict:
        """Constructs the map topology payload for CARINA."""
        geometry_shapes = []
        for e in self.app_state.get_all_edges():
            if e.shape and len(e.shape) > 0:
                flat_coords = [coord for point in e.shape for coord in point]
                geometry_shapes.append({"edge_id": e.id, "coords": flat_coords})

        return {
            "map_hash": "v1.0_synapse_auto",
            "map_file_path": raw_path, 
            "peak_schedule_json": self._load_peak_schedule_json(), 
            "graph": {
                "nodes": [{
                    "id": n.id, "type": n.node_type, "x": n.x, "y": n.y, "tl_logic_id": n.tl_logic_id 
                } for n in self.app_state.get_all_nodes()],
                "edges": [{
                    "id": e.id, "source_node": e.from_node, "target_node": e.to_node
                } for e in self.app_state.get_all_edges()]
            },
            "geometry": {"shapes": geometry_shapes}
        }

    def _load_peak_schedule_json(self) -> str:
        """Helper to read the peak schedule JSON from the config directory."""
        try:
            home_dir = os.path.expanduser("~")
            base_dir = os.path.join(home_dir, "Documentos", "Synapse")
            if not os.path.exists(base_dir):
                base_dir = os.path.join(home_dir, "Documents", "Synapse")
            
            json_path = os.path.join(base_dir, "data", "config", "peak_schedule.json")
            if os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    return f.read()
            else:
                self.log_message.emit(f"[RuntimeConnector] ⚠️ No peak_schedule.json found at {json_path}")
                return ""
        except Exception as e:
            self.log_message.emit(f"[RuntimeConnector] ❌ Error reading peak_schedule.json: {e}")
            return ""