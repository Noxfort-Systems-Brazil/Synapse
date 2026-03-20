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
# File: src/managers/kse_manager.py
# Author: Gabriel Moraes
# Date: 2026-02-13

import time
import traceback
from typing import Optional, Dict
from datetime import datetime

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot, QTimer, Qt

from src.domain.app_state import AppState
from src.utils.debug_logger import carina_logger

# --- Lazy Imports to avoid circular deps ---
from src.services.historical_manager import HistoricalManager

class KSEManager(QObject):
    """
    Kalman State Estimation (KSE) Manager - The Physics Engine.
    
    Refactored V42 (PyQt6 Fix):
    - Fixed TypeError in setTimerType by using strict Qt.TimerType Enum.
    - Implements the 'Elastic Transmission Window' (150ms - 250ms).
    """

    # Output: The physics packet ready for HFT transmission
    data_ready_for_transmission = pyqtSignal(dict)
    
    # Telemetry
    log_message = pyqtSignal(str)
    mode_changed = pyqtSignal(str) # "REALTIME" or "HISTORICAL"

    def __init__(self, app_state: AppState):
        super().__init__()
        self.app_state = app_state
        self.historical_manager = HistoricalManager(app_state)
        
        # State
        self.running = False
        self.current_mode = "REALTIME"
        
        # Physics State
        self.last_sensor_update_time = 0.0 # When we received data from sensors
        self.last_transmission_time = 0.0  # When we last sent to CARINA
        self.has_new_processed_data = False
        
        self.last_known_state = {} # Stores last valid traffic state
        
        # Parameters (The Elastic Window)
        self.min_window_ms = 0.150 # 150ms
        self.max_window_ms = 0.250 # 250ms
        self.sensor_timeout_threshold = 0.5 # 0.5s without data -> Switch to History (was 1.5s)
        
        # The Monitor Timer (High Resolution Checker)
        # Checks window conditions every 10ms
        self.monitor = QTimer(self)
        
        # FIX: PyQt6 requires the Enum object, not an integer
        self.monitor.setTimerType(Qt.TimerType.PreciseTimer)
        
        self.monitor.timeout.connect(self._check_elastic_window)

    def start(self):
        if self.running: return
        self.running = True
        self.last_transmission_time = time.time()
        
        # Auto-detect: If no data sources are configured, start directly in HISTORICAL mode
        all_sources = self.app_state.get_all_data_sources()
        if not all_sources:
            self.current_mode = "HISTORICAL"
            self.mode_changed.emit("HISTORICAL")
            self.log_message.emit("[KSE] ⚡ Starting Physics Engine in HISTORICAL mode (no sensors configured).")
            self.log_message.emit("[KSE] 📊 Gold Database will be used as data source within 150-250ms window.")
            
            # Proactive: Pre-fetch first frame from DB immediately
            try:
                initial_data = self.historical_manager.get_current_state_prediction()
                if initial_data:
                    self.last_known_state = initial_data
                    self.has_new_processed_data = True
                    self.log_message.emit(f"[KSE] ✅ Initial DB frame loaded ({len(initial_data)} edges).")
            except Exception as e:
                self.log_message.emit(f"[KSE] ⚠️ Initial DB fetch failed: {e}")
        else:
            self.log_message.emit(f"[KSE] ⚡ Starting Physics Engine (Elastic Window 150-250ms). Sources: {len(all_sources)}")
        
        self.monitor.start(10) # 10ms resolution check

    def stop(self):
        self.running = False
        self.monitor.stop()
        self.log_message.emit("[KSE] Physics Engine Stopped.")

    @pyqtSlot(dict)
    def sync_with_reality(self, sensor_snapshot: dict):
        """
        Called when InferenceEngine completes processing a frame.
        This signals that 'processing is done' and data is ready.
        """
        self.last_sensor_update_time = time.time()
        self.last_known_state = sensor_snapshot
        self.has_new_processed_data = True # Mark as ready for the opportunity window
        
        # Check logic immediately to minimize latency if window is already open
        if self.running:
            self._check_elastic_window()

        # Restore Realtime mode if needed
        if self.current_mode != "REALTIME":
            self.current_mode = "REALTIME"
            self.mode_changed.emit("REALTIME")
            self.log_message.emit("[KSE] 🟢 Sensor signal restored. Mode: REALTIME.")

    @pyqtSlot()
    def _check_elastic_window(self):
        """
        The core Logic of the Elastic Window.
        
        Architecture Fix V2: ALWAYS uses last_known_state from the neural pipeline
        (fed via sync_with_reality). Historical data is now routed through
        TrafficNode.ghost_step() → KSE → TCN → pipeline → sync_with_reality,
        so no direct bypass is needed.
        """
        if not self.running: return

        try:
            now = time.time()
            dt_transmission = now - self.last_transmission_time
            dt_sensor = now - self.last_sensor_update_time
            
            should_send = False

            # --- ELASTIC WINDOW LOGIC ---
            
            # 1. OPPORTUNITY WINDOW (150ms <= T < 250ms)
            # If 150ms have passed, we act.
            if dt_transmission >= self.min_window_ms:
                if self.has_new_processed_data:
                    # We have real neural data! Send it now.
                    should_send = True
                    packet_source = self.current_mode.lower()
                else:
                    # We hit 150ms but the neural engine hasn't finished the next frame.
                    # FORCE KSE DEAD RECKONING. This guarantees we send data WELL BEFORE 
                    # the 250ms maximum limit.
                    should_send = True
                    packet_source = "kse_dead_reckoning"

            # --- PAYLOAD BUILDER ---
            if should_send:
                # Mode Tracking (Telemetry only — does NOT change data source)
                if dt_sensor > self.sensor_timeout_threshold:
                    if self.current_mode == "REALTIME":
                        self.current_mode = "HISTORICAL"
                        self.mode_changed.emit("HISTORICAL")
                        self.log_message.emit(f"[KSE] ⚠️ Signal Lost ({dt_sensor:.1f}s). Mode: HISTORICAL (ghost_step active).")
                
                # If we are doing dead reckoning, extrapolate the physics by dt
                # so the cars keep moving smoothly on the map instead of freezing
                if packet_source == "kse_dead_reckoning":
                    payload_data = self._apply_kse_extrapolation(self.last_known_state, dt_transmission)
                else:
                    payload_data = self.last_known_state

                # Send and Reset
                self._transmit(payload_data, packet_source)

        except Exception as e:
            self.log_message.emit(f"[KSE] ❌ Monitor Error: {e}")
            traceback.print_exc()

    def _apply_kse_extrapolation(self, base_state: dict, dt: float) -> dict:
        """Projects traffic metrics forward locally using KSE physics velocities."""
        extrapolated = {}
        for source_id, metrics in base_state.items():
            new_metrics = dict(metrics)
            
            # Use physics 'v' (velocity) added by SnapshotBuilder to drift the density/value
            if "physics" in metrics:
                v = metrics["physics"].get("v", 0.0)
                # Simple linear dead reckoning projection (state = state + v * dt)
                # This ensures the HFT stream looks alive and fluid
                new_metrics["value"] = max(0.0, metrics.get("value", 0.0) + (v * dt))
                
            extrapolated[source_id] = new_metrics
            
        return extrapolated

    def _transmit(self, data: dict, source: str):
        """Helper to format and emit the packet."""
        _build_start = time.time()
        _ts_start = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        
        # Build strict schema for HFT
        traffic_list = []
        if data:
            for source_id, metrics in data.items():
                if "density" in metrics or "speed" in metrics:
                    # Data from Historical DB (already Edge formatted)
                    traffic_list.append({
                        "edge_id": str(source_id),
                        "density": float(metrics.get("density", 0.0)),
                        "speed": float(metrics.get("speed", 0.0)),
                        "queue": float(metrics.get("queue", 0.0))
                    })
                else:
                    # Data from Neural Pipeline (Node formatted)
                    speed = 0.0
                    if "physics" in metrics:
                        speed = float(metrics["physics"].get("v", 0.0))
                        
                    traffic_list.append({
                        "edge_id": str(source_id), # Map sensor/node to edge dynamically
                        "density": float(metrics.get("value", 0.0)),
                        "speed": speed,
                        "queue": 0.0
                    })
        
        packet = {
            "timestamp": time.time(),
            "source": source,
            "traffic": traffic_list
        }
        
        self.data_ready_for_transmission.emit(packet)
        
        # Reset State
        self.last_transmission_time = time.time()
        self.has_new_processed_data = False # Consumed
        
        # --- Debug Log: KSE_BUILD ---
        _build_end = time.time()
        _ts_end = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        carina_logger.info(
            f"KSE_BUILD | mode={source} | edges={len(traffic_list)} "
            f"| start={_ts_start} | end={_ts_end} "
            f"| build_time={(_build_end - _build_start)*1000:.2f}ms"
        )