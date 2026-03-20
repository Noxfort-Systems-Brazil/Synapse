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
# File: src/infrastructure/api_ingestor.py
# Author: Gabriel Moraes
# Date: 2025-11-26

import time
import httpx
import random
import math
from typing import Any, Optional
from PyQt6.QtCore import QThread, pyqtSignal

class APIIngestor(QThread):
    """
    Infrastructure adapter for Polling-based Data Sources (e.g., REST APIs).
    
    Responsibility:
    - Connects to external web services (Pull mechanism).
    - Performs periodic polling (default 60s) to fetch snapshots.
    - Extracts numeric data from JSON responses.
    - Feeds the Inference Engine via signals.
    
    Supports both Real HTTP requests and Mock simulation for testing.
    """
    
    # Signal: (source_id, value)
    data_received = pyqtSignal(str, float)
    # Signal: (source_id, error_message)
    connection_error = pyqtSignal(str, str)

    def __init__(self, source_id: str, url: str, interval: int = 60):
        """
        Args:
            source_id: The ID of the data source in the AppState.
            url: The endpoint URL (e.g., "https://api.tomtom.com/...") 
                 or "mock://..." for simulation.
            interval: Polling frequency in seconds (default 60).
        """
        super().__init__()
        self.source_id = source_id
        self.url = url
        self.interval = interval
        self.is_running = True
        
        # Simulation state (for mock mode)
        self._sim_step = 0

    def run(self):
        """Main polling loop."""
        print(f"[APIIngestor] Started polling for '{self.source_id}' every {self.interval}s.")
        
        while self.is_running:
            try:
                value = None
                
                # --- Strategy: Mock vs Real ---
                if self.url.startswith("mock://"):
                    value = self._fetch_mock_data()
                else:
                    value = self._fetch_real_data()
                
                # Emit if we got a valid number
                if value is not None:
                    self.data_received.emit(self.source_id, float(value))
                
            except Exception as e:
                self.connection_error.emit(self.source_id, str(e))
                # Backoff slightly on error to avoid spamming logs
                time.sleep(5)

            # Sleep for the polling interval (interruptible)
            # We split sleep into small chunks to allow faster stop()
            for _ in range(self.interval * 10): 
                if not self.is_running: break
                time.sleep(0.1)

    def stop(self):
        """Stops the thread safely."""
        self.is_running = False
        self.wait()
        print(f"[APIIngestor] Stopped polling '{self.source_id}'.")

    def _fetch_mock_data(self) -> float:
        """
        Generates a realistic traffic flow value based on time of day.
        Simulates a daily cycle with Morning/Evening peaks.
        """
        # Simulate a 24h cycle compressed into shorter steps
        t = self._sim_step
        
        # Formula: Base + Peaks + Noise
        morning_peak = 50 * math.exp(-((t % 100 - 30) ** 2) / 200)
        evening_peak = 60 * math.exp(-((t % 100 - 70) ** 2) / 200)
        base_flow = 20 + morning_peak + evening_peak
        noise = random.uniform(-5, 5)
        
        self._sim_step += 1
        return max(0.0, base_flow + noise)

    def _fetch_real_data(self) -> Optional[float]:
        """
        Executes a real HTTP GET request and parses the response.
        """
        try:
            # Use httpx for modern sync request (inside thread is fine)
            response = httpx.get(self.url, timeout=10.0)
            response.raise_for_status()
            
            data = response.json()
            return self._extract_first_numeric(data)
            
        except httpx.RequestError as e:
            print(f"[APIIngestor] Network error for {self.source_id}: {e}")
            return None
        except Exception as e:
            print(f"[APIIngestor] Parsing error for {self.source_id}: {e}")
            return None

    def _extract_first_numeric(self, data: Any) -> Optional[float]:
        """
        Heuristic parser: Recursively finds the first float/int in a JSON.
        In a full version, the Linguist Agent would tell us WHICH key to read.
        """
        if isinstance(data, (int, float)) and not isinstance(data, bool):
            return float(data)
        
        if isinstance(data, dict):
            for key, val in data.items():
                # Heuristic: Prioritize keys that look like traffic data
                if "speed" in key.lower() or "flow" in key.lower() or "value" in key.lower():
                    res = self._extract_first_numeric(val)
                    if res is not None: return res
                
            # Fallback: check all keys
            for val in data.values():
                res = self._extract_first_numeric(val)
                if res is not None: return res
                
        if isinstance(data, list):
            for item in data:
                res = self._extract_first_numeric(item)
                if res is not None: return res
                
        return None