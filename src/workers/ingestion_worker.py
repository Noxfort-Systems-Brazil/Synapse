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
# File: src/workers/ingestion_worker.py
# Author: Gabriel Moraes
# Date: 2025-12-25

import time
import requests
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

# --- Infrastructure & Domain ---
from src.infrastructure.sensor_gateway import SensorGateway
from src.engine.ingestion_pipeline import IngestionPipeline
from src.domain.app_state import AppState

# --- Utils ---
from src.utils.logging_setup import logger

class IngestionWorker(QObject):
    """
    Worker dedicated to Data Ingestion (The 'Mouth' of the System).
    
    Responsibility (SRP):
    1. Manage the Local Gateway (Push Strategy) - Port 8080.
    2. Manage HTTP Pollers (Pull Strategy) - External APIs.
    3. Run the Zero Trust Pipeline (Filtering).
    
    Timing Logic:
    - Starts Request (Global) fetching IMMEDIATELY upon start.
    - Then waits for the configured interval (2.5 min) for subsequent fetches.
    """
    
    # Signal emitted when valid data passes the pipeline
    # Arguments: source_id (str), payload (dict)
    data_ready = pyqtSignal(str, object)

    def __init__(self, app_state: AppState):
        super().__init__()
        self.app_state = app_state
        
        # 1. The Filter (Zero Trust Logic)
        self.pipeline = IngestionPipeline(self.app_state)
        
        # 2. PUSH Strategy (Gateway)
        # Listens for incoming POST requests on port 8080
        self.gateway = SensorGateway()
        self.gateway.data_received.connect(self._handle_incoming_data)
        self.gateway.server_error.connect(lambda e: logger.error(f"[Ingestion] Gateway Error: {e}"))
        
        # 3. PULL Strategy (HTTP Workers)
        # Manages background threads for fetching data from URLs
        self.http_workers = ThreadPoolExecutor(max_workers=4)
        self.last_fetch_time = 0.0
        
        # Configuration: Fetch Global Data every 150 seconds (2.5 minutes)
        # BUT we will trigger the first one immediately in start()
        self.fetch_interval = 150.0 

    def start(self):
        """Starts the active listening components and triggers initial fetch."""
        logger.info("[IngestionWorker] Starting Sensor Gateway...")
        self.gateway.start()
        
        # --- MODIFICAÇÃO: Disparo Imediato ---
        logger.info("[IngestionWorker] 🚀 Triggering IMMEDIATE Global Fetch (Start-up)...")
        self._trigger_fetch()
        # Atualiza o tempo para que a próxima busca ocorra só daqui a 2.5 min
        self.last_fetch_time = time.time()

    def stop(self):
        """Stops gateway and thread pool."""
        logger.info("[IngestionWorker] Stopping network services...")
        if self.gateway.isRunning():
            self.gateway.stop()
        self.http_workers.shutdown(wait=False)

    def check_global_fetch(self, current_time: float):
        """
        Called by the main cycle to see if it's time to request external data.
        Non-blocking check.
        """
        if current_time - self.last_fetch_time > self.fetch_interval:
            self._trigger_fetch()
            self.last_fetch_time = current_time

    def get_pipeline(self) -> IngestionPipeline:
        """Exposes the pipeline for Linguist/Quarantine checks."""
        return self.pipeline

    # --- INTERNAL LOGIC ---

    def _trigger_fetch(self):
        """Finds all GLOBAL sources and schedules a fetch task."""
        sources = self.app_state.get_all_data_sources()
        count = 0
        for src in sources:
            # Check if it is a Global Source (Pull) AND has a valid URL
            if not src.is_local and src.connection_string and src.connection_string.startswith("http"):
                logger.info(f"[IngestionWorker] ☁️ Fetching data from Global Source: '{src.name}'...")
                # Dispatch to thread pool
                self.http_workers.submit(self._worker_fetch, src.id, src.connection_string)
                count += 1
        
        if count == 0:
            # Silent debug log if nothing to fetch
            pass

    def _worker_fetch(self, source_id: str, url: str):
        """
        Background thread logic to fetch data via HTTP GET.
        Handles JSON and simple CSV/Text responses.
        """
        try:
            # 10s timeout to prevent hanging threads
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                payload = {}
                
                # A. Try JSON
                try:
                    payload = response.json()
                except:
                    # B. Fallback to Text/CSV
                    text_data = response.text.strip()
                    if ',' in text_data or '\n' in text_data:
                         payload = {"raw_csv": text_data}
                         # Simple heuristic parsing (Key=Value or CSV)
                         parts = text_data.split(',')
                         for p in parts:
                             if '=' in p:
                                 k, v = p.split('=', 1)
                                 payload[k.strip()] = v.strip()
                    else:
                        payload = {"value": text_data}
                
                # C. Zero Trust Injection
                # If the external API didn't send the ID, we inject it so the pipeline knows who it is.
                if isinstance(payload, dict) and 'source_id' not in payload:
                    payload['source_id'] = source_id

                # D. Send to Main Pipeline
                # We call the handler directly (thread-safe due to internal locks or signal emission downstream)
                self._handle_incoming_data(source_id, payload)
                
            else:
                logger.warning(f"[IngestionWorker] Fetch failed for {source_id}: HTTP {response.status_code}")

        except Exception as e:
            logger.warning(f"[IngestionWorker] Connection error for {source_id}: {e}")

    @pyqtSlot(str, object)
    def _handle_incoming_data(self, source_id: str, payload: Any):
        """
        The Central Funnel. 
        All data (Push or Pull) ends up here to be filtered by the Pipeline.
        """
        # 1. Pipeline Validation (Zero Trust)
        is_accepted = self.pipeline.process_packet(source_id, payload)
        
        # 2. Emission
        if is_accepted:
            # Only emit if the data passed the strict security checks
            self.data_ready.emit(source_id, payload)