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
# File: src/services/telemetry_service.py
# Author: Gabriel Moraes
# Date: 2025-11-23

import time
import threading
import psutil
import logging
import os
from pathlib import Path
from prometheus_client import start_http_server, Counter, Gauge, Histogram
from torch.utils.tensorboard import SummaryWriter

# Import our logger setup
from src.utils.logging_setup import setup_logger

class TelemetryService:
    """
    Central Telemetry Service.
    
    Responsibility:
    - Exposes internal system metrics to Prometheus/Grafana.
    - Monitors System Health (CPU, Memory).
    - Tracks Business Metrics (Inference Latency, Data Throughput, Auditor Vetos).
    - Runs an embedded HTTP server to serve these metrics.
    """

    def __init__(self, port: int = 8000):
        """
        Args:
            port: The HTTP port to expose metrics on (default 8000).
        """
        self.port = port
        self.logger = setup_logger("TelemetryService")
        self._is_running = False
        
        # --- METRICS DEFINITIONS ---
        
        # 1. System Metrics
        self.system_cpu_usage = Gauge('synapse_system_cpu_percent', 'Current system CPU usage')
        self.system_memory_usage = Gauge('synapse_system_memory_percent', 'Current system Memory usage')
        self.app_uptime = Gauge('synapse_uptime_seconds', 'Application uptime in seconds')
        
        # 2. Data Ingestion Metrics
        self.data_ingested = Counter('synapse_data_ingested_total', 'Total data points received', ['source_id'])
        
        # 3. AI Performance Metrics
        # Buckets suitable for sub-second inference (0.01s to 1.0s)
        self.inference_latency = Histogram(
            'synapse_ai_inference_duration_seconds', 
            'Time spent in AI inference', 
            ['model_type'] # e.g., 'tcn', 'gat', 'itransformer'
        )
        
        # 4. Safety Metrics (Critical)
        self.auditor_vetos = Counter('synapse_auditor_veto_total', 'Total number of actions vetoed by Safety AE')
        self.auditor_checks = Counter('synapse_auditor_checks_total', 'Total number of states audited')

        self.start_time = time.time()
        
        # 5. TensorBoard Engine (Initialized on start)
        self.tb_writer = None

    def _init_tensorboard(self):
        """Initializes TensorBoard SummaryWriter pointing to the shared volume logic."""
        try:
            # We map this to the universal installer path for tensorboard docker composition
            tb_log_dir = os.path.expanduser("~/.local/share/synapse/logs/tensorboard")
            os.makedirs(tb_log_dir, exist_ok=True)
            self.tb_writer = SummaryWriter(log_dir=tb_log_dir)
            self.logger.info(f"TensorBoard SummaryWriter initialized at {tb_log_dir}")
        except Exception as e:
            self.logger.error(f"Failed to initialize TensorBoard SummaryWriter: {e}")

    def start(self):
        """Starts the Prometheus HTTP server and the system monitoring thread."""
        try:
            # Start Prometheus Server (Non-blocking, it spawns its own thread)
            start_http_server(self.port)
            self.logger.info(f"Prometheus metrics server started on port {self.port}")
            
            # Start internal monitoring loop
            self._is_running = True
            self._init_tensorboard()
            self.monitor_thread = threading.Thread(target=self._run_system_monitor, daemon=True)
            self.monitor_thread.start()
            
        except Exception as e:
            self.logger.error(f"Failed to start Telemetry Service: {e}")

    def stop(self):
        """Stops the internal monitoring loop."""
        self._is_running = False
        if self.tb_writer:
            self.tb_writer.close()
        self.logger.info("Telemetry Service stopped.")

    def _run_system_monitor(self):
        """Periodically updates system-level metrics (CPU/RAM)."""
        while self._is_running:
            try:
                # Update Gauge values
                self.system_cpu_usage.set(psutil.cpu_percent())
                self.system_memory_usage.set(psutil.virtual_memory().percent)
                self.app_uptime.set(time.time() - self.start_time)
                
                time.sleep(5) # Update every 5 seconds
            except Exception as e:
                self.logger.error(f"Error in system monitor: {e}")
                time.sleep(5)

    # --- Public Recording Methods ---

    def record_ingestion(self, source_id: str):
        """Increment data counter for a specific source."""
        self.data_ingested.labels(source_id=source_id).inc()

    def record_inference_time(self, model_type: str, duration: float):
        """Record latency observation."""
        self.inference_latency.labels(model_type=model_type).observe(duration)

    def record_audit_result(self, is_safe: bool):
        """Record safety check result."""
        self.auditor_checks.inc()
        if not is_safe:
            self.auditor_vetos.inc()

    def record_loss(self, model_name: str, epoch: int, loss: float):
        """
        Record a training or inference loss scalar to TensorBoard.
        Args:
            model_name (str): Identifier for the neural network (e.g. 'fuser_itransformer')
            epoch (int): The current training step or inference sequence frame index.
            loss (float): The MSE or computed loss value.
        """
        if self.tb_writer:
            # e.g. Loss/fuser_itransformer
            self.tb_writer.add_scalar(f"Loss/{model_name}", loss, epoch)