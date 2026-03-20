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
# File: src/workers/hft_bootstrapper.py
# Author: Gabriel Moraes
# Date: 2025-12-06

import asyncio
from typing import Dict, Any, Optional
from PyQt6.QtCore import QThread, pyqtSignal, pyqtSlot, QObject

from src.infrastructure.grpc_connector import GrpcConnector

class HFTBootstrapper(QThread):
    """
    Async Worker dedicated to the HFT Initialization Sequence.
    
    Refactored V5 (Non-Blocking Task Architecture):
    - Solves 'QThread Destroyed' crash by running the Persistence Loop as a 
      cancellable background task.
    - Ensures 'STOP_THREAD' command is processed immediately, even during retries.
    - Improved cleanup logic in run() and __del__.
    """

    # --- Signals to MainController ---
    log_message = pyqtSignal(str)
    
    worker_ready = pyqtSignal()
    connection_success = pyqtSignal()
    connection_failed = pyqtSignal()
    map_upload_success = pyqtSignal()
    map_upload_failed = pyqtSignal(str)
    system_armed_success = pyqtSignal()
    system_armed_failed = pyqtSignal(str)
    
    finished = pyqtSignal()

    def __init__(self, endpoint: str = "localhost:50051"):
        super().__init__()
        self.endpoint = endpoint
        self.connector: Optional[GrpcConnector] = None
        self._loop = None
        
        # Internal State
        self._command_queue = asyncio.Queue()
        self._keep_running = True
        
        # Track background tasks (like the Map Retry Loop)
        self._current_task: Optional[asyncio.Task] = None

    def __del__(self):
        """
        Safety Net: Ensures thread stops before destruction.
        """
        if self.isRunning():
            self._keep_running = False
            try:
                self.wait(2000) 
            except: pass

    def run(self):
        """
        The Thread Entry Point.
        """
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        
        self.connector = GrpcConnector(self.endpoint)
        self.worker_ready.emit()
        
        try:
            self._loop.run_until_complete(self._process_commands())
        except Exception as e:
            self.log_message.emit(f"[HFT-Boot] 💥 Critical Thread Error: {e}")
        finally:
            # Cleanup Phase
            if self._current_task and not self._current_task.done():
                self._current_task.cancel()
                
            if self.connector:
                try:
                    self._loop.run_until_complete(self.connector.close())
                except: pass
            
            # Cancel all pending tasks to prevent "Task was destroyed but it is pending"
            try:
                pending = asyncio.all_tasks(self._loop)
                for task in pending:
                    task.cancel()
                self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except: pass
            
            self._loop.close()
            self.finished.emit()

    async def _process_commands(self):
        """
        Main Event Loop. 
        Now handles commands IMMEDIATELY, delegating long operations to Tasks.
        """
        await self.connector.start_channel()
        
        while self._keep_running:
            try:
                # Wait for command
                cmd_type, payload = await self._command_queue.get()
            except asyncio.CancelledError:
                break
            
            # --- COMMAND HANDLING ---
            
            if cmd_type == "STOP_THREAD":
                # Cancel any running background operation (e.g., Map Loop)
                if self._current_task and not self._current_task.done():
                    self._current_task.cancel()
                    try:
                        await self._current_task
                    except asyncio.CancelledError: pass
                break
                
            elif cmd_type == "PING":
                # Short operation, can await directly
                await self._handle_ping()
                
            elif cmd_type == "UPLOAD_MAP":
                # Long operation (Loop): Spawn as Task so we don't block STOP commands
                if self._current_task and not self._current_task.done():
                    self._current_task.cancel()
                self._current_task = asyncio.create_task(self._upload_persistence_task(payload))
                
            elif cmd_type == "ARM_SYSTEM":
                # Short operation
                await self._handle_arm()

            elif cmd_type == "SEND_FRAME":
                # Fast path: Route traffic packet to the gRPC stream
                await self.connector.enqueue_traffic_frame(payload)
                
            self._command_queue.task_done()

    # --- Handlers ---

    async def _handle_ping(self):
        try:
            success = await self.connector.ping()
            if success:
                self.connection_success.emit()
            else:
                self.connection_failed.emit()
        except Exception:
            self.connection_failed.emit()

    async def _upload_persistence_task(self, map_data: Dict):
        """
        The Background Task that loops until Carina accepts the map.
        Can be cancelled instantly by _process_commands if STOP is received.
        """
        self.log_message.emit("[HFT] 🗺️ Entering Topology Persistence Loop...")
        attempt_delay = 2.0 
        
        try:
            while True:
                try:
                    success = await self.connector.send_scenario(map_data)
                    if success:
                        self.log_message.emit("[HFT] ✅ Topology Confirmed by Carina. Green Light!")
                        self.map_upload_success.emit()
                        return # Task Complete
                    else:
                        self.log_message.emit(f"[HFT] ⏳ Carina busy/rejected. Retrying in {attempt_delay}s...")
                except Exception:
                     self.log_message.emit(f"[HFT] 🔌 Connection unstable. Retrying in {attempt_delay}s...")
                
                # Check for cancellation during sleep
                await asyncio.sleep(attempt_delay)
                
        except asyncio.CancelledError:
            self.log_message.emit("[HFT] 🛑 Upload Task Cancelled.")
            raise # Re-raise to let asyncio handle task state

    async def _handle_arm(self):
        try:
            success = await self.connector.set_system_state("START")
            if success:
                self.system_armed_success.emit()
            else:
                self.system_armed_failed.emit("Server refused START command")
        except Exception as e:
            self.system_armed_failed.emit(str(e))

    # --- Public API (Thread-Safe) ---

    def request_ping(self):
        self._safe_put("PING", None)

    def request_map_upload(self, map_data: Dict):
        self._safe_put("UPLOAD_MAP", map_data)

    def request_system_arm(self):
        self._safe_put("ARM_SYSTEM", None)

    def send_runtime_command(self, packet: dict):
        """Bridges KSE traffic packets from Qt thread into the async gRPC stream."""
        self._safe_put("SEND_FRAME", packet)

    def _safe_put(self, cmd: str, payload: Any):
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(
                self._command_queue.put_nowait, (cmd, payload)
            )

    def stop_worker(self):
        """Stops the thread gracefully."""
        if not self.isRunning(): return
        
        self._keep_running = False
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(
                self._command_queue.put_nowait, ("STOP_THREAD", None)
            )
        
        # Block until thread finishes. 
        # Since logic is non-blocking now, this should return instantly.
        self.wait()