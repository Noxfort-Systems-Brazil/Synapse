# SYNAPSE - A Gateway of Intelligent Perception for Traffic Management
# Copyright (C) 2026 Noxfort Systems
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
# File: src/infrastructure/grpc_connector.py
# Author: Gabriel Moraes
# Date: 2026-02-21

import grpc
import asyncio
import sys
import os
import time
from typing import Dict, Any, Optional
from datetime import datetime

# Ensure proto modules are reachable
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

from proto import synapse_hft_pb2
from proto import synapse_hft_pb2_grpc

# SRP Modules
from src.infrastructure.hft_serializers import HFTSerializer
from src.infrastructure.hft_streamer import HFTStreamer
from src.utils.debug_logger import carina_logger

class GrpcConnector:
    """
    The High-Frequency Transport (HFT) Connector - Client Side.
    
    Refactored V16 (Auto-Recovery Feature):
    - Acts as the Facade/Orchestrator for the HFT Link.
    - Automatically caches the scenario payload for transparent reconnections.
    - If the connection drops, it pings until Carina returns, re-uploads map, and restarts.
    """

    def __init__(self, endpoint: str = "localhost:50051"):
        self.endpoint = endpoint
        self.channel = None
        self.stub = None
        self.is_running = False
        self.is_connected = False
        
        # Async Task tracking
        self._stream_task: Optional[asyncio.Task] = None
        
        # Sub-components
        self.streamer = HFTStreamer()
        
        # Recovery Cache
        self._recovery_payload: Optional[Dict] = None

    def set_recovery_payload(self, map_data: Dict[str, Any]):
        """Stores the map definition to be re-sent if the server restarts."""
        self._recovery_payload = map_data

    async def start_channel(self):
        """Initializes the raw gRPC Channel with Large File Support."""
        if self.channel: return
        
        print(f"[HFT-Link] 🔗 Opening Channel to {self.endpoint}...")
        
        MAX_MESSAGE_LENGTH = 50 * 1024 * 1024 # 50 MB
        opts = [
            ('grpc.keepalive_time_ms', 5000),
            ('grpc.keepalive_timeout_ms', 2000),
            ('grpc.keepalive_permit_without_calls', 1), 
            ('grpc.http2.max_pings_without_data', 0),
            ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
            ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
        ]
        self.channel = grpc.aio.insecure_channel(self.endpoint, options=opts)
        self.stub = synapse_hft_pb2_grpc.HFTLinkStub(self.channel)
        self.is_running = True

    async def start_streaming(self):
        """Starts the producer loop and enables the streamer component."""
        if not self.is_running:
            await self.start_channel()
            
        self.streamer.start()
        
        if not self._stream_task or self._stream_task.done():
            print("[HFT-Link] 🚀 Manual Stream Start Requested.")
            self._stream_task = asyncio.create_task(self._traffic_producer())

    async def close(self):
        """Graceful shutdown."""
        self.is_running = False
        self.is_connected = False
        self.streamer.stop()
        
        if self._stream_task:
            self._stream_task.cancel()
            try: await self._stream_task
            except asyncio.CancelledError: pass
            
        if self.channel:
            await self.channel.close()
            self.channel = None
            print("[HFT-Link] 🔌 Tunnel Closed.")

    # --- STAGE 1: HANDSHAKE ---

    async def ping(self) -> bool:
        if not self.channel: await self.start_channel()
        try:
            await self.stub.Ping(synapse_hft_pb2.Empty(), timeout=2.0)
            self.is_connected = True
            return True
        except grpc.aio.AioRpcError:
            self.is_connected = False
            return False

    # --- STAGE 2: TOPOLOGY ---

    async def send_scenario(self, map_data: Dict[str, Any]) -> bool:
        # Automatically cache the map data so we can reuse it if Carina restarts
        self.set_recovery_payload(map_data)

        if not self.is_connected: 
            print("[HFT-Link] ⚠️ Cannot send scenario: Not connected.")
            return False
            
        print(f"[HFT-Link] 🗺️  Uploading Scenario File (Binary)...")
        _rpc_start = time.time()
        _ts_start = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        try:
            # Delegate packing to Serializer
            proto_scenario = HFTSerializer.pack_scenario(map_data)
            
            # gRPC Call
            response = await self.stub.LoadScenario(proto_scenario, timeout=60.0)
            
            # --- Debug Log: GRPC_SCENARIO ---
            _ts_end = datetime.now().strftime('%H:%M:%S.%f')[:-3]
            _graph = map_data.get('graph', {})
            carina_logger.info(
                f"GRPC_SCENARIO | nodes={len(_graph.get('nodes',[]))} "
                f"| edges={len(_graph.get('edges',[]))} "
                f"| start={_ts_start} | end={_ts_end} "
                f"| latency={(time.time()-_rpc_start)*1000:.1f}ms "
                f"| accepted={response.accepted}"
            )
            
            if response.accepted:
                print(f"[HFT-Link] ✅ Scenario Upload Accepted.")
                return True
            else:
                print(f"[HFT-Link] ❌ Scenario Rejected: {response.message}")
                return False
        except Exception as e:
            _ts_end = datetime.now().strftime('%H:%M:%S.%f')[:-3]
            carina_logger.info(
                f"GRPC_SCENARIO | ERROR | start={_ts_start} | end={_ts_end} "
                f"| latency={(time.time()-_rpc_start)*1000:.1f}ms | error={e}"
            )
            print(f"[HFT-Link] 💥 Upload Failed: {e}")
            return False

    # --- STAGE 3: CONTROL ---

    async def set_system_state(self, command: str) -> bool:
        if not self.is_connected: return False
        
        cmd_enum = synapse_hft_pb2.ControlCommand.UNKNOWN
        if command == "START": cmd_enum = synapse_hft_pb2.ControlCommand.START
        elif command == "STOP": cmd_enum = synapse_hft_pb2.ControlCommand.STOP
        elif command == "PAUSE": cmd_enum = synapse_hft_pb2.ControlCommand.PAUSE
        
        print(f"[HFT-Link] 🎮 Sending Command: {command}")
        _rpc_start = time.time()
        _ts_start = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        try:
            req = synapse_hft_pb2.ControlCommand(action=cmd_enum)
            resp = await self.stub.SystemControl(req, timeout=5.0)
            
            # --- Debug Log: GRPC_COMMAND ---
            _ts_end = datetime.now().strftime('%H:%M:%S.%f')[:-3]
            carina_logger.info(
                f"GRPC_COMMAND | cmd={command} "
                f"| start={_ts_start} | end={_ts_end} "
                f"| latency={(time.time()-_rpc_start)*1000:.1f}ms "
                f"| success={resp.success}"
            )
            
            if resp.success:
                print(f"[HFT-Link] -> System State is now: {resp.new_state}")
                if command == "START": await self.start_streaming()
                return True
            return False
        except Exception as e:
            _ts_end = datetime.now().strftime('%H:%M:%S.%f')[:-3]
            carina_logger.info(
                f"GRPC_COMMAND | cmd={command} | ERROR "
                f"| start={_ts_start} | end={_ts_end} "
                f"| latency={(time.time()-_rpc_start)*1000:.1f}ms | error={e}"
            )
            print(f"[HFT-Link] ❌ Command Failed: {e}")
            return False

    # --- STAGE 4: STREAMING ORCHESTRATION ---

    async def enqueue_traffic_frame(self, frame_data: Dict[str, Any]):
        """
        Public API: Receives raw dict, serializes it, and buffers it.
        """
        if not self.is_running: return
        _enqueue_start = time.time()
        _ts_start = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        try:
            # 1. Serialize
            proto_frame = HFTSerializer.pack_traffic_frame(frame_data)
            # 2. Buffer (Delegated to Streamer)
            await self.streamer.enqueue_frame(proto_frame)
            
            # --- Debug Log: GRPC_FRAME ---
            _ts_end = datetime.now().strftime('%H:%M:%S.%f')[:-3]
            carina_logger.info(
                f"GRPC_FRAME | edges={len(frame_data.get('traffic',[]))} "
                f"| source={frame_data.get('source','?')} "
                f"| start={_ts_start} | end={_ts_end} "
                f"| serialize+enqueue={(time.time()-_enqueue_start)*1000:.2f}ms"
            )
        except Exception: 
            pass

    async def _traffic_producer(self):
        """
        The Main Loop that connects the Streamer Generator to the gRPC Stub.
        Handles connection errors and triggers recovery automatically.
        """
        print("[HFT-Link] 🌊 Traffic Stream Loop Initiated.")
        backoff = 1.0
        
        while self.is_running:
            try:
                if not self.stub: await self.start_channel()
                backoff = 1.0 
                
                # Connect the Streamer's generator directly to the gRPC call
                # This blocks until the stream breaks or is closed server-side
                response = await self.stub.StreamTraffic(self.streamer.frame_generator())
                print(f"[HFT-Link] 🏁 Stream Finished by Server: {response.state}")
                
                # If the loop is still running but the server finished the stream,
                # it means Carina closed gracefully. We should trigger recovery to reconnect.
                if self.is_running:
                    print("[HFT-Link] ⚠️ Stream ended unexpectedly. Starting Auto-Recovery...")
                    await self._handle_disconnection()
                    
            except grpc.aio.AioRpcError as e:
                if not self.is_running: break
                
                # Handle Connection Loss (UNAVAILABLE or CANCELLED)
                if e.code() in [grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.CANCELLED]:
                    print(f"[HFT-Link] ⚠️ Connection Lost ({e.code().name}). Starting Auto-Recovery...")
                    await self._handle_disconnection()
                else:
                    print(f"[HFT-Link] 💥 RPC Error: {e.code().name} - {e.details()}")
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 1.5, 5.0)
            
            except Exception as e:
                if self.is_running:
                    print(f"[HFT-Link] 💥 Generic Error: {e}")
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 1.5, 5.0)

        print("[HFT-Link] 🌊 Traffic Stream Loop Ended.")

    async def _handle_disconnection(self):
        """Helper to pause stream, recover, and resume seamlessly."""
        recovered = await self._perform_auto_recovery()
        if recovered:
            print(f"[HFT-Link] ♻️ Recovery Successful. Resuming Stream seamlessly.")
        else:
            print(f"[HFT-Link] ❌ Recovery Failed or Aborted.")
            await asyncio.sleep(2.0)

    async def _perform_auto_recovery(self) -> bool:
        """Attempts to reconnect, re-upload map, and re-arm the system without destroying channel."""
        print("[HFT-Link] 🩺 Polling for Server (Ping Loop)...")
        
        # 1. Ping Loop: Wait until Carina is back online
        while self.is_running:
            if await self.ping():
                print("[HFT-Link] ✅ Server is BACK online.")
                break
            await asyncio.sleep(2.0)
            
        if not self.is_running: return False

        # 2. Re-Upload Map transparently
        if self._recovery_payload:
            print("[HFT-Link] 🔄 Re-Uploading Map to new server instance...")
            if not await self.send_scenario(self._recovery_payload):
                return False
        else:
            print("[HFT-Link] ⚠️ No Recovery Payload! Carina might have an empty map.")
            return False

        # 3. Re-Arm System (Send START)
        print("[HFT-Link] 🎮 Re-Arming System...")
        return await self.set_system_state("START")