# SYNAPSE - A Gateway of Intelligent Perception for Traffic Management
# Copyright (C) 2025 Noxfort Labs
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
# File: src/infrastructure/hft_streamer.py
# Author: Gabriel Moraes
# Date: 2026-02-14

import asyncio
import time
import logging
import sys
import os
from pathlib import Path
from typing import AsyncGenerator, Optional

# Ensure proto modules are reachable
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

from proto import synapse_hft_pb2

class HFTStreamer:
    """
    Manages the high-frequency packet stream logic.
    
    Responsibilities:
    1. Buffering: Manages an asyncio Queue for outgoing frames.
    2. Throttling: Enforces minimum intervals (150ms) to prevent network flooding.
    3. Heartbeats: Generates keep-alive frames when idle (Every 250ms).
    4. Telemetry: Logs transmission frequency rates for auditing.
    
    Refactored V2 (Aggressive Heartbeat):
    - Heartbeat interval reduced to 250ms for ultra-low latency connection checks.
    """

    def __init__(self, queue_size: int = 200, min_interval_s: float = 0.150):
        self.transmit_queue: asyncio.Queue = asyncio.Queue(maxsize=queue_size)
        self.is_streaming = False
        
        # Throttling Config
        self.min_interval = min_interval_s
        
        # Heartbeat Config: 250ms (0.25s)
        # If no real traffic is sent for 250ms, a dummy frame is sent.
        self.heartbeat_interval = 0.25 
        
        # Logging
        self.freq_logger = self._setup_frequency_logger()

    def _setup_frequency_logger(self) -> logging.Logger:
        """Configures a dedicated logger to track transmission intervals."""
        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "hft_transmission_rate.log"

        logger = logging.getLogger("HFT_Frequency")
        logger.setLevel(logging.INFO)
        
        # Avoid duplicate handlers
        if not logger.handlers:
            fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
            formatter = logging.Formatter('%(asctime)s - %(message)s')
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        
        return logger

    def start(self):
        self.is_streaming = True

    def stop(self):
        self.is_streaming = False

    async def enqueue_frame(self, proto_frame: synapse_hft_pb2.TrafficFrame):
        """
        Adds a pre-serialized frame to the buffer.
        Drops oldest if full (Traffic data is time-sensitive; old data is useless).
        """
        if not self.is_streaming:
            return

        try:
            if self.transmit_queue.full():
                try:
                    self.transmit_queue.get_nowait() # Drop oldest
                except asyncio.QueueEmpty:
                    pass
            self.transmit_queue.put_nowait(proto_frame)
        except Exception:
            # Failsafe for queue errors
            pass

    async def frame_generator(self) -> AsyncGenerator[synapse_hft_pb2.TrafficFrame, None]:
        """
        The Core Generator that yields frames to the gRPC Stub.
        Implements Throttling and fast Heartbeats.
        """
        self.freq_logger.info("Stream Started (Generator Init)")
        last_send_time = time.time()
        
        # Initial Heartbeat to wake up server immediately
        yield self._create_heartbeat()

        while self.is_streaming:
            try:
                # 1. Fetch Phase
                try:
                    # Wait for real data, but timeout quickly (250ms) to send heartbeat
                    frame = await asyncio.wait_for(
                        self.transmit_queue.get(), 
                        timeout=self.heartbeat_interval
                    )
                except asyncio.TimeoutError:
                    # Idle timeout (250ms passed without data) -> Send Heartbeat
                    frame = self._create_heartbeat()

                # 2. Throttling Phase
                # Even heartbeats respect the min_interval if they flood too fast,
                # but since min_interval (150ms) < heartbeat (250ms), usually no sleep needed for HB.
                now = time.time()
                elapsed = now - last_send_time
                
                if elapsed < self.min_interval:
                    sleep_time = self.min_interval - elapsed
                    await asyncio.sleep(sleep_time)
                    # Recalculate now after sleep
                    now = time.time()
                    elapsed = now - last_send_time

                # 3. Transmission Phase
                yield frame
                
                # 4. Audit Phase (Only log real frames to avoid spamming logs with HB)
                if frame.sequence_id != 0:
                    self.freq_logger.info(f"Sent Delta: {elapsed*1000:.1f}ms | Seq: {frame.sequence_id}")
                
                last_send_time = now

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[HFT-Streamer] Gen Error: {e}")
                await asyncio.sleep(0.5)

    def _create_heartbeat(self) -> synapse_hft_pb2.TrafficFrame:
        """Creates an empty frame to keep connection alive."""
        frame = synapse_hft_pb2.TrafficFrame()
        frame.timestamp = time.time()
        # Sequence ID 0 is the universal signal for "Heartbeat/Keep-Alive"
        frame.sequence_id = 0 
        return frame