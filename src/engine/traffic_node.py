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
# File: src/engine/traffic_node.py
# Author: Gabriel Moraes
# Date: 2026-01-09

import torch
import numpy as np
import time
from typing import Optional, Dict, Any, List

# Import Domain
from src.domain.entities import SourceStatus

# Import Memory & AI
from src.memory.temporal_memory import TemporalMemory
from src.agents.specialist_agent import SpecialistAgent

# Import Managers for Fallback Logic
from src.services.historical_manager import HistoricalManager

# Import Physics Engine (New SOLID Architecture)
from src.kse.filter import RobustKalmanFilter
from src.kse.definitions import PROFILES, KineticState

class TrafficNode:
    """
    Represents a single traffic sensing node (e.g., an intersection or camera).
    
    Refactored V8 (SOLID Architecture):
    - [DIP] Receives `physics_engine` initialized via constructor (No longer creates KEF or hardcodes KEF Profiles).
    - [SRP] Delegates PyTorch state parsing and Numpy Transposition completely to the `SpecialistAgent`.
    """

    def __init__(
        self, 
        source_id: str, 
        memory: TemporalMemory, 
        agent: SpecialistAgent,
        historical_manager: HistoricalManager,
        physics_engine: Any,
        graph_manager: Optional[Any] = None
    ):
        self.source_id = source_id
        self.memory = memory
        self.agent = agent
        self.historical_manager = historical_manager
        self.graph_manager = graph_manager
        
        # --- Physics Engine (Digital Twin) ---
        # [DIP Fix] Receives an active, already-profiled Kalmar Filter
        self.kse = physics_engine
        
        self.last_value = 0.0
        self.last_timestamp = time.time()
        
        # --- Embedding Cache (Architecture Fix) ---
        # Stores the last TCN output so SnapshotBuilder doesn't re-run TCN
        output_dim = getattr(agent, 'output_dim', 32)
        self.last_embedding = np.zeros(output_dim)
        
        # --- Cycle Tracking ---
        # Tracks whether step() was called this cycle (sensor sent data)
        # If not, tick() will call ghost_step() to keep KSE alive
        self._updated_this_cycle = False
        
        self.steps_processed = 0
        self.fallback_steps = 0

    @property
    def is_ready(self) -> bool:
        """Returns True if the node has enough data to start inference."""
        return self.memory.is_ready()

    def step(self, value: float) -> Dict[str, Any]:
        """
        Processes a REAL data point (Ground Truth from Sensor).
        Updates both AI Memory and Kinetic Physics Model.
        """
        # 1. Rollback Correction (If recovering from silence)
        if self.fallback_steps > 0:
            print(f"[TrafficNode] 🏥 REANIMATION: Sensor '{self.source_id}' recovered. Rolling back {self.fallback_steps} synthetic steps.")
            self.memory.rollback(self.fallback_steps)
            self.fallback_steps = 0

        # 2. Physics Update (KSE Correction)
        now = time.time()
        dt = now - self.last_timestamp
        
        # Predict State at current time (Dead Reckoning)
        if dt > 10.0: dt = 0.1 
        self.kse.predict(dt)
        
        # Fuse Sensor Data into Kalman Filter (Correction)
        # This aligns the physics model with the real world
        self.kse.update(measurement=value)
        
        # Update local tracking
        self.last_value = value
        self.last_timestamp = now

        # 3. Normal Ingest (AI Memory)
        self.memory.push([value])
        self.steps_processed += 1
        
        if not self.memory.is_ready():
            return {
                "source_id": self.source_id,
                "status": "Warming up",
                "buffer_size": len(self.memory.buffer),
                "ready": False
            }

        # 4. Online Learning (Adaptation)
        current_window_tensor = self.memory.get_tensor()
        loss = self.agent.train(current_window_tensor, current_window_tensor)

        # 5. Inference (SRP Fix: Delegate tensor shapes to Agent)
        history_np = self.memory.get_numpy()
        current_embedding = self.agent.predict(history_np)
        
        # Cache embedding for SnapshotBuilder
        if current_embedding is not None:
            self.last_embedding = current_embedding
        
        # Mark as updated this cycle
        self._updated_this_cycle = True

        # Get Physics State for Reporting
        kse_snapshot = self.kse.get_kinetic_snapshot()
        physics_report = {
            "position": kse_snapshot.p,
            "velocity": kse_snapshot.v,
            "acceleration": kse_snapshot.a,
            "confidence": kse_snapshot.confidence
        }

        return {
            "source_id": self.source_id,
            "status": "Active",
            "type": "real",
            "ready": True,
            "loss": loss,
            "embedding": current_embedding,
            "value": value,
            "physics": physics_report,
            "steps": self.steps_processed
        }

    def ghost_step(self) -> Dict[str, Any]:
        """
        Executes the SWITCHING LOGIC (No Sensor Available).
        
        Logic:
        1. Check Historical DB for EXACT match (Truth).
           - IF FOUND: Use it, update KSE (Correct), and send.
        2. IF NOT FOUND (Hole): 
           - Use KSE Physics (Predict/Dead Reckoning) and send.
        """
        if not self.memory.is_ready():
             return {"source_id": self.source_id, "ready": False, "status": "Waiting for History"}

        now = time.time()
        dt = now - self.last_timestamp
        self.last_timestamp = now 

        # --- PHASE 1: CHECK GOLDEN DATABASE (The Truth) ---
        # Tolerance: 0.25s (250ms) to find a record
        historical_val = self.historical_manager.get_exact_reading(self.source_id, now, tolerance=0.25)
        
        if historical_val is not None:
            # HIT! We have a recorded truth for this moment.
            # We treat this almost like a real sensor reading.
            
            # 1. Correct the Physics Model (Snap to Truth)
            # We predict first to advance time, then update with historical value
            self.kse.predict(dt)
            self.kse.update(historical_val)
            
            final_val = historical_val
            method_used = "Historical (DB Match)"
            
        else:
            # MISS! We are in a data hole.
            # --- PHASE 2: KSE PHYSICS (The Filler) ---
            
            # Advance physics purely by momentum (Dead Reckoning)
            self.kse.predict(dt)
            
            # We trust the physics projection
            kse_snapshot = self.kse.get_kinetic_snapshot()
            final_val = kse_snapshot.p
            method_used = "KSE (Physics Projection)"

        # Final Safety: Non-negative
        final_val = max(0.0, final_val)
        
        # Log occasionally
        if self.fallback_steps % 10 == 0:
            print(f"[TrafficNode] 🛡️ FALLBACK: {method_used} | Val: {final_val:.2f}")

        # Update Memory (Synthetic Data)
        self.memory.push([final_val])
        self.fallback_steps += 1
        
        # Inference on Synthetic Data (Keep Brain Alive)
        # (SRP Fix: Delegate tensor shapes to Agent)
        history_np = self.memory.get_numpy()
        current_embedding = self.agent.predict(history_np)
        
        # Cache embedding for SnapshotBuilder
        if current_embedding is not None:
            self.last_embedding = current_embedding
        
        kse_snapshot = self.kse.get_kinetic_snapshot()
        physics_report = {
            "position": kse_snapshot.p,
            "velocity": kse_snapshot.v,
            "acceleration": kse_snapshot.a,
            "confidence": kse_snapshot.confidence
        }

        return {
            "source_id": self.source_id,
            "status": f"Fallback: {method_used}",
            "type": "synthetic",
            "ready": True,
            "loss": 0.0,
            "embedding": current_embedding,
            "value": final_val,
            "physics": physics_report,
            "fallback_count": self.fallback_steps
        }

    def tick(self):
        """
        Called once per global cycle. If no sensor data arrived this cycle,
        triggers ghost_step() to keep KSE alive (dead reckoning) and TCN updated.
        Resets the cycle flag after execution.
        """
        if not self._updated_this_cycle:
            self.ghost_step()
        self._updated_this_cycle = False

    # --- Persistence Methods (Unchanged) ---

    def get_state(self) -> Dict[str, Any]:
        """Exports the full internal state (Brain + Memory + Physics)."""
        # [SRP Fix] We no longer access internal PyTorch subcomponents
        return {
            "source_id": self.source_id,
            "steps_processed": self.steps_processed,
            "memory_buffer": [x.tolist() for x in self.memory.buffer],
            "agent_state": self.agent.get_state(),
            "kse_state": {
                "x": self.kse.x.tolist(),
                "P": self.kse.P.tolist(),
                "last_time": self.kse.last_update_time
            }
        }

    def set_state(self, state: Dict[str, Any]):
        """Restores the state from a checkpoint."""
        try:
            self.steps_processed = state.get("steps_processed", 0)
            
            raw_buffer = state.get("memory_buffer", [])
            self.memory.clear()
            for item in raw_buffer:
                self.memory.push(item) 
            
            # [SRP Fix] Pass dict to Agent
            agent_data = state.get("agent_state", {})
            self.agent.set_state(agent_data)

            kse_data = state.get("kse_state", None)
            if kse_data:
                self.kse.x = np.array(kse_data["x"])
                self.kse.P = np.array(kse_data["P"])
                self.kse.last_update_time = kse_data.get("last_time", time.time())
                self.last_value = float(self.kse.x[0,0])
                
        except Exception as e:
            print(f"[TrafficNode] Failed to restore state for {self.source_id}: {e}")