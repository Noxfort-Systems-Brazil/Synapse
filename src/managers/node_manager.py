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
# File: src/managers/node_manager.py
# Author: Gabriel Moraes
# Date: 2025-12-03

import torch
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

# Import Domain
from src.engine.traffic_node import TrafficNode
from src.memory.temporal_memory import TemporalMemory
from src.factories.agent_factory import AgentFactory
from src.services.historical_manager import HistoricalManager
from src.managers.graph_manager import GraphManager
from src.managers.storage_manager import StorageManager
from src.kse.filter import RobustKalmanFilter
from src.kse.definitions import PROFILES

class NodeManager:
    """
    Manages the lifecycle and state of Traffic Nodes (Sensors).
    
    Refactored V3 (Fallback Support):
    - Added 'trigger_fallback()' to handle sensor errors reported by the Pipeline.
    - This delegates to the node's 'ghost_step()' method (MEH/Historical data).
    """

    def __init__(self, config: dict, device: torch.device, 
                 historical_manager: HistoricalManager, 
                 graph_manager: Optional[GraphManager] = None):
        
        self.config = config
        self.device = device
        self.historical_manager = historical_manager
        self.graph_manager = graph_manager
        
        self.storage = StorageManager()
        self._nodes: Dict[str, TrafficNode] = {}
        
        # Probation State: {source_id: start_timestamp}
        self.probation_nodes: Dict[str, datetime] = {}
        self.probation_duration = 60.0 # 1 minute test duration
        
        self.seq_len = 60
        self.feature_dim = 1
        self.embedding_dim = 32

    def register_node(self, source_id: str) -> bool:
        """
        Creates a new node. Tries to restore previous state (brain) if available.
        """
        if source_id in self._nodes:
            return False
            
        memory = TemporalMemory(self.seq_len, self.feature_dim)
        agent = AgentFactory.create_specialist(
            config=self.config,
            input_dim=self.feature_dim,
            output_dim=self.embedding_dim
        )
        agent.to(self.device)
        
        # Determine Sensor Profile based on ID naming convention (Moved here for DIP)
        profile = PROFILES["DEFAULT"]
        if "cam" in source_id.lower(): profile = PROFILES["CAMERA"]
        elif "loop" in source_id.lower(): profile = PROFILES["INDUCTIVE"]
        kse = RobustKalmanFilter(node_id=source_id, initial_val=0.0, profile=profile)

        node = TrafficNode(
            source_id=source_id,
            memory=memory,
            agent=agent,
            historical_manager=self.historical_manager,
            physics_engine=kse,
            graph_manager=self.graph_manager
        )
        
        # Restore State (Hibernation Wake-up)
        if self._load_node_checkpoint(node):
            print(f"[NodeManager] 🕯️ Node '{source_id}' restored. Entering Probation.")
            self.probation_nodes[source_id] = datetime.now()
        
        self._nodes[source_id] = node
        return True

    def remove_node(self, source_id: str):
        if source_id in self._nodes:
            del self._nodes[source_id]
            if source_id in self.probation_nodes:
                del self.probation_nodes[source_id]
            print(f"[NodeManager] Removed Node: {source_id}")

    def update_node(self, source_id: str, value: float) -> Optional[Dict[str, Any]]:
        """
        Orchestrates the NORMAL data step for a node.
        """
        if source_id not in self._nodes:
            return None
            
        self._check_probation(source_id)
            
        node = self._nodes[source_id]
        result = node.step(value)
        
        # Inject Status Override for UI
        if source_id in self.probation_nodes:
            elapsed = (datetime.now() - self.probation_nodes[source_id]).total_seconds()
            remaining = int(self.probation_duration - elapsed)
            result["status"] = f"Probation ({remaining}s)"
            
        return result

    def trigger_fallback(self, source_id: str, error_msg: str) -> Optional[Dict[str, Any]]:
        """
        Orchestrates the FALLBACK step (Ghost Step) when a sensor fails.
        Uses MEH (Historical Data) to keep the TCN/Transformer running.
        """
        if source_id not in self._nodes:
            return None
            
        print(f"[NodeManager] 🚑 Triggering Fallback for {source_id}: {error_msg}")
        node = self._nodes[source_id]
        
        # Ghost Step uses historical data
        result = node.ghost_step()
        return result

    def save_all_nodes(self):
        """Persists the state of all active nodes to disk."""
        if not self._nodes: return
        
        print("[NodeManager] 💾 Hibernating: Saving state for all nodes...")
        ckpt_dir = Path(self.storage.get_checkpoints_path())
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        
        count = 0
        for nid, node in self._nodes.items():
            try:
                state = node.get_state()
                filename = f"state_{nid}.pth"
                path = ckpt_dir / filename
                torch.save(state, str(path))
                count += 1
            except Exception as e:
                print(f"[NodeManager] Failed to save {nid}: {e}")
                
        print(f"[NodeManager] Saved {count} nodes to {ckpt_dir}.")

    # --- Internal Helpers ---

    def _load_node_checkpoint(self, node: TrafficNode) -> bool:
        filename = f"state_{node.source_id}.pth"
        path = Path(self.storage.get_checkpoints_path()) / filename
        
        if path.exists():
            try:
                state = torch.load(str(path))
                node.set_state(state)
                return True
            except Exception as e:
                print(f"[NodeManager] Corrupted checkpoint for {node.source_id}: {e}")
        return False

    def _check_probation(self, source_id: str):
        if source_id in self.probation_nodes:
            start_time = self.probation_nodes[source_id]
            elapsed = (datetime.now() - start_time).total_seconds()
            
            if elapsed > self.probation_duration:
                print(f"[NodeManager] 🎉 Node '{source_id}' passed probation!")
                del self.probation_nodes[source_id]

    # --- Accessors ---

    def get_node(self, source_id: str) -> Optional[TrafficNode]:
        return self._nodes.get(source_id)

    def get_all_nodes(self) -> Dict[str, TrafficNode]:
        return self._nodes

    def get_ready_nodes_ids(self) -> List[str]:
        return sorted([nid for nid, node in self._nodes.items() if node.is_ready])

    def get_agents_for_pbt(self) -> Dict[str, Any]:
        return {nid: node.agent for nid, node in self._nodes.items()}