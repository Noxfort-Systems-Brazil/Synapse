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
# File: src/engine/snapshot_builder.py
# Author: Gabriel Moraes
# Date: 2026-02-14

import numpy as np
import logging
from typing import Dict, Any

# --- Domain Imports ---
from src.domain.app_state import AppState
from src.managers.graph_manager import GraphManager

class SnapshotBuilder:
    """
    Responsible for constructing the Global Network Snapshot.
    
    Refactored V4 (Architecture Fix — No Duplicate TCN):
    - Reads `traffic_node.last_embedding` instead of running TCN again.
    - TrafficNode.step() / ghost_step() already compute the TCN embedding.
    - This class only COLLECTS the cached results into a snapshot dict.
    """

    def __init__(self, app_state: AppState, graph_manager: GraphManager, embedding_dim: int = 32):
        self.logger = logging.getLogger(__name__)
        self.app_state = app_state
        self.graph_manager = graph_manager
        self.embedding_dim = embedding_dim
        
    def gather_snapshot(self) -> Dict[str, Any]:
        """
        Iterates over all nodes to build a complete state snapshot for the cycle.
        
        Architecture Fix: Uses the CACHED embedding from TrafficNode
        (computed by step() or ghost_step()) instead of re-running TCN.
        
        Returns:
            Dict mapping NodeID -> {embedding, value, timestamp}
        """
        snapshot = {}
        
        # Get static nodes from AppState (The Source of Truth for Topology)
        map_nodes = self.app_state.get_all_nodes()
        
        for map_node in map_nodes:
            # Retrieve the RUNTIME TrafficNode from GraphManager using ID
            traffic_node = self.graph_manager.get_node(map_node.id)
            
            # Prepare default payload (Zero vector)
            payload = {
                "embedding": np.zeros(self.embedding_dim),
                "value": 0.0,
                "timestamp": 0.0
            }
            
            # If GraphManager hasn't initialized this node yet, use the default
            if not traffic_node:
                snapshot[map_node.id] = payload
                continue

            try:
                # Architecture Fix: Read CACHED embedding (already computed by step/ghost_step)
                payload["embedding"] = traffic_node.last_embedding
                payload["value"] = traffic_node.last_value
                payload["timestamp"] = traffic_node.last_timestamp
                
                # Fetch recent Physics (Kalman Filter State) for Dead Reckoning extrapolation
                kse_snap = traffic_node.kse.get_kinetic_snapshot()
                payload["physics"] = {
                    "v": kse_snap.v,
                    "a": kse_snap.a
                }
                
                snapshot[traffic_node.source_id] = payload
                
            except Exception as e:
                snapshot[traffic_node.source_id] = payload
                
        return snapshot