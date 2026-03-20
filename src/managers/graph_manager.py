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
# File: src/managers/graph_manager.py
# Author: Gabriel Moraes
# Date: 2026-01-09

import torch
from typing import List, Tuple, Dict, Optional, Any

# --- Domain & Entities ---
from src.domain.app_state import AppState
from src.domain.entities import MapNode, MapEdge

# --- Components for TrafficNode Injection ---
from src.engine.traffic_node import TrafficNode
from src.memory.spatial_memory import SpatialMemory
from src.memory.temporal_memory import TemporalMemory
from src.agents.specialist_agent import SpecialistAgent
from src.services.historical_manager import HistoricalManager
from src.kse.filter import RobustKalmanFilter
from src.kse.definitions import PROFILES

class GraphManager:
    """
    Manages the Graph Topology and Traffic Node State.
    
    Refactored V5 (Added get_ordered_node_ids):
    - Explicitly stores and exposes the node order to ensure Tensor alignment.
    - Instantiates HistoricalManager using AppState.
    """

    def __init__(self, app_state: AppState, embedding_dim: int = 64, device: torch.device = None):
        """
        Args:
            app_state: Reference to the central application state.
            embedding_dim: Size of the feature vector per node (default: 64).
            device: Torch device (CPU/CUDA). Auto-detects if None.
        """
        self.app_state = app_state
        self.embedding_dim = embedding_dim
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Internal State
        self.nodes: Dict[str, TrafficNode] = {}
        self.spatial_memory: Optional[SpatialMemory] = None
        self.edge_index: Optional[torch.Tensor] = None
        self.ordered_node_ids: List[str] = [] # Stores the strict order of nodes for tensors
        
        # Shared Service for all nodes
        try:
            self.historical_manager = HistoricalManager(app_state)
        except TypeError:
            self.historical_manager = HistoricalManager()

        # Initialize Graph structures immediately
        self.rebuild_graph()

    def rebuild_graph(self):
        """
        Syncs internal graph structures with AppState.
        Instantiates complex TrafficNode objects for every map node.
        """
        map_nodes = self.app_state.get_all_nodes()
        map_edges = self.app_state.get_all_edges()
        
        # 0. Define and Store Order (CRITICAL for GNN Alignment)
        self.ordered_node_ids = [n.id for n in map_nodes]
        
        # 1. Instantiate TrafficNodes (Factory Logic)
        self.nodes = {}
        for mn in map_nodes:
            # A. Create Dependencies
            # Each node gets its own Memory Buffer (Seq Length 60)
            mem = TemporalMemory(feature_dim=1, max_len=60)
            
            # Each node gets its own Local Specialist Brain (TCN)
            # Input dim 1 (flow), Output dim 32 (latent embedding)
            agent = SpecialistAgent(input_dim=1, output_dim=32, num_channels=[16, 32])
            agent.to(self.device)
            
            # C. Inject Physics Engine (DIP)
            profile = PROFILES["DEFAULT"]
            if "cam" in mn.id.lower(): profile = PROFILES["CAMERA"]
            elif "loop" in mn.id.lower(): profile = PROFILES["INDUCTIVE"]
            kse = RobustKalmanFilter(node_id=mn.id, initial_val=0.0, profile=profile)
            
            # D. Assemble TrafficNode (Injection)
            t_node = TrafficNode(
                source_id=mn.id,
                memory=mem,
                agent=agent,
                historical_manager=self.historical_manager, # Injected Shared Service
                physics_engine=kse,
                graph_manager=self
            )
            
            self.nodes[mn.id] = t_node
            
        # 2. Build Spatial Memory (Static Feature Matrix Wrapper for GNN)
        # We use the explicitly stored order
        self.spatial_memory = SpatialMemory(self.ordered_node_ids, feature_dim=self.embedding_dim)
        
        # 3. Build Connectivity (Edge Index)
        self.edge_index = self._build_edge_index(map_nodes, map_edges)
        
        print(f"[GraphManager] Graph Rebuilt: {len(self.nodes)} TrafficNodes Active.")

    def get_node(self, node_id: str) -> Optional[TrafficNode]:
        """Returns the active TrafficNode object (for XAI/TCN)."""
        return self.nodes.get(node_id)
        
    def get_ordered_node_ids(self) -> List[str]:
        """
        Returns the list of Node IDs in the exact order used to construct tensors.
        Used by CycleProcessor to align inputs.
        """
        return self.ordered_node_ids

    def update_node_memory(self, source_id: str, payload: Any):
        """
        Injects sensor data into the corresponding Node's memory.
        """
        # Resolving Association: Sensor -> Node
        node_id = self.app_state.get_element_for_source(source_id)
        if not node_id:
            node_id = source_id
            
        if node_id in self.nodes:
            # Parse payload
            val = 0.0
            try:
                if isinstance(payload, (int, float)):
                    val = float(payload)
                elif isinstance(payload, list) and len(payload) > 0:
                    val = float(payload[0])
            except: val = 0.0
            
            # 1. Execute Node Step (Includes Physics KSE + AI Memory)
            self.nodes[node_id].step(val)
            
            # 2. Update Spatial Memory (for GATv2 Global View)
            current_feats = [0.0] * self.embedding_dim
            current_feats[0] = val
            self.spatial_memory.update_node(node_id, current_feats)

    def get_graph_snapshot(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the current state of the graph for GAT inference.
        Returns: (x, edge_index)
        """
        if not self.spatial_memory:
            return torch.empty(0), torch.empty(0)
            
        x = self.spatial_memory.get_node_features_tensor().to(self.device)
        return x, self.edge_index

    def _build_edge_index(self, nodes: List[MapNode], edges: List[MapEdge]) -> torch.Tensor:
        """
        Constructs the Edge Index tensor required by PyTorch Geometric.
        """
        edge_list = []
        # Crucial: ID Map must follow the same order as ordered_node_ids
        # Since self.ordered_node_ids comes from 'nodes' list passed here (in rebuild_graph), we are safe.
        id_map = {n.id: i for i, n in enumerate(nodes)}
        
        if not edges:
            return torch.empty((2, 0), dtype=torch.long).to(self.device)
        else:
            for e in edges:
                if e.from_node in id_map and e.to_node in id_map:
                    src_idx = id_map[e.from_node]
                    dst_idx = id_map[e.to_node]
                    edge_list.append([src_idx, dst_idx])
        
        if not edge_list:
             return torch.empty((2, 0), dtype=torch.long).to(self.device)
            
        tensor = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        return tensor.to(self.device)