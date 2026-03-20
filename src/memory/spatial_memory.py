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
# File: src/memory/spatial_memory.py
# Author: Gabriel Moraes
# Date: 2025-11-22

import torch
import numpy as np
from typing import Dict, List, Tuple

class SpatialMemory:
    """
    Manages the spatial state of the network (Snapshot Buffer).
    
    Used by:
    - Coordinator Agent (GATv2 input).
    
    Responsibility:
    - Stores the latest known feature vector for every node (intersection) in the graph.
    - Assembles the 'Node Feature Matrix' (X) required by Graph Neural Networks.
    """

    def __init__(self, node_ids: List[str], feature_dim: int):
        """
        Args:
            node_ids: List of all intersection IDs in the managed region.
            feature_dim: Number of features per node.
        """
        self.node_ids = node_ids
        self.feature_dim = feature_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Map node_id -> index (for matrix construction)
        self.id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
        
        # Storage: [Num_Nodes, Features]
        # Initialize with zeros
        self.state_matrix = np.zeros((len(node_ids), feature_dim), dtype=np.float32)

    def update_node(self, node_id: str, features: List[float]):
        """
        Updates the state of a single node (e.g., when a sensor sends data).
        """
        if node_id in self.id_to_idx:
            idx = self.id_to_idx[node_id]
            self.state_matrix[idx] = np.array(features, dtype=np.float32)

    def get_node_features_tensor(self) -> torch.Tensor:
        """
        Returns the Node Feature Matrix (X) for the GAT.
        
        Returns:
            Tensor of shape [Num_Nodes, Features].
        """
        return torch.FloatTensor(self.state_matrix).to(self.device)

    def get_missing_nodes(self) -> List[str]:
        """
        Helper to identify nodes that haven't reported data yet (still zeros).
        (Heuristic: checks if row is all zeros).
        """
        missing = []
        for nid, idx in self.id_to_idx.items():
            if np.all(self.state_matrix[idx] == 0):
                missing.append(nid)
        return missing