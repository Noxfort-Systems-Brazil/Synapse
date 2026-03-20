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
# File: src/models/gatv2_lite.py
# Author: Gabriel Moraes
# Date: 2025-11-22

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

class SpatialGAT(nn.Module):
    """
    Spatial Graph Attention Network V2 (Lite Version).
    
    Operates at Level 2 (Regional Perception).
    It treats the city as a graph where:
    - Nodes: Intersections/Junctions.
    - Edges: Roads connecting them.
    
    This 'Lite' version uses GATv2Conv to learn dynamic weights (attention) 
    efficiently with a shallow architecture suitable for real-time inference.
    """

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, heads: int = 4, dropout: float = 0.6):
        """
        Initializes the GATv2 architecture.

        Args:
            in_channels: Size of input features per node (e.g., embedding from TCN).
            hidden_channels: Size of hidden node embeddings.
            out_channels: Size of the final output vector per node.
            heads: Number of multi-head attentions. 
                   (Allows the model to focus on different aspects of neighbors simultaneously).
            dropout: Dropout probability for regularization.
        """
        super(SpatialGAT, self).__init__()
        
        self.dropout_rate = dropout

        # --- Layer 1: Input -> Hidden ---
        # GATv2Conv automatically handles the message passing logic.
        # We use multi-head attention to capture diverse spatial relationships.
        # concat=True means output dimension will be hidden_channels * heads
        self.conv1 = GATv2Conv(
            in_channels, 
            hidden_channels, 
            heads=heads, 
            dropout=dropout,
            concat=True 
        )

        # --- Layer 2: Hidden -> Output ---
        # We project the concatenated heads down to the desired output dimension.
        # For the final layer, we usually set concat=False to average the heads or output a single vector.
        # Input dim is (hidden_channels * heads) from the previous layer.
        self.conv2 = GATv2Conv(
            hidden_channels * heads, 
            out_channels, 
            heads=1, 
            concat=False, 
            dropout=dropout
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass logic.

        Args:
            x: Node feature matrix [Num_Nodes, In_Channels].
            edge_index: Graph connectivity in COO format [2, Num_Edges].
            
        Returns:
            New node embeddings [Num_Nodes, Out_Channels].
        """
        # 1. First Graph Attention Layer
        # Dropout on input features is common in GATs to prevent overfitting on specific nodes
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x) # ELU (Exponential Linear Unit) is standard for GATs

        # 2. Second Graph Attention Layer (Output)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv2(x, edge_index)
        
        # We don't apply softmax here because this output might be used for regression 
        # (predicting flow values) or fed into the Global Fusion engine.
        return x