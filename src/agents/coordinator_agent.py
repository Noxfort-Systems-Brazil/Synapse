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
# File: src/agents/coordinator_agent.py
# Author: Gabriel Moraes
# Date: 2026-02-23

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Any, Tuple, Optional, Dict
from torch.amp import autocast, GradScaler

# Import Base Agent
from src.agents.base_agent import BaseAgent

# --- PyTorch Geometric Imports (Safety Check) ---
try:
    from torch_geometric.data import Data, Batch
    # IMPORT CORRECTION: Importing 'SpatialGAT' as defined in existing codebase
    from src.models.gatv2_lite import SpatialGAT
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    Data = None
    Batch = None
    SpatialGAT = None

class CoordinatorAgent(BaseAgent):
    """
    The Coordinator Agent ('O Estrategista').
    
    Responsibilities:
    1. Spatial Reasoning: Understands traffic flow across the graph topology.
    2. Global Embedding: Produces a latent vector representing the entire network state.
    
    Refactored V6 (Topology Context Memory):
    - Uses 'SpatialGAT' class name correctly.
    - Includes PyTorch Geometric availability check.
    - Caches the edge_index (topology) in memory so the agent retains context.
    - Implements 'forward_pass' for CycleProcessor compatibility.
    """

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, heads: int = 2, dropout: float = 0.1, learning_rate: float = 0.001):
        """
        Initializes the GATv2 model (SpatialGAT).
        """
        if not PYG_AVAILABLE:
            raise ImportError("CoordinatorAgent requires 'torch_geometric' installed.")

        if SpatialGAT is None:
             raise ImportError("Could not import 'SpatialGAT' from 'src.models.gatv2_lite'. Check file structure.")

        # 1. Initialize Model
        model = SpatialGAT(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            heads=heads,
            dropout=dropout
        )
        
        # 2. Base Init
        super().__init__(model=model, name="CoordinatorAgent")
        
        # 3. Memory Context (Graph Topology)
        self.cached_edge_index: Optional[torch.Tensor] = None
        
        # 4. Optimization
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scaler = GradScaler(device='cuda' if torch.cuda.is_available() else 'cpu')
        self.criterion = nn.MSELoss()

    def set_topology(self, edge_index: torch.Tensor):
        """
        Injects the graph topology into the agent's memory.
        This allows the agent to reason about the network without needing the raw map file.
        """
        self.cached_edge_index = edge_index.to(self.device)
        print(f"[{self.name}] Topology context injected into memory successfully.")

    def inference(self, input_data: Dict[str, Any]) -> Any:
        """
        Executes GAT inference bridging the base signature to the forward pass.
        Expects a dict with: 'x_spatial' and 'edge_index'.
        """
        x = input_data.get("x_spatial")
        edge_index = input_data.get("edge_index")
        
        if x is None:
             raise ValueError("Coordinator inference requires 'x_spatial'.")
             
        try:
             return self.forward_pass(x, edge_index)
        except Exception as e:
             import logging
             logging.error(f"[Coordinator] Inference failed: {e}")
             return torch.zeros((1, 1))

    def forward_pass(self, x: torch.Tensor, edge_index: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Specific Interface for Graph Processing.
        Called by CycleProcessor.
        
        Args:
            x: Node features [Num_Nodes, Features]
            edge_index: Graph connectivity [2, Num_Edges]. Uses cached if None.
        """
        self.model.eval()
        
        # Ensure inputs are on the correct device
        device = self.device
        x = x.to(device)
        
        # Resolve topology context
        current_edge_index = edge_index if edge_index is not None else self.cached_edge_index
        
        if current_edge_index is None:
            raise ValueError(f"[{self.name}] Missing topology context. Provide edge_index or call set_topology().")
            
        current_edge_index = current_edge_index.to(device)
        
        with torch.no_grad():
            # Use Mixed Precision if available
            with autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                # GATv2 Forward
                out = self.model(x, current_edge_index)
                
                # Handle potential tuple return (embedding, attention_weights)
                if isinstance(out, tuple):
                    return out[0]
                return out

    def train_step(self, batch_data: Any) -> float:
        """
        Training step for Graph Data.
        Expects batch_data to be a Geometric Data object or tuple.
        """
        self.model.train()
        
        # Unpack Data
        if hasattr(batch_data, 'x') and hasattr(batch_data, 'edge_index'):
            x = batch_data.x.to(self.device)
            edge_index = batch_data.edge_index.to(self.device)
            y = batch_data.y.to(self.device) if hasattr(batch_data, 'y') else x # Self-supervised fallback
        elif isinstance(batch_data, (tuple, list)):
            x, edge_index, y = batch_data
            x, edge_index, y = x.to(self.device), edge_index.to(self.device), y.to(self.device)
        else:
            return 0.0

        self.optimizer.zero_grad()

        with autocast(device_type=self.device.type, enabled=(self.device.type == 'cuda')):
            out = self.model(x, edge_index)
            
            # Handle tuple output in training too
            if isinstance(out, tuple):
                out = out[0]
                
            loss = self.criterion(out, y)

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return loss.item()  