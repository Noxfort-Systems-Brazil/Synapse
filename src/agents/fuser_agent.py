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
# File: src/agents/fuser_agent.py
# Author: Gabriel Moraes
# Date: 2026-03-02

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional, Dict, Any
from torch.amp import autocast, GradScaler

# Import Base & Model
from src.agents.base_agent import BaseAgent
from src.models.itransformer import iTransformer

class FuserAgent(BaseAgent):
    """
    The Fuser Agent ('O Sintetizador' - Formerly 'O Vidente').
    
    Refactored V4 (Present State Estimator):
    - ROLE CHANGE: No longer predicts the future. Now fuses historical noise
      to generate a "Perfect Snapshot" of the PRESENT state [t].
    - Configures iTransformer with pred_len=1 (Current Step Fusion).
    - Maintains Dynamic Device Detection for stability.
    """

    def __init__(self, num_variates: int, seq_len: int, pred_len: int = 1, d_model: int = 512, n_heads: int = 8, layers: int = 2, learning_rate: float = 0.0001):
        """
        Initializes the agent.
        
        Args:
            pred_len: Defaults to 1. This implies we are reconstructing the 
                      current single time step, not forecasting a sequence.
        """
        # Enforce pred_len=1 logic if the user tries to pass a forecast horizon
        # (Unless explicitly training for next-step prediction as a proxy for current state)
        effective_pred_len = 1 
        
        # 1. Create Model
        model = iTransformer(
            num_variates=num_variates,
            seq_len=seq_len,
            pred_len=effective_pred_len,
            d_model=d_model,
            n_heads=n_heads,
            e_layers=layers
        )
        
        # 2. Initialize Base
        super().__init__(model=model, name="FuserAgent")

        # 3. Setup Optimizer & Scaler
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Initialize Scaler safely
        self.scaler = GradScaler('cuda' if torch.cuda.is_available() else 'cpu')
        self.criterion = nn.MSELoss()
        
        # --- Semantic Registry ---
        self.source_registry: Dict[str, Dict[str, str]] = {}

    def _get_current_device(self) -> torch.device:
        """
        Introspects the model to find its actual physical location.
        """
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def inference(self, input_data: Dict[str, Any]) -> Any:
        """
        Predicts future states from sequence history using iTransformer.
        Expects a dict containing 'x_temporal' and optionally 'spatial_context'.
        """
        if "x_temporal" not in input_data:
             raise ValueError("Fuser inference requires 'x_temporal'.")
             
        x_temporal = input_data["x_temporal"]
        spatial_context = input_data.get("spatial_context", None)
        
        # Compatibility cast for predict_state
        if isinstance(x_temporal, torch.Tensor):
             x_temporal = x_temporal.cpu().numpy()
             
        return self.fuse_state(x_temporal, spatial_context=spatial_context)

    def fuse_state(self, current_history: np.ndarray, spatial_context: torch.Tensor = None) -> np.ndarray:
        """
        Generates the 'Perfect Present State' by fusing temporal history
        with spatial topology context from the Coordinator.
        
        Args:
            current_history: Noisy history [Seq_Len, Variates] or [Variates, Seq_Len]
            spatial_context: Optional spatial embeddings from Coordinator
                           [Nodes, SpatialDim] or [Batch, Nodes, SpatialDim]
            
        Returns:
            Refined Vector [Variates] representing the state at time T.
        """
        self.model.eval()
        device = self._get_current_device() # Dynamic Check

        with torch.no_grad():
            # Handle conversion
            if isinstance(current_history, np.ndarray):
                tensor_x = torch.FloatTensor(current_history).to(device)
            elif isinstance(current_history, torch.Tensor):
                tensor_x = current_history.to(device)
            else:
                tensor_x = torch.tensor(current_history, dtype=torch.float).to(device)
            
            # Ensure batch dimension [1, Seq, Variates]
            if tensor_x.dim() == 2:
                tensor_x = tensor_x.unsqueeze(0)
            
            # Move spatial context to device if provided
            if spatial_context is not None:
                spatial_context = spatial_context.to(device)
            
            device_type = device.type if device.type != 'mps' else 'cpu'
            
            with autocast(device_type=device_type, enabled=(device.type == 'cuda')):
                # iTransformer Forward Pass (with optional spatial cross-attention)
                # Output shape: [Batch, Pred_Len, Variates] -> [1, 1, Variates]
                refined_state = self.model(tensor_x, spatial_context=spatial_context)
            
            # Flatten to [Variates]
            return refined_state.cpu().float().numpy().flatten()

    # Alias for backward compatibility with CycleProcessor
    def predict_state(self, current_history: np.ndarray, spatial_context: torch.Tensor = None) -> np.ndarray:
        return self.fuse_state(current_history, spatial_context=spatial_context)

    def train_step(self, batch_data: Any) -> float:
        """
        Standard Training Step.
        Enforces strict mathematical shape alignment between inputs and targets.
        """
        self.model.train()
        device = self._get_current_device()
        
        inputs, targets = batch_data
        
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        self.optimizer.zero_grad()
        
        device_type = device.type if device.type != 'mps' else 'cpu'
        
        with autocast(device_type=device_type, enabled=(device.type == 'cuda')):
            outputs = self.model(inputs)
            
            # Dimensionality strict lock to prevent PyTorch Broadcasting Warnings
            batch_size, pred_len, num_variates = outputs.shape
            
            if targets.dim() == 2: 
                # Target is [Batch, Variates]. Reshape to [Batch, 1, Variates]
                targets = targets.view(batch_size, 1, num_variates)
                
            elif targets.dim() == 3 and targets.size(1) != pred_len:
                # Target is [Batch, Seq_Len, Variates]. 
                # Slice only the last 'pred_len' steps to perfectly match the output horizon
                targets = targets[:, -pred_len:, :]
                
            # Ultimate safety check to guarantee MSELoss behaves mathematically correctly
            if targets.shape != outputs.shape:
                raise RuntimeError(f"[{self.name}] Shape mismatch after alignment! Output: {outputs.shape}, Target: {targets.shape}")
            
            loss = self.criterion(outputs, targets)
        
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return loss.item()

    def register_source_metadata(self, source_id: str, semantic_type: str, unit: str):
        """
        Called by the Inference Engine when the Linguist Agent approves a source.
        """
        self.source_registry[source_id] = {
            "type": semantic_type,
            "unit": unit
        }

    def train(self, inputs: torch.Tensor, targets: torch.Tensor, epochs: int = 50, batch_size: int = 16):
        device = self._get_current_device()
        print(f"[{self.name}] Training on {device}...")
        dataset_size = inputs.size(0)
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            permutation = torch.randperm(dataset_size)
            
            for i in range(0, dataset_size, batch_size):
                indices = permutation[i : i + batch_size]
                batch_x = inputs[indices]
                batch_y = targets[indices]
                
                loss = self.train_step((batch_x, batch_y))
                epoch_loss += loss
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Loss: {epoch_loss:.4f}")