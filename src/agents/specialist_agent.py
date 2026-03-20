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
# File: src/agents/specialist_agent.py
# Author: Gabriel Moraes
# Date: 2026-02-14

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Any
from torch.amp import autocast, GradScaler

# Import Base & Model
from src.agents.base_agent import BaseAgent
from src.models.tcn_model import TemporalConvNet

class SpecialistAgent(BaseAgent):
    """
    The Specialist Agent ('O Tático').
    
    Refactored V3 (Device Safe):
    - Implements Dynamic Device Detection to prevent CPU/GPU conflicts.
    - Ensures TCN inference respects the actual location of model weights.
    - Maintains PBT primitives (copy/mutate) robustly.
    """

    def __init__(self, input_dim: int, output_dim: int, num_channels: List[int], kernel_size: int = 2, dropout: float = 0.2, learning_rate: float = 0.001):
        """
        Initializes the agent and the TCN network.
        """
        # 1. Hyperparameters
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        
        # 2. Build Components
        tcn = TemporalConvNet(
            num_inputs=input_dim,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout
        )
        
        decoder = nn.Linear(num_channels[-1], output_dim)
        
        # Reconstruction Head: Embedding -> Input
        reconstruction_head = nn.Linear(output_dim, input_dim)
        
        # 3. Bundle into ModuleDict
        model = nn.ModuleDict({
            "tcn": tcn,
            "decoder": decoder,
            "reconstruction_head": reconstruction_head
        })
        
        # 4. Initialize Base
        super().__init__(model=model, name="SpecialistAgent")
        
        # 5. Shortcuts for readability
        self.tcn = self.model["tcn"]
        self.decoder = self.model["decoder"]
        self.reconstruction_head = self.model["reconstruction_head"]
        
        # 6. Optimizer & Scaler
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Initialize Scaler safely
        self.scaler = GradScaler(device='cuda' if torch.cuda.is_available() else 'cpu')
        self.criterion = nn.MSELoss()
        
        # PBT Metrics
        self.running_loss = 0.0
        self.steps = 0

    def _get_current_device(self) -> torch.device:
        """
        Introspects the model to find its actual physical location.
        """
        try:
            # Check the first parameter of the TCN submodule
            return next(self.tcn.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def inference(self, input_data: Any) -> Any:
        """
        Standard Interface Implementation.
        Wrapper for predict logic.
        """
        return self.predict(input_data)

    def predict(self, input_sequence: np.ndarray) -> np.ndarray:
        """
        Forecasts next steps/embeddings using AMP and Safe Device placement.
        Refactored V4 (SRP Encapsulated): Handles T-transposition and [-1] slicing natively.
        """
        self.model.eval()
        device = self._get_current_device() # Dynamic Check

        with torch.no_grad():
            # [SRP Fix] Auto-handle TrafficNode's (Seq_Len, Channels) numpy array
            if getattr(input_sequence, "ndim", 0) == 2 and input_sequence.shape[0] > input_sequence.shape[1]:
                 # Typical memory buffer is [Time, Features], TCN needs [Features, Time]
                 input_sequence = input_sequence.T

            # Handle conversion
            if isinstance(input_sequence, np.ndarray):
                tensor_x = torch.FloatTensor(input_sequence).to(device)
            elif isinstance(input_sequence, torch.Tensor):
                tensor_x = input_sequence.to(device)
            else:
                tensor_x = torch.tensor(input_sequence, dtype=torch.float).to(device)
            
            # Ensure shape [1, Channels, Seq_Len]
            if tensor_x.dim() == 2: # [Channels, Seq]
                tensor_x = tensor_x.unsqueeze(0)
            
            device_type = device.type if device.type != 'mps' else 'cpu'
            
            with autocast(device_type=device_type, enabled=(device.type == 'cuda')):
                tcn_out = self.tcn(tensor_x)
                
                # Decoder expects [Batch, Seq, Channels], but TCN outputs [Batch, Channels, Seq]
                tcn_out_transposed = tcn_out.transpose(1, 2)
                output = self.decoder(tcn_out_transposed)
                
                # Return to [Batch, Channels, Seq]
                output = output.transpose(1, 2)
            
            # [SRP Fix] Return only the final embedding vector for the current state [Channels] The Node doesn't need the whole sequence history trajectory.
            full_seq = output.cpu().float().numpy().squeeze(0)
            if full_seq.ndim == 2:
                return full_seq[:, -1]
            return full_seq

    def train_step(self, batch_data: Any) -> float:
        """
        Standard Training Step.
        """
        self.model.train()
        device = self._get_current_device()

        # Unpack
        if isinstance(batch_data, (tuple, list)):
            inputs, targets = batch_data
        else:
            # Self-supervised fallback
            inputs, targets = batch_data, batch_data
        
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        self.optimizer.zero_grad()
        
        device_type = device.type if device.type != 'mps' else 'cpu'
        
        with autocast(device_type=device_type, enabled=(device.type == 'cuda')):
            # 1. TCN Forward
            # Input: [Batch, Seq, Feat] -> TCN needs [Batch, Feat, Seq]
            if inputs.shape[-1] == self.input_dim and inputs.ndim == 3:
                 batch_x_t = inputs.permute(0, 2, 1)
            else:
                 batch_x_t = inputs

            tcn_out = self.tcn(batch_x_t)
            
            # 2. Decode
            tcn_out_transposed = tcn_out.transpose(1, 2)
            embedding = self.decoder(tcn_out_transposed)
            
            pred = embedding
            
            # 3. Dynamic Loss Handling
            # If target is raw input (Reconstruction)
            if targets.shape[-1] == self.input_dim and self.output_dim != self.input_dim:
                pred = self.reconstruction_head(embedding)
            # If target is embedding but transposed
            elif targets.shape[-1] != self.output_dim and targets.shape[1] == self.output_dim:
                pred = pred.transpose(1, 2)

            loss = self.criterion(pred, targets)
        
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return loss.item()

    def train(self, inputs: torch.Tensor, targets: torch.Tensor, epochs: int = 1, batch_size: int = 32) -> float:
        """
        Legacy wrapper compatible with NodeManager.
        """
        dataset_size = inputs.size(0)
        total_loss = 0.0
        effective_batch = min(dataset_size, batch_size)
        permutation = torch.randperm(dataset_size)
        
        for i in range(0, dataset_size, effective_batch):
            indices = permutation[i : i + effective_batch]
            batch_x = inputs[indices]
            batch_y = targets[indices]
            
            loss = self.train_step((batch_x, batch_y))
            total_loss += loss
        
        avg_loss = total_loss / max(1, (dataset_size / effective_batch))
        
        # PBT Metric Update
        self.running_loss = 0.9 * self.running_loss + 0.1 * avg_loss if self.steps > 0 else avg_loss
        self.steps += 1
        
        return avg_loss

    # --- PBT Primitives ---

    def copy_from(self, other_agent: 'SpecialistAgent'):
        """Exploit Step."""
        self.model.load_state_dict(other_agent.model.state_dict())
        self.learning_rate = other_agent.learning_rate
        self.dropout = other_agent.dropout
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.learning_rate
            
        self.running_loss = 0.0
        self.steps = 0

    def mutate(self):
        """Explore Step."""
        factor = np.random.choice([0.8, 1.2])
        self.learning_rate *= factor
        
        self.dropout += np.random.normal(0, 0.05)
        self.dropout = np.clip(self.dropout, 0.0, 0.5)
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.learning_rate
            
        self._update_dropout_layers()

    def _update_dropout_layers(self):
        for m in self.tcn.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout1d, nn.Dropout2d)):
                m.p = self.dropout

    # --- Persistence Primitives (SOLID SRP) ---

    def get_state(self) -> dict:
        """
        Returns the agent's internal neural state dictionary, 
        shielding callers from knowing about TCN/Decoders.
        """
        return {
            "agent_weights": {
                "tcn": self.tcn.state_dict(),
                "decoder": self.decoder.state_dict(),
                "head": self.reconstruction_head.state_dict()
            },
            "agent_config": {
                "lr": self.learning_rate,
                "dropout": self.dropout
            }
        }

    def set_state(self, state: dict):
        """Loads weights safely without exposing components."""
        weights = state.get("agent_weights", {})
        if "tcn" in weights:
            self.tcn.load_state_dict(weights["tcn"])
        if "decoder" in weights:
            self.decoder.load_state_dict(weights["decoder"])
        if "head" in weights:
            self.reconstruction_head.load_state_dict(weights["head"])

        conf = state.get("agent_config", {})
        self.learning_rate = conf.get("lr", 0.001)
        self.dropout = conf.get("dropout", 0.2)
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.learning_rate
        self._update_dropout_layers()