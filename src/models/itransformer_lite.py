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
# File: src/models/itransformer_lite.py
# Author: Gabriel Moraes
# Date: 2026-03-02

import torch
import torch.nn as nn

class iTransformerLite(nn.Module):
    """
    A static, lightweight adaptation of the iTransformer attention mechanism.
    Designed exclusively for Multimodal Data Fusion before temporal analysis.
    It acts as a pre-processor, fusing multivariate heterogeneous sensor data 
    (e.g., Camera Volume + Waze Speed) into a single 1D "Traffic Stress Signal" 
    using a static, low-overhead Multi-Head Attention mechanism.
    """
    def __init__(self, num_sensors: int, d_model: int = 32, n_heads: int = 2):
        """
        Static Hyperparameters to avoid Optuna overhead:
        - num_sensors: How many columns are coming in (e.g., 2 for Volume and Speed).
        - d_model: Tiny hidden dimension (32 is enough for fusion).
        - n_heads: 2 heads to cross-reference Volume vs Speed.
        """
        super(iTransformerLite, self).__init__()
        
        self.num_sensors = num_sensors
        
        # 1. Feature Projection: Maps raw sensor data to a richer hidden space
        self.feature_projection = nn.Linear(num_sensors, d_model)
        
        # 2. The "Brain": Lightweight Self-Attention
        # It looks at the whole sequence window to understand context (e.g., "Is traffic slowing down?")
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        
        # 3. Output Condenser: Smashes the 32 dimensions down to a single 1D Stress Signal
        self.stress_out = nn.Linear(d_model, 1)
        
        # Activation and Normalization for stability
        self.gelu = nn.GELU()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [Batch, Seq_Len, Num_Sensors]
               Example: [1, 96, 2] (96 hours of Volume and Speed)
               
        Returns:
            stress_signal: Tensor of shape [Batch, Seq_Len, 1]
        """
        # Step 1: Project inputs to hidden dimension
        # Shape becomes: [Batch, Seq_Len, d_model]
        hidden = self.gelu(self.feature_projection(x))
        
        # Step 2: Apply Cross-Context Attention
        # The network learns to pay attention to Speed when Volume drops to 0 during a jam.
        attn_out, _ = self.attention(hidden, hidden, hidden)
        
        # Add & Norm (Residual connection for stable gradients)
        fused_hidden = self.norm(hidden + attn_out)
        
        # Step 3: Condense to a single "Stress Index" feature
        # Shape becomes: [Batch, Seq_Len, 1]
        stress_signal = self.stress_out(fused_hidden)
        
        return stress_signal