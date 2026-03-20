# SYNAPSE - A Gateway of Intelligent Perception for Traffic Management
# Copyright (C) 2025 Noxfort Labs
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
# File: src/models/timegan.py
# Author: Gabriel Moraes
# Date: 2026-02-15

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.backends.cudnn as cudnn

class TimeGAN(nn.Module):
    """
    Time-series Generative Adversarial Network (TimeGAN) implementation.
    
    Reference: "Time-series Generative Adversarial Networks" (Jinsung Yoon et al., NeurIPS 2019).
    
    Refactored V5 (Nuclear Stability Mode):
    - Explicitly DISABLES cuDNN for GRU layers using a context manager.
    - This bypasses the 'CUDNN_STATUS_NOT_SUPPORTED' error caused by 
      WeightNorm/Memory fragmentation in checkpoints.
    - Forces the use of PyTorch's native ATen backend for RNNs.
    """

    def __init__(self, feature_dim: int, hidden_dim: int, num_layers: int, padding_value: float = 0.0):
        """
        Initializes the TimeGAN architecture.
        """
        super(TimeGAN, self).__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.padding_value = padding_value

        # --- 1. Autoencoder Components (Embedder & Recovery) ---
        
        # Embedder: Input (Batch, Seq, Feat) -> Latent (Batch, Seq, Hidden)
        self.embedder = nn.GRU(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.embedder_out = nn.Linear(hidden_dim, hidden_dim)

        # Recovery: Latent (Batch, Seq, Hidden) -> Output (Batch, Seq, Feat)
        self.recovery = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.recovery_out = nn.Linear(hidden_dim, feature_dim)

        # --- 2. Generator Components (Generator & Supervisor) ---
        
        # Generator: Random Noise (Batch, Seq, Hidden) -> Latent (Batch, Seq, Hidden)
        self.generator = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.generator_out = nn.Linear(hidden_dim, hidden_dim)

        # Supervisor: Latent (Batch, Seq, Hidden) -> Next Latent (Batch, Seq, Hidden)
        supervisor_layers = max(1, num_layers - 1)
        
        self.supervisor = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=supervisor_layers, 
            batch_first=True
        )
        self.supervisor_out = nn.Linear(hidden_dim, hidden_dim)

        # --- 3. Discriminator Component ---
        
        # Discriminator: Latent (Batch, Seq, Hidden) -> Score (Batch, Seq, 1)
        self.discriminator = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.discriminator_out = nn.Linear(hidden_dim, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Applies Xavier Uniform initialization to Linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)

    def _forward_gru_safe(self, gru_layer, x):
        """
        Helper to execute GRU layers safely by temporarily disabling cuDNN.
        This resolves conflicts with WeightNorm hooks in checkpoints.
        """
        # Save current state
        original_cudnn_state = cudnn.enabled
        
        try:
            # Force cuDNN OFF -> Use safe ATen implementation
            cudnn.enabled = False
            
            # Ensure float32 and contiguous
            x = x.float().contiguous()
            
            # Execute
            out, _ = gru_layer(x)
            return out
            
        finally:
            # Restore original state
            cudnn.enabled = original_cudnn_state

    # --- Forward Passes with Safety Wrapper ---

    def forward_embedder(self, x):
        """Encodes features into latent space."""
        # x: [Batch, Seq, Feat]
        # Disable autocast to ensure we stay in FP32
        with torch.autocast(device_type=x.device.type, enabled=False):
            h = self._forward_gru_safe(self.embedder, x)
            h = torch.sigmoid(self.embedder_out(h)) 
            return h

    def forward_recovery(self, h):
        """Decodes latent space back to features."""
        # h: [Batch, Seq, Hidden]
        with torch.autocast(device_type=h.device.type, enabled=False):
            x_tilde = self._forward_gru_safe(self.recovery, h)
            x_tilde = self.recovery_out(x_tilde) 
            return x_tilde

    def forward_generator(self, z):
        """Generates synthetic latent codes from noise."""
        # z: [Batch, Seq, Hidden]
        with torch.autocast(device_type=z.device.type, enabled=False):
            e = self._forward_gru_safe(self.generator, z)
            e = torch.sigmoid(self.generator_out(e))
            return e

    def forward_supervisor(self, h):
        """Generates the next likely latent state sequence."""
        # h: [Batch, Seq, Hidden]
        with torch.autocast(device_type=h.device.type, enabled=False):
            s = self._forward_gru_safe(self.supervisor, h)
            s = torch.sigmoid(self.supervisor_out(s))
            return s

    def forward_discriminator(self, h):
        """Classifies latent codes as Real or Fake."""
        # h: [Batch, Seq, Hidden]
        with torch.autocast(device_type=h.device.type, enabled=False):
            y_hat = self._forward_gru_safe(self.discriminator, h)
            y_hat = self.discriminator_out(y_hat) 
            return y_hat