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
# File: src/models/tcn_model.py
# Author: Gabriel Moraes
# Date: 2025-11-22

import torch
import torch.nn as nn

# Updated import to avoid FutureWarning in newer PyTorch versions
try:
    from torch.nn.utils.parametrizations import weight_norm
except ImportError:
    from torch.nn.utils import weight_norm

class Chomp1d(nn.Module):
    """
    Utility layer to crop the output of a Conv1d to ensure causal padding.
    
    PyTorch's Conv1d with padding adds zeros to BOTH sides.
    For time-series forecasting, we only want padding on the LEFT (past).
    This layer removes the extra padding from the RIGHT (future).
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        # Slice the tensor to remove the last 'chomp_size' elements
        return x[:, :, :-self.chomp_size]

class TemporalBlock(nn.Module):
    r"""
    A single residual block for the TCN.
    
    Structure:
    Input -> [Dilated Conv -> ReLU -> Dropout] x2 -> Output
          \_____________________________________/
                          |
                     Residual Link
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        
        # --- First Conv Layer ---
        # Weight Normalization is standard for TCNs to stabilize training
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        # Remove future leakage
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # --- Second Conv Layer ---
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # Container for the sequential operations
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        
        # --- Residual Connection ---
        # If input dim != output dim, we need a 1x1 conv to match dimensions for addition
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        """Initialize weights with a normal distribution."""
        # Note: with new weight_norm, we might need to access the weight via the original parameter
        # but for initialization simple normal_ usually works on the layer's weight attribute.
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        # Standard path
        out = self.net(x)
        # Residual path
        res = x if self.downsample is None else self.downsample(x)
        # Combine
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    """
    The Main TCN Model.
    
    Consists of a stack of TemporalBlocks with exponentially increasing dilation.
    """
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        """
        Args:
            num_inputs: Number of input features (channels).
            num_channels: List of integers defining the hidden size of each layer.
                          Example: [25, 25, 25] creates a 3-layer TCN with 25 channels each.
            kernel_size: Size of the sliding window.
            dropout: Dropout probability.
        """
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i # 1, 2, 4, 8...
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            # Padding is calculated to maintain sequence length: (k-1) * d
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                     dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size,
                                     dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: Input tensor [Batch, Channels (Features), Sequence Length]
        Returns:
            Output tensor [Batch, Channels, Sequence Length]
        """
        return self.network(x)