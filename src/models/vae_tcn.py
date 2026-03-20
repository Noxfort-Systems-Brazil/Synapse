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
# File: src/models/vae_tcn.py
# Author: Gabriel Moraes
# Date: 2026-03-02

import torch
import torch.nn as nn
import torch.nn.functional as F

# Using the modern parametrizations API to avoid deprecation warnings
from torch.nn.utils.parametrizations import weight_norm

class Chomp1d(nn.Module):
    """
    Removes the extra padding added by convolution to maintain causality.
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    """
    A single TCN block consisting of dilated convolutions, non-linearity, and dropout.
    Includes residual connection.
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        
        # 1st Conv Layer - Applied modern weight_norm
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # 2nd Conv Layer - Applied modern weight_norm
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        
        # Residual connection matching (1x1 conv if dimensions differ)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class VAETCN(nn.Module):
    """
    Variational Autoencoder with Temporal Convolutional Network backbone.
    
    Structure:
    - Encoder: TCN reduces input to latent distribution parameters (Mu, LogVar).
    - Reparameterization: z = mu + sigma * epsilon
    - Decoder: TCN reconstructs the sequence from z.
    """
    def __init__(self, input_channels, hidden_channels, latent_channels, kernel_size=3, dropout=0.2):
        super(VAETCN, self).__init__()
        
        # --- Encoder ---
        # Reduces dimensionality and captures temporal context
        self.encoder_tcn = nn.Sequential(
            TemporalBlock(input_channels, hidden_channels, kernel_size, stride=1, dilation=1, padding=(kernel_size-1)*1, dropout=dropout),
            TemporalBlock(hidden_channels, hidden_channels, kernel_size, stride=1, dilation=2, padding=(kernel_size-1)*2, dropout=dropout),
            TemporalBlock(hidden_channels, hidden_channels, kernel_size, stride=1, dilation=4, padding=(kernel_size-1)*4, dropout=dropout)
        )
        
        # Projections to Latent Space (Mean and Log-Variance)
        # We perform 1x1 conv to map hidden channels to latent channels
        self.fc_mu = nn.Conv1d(hidden_channels, latent_channels, 1)
        self.fc_logvar = nn.Conv1d(hidden_channels, latent_channels, 1)

        # --- Decoder ---
        # Reconstructs from latent Z
        # We perform 1x1 conv to map latent back to hidden
        self.fc_decode = nn.Conv1d(latent_channels, hidden_channels, 1)
        
        self.decoder_tcn = nn.Sequential(
            TemporalBlock(hidden_channels, hidden_channels, kernel_size, stride=1, dilation=4, padding=(kernel_size-1)*4, dropout=dropout),
            TemporalBlock(hidden_channels, hidden_channels, kernel_size, stride=1, dilation=2, padding=(kernel_size-1)*2, dropout=dropout),
            TemporalBlock(hidden_channels, hidden_channels, kernel_size, stride=1, dilation=1, padding=(kernel_size-1)*1, dropout=dropout)
        )
        
        # Final projection to original input dimension
        self.final_layer = nn.Conv1d(hidden_channels, input_channels, 1)

    def reparameterize(self, mu, logvar):
        """
        The VAE Reparameterization Trick: z = mu + std * epsilon
        Allows backprop through stochastic node.
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def forward(self, x):
        """
        Args:
            x: Input tensor [Batch, Channels, Sequence_Length]
        Returns:
            recon: Reconstructed sequence
            mu: Latent mean
            logvar: Latent log variance
        """
        # 1. Encode
        enc_out = self.encoder_tcn(x)
        
        # 2. Latent Distribution
        mu = self.fc_mu(enc_out)
        logvar = self.fc_logvar(enc_out)
        
        # 3. Sample Z
        z = self.reparameterize(mu, logvar)
        
        # 4. Decode
        dec_in = self.fc_decode(z)
        dec_out = self.decoder_tcn(dec_in)
        
        # 5. Reconstruct
        recon = self.final_layer(dec_out)
        
        return recon, mu, logvar