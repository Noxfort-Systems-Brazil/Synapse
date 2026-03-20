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
# File: src/models/wavelet_ae_occ.py
# Author: Gabriel Moraes
# Date: 2026-03-02

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Try importing Kymatio for Wavelet Scattering
try:
    from kymatio.torch import Scattering1D
    KYMATIO_AVAILABLE = True
except ImportError:
    KYMATIO_AVAILABLE = False

class WaveletAEOCC(nn.Module):
    """
    Hybrid Neural Architecture for Anomaly Detection.
    
    Components:
    1. Fixed Feature Extractor: Wavelet Scattering Transform (Invariant to translation/noise).
    2. Compressor: Lightweight Autoencoder (Bottle-neck).
    3. Classifier: Adaptive One-Class Classifier (Deep SVDD style with EMA threshold).
    """

    def __init__(self, input_len: int, J: int = 2, Q: int = 1, latent_dim: int = 16):
        """
        Args:
            input_len: Length of the time-series window (T).
            J: Scale of scattering (2^J must be <= padded_len).
            Q: Quality factor (filters per octave).
            latent_dim: Dimension of the AE bottleneck.
        """
        super().__init__()
        
        self.input_len = input_len
        self.latent_dim = latent_dim
        self.center_initialized = False
        
        # --- Padding Strategy to Avoid Border Effects ---
        # We artificially extend the signal using reflection padding so Kymatio
        # has enough support to apply the Wavelet Transform without edge distortion.
        # We pad to the next power of 2 that is strictly greater than input_len.
        self.padded_len = 2 ** int(np.ceil(np.log2(input_len)) + 1)
        
        # 1. Wavelet Scattering Setup
        if KYMATIO_AVAILABLE:
            # T=padded_len ensures global pooling over the entire extended sequence
            self.scattering = Scattering1D(J=J, shape=(self.padded_len,), Q=Q, T=self.padded_len)
            
            # Dynamic dimension calculation
            # We run a dummy pass on CPU to determine output size
            with torch.no_grad():
                dummy_input = torch.zeros(1, self.padded_len)
                dummy_out = self.scattering(dummy_input) # [1, Coeffs, 1]
                self.scat_dim = dummy_out.shape[1] 
            self.using_wavelets = True
        else:
            # Fallback: Identity (Raw input, but padded to match dimensions)
            self.scat_dim = self.padded_len
            self.scattering = nn.Identity()
            self.using_wavelets = False

        # 2. Encoder (Features -> Latent)
        self.encoder = nn.Sequential(
            nn.Linear(self.scat_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        
        # 3. Decoder (Latent -> Reconstruction)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, self.scat_dim)
        )
        
        # 4. One-Class State (Buffers ensure they are saved with model but not trained by SGD)
        # Center 'c' of the hypersphere
        self.register_buffer('center', torch.zeros(1, latent_dim))
        
        # Adaptive Threshold (EMA updated)
        self.register_buffer('threshold', torch.tensor(0.5))
        self.momentum = 0.1 

    def init_center(self, z: torch.Tensor):
        """
        Initialize the hypersphere center 'c' as the mean of the first batch.
        Prevents mode collapse where c=0 and z=0 trivially.
        """
        with torch.no_grad():
            self.center = torch.mean(z, dim=0, keepdim=True)
            self.center_initialized = True

    def update_threshold(self, anomaly_scores: torch.Tensor):
        """
        Updates the anomaly threshold based on the statistics of the current batch (EMA).
        Threshold ~= Mean + 2 * StdDev (covers ~95% of normal data)
        """
        with torch.no_grad():
            current_limit = anomaly_scores.mean() + 2 * anomaly_scores.std()
            # EMA: New = (1-m)*Old + m*Current
            self.threshold = (1 - self.momentum) * self.threshold + self.momentum * current_limit

    def forward(self, x: torch.Tensor):
        """
        Returns:
            feats: Extracted features (Wavelet or Raw).
            z: Latent representation.
            rec_feats: Reconstructed features.
        """
        # Ensure correct dimensionality [Batch, Time]
        if x.ndim == 3: 
            x = x.squeeze(-1)
            
        # --- Apply Reflection Padding ---
        pad_size = self.padded_len - self.input_len
        if pad_size > 0:
            pad_left = pad_size // 2
            pad_right = pad_size - pad_left
            # mode='reflect' mirrors the data at the boundaries, preventing sharp drops
            x = F.pad(x, (pad_left, pad_right), mode='reflect')
        
        # 1. Feature Extraction
        if self.using_wavelets:
            feats = self.scattering(x)
            # Flatten: [Batch, Coeffs, 1] -> [Batch, Coeffs]
            feats = feats.view(x.size(0), -1)
        else:
            feats = x
            
        # 2. Encode
        z = self.encoder(feats)
        
        # 3. Decode
        rec_feats = self.decoder(z)
        
        return feats, z, rec_feats