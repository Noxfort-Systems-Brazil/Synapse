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
# File: src/agents/auditor_agent.py
# Author: Gabriel Moraes
# Date: 2026-03-02

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from typing import Dict, Any, Union, Tuple
from torch.amp import autocast, GradScaler

from src.agents.base_agent import BaseAgent
from src.models.wavelet_ae_occ import WaveletAEOCC

logger = logging.getLogger("Synapse.AuditorAgent")

class AuditorAgent(BaseAgent):
    """
    The Auditor Agent ('O Segurança').
    
    Responsibilities:
    1. Anomaly Detection in Time-Series (Traffic Patterns).
    2. Security Auditing of Sensor Data Inputs.
    
    Architecture:
    - Wraps the WaveletAEOCC model.
    - Uses AMP (Automatic Mixed Precision) for Tensor Core acceleration.
    """

    def __init__(self, input_len: int, J: int = 2, Q: int = 1, latent_dim: int = 16, learning_rate: float = 1e-3):
        """
        Args:
            input_len: Window size of the time series.
            J: Scattering scale.
            Q: Scattering quality factor.
            latent_dim: Autoencoder bottleneck size.
        """
        # --- Safety Fix: Validate Wavelet Scale (J) vs Padded Input Length ---
        # The model now applies reflection padding to the next power of 2.
        # We calculate the padded length to validate J correctly and avoid warnings.
        padded_len = 2 ** int(np.ceil(np.log2(input_len)) + 1)
        max_safe_j = int(np.log2(padded_len))
        
        if J > max_safe_j:
            safe_J = max(1, max_safe_j)
            # Only warn if strictly necessary to avoid log spam during AutoML
            if J > safe_J: 
                logger.info(f"[AuditorAgent] Auto-adjusting J from {J} to {safe_J} for stability (padded_len={padded_len}).")
            J = safe_J

        # 1. Initialize Model
        model = WaveletAEOCC(
            input_len=input_len,
            J=J,
            Q=Q,
            latent_dim=latent_dim
        )
        
        # 2. Base Agent Init
        super().__init__(model=model, name="AuditorAgent")
        
        # 3. Optimization Setup
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5) 
        self.criterion_rec = nn.MSELoss()
        
        # 4. AMP Scaler
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = GradScaler('cuda' if self.device.type == 'cuda' else 'cpu')
        
        # Move to device
        self.model.to(self.device)

    def train_step(self, input_data: Union[torch.Tensor, Tuple[torch.Tensor, Any]]) -> float:
        """
        Performs a single training step optimizing for both:
        1. Reconstruction Quality (MSE).
        2. One-Class Compactness (Distance to Center).
        
        Includes Stabilization for AutoML (NaN checks, Instance Norm).
        """
        self.model.train()
        
        # Unpack tuple if coming from DataLoader
        if isinstance(input_data, (tuple, list)):
            x = input_data[0]
        else:
            x = input_data
            
        x = x.to(self.device)

        # --- Stability Fix: Instance Normalization ---
        # Ensures that each window has mean 0 and std 1 before entering the AE.
        # This prevents "Exploding Gradients" when raw sensor data has high variance.
        eps = 1e-5
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True)
        x = (x - mean) / (std + eps)
        
        self.optimizer.zero_grad()
        
        device_type = self.device.type if self.device.type != 'mps' else 'cpu'
        
        # --- Mixed Precision Context ---
        with autocast(device_type=device_type, enabled=(self.device.type == 'cuda')):
            # Forward Pass
            feats, z, rec_feats = self.model(x)
            
            # Initialize Hypersphere Center on first batch
            if not self.model.center_initialized:
                self.model.init_center(z)
                
            # --- Loss Calculation ---
            # 1. Reconstruction Loss
            loss_rec = self.criterion_rec(rec_feats, feats)
            
            # 2. Compactness Loss (Deep SVDD)
            dist = torch.sum((z - self.model.center) ** 2, dim=1)
            loss_occ = torch.mean(dist)
            
            # Weighted Total Loss
            loss = loss_rec + (0.1 * loss_occ)
        
        # --- Safety Check: NaN/Inf Guard ---
        if not torch.isfinite(loss):
            # If loss explodes, skip update and return high penalty
            # This allows Optuna to prune this bad trial without crashing the study
            self.scaler.update() 
            return 100.0 # Arbitrary high loss

        # --- Scaled Backward Pass ---
        self.scaler.scale(loss).backward()
        
        # Unscale for clipping
        self.scaler.unscale_(self.optimizer)
        
        # Stronger Gradient Clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
        
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        # --- Adaptive Threshold Update ---
        with torch.no_grad():
            rec_err = torch.mean((rec_feats - feats) ** 2, dim=1)
            batch_scores = rec_err + (0.1 * dist)
            self.model.update_threshold(batch_scores)

        return loss.item()

    def inference(self, input_data: Union[Dict[str, Any], torch.Tensor]) -> Dict[str, Any]:
        """
        Reconstructs the embedding to check for anomalies. 
        Supports both raw Tensor inputs and standardized Domain payloads.
        """
        self.model.eval()
        
        # Input Handling
        if isinstance(input_data, dict):
            if "signature" in input_data:
                x = input_data["signature"]
            else:
                x = input_data.get('data') or input_data.get('window')
                
            if x is None: return {"error": "No data provided"}
            x = torch.tensor(x) if not isinstance(x, torch.Tensor) else x
        else:
            x = input_data
            
        x = x.to(self.device)
        
        if x.ndim == 1:
            x = x.unsqueeze(0)
            
        # Optional: Safety casting inside the agent instead of the engine
        if hasattr(self.model, 'encoder'):
             auditor_in = self.model.encoder[0].in_features
             sig_dim = x.shape[1]
             
             if sig_dim != auditor_in:
                 if sig_dim < auditor_in:
                      import torch.nn.functional as F
                      x = F.pad(x, (0, auditor_in - sig_dim))
                 else:
                      x = x[:, :auditor_in]
            
        # Apply same Instance Norm as training
        eps = 1e-5
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True)
        x = (x - mean) / (std + eps)
        
        device_type = self.device.type if self.device.type != 'mps' else 'cpu'
        
        with torch.no_grad():
            with autocast(device_type=device_type, enabled=(self.device.type == 'cuda')):
                feats, z, rec_feats = self.model(x)
                
                # Calculate Anomaly Score
                rec_err = torch.mean((rec_feats - feats) ** 2, dim=1)
                dist_center = torch.sum((z - self.model.center) ** 2, dim=1)
                
                total_score = rec_err + (0.1 * dist_center)
                
                threshold = self.model.threshold
                is_anomaly = total_score > threshold

        return {
            "is_anomaly": bool(is_anomaly.item()),
            "score": float(total_score.item()),
            "threshold": float(threshold.item()),
            "status": "audited"
        }