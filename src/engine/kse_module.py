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
# File: src/engine/kse_module.py
# Author: Gabriel Moraes
# Date: 2026-02-15

import torch
import torch.nn as nn
from typing import Optional, Tuple
from torch.amp import autocast

class KineticStateEstimator(nn.Module):
    """
    GPU-Accelerated Linear Kalman Filter Module (KSE).
    
    Architecture:
    - Parallelized Batch Processing: Handles N sensors simultaneously.
    - Tensor Core Optimized: Uses AMP (Mixed Precision) for matrix multiplications.
    - Numerical Stability: Falls back to FP32 for Cholesky/Inverse operations.
    
    State Vector (per sensor): [Value, Rate_of_Change]
    """

    def __init__(self, batch_size: int, dt: float = 1.0, process_noise: float = 1e-4, measure_noise: float = 1e-2):
        super().__init__()
        
        self.batch_size = batch_size
        self.dt = dt
        
        # Determine device automatically
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # --- System Matrices (Fixed) ---
        # F: State Transition Matrix [[1, dt], [0, 1]] (Physics Model)
        self.register_buffer('F', torch.tensor([[1.0, dt], [0.0, 1.0]], device=self.device))
        
        # H: Measurement Matrix [[1, 0]] (We only measure the value, not the rate)
        self.register_buffer('H', torch.tensor([[1.0, 0.0]], device=self.device))
        
        # Q: Process Noise Covariance (Uncertainty in the model)
        self.register_buffer('Q', torch.eye(2, device=self.device) * process_noise)
        
        # R: Measurement Noise Covariance (Uncertainty in sensors)
        # Note: R is scalar here as we have 1 measurement per sensor, but kept as matrix for logic genericism
        self.register_buffer('R', torch.tensor([[measure_noise]], device=self.device))
        
        # I: Identity Matrix
        self.register_buffer('I', torch.eye(2, device=self.device))

        # --- Dynamic State (Mutable) ---
        # x: State Vector [Batch, 2, 1]
        self.x = nn.Parameter(torch.zeros(batch_size, 2, 1, device=self.device), requires_grad=False)
        
        # P: Error Covariance Matrix [Batch, 2, 2]
        # Initial high uncertainty
        self.P = nn.Parameter(torch.eye(2, device=self.device).unsqueeze(0).repeat(batch_size, 1, 1) * 10.0, requires_grad=False)
        
        # Move everything to correct device immediately
        self.to(self.device)

    def initialize_states(self, initial_values: torch.Tensor):
        """
        Resets states for specific indices or bulk initialization.
        Args:
            initial_values: Tensor [Batch, 1] of sensor readings.
        """
        with torch.no_grad():
            initial_values = initial_values.to(self.device)
            # Set Position to value, Velocity to 0
            self.x[:, 0, 0] = initial_values.squeeze()
            self.x[:, 1, 0] = 0.0
            
            # Reset Covariance
            self.P.copy_(torch.eye(2, device=self.device).unsqueeze(0).repeat(self.batch_size, 1, 1) * 10.0)

    def predict(self) -> torch.Tensor:
        """
        Prediction Step (A Priori).
        Utilizes Tensor Cores via AMP.
        x = Fx
        P = FPF' + Q
        """
        device_type = self.device.type if self.device.type != 'mps' else 'cpu'
        
        with autocast(device_type=device_type, enabled=(self.device.type == 'cuda')):
            # Expand F for batch matmul if needed, or rely on broadcasting
            # F is [2, 2], x is [B, 2, 1] -> broadcast works
            
            # 1. Predict State
            self.x.copy_(self.F @ self.x)
            
            # 2. Predict Covariance
            # P = F @ P @ F.T + Q
            # Q needs expansion to [B, 2, 2]
            Q_expanded = self.Q.unsqueeze(0)
            
            P_new = self.F @ self.P @ self.F.T + Q_expanded
            self.P.copy_(P_new)
            
            return self.x[:, 0, :] # Return just the predicted values

    def correct(self, z: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Correction Step (A Posteriori).
        Integrates new measurements.
        
        Args:
            z: Measurements [Batch, 1]
            mask: Boolean Tensor [Batch] indicating which sensors actually sent data.
                  If None, assumes all sent data.
        """
        device_type = self.device.type if self.device.type != 'mps' else 'cpu'
        
        z = z.to(self.device).unsqueeze(-1) # [B, 1, 1]
        
        # Handle Masking (Only update sensors that have data)
        # We process everything but only apply changes to masked ones to keep batch size fixed
        # Or simpler: we assume z has NaNs for missing data and filter? 
        # For GPU performance, usually better to compute all and mask write-back, 
        # or gather active indices. Let's assume mask is passed.

        with autocast(device_type=device_type, enabled=(self.device.type == 'cuda')):
            # 1. Measurement Residual
            # y = z - Hx
            y = z - (self.H @ self.x)
            
            # 2. Residual Covariance
            # S = HPH' + R
            PHT = self.P @ self.H.T
            S = self.H @ PHT + self.R
            
            # 3. Optimal Kalman Gain
            # K = P H' S^-1
            # CRITICAL: Matrix Inversion is unstable in FP16.
            # We temporarily force FP32 for the solve/inverse step.
        
        # --- Stability Block (FP32) ---
        with torch.autocast(device_type=device_type, enabled=False):
            S_fp32 = S.float()
            PHT_fp32 = PHT.float()
            
            # Use linalg.solve instead of inverse() for better numerical stability
            # K = PHT * inv(S) is equivalent to solving S * K.T = PHT.T
            # Since S is scalar (1x1 measurement), simple division or inverse works well.
            # But implementing generic matrix formulation:
            
            try:
                # Adding small epsilon to diagonal for stability
                epsilon = 1e-6
                S_fp32 = S_fp32 + (torch.eye(1, device=self.device) * epsilon)
                
                S_inv = torch.linalg.inv(S_fp32)
                K = PHT_fp32 @ S_inv
            except RuntimeError:
                # Fallback if singular matrix (rare in 1D case but good practice)
                K = torch.zeros_like(PHT_fp32)

        # Resume AMP/FP16 logic or cast back
        K = K.to(self.x.dtype)
        
        with autocast(device_type=device_type, enabled=(self.device.type == 'cuda')):
            # 4. Update State
            # x = x + Ky
            dx = K @ y
            x_new = self.x + dx
            
            # 5. Update Covariance
            # P = (I - KH)P
            I_KH = self.I - (K @ self.H)
            P_new = I_KH @ self.P
            
            # Apply Mask if provided
            if mask is not None:
                mask = mask.view(-1, 1, 1) # [B, 1, 1]
                self.x.copy_(torch.where(mask, x_new, self.x))
                self.P.copy_(torch.where(mask, P_new, self.P))
            else:
                self.x.copy_(x_new)
                self.P.copy_(P_new)

        return self.x[:, 0, :] # Return filtered values

    def forward(self, z: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Single Cycle: Predict -> (Optional) Correct
        """
        # Always predict physics
        self.predict()
        
        # Correct if we have data
        if z is not None:
            return self.correct(z, mask)
        
        return self.x[:, 0, :]