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
# File: src/agents/imputer_agent.py
# Author: Gabriel Moraes
# Date: 2026-02-28

import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Any, Dict
from torch.amp import autocast, GradScaler

from src.agents.base_agent import BaseAgent
from src.models.timegan import TimeGAN

logger = logging.getLogger(__name__)

class ImputerAgent(BaseAgent):
    """
    The Imputer Agent ('O Reconstrutor').
    
    Responsibilities:
    1. Data Recovery: Fills long gaps (NaNs) in sensor time series using TimeGAN.
    2. Synthetic Generation: Creates realistic data for Phase Zero re-training.
    
    Refactored V8 (Sequence Chunking Anti-OOM):
    - Implements sequence chunking during inference to avoid GRU memory overflow.
    - Performs reshaping and contiguity checks in NumPy BEFORE Tensor conversion.
    - Explicitly disables autocast for Inference to avoid cuDNN FP16 conflicts.
    - Ensures strict 'float32' types for stability.
    """

    def __init__(self, feature_dim: int = 4, hidden_dim: int = 24, seq_len: int = 24, num_layers: int = 3, learning_rate: float = 0.001):
        """
        Initializes the TimeGAN model and its adversarial optimizers.
        """
        # 1. Create the Composite Model
        model = TimeGAN(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        )
        
        # 2. Base Init
        super().__init__(model=model, name="ImputerAgent")
        
        # 3. Attributes
        self.feature_dim = feature_dim
        self.seq_len = seq_len
        
        # 4. Multi-Optimizers (GAN Pattern)
        self.opt_g = optim.Adam(self.model.generator.parameters(), lr=learning_rate)
        self.opt_d = optim.Adam(self.model.discriminator.parameters(), lr=learning_rate)
        
        # 5. Scalers for mixed precision
        self.scaler_g = GradScaler()
        self.scaler_d = GradScaler()
        
        self.criterion = nn.BCEWithLogitsLoss()
        
        # 6. Device Setup (Critical for Phase 1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def inference(self, input_data: Any) -> Any:
        """
        Standard Interface: Reconstructs missing values.
        """
        return self.impute(input_data)

    def impute(self, incomplete_seq: np.ndarray, chunk_size: int = 4096) -> np.ndarray:
        """
        Uses the Generator to fill gaps in a sequence.
        Processes the sequence in chunks to avoid CUDA OOM in GRU layers.
        
        Args:
            incomplete_seq: Numpy array containing NaNs. Usually [Seq_Len, Features].
            chunk_size: Maximum sequence length to process at once in the GPU.
        """
        self.model.eval()
        
        # --- Pre-processing: Handle NaNs & Layout ---
        working_seq = incomplete_seq.copy()
        mask = np.isnan(working_seq)
        working_seq[mask] = 0.0 # Zero-fill for math stability

        is_2d = (working_seq.ndim == 2)
        working_seq = np.ascontiguousarray(working_seq, dtype=np.float32)
        
        reconstructed_results = []
        total_length = working_seq.shape[0] if is_2d else working_seq.shape[1]
        
        logger.info(f"[{self.name}] Starting chunked imputation. Total length: {total_length}, Chunk size: {chunk_size}")

        with torch.no_grad():
            if is_2d:
                # Process continuous sequence in sliding windows/chunks
                for i in range(0, total_length, chunk_size):
                    chunk = working_seq[i:i+chunk_size]
                    
                    # Force Batch Dimension: [Chunk, Feat] -> [1, Chunk, Feat]
                    chunk_3d = chunk[np.newaxis, :, :]
                    tensor_x = torch.from_numpy(chunk_3d).to(self.device)

                    # Disable autocast for inference to prevent cuDNN type mismatches
                    h_latent = self.model.forward_embedder(tensor_x)
                    x_reconstructed = self.model.forward_recovery(h_latent)
                    
                    # Convert back to Numpy [Chunk, Feat]
                    recon_np = x_reconstructed.cpu().float().numpy().squeeze(0)
                    reconstructed_results.append(recon_np)
                    
                    # Memory Cleanup
                    del tensor_x, h_latent, x_reconstructed
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()
                        
                # Reassemble the entire sequence
                reconstructed = np.concatenate(reconstructed_results, axis=0)
            else:
                # If it's already 3D (Batch, Seq, Feat), process by batches
                batch_size_3d = 128
                for i in range(0, working_seq.shape[0], batch_size_3d):
                    chunk = working_seq[i:i+batch_size_3d]
                    tensor_x = torch.from_numpy(chunk).to(self.device)
                    
                    h_latent = self.model.forward_embedder(tensor_x)
                    x_reconstructed = self.model.forward_recovery(h_latent)
                    
                    reconstructed_results.append(x_reconstructed.cpu().float().numpy())
                    
                    del tensor_x, h_latent, x_reconstructed
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()
                        
                reconstructed = np.concatenate(reconstructed_results, axis=0)

        # --- Post-processing: Patching ---
        final_output = incomplete_seq.copy()
        final_output[mask] = reconstructed[mask]
        
        logger.info(f"[{self.name}] Imputation completed successfully without memory overflow.")
        return final_output

    def train_step(self, batch_data: torch.Tensor) -> Dict[str, float]:
        """
        Adversarial Training Step for TimeGAN.
        """
        self.model.train()
        
        # Ensure batch is on device
        real_data = batch_data.to(self.device)
        
        # Handle potential NaNs
        if torch.isnan(real_data).any():
             real_data = torch.nan_to_num(real_data, nan=0.0)

        # CRITICAL: Force contiguity for Training data too
        if not real_data.is_contiguous():
            real_data = real_data.contiguous()

        batch_size = real_data.size(0)
        
        # --- Discriminator Training ---
        self.opt_d.zero_grad()
        with autocast(device_type=self.device.type, enabled=(self.device.type == 'cuda')):
            # Generate Random Noise
            z = torch.randn(batch_size, self.seq_len, self.model.hidden_dim).to(self.device)
            
            # Generate Fake Latent
            e_fake = self.model.forward_generator(z)
            
            # Embed Real Data
            h_real = self.model.forward_embedder(real_data)
            
            # Discriminator Scores
            y_real = self.model.forward_discriminator(h_real)
            y_fake = self.model.forward_discriminator(e_fake.detach())
            
            loss_d_real = self.criterion(y_real, torch.ones_like(y_real))
            loss_d_fake = self.criterion(y_fake, torch.zeros_like(y_fake))
            loss_d = loss_d_real + loss_d_fake

        self.scaler_d.scale(loss_d).backward()
        self.scaler_d.step(self.opt_d)
        self.scaler_d.update()

        # --- Generator Training ---
        self.opt_g.zero_grad()
        with autocast(device_type=self.device.type, enabled=(self.device.type == 'cuda')):
            # We want Discriminator to think Fake is Real
            y_fake_g = self.model.forward_discriminator(e_fake)
            loss_g_adv = self.criterion(y_fake_g, torch.ones_like(y_fake_g))
            loss_g = loss_g_adv

        self.scaler_g.scale(loss_g).backward()
        self.scaler_g.step(self.opt_g)
        self.scaler_g.update()

        return {"d_loss": loss_d.item(), "g_loss": loss_g.item()}

    def generate_synthetic(self, num_samples: int) -> np.ndarray:
        """
        Generates completely new data for re-training other agents.
        """
        self.model.eval()
        
        with torch.no_grad():
            z = torch.randn(num_samples, self.seq_len, self.model.hidden_dim).to(self.device)
            e_hat = self.model.forward_generator(z)
            x_hat = self.model.forward_recovery(e_hat)
            
        return x_hat.cpu().numpy()