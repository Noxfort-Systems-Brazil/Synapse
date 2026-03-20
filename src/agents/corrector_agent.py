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
# File: src/agents/corrector_agent.py
# Author: Gabriel Moraes
# Date: 2026-03-02

import os
import logging
import torch
import numpy as np
from typing import Dict, Optional, Any

from src.agents.base_agent import BaseAgent
from src.models.vae_tcn import VAETCN

logger = logging.getLogger(__name__)

class CorrectorAgent(BaseAgent):
    """
    The Corrector Agent (Zelador).
    Uses a VAE-TCN architecture to filter noise, reconstruct corrupted data,
    and generate a reliable Golden Dataset.
    Includes built-in Z-Score Normalization to prevent Exploding Gradients.
    """
    
    def __init__(self, input_dim: int, seq_len: int = 10, hidden_dim: int = 64, latent_dim: int = 32, kernel_size: int = 3, learning_rate: float = 1e-3, device: Optional[str] = None):
        vae_model = VAETCN(
            input_channels=input_dim, 
            hidden_channels=hidden_dim,
            latent_channels=latent_dim,
            kernel_size=kernel_size
        )
        
        super().__init__(model=vae_model, name="CorrectorAgent")
        
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.kernel_size = kernel_size
        self.learning_rate = learning_rate
        
        if device is None:
            target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            target_device = torch.device(device)
            
        self.to(target_device)
        logger.info(f"[{self.name}] Initializing VAE-TCN on device: {self.device}")
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def inference(self, input_data: Any, batch_size: int = 128) -> np.ndarray:
        """
        Performs inference with dynamic Z-score normalization to match training distribution.
        """
        self.model.eval()
        original_dim = len(input_data.shape)
        
        if original_dim == 2:
            input_data = np.expand_dims(input_data, axis=-1)
        elif original_dim == 3:
            input_data = np.transpose(input_data, (0, 2, 1))
            
        dataset_size = input_data.shape[0]
        reconstructed_results = []
        
        try:
            x_tensor_cpu = torch.tensor(input_data, dtype=torch.float32)
            
            with torch.no_grad():
                for i in range(0, dataset_size, batch_size):
                    batch_x = x_tensor_cpu[i:i+batch_size].to(self.device)
                    
                    # --- APPLIED FIX: Z-Score Inference Normalization ---
                    # Prevents VAE from seeing extremely unscaled data
                    eps = 1e-5
                    mean = batch_x.mean(dim=(1, 2), keepdim=True)
                    std = batch_x.std(dim=(1, 2), keepdim=True)
                    batch_x_norm = (batch_x - mean) / (std + eps)
                    
                    recon_x_norm, mu, logvar = self.model(batch_x_norm)
                    
                    # Revert normalization back to physical scale
                    recon_x = (recon_x_norm * (std + eps)) + mean
                    
                    reconstructed_results.append(recon_x.cpu().numpy())
                    
                    del batch_x, batch_x_norm, recon_x, recon_x_norm, mu, logvar
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()
                        
            final_reconstruction = np.concatenate(reconstructed_results, axis=0)
            
            if original_dim == 2:
                final_reconstruction = np.squeeze(final_reconstruction, axis=-1)
            elif original_dim == 3:
                final_reconstruction = np.transpose(final_reconstruction, (0, 2, 1))
                
            return final_reconstruction
            
        except Exception as e:
            logger.error(f"[{self.name}] Error during inference: {str(e)}")
            raise

    def train_step(self, batch_data: Any) -> float:
        """
        Single training step with strictly enforced Z-Score Normalization
        to eradicate Exploding Gradients and inf/NaN loss outputs.
        """
        self.model.train()
        
        if not isinstance(batch_data, torch.Tensor):
            batch_data = torch.tensor(batch_data, dtype=torch.float32)
            
        if batch_data.dim() == 2:
            batch_data = batch_data.unsqueeze(-1)
        elif batch_data.dim() == 3:
            batch_data = batch_data.permute(0, 2, 1)
            
        batch_data = batch_data.to(self.device)
        
        if torch.isnan(batch_data).any() or torch.isinf(batch_data).any():
            batch_data = torch.nan_to_num(batch_data, nan=0.0, posinf=1.0, neginf=-1.0)
            
        # --- APPLIED FIX: Z-Score Training Normalization ---
        # This is the silver bullet against VAE explosion. 
        # Forces the physical sensor data to have Mean=0 and Std=1
        eps = 1e-5
        mean = batch_data.mean(dim=(1, 2), keepdim=True)
        std = batch_data.std(dim=(1, 2), keepdim=True)
        batch_data_norm = (batch_data - mean) / (std + eps)
        
        self.optimizer.zero_grad()
        
        # Feed normalized data to the network
        recon_x_norm, mu, logvar = self.model(batch_data_norm)
        
        # Calculate loss on the normalized space
        recon_loss = torch.nn.functional.mse_loss(recon_x_norm, batch_data_norm, reduction='mean')
        
        logvar_clamped = torch.clamp(logvar, min=-10.0, max=5.0)
        mu_clamped = torch.clamp(mu, min=-20.0, max=20.0)
        
        kl_div = -0.5 * torch.mean(1 + logvar_clamped - mu_clamped.pow(2) - logvar_clamped.exp())
        
        beta = 0.001
        loss = recon_loss + (beta * kl_div)
        
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(f"[{self.name}] NaNs/Infs detected in loss computation. Skipping optimization step.")
            return 1e6 # Return a highly penalized valid float instead of inf to keep AutoML running cleanly
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)
        self.optimizer.step()
        
        return loss.item()

    def train(self, data: np.ndarray, epochs: int = 50, batch_size: int = 64) -> Dict[str, list]:
        """Legacy helper method."""
        dataset_size = data.shape[0]
        history = {'loss': []}
        x_tensor_cpu = torch.tensor(data, dtype=torch.float32)
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            indices = torch.randperm(dataset_size)
            
            for i in range(0, dataset_size, batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_x = x_tensor_cpu[batch_indices]
                
                step_loss = self.train_step(batch_x)
                if step_loss < 1e5: # Only accumulate valid losses
                    epoch_loss += step_loss
                
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                    
            avg_loss = epoch_loss / max(1, (dataset_size // batch_size))
            history['loss'].append(avg_loss)
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(f"[{self.name}] Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")
                
        return history

    def save_weights(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        super().save_weights(path)