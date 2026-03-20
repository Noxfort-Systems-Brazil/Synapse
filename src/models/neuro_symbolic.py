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
# File: src/models/neuro_symbolic.py
# Author: Gabriel Moraes
# Date: 2026-02-14

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class NeuroSymbolicModel(nn.Module):
    """
    The Neuro-Symbolic Model backbone for the Linguist Agent.
    
    Architecture:
    1. Symbolic/Semantic Extractor: Transformer (DistilRoBERTa) - Frozen by default.
    2. Neural Reasoner: TCN-Autoencoder (Temporal Convolutional AE).
    
    The model learns to reconstruct 'normal' semantic patterns. High reconstruction 
    error indicates a semantic anomaly (e.g., a log message that makes grammatical 
    sense but appears in an impossible context).
    """

    def __init__(self, model_name: str = "distilroberta-base", freeze_transformer: bool = True, latent_dim: int = 64):
        super(NeuroSymbolicModel, self).__init__()
        
        # 1. Semantic Backbone (Transformer)
        # We load config first to get hidden size without downloading model if not needed immediately
        self.config = AutoConfig.from_pretrained(model_name)
        self.hidden_size = self.config.hidden_size
        
        self.transformer = AutoModel.from_pretrained(model_name)
        
        if freeze_transformer:
            for param in self.transformer.parameters():
                param.requires_grad = False
                
        # 2. Neural Reasoner (TCN Autoencoder)
        # We process the sequence of embeddings [Batch, Seq_Len, Hidden]
        # Using 1D Convolutions to capture local context in the sentence structure
        
        # Encoder: Project down to latent space
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=self.hidden_size, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Conv1d(in_channels=256, out_channels=latent_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Decoder: Reconstruct original embeddings
        self.decoder = nn.Sequential(
            nn.Conv1d(in_channels=latent_dim, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Conv1d(in_channels=256, out_channels=self.hidden_size, kernel_size=3, padding=1)
        )
        
        # Grouping for easy optimization in Agent
        self.ae = nn.ModuleDict({
            "encoder": self.encoder,
            "decoder": self.decoder
        })

    def forward(self, input_ids, attention_mask):
        """
        Forward pass through Transformer -> TCN-AE.
        
        Args:
            input_ids: [Batch, Seq_Len]
            attention_mask: [Batch, Seq_Len]
            
        Returns:
            reconstruction: [Batch, Seq_Len, Hidden]
            original_embeddings: [Batch, Seq_Len, Hidden]
        """
        # 1. Get Semantic Embeddings
        # Transformer outputs: last_hidden_state [Batch, Seq_Len, Hidden]
        with torch.set_grad_enabled(not self.transformer.training): # Respect freeze logic
            outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        
        original_embeddings = outputs.last_hidden_state
        
        # 2. Prepare for TCN (Conv1d expects [Batch, Channels, Seq_Len])
        # Permute: [Batch, Seq, Hidden] -> [Batch, Hidden, Seq]
        x = original_embeddings.permute(0, 2, 1)
        
        # 3. Autoencoder Pass
        latent = self.encoder(x)
        reconstructed_x = self.decoder(latent)
        
        # 4. Restore shape: [Batch, Hidden, Seq] -> [Batch, Seq, Hidden]
        reconstruction = reconstructed_x.permute(0, 2, 1)
        
        return reconstruction, original_embeddings