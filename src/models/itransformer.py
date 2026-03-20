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
# File: src/models/itransformer.py
# Author: Gabriel Moraes
# Date: 2025-11-22

import torch
import torch.nn as nn
import torch.nn.functional as F

class iTransformer(nn.Module):
    """
    Inverted Transformer (iTransformer) for Global Traffic Fusion.
    
    Operates at Level 3 (Global Perception).
    
    Core Innovation:
    Instead of embedding time steps as tokens (like NLP Transformers), 
    the iTransformer embeds the entire time series of each variate (sensor/feature)
    as a single token.
    
    V2 (Spatio-Temporal Cross-Attention):
    After self-attention learns temporal inter-variate correlations, a cross-attention
    layer enriches each variate's representation with spatial context from the 
    Coordinator Agent (GATv2 graph neural network). This creates a true 
    spatio-temporal fusion — the "Gold Standard" for traffic prediction.
    
    Architecture:
    1. Input: [Batch, Seq_Len, Num_Variates] + optional spatial_context [Nodes, SpatialDim]
    2. Inversion: Treat 'Num_Variates' as tokens, 'Seq_Len' as features
    3. Self-Attention: Learn temporal correlations between variates
    4. Cross-Attention: Enrich with spatial topology context (Variates attend to Nodes)
    5. Projection: Map to prediction
    """

    def __init__(self, num_variates: int, seq_len: int, pred_len: int, 
                 d_model: int = 512, n_heads: int = 8, e_layers: int = 2, 
                 dropout: float = 0.1, spatial_dim: int = 32):
        """
        Initializes the iTransformer architecture.

        Args:
            num_variates: Number of sensors/features in the system (The 'Vocab Size').
            seq_len: Length of the input history lookback window.
            pred_len: Length of the forecast horizon.
            d_model: Latent dimension size.
            n_heads: Number of attention heads.
            e_layers: Number of encoder layers.
            dropout: Dropout probability.
            spatial_dim: Dimension of spatial embeddings from Coordinator (GATv2 out_channels).
        """
        super(iTransformer, self).__init__()
        
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model
        
        # --- 1. Embedding (Inverted) ---
        # Maps the raw time series (length seq_len) to the latent dimension (d_model).
        # Applied independently to each variate.
        self.enc_embedding = nn.Linear(seq_len, d_model)
        
        # --- 2. Encoder (Self-Attention: Temporal Inter-Variate) ---
        # Standard Transformer Encoder Stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            dim_feedforward=d_model * 4, 
            dropout=dropout,
            batch_first=True, # Critical: We arrange inputs as [Batch, Variates, d_model]
            activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)
        
        # --- 3. Cross-Attention (Spatial Context Injection) ---
        # Projects Coordinator's spatial embeddings to d_model space
        self.spatial_proj = nn.Linear(spatial_dim, d_model)
        self.spatial_norm = nn.LayerNorm(d_model)
        
        # Cross-Attention: Q=temporal tokens, K/V=spatial context
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.cross_norm = nn.LayerNorm(d_model)
        self.cross_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.cross_ffn_norm = nn.LayerNorm(d_model)
        
        # --- 4. Projection / Head ---
        # Maps the latent representation back to the time domain (prediction length).
        self.projector = nn.Linear(d_model, pred_len, bias=True)
        
        # Normalization usually helps convergence
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, spatial_context: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass logic.

        Args:
            x: Input tensor [Batch, Seq_Len, Num_Variates].
            spatial_context: Optional spatial embeddings from Coordinator 
                           [Batch, Nodes, SpatialDim] or [Nodes, SpatialDim].
                           When provided, cross-attention enriches temporal features
                           with spatial topology awareness.
            
        Returns:
            Forecast tensor [Batch, Pred_Len, Num_Variates].
        """
        # x shape: [Batch, Seq_Len, Num_Variates]
        
        # 1. Inversion (Transpose)
        # We want: [Batch, Num_Variates, Seq_Len]
        # So that 'Num_Variates' becomes the 'Sequence' for the Transformer
        x_enc = x.permute(0, 2, 1) 
        
        # 2. Embedding
        # Linear layer applies to the last dim (Seq_Len) -> d_model
        # Result: [Batch, Num_Variates, d_model]
        enc_out = self.enc_embedding(x_enc)
        enc_out = self.layer_norm(enc_out)
        
        # 3. Self-Attention (Temporal Inter-Variate Correlations)
        # Learns which sensors correlate with each other over time
        # Result: [Batch, Num_Variates, d_model]
        enc_out = self.encoder(enc_out)
        
        # 4. Cross-Attention (Spatial Context Injection)
        # Each variate "asks" the spatial graph: "how do my neighbors affect me?"
        if spatial_context is not None:
            # Ensure batch dimension
            if spatial_context.dim() == 2:
                spatial_context = spatial_context.unsqueeze(0)  # [1, Nodes, SpatialDim]
            
            # Expand spatial context to match batch size
            if spatial_context.size(0) == 1 and enc_out.size(0) > 1:
                spatial_context = spatial_context.expand(enc_out.size(0), -1, -1)
            
            # Project spatial embeddings to d_model space
            spatial_kv = self.spatial_proj(spatial_context)  # [B, Nodes, d_model]
            spatial_kv = self.spatial_norm(spatial_kv)
            
            # Cross-Attention: Q=temporal, K=V=spatial
            cross_out, _ = self.cross_attn(
                query=enc_out,      # [B, Variates, d_model]
                key=spatial_kv,     # [B, Nodes, d_model]
                value=spatial_kv    # [B, Nodes, d_model]
            )
            enc_out = self.cross_norm(enc_out + cross_out)  # Residual connection
            
            # Feed-forward after cross-attention
            enc_out = self.cross_ffn_norm(enc_out + self.cross_ffn(enc_out))
        
        # 5. Projection (Prediction)
        # Maps d_model -> Pred_Len
        # Result: [Batch, Num_Variates, Pred_Len]
        dec_out = self.projector(enc_out)
        
        # 6. Reversion (Transpose Back)
        # We want: [Batch, Pred_Len, Num_Variates] to match standard output format
        dec_out = dec_out.permute(0, 2, 1)
        
        return dec_out