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
# File: src/models/sinkhorn_cross_attention.py
# Author: Gabriel Moraes
# Date: 2026-03-08

"""
Sinkhorn Cross-Attention — SOTA Graph Matching Architecture.

A Siamese GATv2 encoder with bidirectional Cross-Attention and 
Sinkhorn doubly-stochastic normalization for optimal 1-to-1 assignment.

Paper References:
    - GATv2: Brody et al. "How Attentive are Graph Attention Networks?" (ICLR 2022)
    - SuperGlue: Sarlin et al. "SuperGlue: Learning Feature Matching" (CVPR 2020)
    - DGMC: Fey et al. "Deep Graph Matching Consensus" (ICLR 2020)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

from safetensors.torch import save_file, load_file

try:
    from torch_geometric.nn import GATv2Conv
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    GATv2Conv = None

# Enable Tensor Cores globally
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ─────────────────────────────────────────────────────────────────────────────
# AUXILIARY BLOCKS (Private)
# ─────────────────────────────────────────────────────────────────────────────

class _EdgeFeatureEncoder(nn.Module):
    """Projects raw feature vector to d_model embedding space."""

    def __init__(self, raw_dim: int, d_model: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(raw_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class _SiameseGATv2Block(nn.Module):
    """
    Multi-layer GATv2 with residual connections and LayerNorm.
    Shared weights — same instance processes both branches.
    """

    def __init__(self, d_model: int, n_heads: int, n_layers: int, dropout: float):
        super().__init__()

        if not PYG_AVAILABLE:
            raise ImportError("_SiameseGATv2Block requires torch_geometric.")

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = dropout

        for _ in range(n_layers):
            gat = GATv2Conv(
                in_channels=d_model,
                out_channels=d_model // n_heads,
                heads=n_heads,
                dropout=dropout,
                concat=True,
            )
            self.layers.append(gat)
            self.norms.append(nn.LayerNorm(d_model))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for gat, norm in zip(self.layers, self.norms):
            residual = x
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = gat(x, edge_index)
            x = F.elu(x)
            x = norm(x + residual)
        return x


class _CrossAttentionBlock(nn.Module):
    """Bidirectional Multi-Head Cross-Attention between two graphs."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = math.sqrt(self.head_dim)

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_out = nn.Linear(d_model, d_model)

        self.norm_a = nn.LayerNorm(d_model)
        self.norm_b = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def _attend(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        N, M = query.size(0), context.size(0)
        H, D = self.n_heads, self.head_dim

        Q = self.W_q(query).view(N, H, D).transpose(0, 1)
        K = self.W_k(context).view(M, H, D).transpose(0, 1)
        V = self.W_v(context).view(M, H, D).transpose(0, 1)

        attn = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.drop(attn)

        out = torch.matmul(attn, V)
        out = out.transpose(0, 1).contiguous().view(N, H * D)
        return self.W_out(out)

    def forward(
        self, emb_a: torch.Tensor, emb_b: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        a_out = self.norm_a(emb_a + self.drop(self._attend(emb_a, emb_b)))
        b_out = self.norm_b(emb_b + self.drop(self._attend(emb_b, emb_a)))
        return a_out, b_out


class _SinkhornHead(nn.Module):
    """Differentiable doubly-stochastic normalization for 1-to-1 matching."""

    def __init__(self, d_model: int, sinkhorn_iters: int, temperature: float):
        super().__init__()
        self.iters = sinkhorn_iters
        self.temp = temperature
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, emb_a: torch.Tensor, emb_b: torch.Tensor) -> torch.Tensor:
        a = self.proj(emb_a)
        b = self.proj(emb_b)
        scores = torch.matmul(a, b.t()) / self.temp

        # Iterative row/column log-normalization
        log_alpha = scores
        for _ in range(self.iters):
            log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=1, keepdim=True)
            log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=0, keepdim=True)

        return torch.exp(log_alpha)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN MODEL
# ─────────────────────────────────────────────────────────────────────────────

class SinkhornCrossAttention(nn.Module):
    """
    Siamese Graph Matching Network with Cross-Attention and Sinkhorn.

    Pipeline:
        Raw Features → Encoder → GATv2 (shared) → CrossAttention → Sinkhorn → Alignment

    Architecture follows the same pattern as gatv2_lite.py and itransformer.py:
    sub-blocks as private classes, single public class with __init__ + forward.
    """

    def __init__(
        self,
        raw_dim: int = 11,
        d_model: int = 64,
        n_heads: int = 4,
        n_gat_layers: int = 2,
        sinkhorn_iters: int = 10,
        dropout: float = 0.1,
        temperature: float = 0.1,
    ):
        """
        Args:
            raw_dim: Size of raw feature vector per Line Graph node.
            d_model: Latent embedding dimension.
            n_heads: Number of attention heads (GATv2 + CrossAttention).
            n_gat_layers: Number of shared GATv2 layers.
            sinkhorn_iters: Number of doubly-stochastic normalization passes.
            dropout: Dropout probability.
            temperature: Sinkhorn temperature (lower = sharper assignments).
        """
        super().__init__()

        self.encoder = _EdgeFeatureEncoder(raw_dim, d_model)
        self.gat = _SiameseGATv2Block(d_model, n_heads, n_gat_layers, dropout)
        self.cross_attn = _CrossAttentionBlock(d_model, n_heads, dropout)
        self.sinkhorn = _SinkhornHead(d_model, sinkhorn_iters, temperature)

    def forward(self, source_data, target_data) -> torch.Tensor:
        """
        Forward pass.

        Args:
            source_data: PyG Data (SUMO Line Graph sub-graph).
            target_data: PyG Data (Sensor Line Graph).

        Returns:
            alignment: [N_source, N_target] doubly-stochastic matrix.
        """
        # 1. Encode raw features
        src_x = self.encoder(source_data.x)
        tgt_x = self.encoder(target_data.x)

        # 2. GATv2 — shared weights, two branches
        src_emb = self.gat(src_x, source_data.edge_index)
        tgt_emb = self.gat(tgt_x, target_data.edge_index)

        # 3. Cross-Attention
        src_rich, tgt_rich = self.cross_attn(src_emb, tgt_emb)

        # 4. Sinkhorn
        return self.sinkhorn(src_rich, tgt_rich)

    # ─── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def get_best_matches(alignment: torch.Tensor) -> List[Tuple[int, int, float]]:
        """Extract best (source_idx, target_idx, confidence) from alignment."""
        vals, idxs = alignment.max(dim=1)
        return [
            (i, idxs[i].item(), vals[i].item())
            for i in range(alignment.size(0))
        ]

    @staticmethod
    def alignment_loss(predicted: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
        """Cross-entropy on alignment rows (each row is a classification)."""
        gt_labels = ground_truth.argmax(dim=1)
        log_pred = torch.log(predicted + 1e-10)
        return F.nll_loss(log_pred, gt_labels)

    # ─── Diploma (Safetensors Persistence) ────────────────────────────────

    def save_diploma(self, path: str):
        """Save trained weights as .safetensors."""
        state = {k: v.contiguous() for k, v in self.state_dict().items()}
        save_file(state, path)

    def load_diploma(self, path: str, device: str = "cpu"):
        """Load trained weights from .safetensors."""
        state = load_file(path, device=device)
        self.load_state_dict(state)
