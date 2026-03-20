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
# File: src/models/timesnet.py
# Author: Gabriel Moraes
# Date: 2026-03-02

import math
import torch
import torch.nn as nn
import torch.fft
import torch.nn.functional as F
from typing import Tuple

# Enable TensorFloat32 (TF32) for Tensor Cores on Ampere+ GPUs
torch.set_float32_matmul_precision('high')

class InceptionBlock(nn.Module):
    """
    Inception block for 2D convolutions to capture variations across different periods.
    """
    def __init__(self, in_channels: int, out_channels: int, num_kernels: int = 3):
        super(InceptionBlock, self).__init__()
        self.num_kernels = num_kernels
        self.kernels = nn.ModuleList()
        
        for i in range(self.num_kernels):
            kernel_size = 2 * i + 1
            padding = kernel_size // 2
            self.kernels.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
                    nn.GELU()
                )
            )
        
        self.projection = nn.Conv2d(out_channels * self.num_kernels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.cat(res_list, dim=1)
        return self.projection(res)

class TimesBlock(nn.Module):
    """
    Core block of TimesNet: FFT frequency extraction, 1D-to-2D reshape, and 2D Inception.
    """
    def __init__(self, seq_len: int, pred_len: int, top_k: int, d_model: int, d_ff: int):
        super(TimesBlock, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.k = top_k
        
        self.conv = nn.Sequential(
            InceptionBlock(in_channels=d_model, out_channels=d_ff),
            nn.GELU(),
            InceptionBlock(in_channels=d_ff, out_channels=d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, channels = x.size()
        
        # cuFFT in half-precision strictly requires power-of-2 sequence lengths.
        # We calculate the next power of 2 to pad the signal dynamically before FFT.
        N = 2 ** math.ceil(math.log2(time_steps)) if time_steps > 0 else 1
        
        if N != time_steps:
            # Pad the time dimension (dim=1). 
            # F.pad format for 3D is: (channels_left, channels_right, time_left, time_right)
            x_fft = F.pad(x, (0, 0, 0, N - time_steps))
        else:
            x_fft = x
            
        # 1. FFT Operation
        # FIXED: We forcefully disable AMP (Autocast) locally for this block.
        # This prevents PyTorch from forcing 'ComplexHalf' (experimental) and crashing.
        device_type = x.device.type if x.device.type in ['cuda', 'cpu'] else 'cuda'
        
        with torch.autocast(device_type=device_type, enabled=False):
            # Now it runs purely in FP32 and safely generates complex64
            x_float32 = x_fft.to(torch.float32)
            xf = torch.fft.rfft(x_float32, dim=1)
            
            # Calculate amplitude
            frequency_list = abs(xf).mean(0).mean(-1)
            frequency_list[0] = 0 # Remove DC component
        
        # Top-k frequencies (Safeguard against extremely small inputs)
        k = min(self.k, frequency_list.shape[0])
        _, top_list = torch.topk(frequency_list, k)
        top_list = top_list.detach().cpu().numpy()
        
        # 2. Reshape and 2D Conv
        res = torch.zeros_like(x)
        
        for freq in top_list:
            # Calculate the period using the padded length N to maintain frequency accuracy
            period = N // freq if freq > 0 else N
            
            # Prevent division by zero mathematically
            if period <= 0:
                period = 1
                
            pad_len = (period - (time_steps % period)) % period
            
            # Pad to make it strictly divisible by period
            x_padded = torch.cat([x, x[:, -pad_len:, :]], dim=1)
            
            # Reshape 1D to 2D
            length = x_padded.size(1)
            x_2d = x_padded.reshape(batch_size, length // period, period, channels)
            
            # Permute for Conv2d: (Batch, Channels, Height, Width)
            x_2d = x_2d.permute(0, 3, 1, 2)
            
            # Apply 2D Convolutions
            out_2d = self.conv(x_2d)
            
            # Reshape 2D back to 1D
            out_2d = out_2d.permute(0, 2, 3, 1).reshape(batch_size, length, channels)
            
            # Truncate padding and accumulate
            res = res + out_2d[:, :time_steps, :]
            
        return res

class TimesNet(nn.Module):
    """
    Main TimesNet architecture replacing standard MLPs for complex temporal feature extraction.
    Designed to work natively with PyTorch AMP (torch.cuda.amp.autocast).
    """
    def __init__(self, 
                 seq_len: int = 96, 
                 pred_len: int = 24, 
                 enc_in: int = 1, 
                 d_model: int = 64, 
                 d_ff: int = 64, 
                 e_layers: int = 2, 
                 top_k: int = 3, 
                 dropout: float = 0.1):
        super(TimesNet, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        self.embedding = nn.Linear(enc_in, d_model)
        
        self.blocks = nn.ModuleList([
            TimesBlock(seq_len, pred_len, top_k, d_model, d_ff)
            for _ in range(e_layers)
        ])
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.projection = nn.Linear(d_model, enc_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [Batch, Sequence_Length, Features]
        
        x_enc = self.embedding(x)
        
        for block in self.blocks:
            x_enc = self.layer_norm(block(x_enc))
            x_enc = self.dropout(x_enc)
            
        out = self.projection(x_enc)
        
        return out

    @torch.autocast(device_type="cuda", dtype=torch.float16)
    def infer_amp(self, x: torch.Tensor) -> torch.Tensor:
        """
        Helper method to run inference enforcing AMP (Automatic Mixed Precision).
        """
        return self.forward(x)