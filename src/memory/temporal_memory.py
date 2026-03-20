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
# File: src/memory/temporal_memory.py
# Author: Gabriel Moraes
# Date: 2025-12-25

import torch
import numpy as np
from collections import deque
from typing import List, Union, Optional

class TemporalMemory:
    """
    Manages a sliding window of temporal data (Time Series).
    
    Refactored V3 (Robust Interface):
    - Implements 'max_len' to match GraphManager's factory.
    - Provides conversion methods (Tensor/Numpy).
    - Supports 'rollback' for Physics Fallback recovery.
    """

    def __init__(self, feature_dim: int, max_len: int = 60):
        """
        Args:
            feature_dim: Number of features per time step (e.g., 1 for Flow).
            max_len: Size of the sliding window (Sequence Length).
        """
        self.feature_dim = feature_dim
        self.max_len = max_len
        
        # Internal buffer (Deque automatically handles sliding window eviction)
        self.buffer = deque(maxlen=max_len)
        
        # Device management (defaults to CPU, agents move data to GPU as needed)
        self.device = torch.device("cpu")

    def push(self, data: Union[List[float], np.ndarray, float]):
        """
        Adds a new time step to the memory.
        Automatically removes the oldest item if full.
        """
        # Normalize input to List[float]
        if isinstance(data, (int, float)):
            clean_data = [float(data)]
        elif isinstance(data, np.ndarray):
            clean_data = data.flatten().tolist()
        elif isinstance(data, list):
            clean_data = data
        else:
            # Fallback for unknown types
            clean_data = [0.0] * self.feature_dim

        # Ensure dimension consistency
        if len(clean_data) != self.feature_dim:
            # Padding or Truncating logic could go here
            # For now, strict check or silent adaptation
            if len(clean_data) < self.feature_dim:
                clean_data += [0.0] * (self.feature_dim - len(clean_data))
            else:
                clean_data = clean_data[:self.feature_dim]

        self.buffer.append(clean_data)

    def is_ready(self) -> bool:
        """Returns True if the buffer is full (Warmed Up)."""
        return len(self.buffer) >= self.max_len

    def get_tensor(self) -> torch.Tensor:
        """
        Returns the buffer as a PyTorch Tensor.
        Shape: (Batch=1, Seq_Len, Features)
        """
        if not self.buffer:
            return torch.zeros((1, self.max_len, self.feature_dim))
            
        # Convert deque to list of lists
        data_list = list(self.buffer)
        
        # Create Tensor
        # Shape: (Seq, Feat) -> (1, Seq, Feat)
        tensor = torch.tensor(data_list, dtype=torch.float32)
        return tensor.unsqueeze(0)

    def get_numpy(self) -> np.ndarray:
        """
        Returns the buffer as a Numpy Array.
        Shape: (Seq_Len, Features)
        """
        if not self.buffer:
            return np.zeros((self.max_len, self.feature_dim))
        return np.array(self.buffer, dtype=np.float32)

    def rollback(self, steps: int):
        """
        Removes the last N steps.
        Used when the Physics Engine detects that synthetic data was generated
        during a sensor outage, and now we want to revert to sync with real data.
        """
        for _ in range(min(steps, len(self.buffer))):
            self.buffer.pop()

    def clear(self):
        """Resets the memory."""
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)