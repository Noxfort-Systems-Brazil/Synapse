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
# File: src/utils/batch_inference.py
# Author: Gabriel Moraes
# Date: 2025-12-26

import torch
import gc
import numpy as np
import logging
from typing import Optional, Callable

# Configura o logger para este módulo
logger = logging.getLogger(__name__)

class BatchInferenceEngine:
    """
    Infrastructure Component.
    
    Responsibility: 
    - Execute ANY PyTorch model on ANY large dataset safely.
    - Encapsulates VRAM Management (CUDA Cache clearing).
    - Handles Data Slicing (Sliding Window) to prevent OOM errors.
    
    Usage:
    engine = BatchInferenceEngine(device, batch_size=256)
    result = engine.run(model, raw_data, seq_len=24)
    """

    def __init__(self, device: torch.device, batch_size: int = 256):
        """
        Args:
            device: The target torch device (CPU/CUDA).
            batch_size: Number of windows to process at once. Decrease if OOM persists.
        """
        self.device = device
        self.batch_size = batch_size

    def run(self, 
            model: torch.nn.Module, 
            data_matrix: np.ndarray, 
            seq_len: int, 
            progress_callback: Optional[Callable[[int], None]] = None) -> np.ndarray:
        """
        Generic execution method for time-series reconstruction/inference.
        
        Args:
            model: The neural network (already loaded).
            data_matrix: The raw input numpy array (N samples, Features).
            seq_len: The sequence length the model expects (Sliding Window size).
            progress_callback: Optional function (accepts int 0-100) to report progress.
            
        Returns:
            np.ndarray: Reconstructed matrix aligned with input size (same N samples).
        """
        num_samples = len(data_matrix)
        
        # Validation
        if num_samples < seq_len:
            logger.warning(f"[BatchEngine] Data too short ({num_samples}) for seq_len {seq_len}.")
            return np.zeros_like(data_matrix)

        reconstructed_list = []
        
        # Prepare Model
        model.eval()
        model.to(self.device)
        
        logger.info(f"[BatchEngine] Processing {num_samples} samples on {self.device} with batch_size {self.batch_size}...")

        # Disable Gradient Calculation to save memory
        with torch.no_grad():
            # Main Loop: Slice data into mini-batches
            for i in range(0, num_samples - seq_len + 1, self.batch_size):
                
                # 1. Define Slicing Indices
                # We want to create a batch of sliding windows
                batch_indices = range(i, min(i + self.batch_size, num_samples - seq_len + 1))
                
                batch_windows = []
                for idx in batch_indices:
                    # Create window [idx : idx + seq_len]
                    window = data_matrix[idx : idx + seq_len]
                    batch_windows.append(window)
                
                if not batch_windows: 
                    break
                
                # 2. Upload to GPU
                # Shape: (Batch, Seq_Len, Features)
                tensor_batch = torch.tensor(np.array(batch_windows)).float().to(self.device)
                
                # 3. Inference
                # Run the forward pass
                recon_batch = model(tensor_batch)
                
                # 4. Extract Strategy (Last Point)
                # Assuming output is (Batch, Seq, Feat), we take the last time step 
                # as the "corrected" value for the current timestamp.
                last_point_recon = recon_batch[:, -1, :]
                
                # 5. Download to CPU & Store
                reconstructed_list.append(last_point_recon.cpu().numpy())
                
                # 6. Immediate Cleanup
                # Delete tensor references to allow PyTorch to reuse memory block
                del tensor_batch, recon_batch
                
                # 7. Deep Clean (Periodic)
                # Every ~5000 samples, force Python GC and Empty CUDA Cache
                # This prevents "Memory Fragmentation" over long runs.
                if i % 5120 == 0:
                    self._force_cleanup()
                    if progress_callback:
                        percent = int((i / num_samples) * 100)
                        progress_callback(percent)

        # 8. Reassemble
        if not reconstructed_list:
            return np.zeros_like(data_matrix)

        reconstructed_matrix = np.concatenate(reconstructed_list, axis=0)
        
        # 9. Time Alignment (Padding)
        # Sliding window reduces output size by (seq_len - 1).
        # We assume the model "predicts" or "corrects" the end of the window.
        # So we pad the beginning with original data to match length.
        pad_width = num_samples - len(reconstructed_matrix)
        
        if pad_width > 0:
            padding = data_matrix[:pad_width]
            final_matrix = np.vstack([padding, reconstructed_matrix])
        else:
            final_matrix = reconstructed_matrix
            
        # Final cleanup before returning
        self._force_cleanup()
        
        return final_matrix

    def _force_cleanup(self):
        """
        Executes aggressive memory release protocols.
        """
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()