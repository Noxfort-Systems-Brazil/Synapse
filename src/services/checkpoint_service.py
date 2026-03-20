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
# File: src/services/checkpoint_service.py
# Author: Gabriel Moraes
# Date: 2025-12-25

import torch
import os
import shutil
from typing import Dict, Any
from src.utils.logging_setup import logger

class CheckpointService:
    """
    Service dedicated to File I/O for Model Weights.
    
    Responsibility (SRP):
    - Handle secure reading/writing of checkpoint files.
    - Ensure 'Atomic Saves' (write to temp -> move) to prevent file corruption
      if the system crashes mid-write.
    """

    def __init__(self, checkpoint_dir: str = "checkpoints/"):
        """
        Args:
            checkpoint_dir: Directory where models will be stored.
        """
        self.checkpoint_dir = checkpoint_dir
        
        # Ensure the directory exists upon initialization
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            logger.info(f"[Checkpoint] 📁 Created checkpoint directory: {self.checkpoint_dir}")

    def save_checkpoint(self, state_dict: Dict[str, Any], filename: str = "best_model.pth"):
        """
        Saves the model state securely.
        
        Strategy:
        1. Write to a temporary file (e.g., model.pth.tmp).
        2. Once write is 100% complete, rename/move it to the final name.
        This prevents a corrupted 'half-written' file if power fails.
        """
        try:
            final_path = os.path.join(self.checkpoint_dir, filename)
            temp_path = final_path + ".tmp"
            
            # 1. Save to temp file
            torch.save(state_dict, temp_path)
            
            # 2. Atomic Move (Overwrite)
            shutil.move(temp_path, final_path)
            
            logger.info(f"[Checkpoint] 💾 Saved successfully to {final_path}")

        except Exception as e:
            logger.error(f"[Checkpoint] ❌ Save failed for {filename}: {e}")

    def load_checkpoint(self, model: torch.nn.Module, filename: str = "best_model.pth") -> bool:
        """
        Loads weights into the provided model instance.
        
        Args:
            model: The PyTorch model instance (nn.Module) to update.
            filename: The filename to load.
            
        Returns:
            bool: True if loaded successfully, False otherwise.
        """
        path = os.path.join(self.checkpoint_dir, filename)
        
        if os.path.exists(path):
            try:
                # Load to CPU to avoid CUDA errors if GPU changed/missing
                checkpoint = torch.load(path, map_location="cpu")
                
                # Logic to handle different saving formats
                # Case A: Saved as a full dict (recommended) -> keys: 'model_state_dict', 'optimizer', etc.
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                
                # Case B: Saved strictly as state_dict
                else:
                    model.load_state_dict(checkpoint)
                    
                logger.info(f"[Checkpoint] 📂 Loaded weights from {filename}")
                return True

            except RuntimeError as e:
                logger.error(f"[Checkpoint] ⚠️ Architecture Mismatch or Corrupt File: {e}")
            except Exception as e:
                logger.error(f"[Checkpoint] ⚠️ Load failed: {e}")
        else:
            logger.debug(f"[Checkpoint] File {filename} not found. Starting with fresh weights.")
            
        return False