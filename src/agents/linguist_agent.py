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
# File: src/agents/linguist_agent.py
# Author: Gabriel Moraes
# Date: 2026-03-02

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from typing import Dict, Any, List
from transformers import AutoTokenizer

from src.agents.base_agent import BaseAgent
from src.models.neuro_symbolic import NeuroSymbolicModel

logger = logging.getLogger("Synapse.Agents.Linguist")

class LinguistAgent(BaseAgent):
    """
    The Linguist Agent ('O Validador Semântico-Físico').
    
    Refactored to use the unified NeuroSymbolicModel.
    This agent processes semantic logs (text), embeds them using a frozen 
    Transformer, and reconstructs the embeddings using a Temporal Convolutional 
    Autoencoder (TCN-AE). Anomalies in reconstruction indicate illogical or 
    hallucinated sensor logs.
    """

    def __init__(self, model_name: str = "distilroberta-base", learning_rate: float = 1e-4):
        # 1. Initialize the unified Neuro-Symbolic Model
        model = NeuroSymbolicModel(model_name=model_name)
        
        # 2. Call BaseAgent constructor
        super().__init__(model=model, name="LinguistAgent")
        
        # 3. Setup Tokenizer for the Transformer backbone
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 4. Setup Optimizer ONLY for the Autoencoder parts (encoder & decoder)
        # We access the ModuleDict 'ae' created inside NeuroSymbolicModel
        self.optimizer = optim.Adam(self.model.ae.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # Threshold for contradiction detection
        self.anomaly_threshold = 0.85 

    def inference(self, input_data: Any) -> Dict[str, Any]:
        """
        Runs the semantic anomaly detection loop.
        
        Args:
            input_data: Dict containing 'text' (str) or a direct string.
        Returns:
            Dict containing 'is_valid', 'error_score', and 'semantic_vector'.
        """
        self.model.eval()
        
        if isinstance(input_data, dict):
            text_claim = input_data.get("text", "")
        else:
            text_claim = str(input_data)
            
        # Tokenize the text for the Transformer
        inputs = self.tokenizer(
            text_claim, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=128
        )
        
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        with torch.no_grad():
            reconstruction, original = self.model(input_ids, attention_mask)
            
            # Calculate Reconstruction Error
            error = torch.nn.functional.mse_loss(reconstruction, original, reduction='none')
            mean_error = error.mean().item()
            
        # If the error surpasses the threshold, the log is grammatically sound 
        # but semantically impossible given the context learned by the AE.
        is_valid = bool(mean_error < self.anomaly_threshold)
        
        return {
            "is_valid": is_valid,
            "error_score": mean_error,
            "semantic_vector": original.cpu().numpy()
        }

    def train_step(self, batch_data: Any) -> float:
        """
        Optimizes ONLY the Temporal Autoencoder part of the model.
        """
        self.model.train()
        
        # HPO Fallback: Optuna (strategies_semantic.py) might pass dummy numeric tensors
        # testing the training loop structure. We handle this to avoid tokenizer crashes.
        if isinstance(batch_data, (np.ndarray, torch.Tensor)):
            batch_size = batch_data.shape[0] if len(batch_data.shape) > 0 else 16
            texts = ["Dummy sensor log for structural HPO optimization."] * batch_size
        elif isinstance(batch_data, list) and isinstance(batch_data[0], str):
            texts = batch_data
        elif isinstance(batch_data, dict) and "text" in batch_data:
            texts = batch_data["text"]
            if isinstance(texts, str): 
                texts = [texts]
        else:
            texts = ["Standard operational traffic log."]
            
        inputs = self.tokenizer(
            texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=128
        )
        
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        self.optimizer.zero_grad()
        
        # Forward pass returning (reconstructed_embeddings, original_embeddings)
        reconstruction, original = self.model(input_ids, attention_mask)
        
        loss = self.criterion(reconstruction, original)
        loss.backward()
        
        # Gradient clipping to ensure stability
        torch.nn.utils.clip_grad_norm_(self.model.ae.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return loss.item()

    def calibrate_threshold(self, validation_texts: List[str]):
        """
        Auto-adjusts the anomaly threshold based on a baseline distribution of normal logs.
        """
        logger.info(f"[{self.name}] Calibrating semantic anomaly threshold...")
        self.model.eval()
        
        if not validation_texts:
            logger.warning(f"[{self.name}] No validation texts provided for calibration.")
            return
            
        inputs = self.tokenizer(
            validation_texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=128
        )
        
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        with torch.no_grad():
             reconstruction, original = self.model(input_ids, attention_mask)
             errors = torch.nn.functional.mse_loss(reconstruction, original, reduction='none')
             
             # Calculate mean error per sequence in batch
             seq_errors = errors.mean(dim=(1, 2)).cpu().numpy()
             
             # Calculate 95th percentile as the new threshold
             self.anomaly_threshold = float(np.percentile(seq_errors, 95))
             
        logger.info(f"[{self.name}] New semantic threshold set to: {self.anomaly_threshold:.4f}")