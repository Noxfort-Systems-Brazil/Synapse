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
# File: src/optimization/strategies_semantic.py
# Author: Gabriel Moraes
# Date: 2026-03-02

import torch
import numpy as np
from typing import Any
import logging

# Import Agents ONLY - Strict Encapsulation
from src.agents.linguist_agent import LinguistAgent
from src.agents.peak_classifier_agent import PeakClassifierAgent

logger = logging.getLogger("Synapse.Strategies.Semantic")

class SemanticStrategies:
    """
    Optimization Strategies for Semantic Agents (Language & Classification).
    
    Contains:
    1. Linguist Strategy (NLP / Neuro-Symbolic)
    2. Classifier Strategy (Peak Detection via encapsulated TimesNet Agent)
    """

    @staticmethod
    def linguist_strategy(trial, data: Any, device: torch.device) -> float:
        """
        Optimizes the Linguist Agent (DistilRoBERTa + TCN-AE).
        Focuses on Reconstruction Loss of the physics/numeric signals (TCN-AE),
        since the Semantic/Grammar Transformer is frozen during HPO.
        """
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        
        try:
            agent = LinguistAgent(learning_rate=lr)
            agent.to(device)
        except Exception as e:
            logger.error(f"[Linguist] Init Error: {e}")
            raise e

        # FIXED: Provide numeric signal data instead of strings.
        # The LinguistAgent only trains the TCN-AE (physics) part.
        if isinstance(data, (np.ndarray, torch.Tensor)):
            train_data = data[:128] if len(data) > 128 else data 
        else:
            # Fallback to a valid numerical dummy tensor if no real data is provided
            train_data = np.random.randn(16, 1, 60).astype(np.float32)

        try:
            total_loss = 0.0
            steps = 5 
            
            for _ in range(steps):
                loss = agent.train_step(train_data)
                total_loss += loss
            
            avg_loss = total_loss / steps
            return avg_loss

        except Exception as e:
            logger.warning(f"[Linguist] Training failed: {e}")
            return float('inf')
        finally:
            del agent
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @staticmethod
    def classifier_strategy(trial, inputs: np.ndarray, targets: np.ndarray, device: torch.device) -> float:
        """
        Optimizes the Peak Classifier Agent.
        Delegates model instantiation, loss calculation, and optimization entirely to the Agent.
        """
        # 1. Hyperparameters customized for Edge Hardware constraints (VRAM Limit: ~6GB)
        # REDUCED search space to prevent CUDA Out of Memory issues
        d_model = trial.suggest_categorical("d_model", [16, 32, 64]) # Reduced from [32, 64, 128]
        e_layers = trial.suggest_int("e_layers", 1, 2)               # Reduced from 1-3
        top_k = trial.suggest_int("top_k", 1, 3)                     # Reduced from 1-5
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        
        # 2. Input Validation
        if inputs is None or len(inputs) == 0:
            return float('inf')
            
        # 3. Instantiate Agent (The agent handles its internal TimesNet creation)
        try:
            itransformer_config = {
                'num_variates': 2,
                'seq_len': 96,
                'pred_len': 96,
                'e_layers': e_layers,
                'd_model': d_model,
                'learning_rate': lr
            }
            timesnet_config = {
                'seq_len': 96,
                'pred_len': 96,
                'e_layers': e_layers,
                'd_model': d_model,
                'd_ff': d_model * 4,
                'top_k': top_k
            }
            
            agent = PeakClassifierAgent(
                itransformer_config=itransformer_config,
                timesnet_config=timesnet_config
            )
        except Exception as e:
            logger.error(f"[Classifier] Init Error: {e}")
            return float('inf')

        # 4. Training Loop delegated to the Agent
        try:
            total_loss = 0.0
            epochs = 5
            
            for _ in range(epochs):
                # Agent internally handles AMP context, tensors mapping, and optimizer steps
                loss = agent.train_step(inputs, targets)
                total_loss += loss
            
            return total_loss / epochs

        except Exception as e:
            logger.warning(f"[Classifier] Training loop error: {e}")
            return float('inf')
        finally:
            del agent
            if torch.cuda.is_available():
                torch.cuda.empty_cache()