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
# File: src/services/tuner_service.py
# Author: Gabriel Moraes
# Date: 2025-12-25

import torch.optim as optim
from typing import Dict, Any

class TunerService:
    """
    Service dedicated to Hyperparameter Tuning logic.
    
    Responsibility (SRP):
    - Manage the Optimizer (Gradient Descent strategy).
    - Manage the Learning Rate Scheduler (Adaptive adjustment).
    - It encapsulates the 'Math' of how the network learns.
    """

    def __init__(self, learning_rate: float = 0.001, weight_decay: float = 1e-5):
        """
        Args:
            learning_rate: The initial step size for the optimizer.
            weight_decay: L2 regularization factor to prevent overfitting.
        """
        self.initial_lr = learning_rate
        self.weight_decay = weight_decay
        
        # State containers
        self.optimizer = None
        self.scheduler = None

    def initialize_optimizer(self, model_params) -> optim.Optimizer:
        """
        Initializes the Optimizer and Scheduler using the provided model parameters.
        
        Strategy:
        - Optimizer: AdamW (Standard for Transformers/LSTMs).
        - Scheduler: ReduceLROnPlateau (Lowers LR if validation loss stalls).
        
        Returns:
            The initialized PyTorch Optimizer.
        """
        # 1. Create Optimizer
        self.optimizer = optim.AdamW(
            model_params, 
            lr=self.initial_lr, 
            weight_decay=self.weight_decay
        )
        
        # 2. Create Scheduler
        # 'patience=5': If loss doesn't improve for 5 checks, reduce LR.
        # 'factor=0.5': Cut LR in half when reducing.
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5,
            verbose=False # Logging handled by OptimizerService
        )
        
        return self.optimizer

    def adjust_learning_rate(self, current_metric: float):
        """
        Steps the scheduler based on the validation metric (usually Loss).
        Should be called at the end of an evaluation cycle.
        """
        if self.scheduler:
            # The scheduler watches 'current_metric'. If it stops going down,
            # it triggers the LR reduction.
            self.scheduler.step(current_metric)

    def get_hyperparams(self) -> Dict[str, float]:
        """
        Returns the current state of hyperparameters for logging/dashboard.
        Useful to see if the LR has decayed.
        """
        current_lr = self.initial_lr
        if self.optimizer:
            # Get the LR from the first parameter group
            current_lr = self.optimizer.param_groups[0]['lr']
            
        return {
            "learning_rate": current_lr,
            "weight_decay": self.weight_decay
        }