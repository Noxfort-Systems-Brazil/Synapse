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
# File: src/optimization/callbacks.py
# Author: Gabriel Moraes
# Date: 2025-12-27

import numpy as np
import optuna
from typing import Optional, Callable, Dict, Any

from src.utils.logging_setup import logger

class DerivativeConvergenceCallback:
    """
    Mathematical Convergence Mechanism for Optuna Studies.
    
    Responsibility:
    - Monitors the optimization loss curve.
    - Calculus-based detection of plateaus (derivative ≈ 0).
    - Stops the study early to save GPU resources.
    """

    def __init__(self, slope_threshold: float = 1e-5, window_size: int = 25, 
                 min_trials: int = 35, signal_emitter: Optional[Callable[[Dict], None]] = None):
        """
        Args:
            slope_threshold: The minimum slope to consider as 'improving'.
            window_size: How many past trials to consider for regression.
            min_trials: Warm-up period before checking convergence.
            signal_emitter: Optional callback to update UI/Logs with trial data.
        """
        self.slope_threshold = slope_threshold
        self.window_size = window_size
        self.min_trials = min_trials
        self.signal_emitter = signal_emitter
        
        self.best_history = []
        self.current_best = float('inf')
    
    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
        """
        Executed at the end of every trial.
        """
        if trial.value is None:
            return

        # 1. Track Best Value
        if trial.value < self.current_best:
            self.current_best = trial.value
        
        self.best_history.append(self.current_best)

        # 2. Emit Telemetry (if UI is listening)
        if self.signal_emitter:
            self.signal_emitter({
                "trial": trial.number,
                "current_loss": trial.value,
                "best_loss": self.current_best,
                "params": trial.params
            })

        # 3. Warm-up Check
        if trial.number < self.min_trials:
            return

        # 4. Mathematical Convergence Check (Linear Regression on Best History)
        recent_history = self.best_history[-self.window_size:]
        x = np.arange(len(recent_history))
        y = np.array(recent_history)

        # Safety check for NaNs or Infs
        if not np.isfinite(y).all(): 
            return

        if len(y) > 1:
            # Calculate Slope (m)
            slope, _ = np.polyfit(x, y, 1)
            
            # If slope is flat (close to 0), we have converged.
            if abs(slope) < self.slope_threshold:
                logger.info(f"   [AutoML] 🛑 Mathematical Convergence. Slope ({slope:.1e}) ≈ 0.")
                study.stop()