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
# File: src/managers/pbt_manager.py
# Author: Gabriel Moraes
# Date: 2025-11-23

import random
import logging
import numpy as np
from collections import deque
from typing import Dict, List
from src.agents.specialist_agent import SpecialistAgent

class PBTManager:
    """
    Population Based Training (PBT) Manager.
    
    Responsibility:
    - Manages the evolution of the SpecialistAgents (TCNs) during Online Operation.
    - Implements 'Exploit' (Copy) and 'Explore' (Mutate) evolutionary strategies.
    - UPDATED: Uses 'Loss Plateau' detection to trigger evolution dynamically.
      Evolution only happens when the population stops learning (stagnates).
    """

    def __init__(self, exploit_ratio: float = 0.2, patience: int = 10, threshold: float = 1e-3, cooldown: int = 30):
        """
        Args:
            exploit_ratio: Fraction of bottom agents to replace (e.g., 0.2).
            patience: Number of steps to wait for improvement before declaring plateau.
            threshold: Minimum loss improvement required to reset patience.
            cooldown: Number of cycles to wait AFTER an evolution before checking again.
        """
        self.exploit_ratio = exploit_ratio
        self.threshold = threshold
        self.cooldown_period = cooldown
        self.current_cooldown = 0
        
        self.logger = logging.getLogger("PBTManager")
        
        # Plateau Detection State
        self.loss_history = deque(maxlen=patience)
        self.best_loss = float('inf')
        self.stagnation_counter = 0

    def step(self, population: Dict[str, SpecialistAgent]):
        """
        Executes one evolutionary check.
        Evolution is only triggered if the population's learning has plateaued.
        
        Args:
            population: Dictionary {source_id: Agent}.
        """
        # 1. Check Cooldown (Give agents time to adapt after a change)
        if self.current_cooldown > 0:
            self.current_cooldown -= 1
            return

        # 2. Gather Population Metrics
        # Filter valid agents (must have some training steps)
        valid_agents = [a for a in population.values() if a.steps > 5]
        if len(valid_agents) < 2:
            return

        # Calculate Population Mean Loss
        current_avg_loss = np.mean([a.running_loss for a in valid_agents])
        
        # 3. Check for Plateau
        if current_avg_loss < (self.best_loss - self.threshold):
            # Improvement detected
            self.best_loss = current_avg_loss
            self.stagnation_counter = 0
        else:
            # No significant improvement
            self.stagnation_counter += 1
            
        # Log status occasionally
        # if self.stagnation_counter % 5 == 0:
        #     self.logger.debug(f"[PBT Monitor] Avg Loss: {current_avg_loss:.4f} | Stagnation: {self.stagnation_counter}/{self.loss_history.maxlen}")

        # 4. Trigger Evolution if Stagnated
        if self.stagnation_counter >= self.loss_history.maxlen:
            self.logger.info(f"[PBT] Plateau Detected (Loss ~{current_avg_loss:.4f}). Triggering Evolution...")
            self._evolve(population)
            
            # Reset State
            self.stagnation_counter = 0
            self.best_loss = current_avg_loss # Reset baseline
            self.current_cooldown = self.cooldown_period # Start cooldown

    def _evolve(self, population: Dict[str, SpecialistAgent]):
        """Performs the Exploit-and-Explore routine."""
        # Rank Population
        valid_agents = [(sid, agent) for sid, agent in population.items() if agent.steps > 5]
        ranked_population = sorted(valid_agents, key=lambda item: item[1].running_loss)
        
        n = len(ranked_population)
        cutoff_index = int(n * self.exploit_ratio)
        if cutoff_index == 0: return

        winners = ranked_population[:cutoff_index]
        losers = ranked_population[-cutoff_index:]
        
        self.logger.info(f"[PBT] Evolution: Replacing {len(losers)} bottom agents with genetics from top {len(winners)}.")

        for loser_id, loser_agent in losers:
            winner_id, winner_agent = random.choice(winners)
            
            # EXPLOIT
            loser_agent.copy_from(winner_agent)
            
            # EXPLORE
            loser_agent.mutate()
            
            self.logger.info(f"[PBT] Agent '{loser_id}' evolved (Parent: '{winner_id}'). "
                             f"New LR: {loser_agent.learning_rate:.6f}, Dropout: {loser_agent.dropout:.2f}")

    def get_best_agent_stats(self, population: Dict[str, SpecialistAgent]) -> dict:
        """Helper to inspect the current best agent."""
        if not population: return {}
        best_agent = min(population.values(), key=lambda a: a.running_loss)
        return {
            "loss": best_agent.running_loss,
            "lr": best_agent.learning_rate,
            "dropout": best_agent.dropout
        }