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
# File: src/engine/neural_factory.py
# Author: Gabriel Moraes
# Date: 2026-02-15

import os
import torch
import logging
from typing import Dict, Any, List

# --- Phase 2 (Online) Agent Imports ---
# Only importing agents strictly defined for the Runtime Phase.
from src.agents.coordinator_agent import CoordinatorAgent # GATv2 Lite
from src.agents.fuser_agent import FuserAgent             # iTransformer
from src.agents.auditor_agent import AuditorAgent         # Wavelet + AE + OCC
from src.agents.linguist_agent import LinguistAgent       # Neuro-symbolic (DistilRoBERTa + TCN-AE)
from src.agents.jurist_agent import JuristAgent           # Qwen 2.5 1.5B
from src.agents.specialist_agent import SpecialistAgent   # TCN (The Analyst)

class NeuralFactory:
    """
    Factory class responsible for instantiating Neural Agents for PHASE 2 (Online).
    
    Refactored V9 (Device Placement Fix):
    - Corrected the device placement logic loop.
    - Now prioritizes calling agent.to(device) for nn.Module instances.
    - This ensures 'self.device' inside agents is correctly updated to CUDA,
      preventing Input(CPU) vs Weight(GPU) conflicts.
    """

    def __init__(self, weights_dir: str = "checkpoints/"):
        self.logger = logging.getLogger(__name__)
        self.weights_dir = weights_dir
        self.device = self._setup_device()
        self.agents: Dict[str, Any] = {}

    def _setup_device(self) -> torch.device:
        """
        Detects the best available hardware accelerator.
        """
        if torch.cuda.is_available():
            self.logger.info(f"Neural Factory: CUDA detected. Using {torch.cuda.get_device_name(0)}")
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.logger.info("Neural Factory: Apple MPS detected.")
            return torch.device("mps")
        else:
            self.logger.warning("Neural Factory: No accelerator detected. Using CPU.")
            return torch.device("cpu")

    def _load_best_hparams(self) -> Dict[str, Any]:
        """
        Loads the best hyperparameters found by Optuna (Phase Zero).
        """
        candidate_paths = [
            os.path.join(self.weights_dir, "best_hparams.pth"),
            os.path.join(os.path.expanduser("~"), "Documentos", "Synapse", "Checkpoint", "best_hparams.pth")
        ]
        
        for hparams_path in candidate_paths:
            if os.path.exists(hparams_path):
                try:
                    best_params = torch.load(hparams_path, map_location=torch.device('cpu'))
                    self.logger.info(f"Neural Factory: Loaded OPTIMIZED parameters from {hparams_path}")
                    return best_params
                except Exception as e:
                    self.logger.error(f"Neural Factory: Failed to load hparams file at {hparams_path}: {e}")
        
        self.logger.info("Neural Factory: Phase Zero file not found. Using Legacy Defaults.")
        return {}

    def build_all(self, app_state: Any) -> Dict[str, Any]:
        """
        Instantiates ONLY the agents required for Online Operation (Phase 2).
        """
        self.logger.info("Neural Factory: Starting Runtime Agent Ecosystem initialization...")
        
        # 1. Topology Analysis
        nodes = app_state.get_all_nodes()
        num_nodes = len(nodes) if nodes else 10
        self.logger.info(f"Neural Factory: Configuring agents for Topology Size: {num_nodes} nodes.")

        # 2. Load Dynamic Hyperparameters
        hparams = self._load_best_hparams()

        try:
            # === PHASE 2 AGENTS ===

            # 1. Specialist Agent (The Analyst - TCN)
            self.agents['specialist'] = SpecialistAgent(
                input_dim=4,  # [Occupancy, Speed, Queue, Density]
                output_dim=hparams.get('specialist_out', 16),
                num_channels=hparams.get('specialist_channels', [16, 32, 64]),
                kernel_size=hparams.get('specialist_kernel', 3),
                dropout=hparams.get('specialist_dropout', 0.2)
            )

            # 2. Coordinator Agent (GATv2 Lite)
            self.agents['coordinator'] = CoordinatorAgent(
                in_channels=4,                                 
                hidden_channels=hparams.get('coordinator_hidden', 32),
                out_channels=hparams.get('coordinator_out', 32),
                heads=hparams.get('coordinator_heads', 2),
                dropout=hparams.get('coordinator_dropout', 0.1)
            )

            # 3. Fuser Agent (iTransformer)
            self.agents['fuser'] = FuserAgent(
                seq_len=hparams.get('seq_len', 60),
                pred_len=hparams.get('pred_len', 12),
                num_variates=num_nodes,
                d_model=hparams.get('fuser_d_model', 64),
                n_heads=hparams.get('fuser_n_heads', 4)
            )

            # 4. Auditor Agent (Wavelet Scattering + AE + OCC)
            self.agents['auditor'] = AuditorAgent(
                input_len=hparams.get('seq_len', 60), 
                J=hparams.get('auditor_J', 2),        
                Q=hparams.get('auditor_Q', 1),        
                latent_dim=hparams.get('auditor_latent', 16)
            )

            # 5. Linguist Agent (Neuro-Symbolic)
            self.agents['linguist'] = LinguistAgent()

            # 6. Jurist Agent (Qwen 2.5 1.5B)
            self.agents['jurist'] = JuristAgent(
                model_id="Qwen/Qwen2.5-1.5B-Instruct" 
            )
            
            # --- Device Placement (CRITICAL FIX) ---
            for name, agent in self.agents.items():
                # Jurist manages its own device map (accelerate/bitsandbytes)
                if name == 'jurist': 
                    continue

                # FIX: Check if it's an nn.Module (BaseAgent inherits from it)
                # Calling .to() on the Agent updates 'self.device' AND moves the weights.
                if isinstance(agent, torch.nn.Module):
                    agent.to(self.device)
                
                # Fallback for agents that might wrap a model but not inherit properly
                elif hasattr(agent, 'model') and agent.model:
                    if hasattr(agent.model, 'to'):
                        agent.model.to(self.device)
            
            self.logger.info(f"Neural Factory: Successfully initialized {len(self.agents)} Runtime Agents on {self.device}.")
            return self.agents

        except Exception as e:
            self.logger.critical(f"Neural Factory: Critical failure during Agent construction. {str(e)}")
            raise e

    def get_device(self) -> torch.device:
        return self.device