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
# File: src/engine/cycle_processor.py
# Author: Gabriel Moraes
# Date: 2026-02-14

import time
import torch
import numpy as np
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional, TYPE_CHECKING

from src.utils.logging_setup import logger
from src.utils.debug_logger import perf_logger

# --- Domain ---
from src.domain.app_state import AppState

# --- Managers & Engine Helpers ---
from src.managers.xai_manager import XAIManager
from src.managers.graph_manager import GraphManager
from src.engine.tensor_builder import TensorBuilder
from src.engine.security_monitor import SecurityMonitor

# --- Lazy Imports (Prevent Circular Dependency) ---
if TYPE_CHECKING:
    from src.agents.coordinator_agent import CoordinatorAgent
    from src.agents.fuser_agent import FuserAgent
    from src.agents.auditor_agent import AuditorAgent

class CycleProcessor:
    """
    Handles the mathematical heavy-lifting of the Global Inference Cycle.
    
    Refactored V9 (SOLID Principles):
    - Uses TensorBuilder for data preprocessing (SRP).
    - Unifies Agent Pipeline abstraction via standard .inference() (LSP/ISP).
    - Uses SecurityMonitor for alerting evaluation (SRP).
    """

    def __init__(self, 
                 app_state: AppState, 
                 device: torch.device,
                 coordinator: 'CoordinatorAgent',
                 fuser: 'FuserAgent',
                 auditor: 'AuditorAgent',
                 xai_manager: XAIManager,
                 graph_manager: GraphManager):
        
        self.app_state = app_state
        self.device = device
        
        # Agents (Injected)
        self.coordinator = coordinator
        self.fuser = fuser
        self.auditor = auditor
        
        # Managers
        self.xai_manager = xai_manager
        self.graph_manager = graph_manager
        
        # --- Engine Domain Extractors (SOLID) ---
        self.tensor_builder = TensorBuilder(device, coordinator.model, graph_manager)
        self.security_monitor = SecurityMonitor(threshold=0.5)

    def run_logic(self, snapshot: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        """
        Executes the main AI pipeline orchestrating standardized agents.
        """
        start_time = time.time()
        
        # 1. Prepare Tensors (Delegated to TensorBuilder - SRP)
        try:
            x_spatial, edge_index, x_temporal = self.tensor_builder.prepare_tensors(snapshot)
        except ValueError as ve:
            return {}, None
        except Exception as e:
            logger.error(f"[Processor] Critical Tensor Failure: {e}")
            return {}, None
        t_tensor = time.time()

        # 2. Coordinator (Spatial Pipeline)
        try:
            spatial_embedding = self.coordinator.inference({
                 "x_spatial": x_spatial, 
                 "edge_index": edge_index
            })
        except Exception as e:
            logger.critical(f"[Processor] Spatial Inference Crash: {e}")
            raise e
        t_coordinator = time.time()
        
        # 3. Fuser (Spatio-Temporal Fusion — Gold Standard)
        # Sequential: Coordinator output feeds into Fuser via cross-attention
        forecast = self.fuser.inference({
            "x_temporal": x_temporal,
            "spatial_context": spatial_embedding  # Cross-Attention context from GATv2
        })
        t_fuser = time.time()
        
        # 4. Auditor (Security Pipeline)
        security_score = 0.0
        if self.auditor:
            try:
                # Compress State: Mean across Embedding Dim -> [1, Nodes]
                signature = spatial_embedding.mean(dim=1).unsqueeze(0) 
                
                # Dynamic Auditor Check is handled natively by the Agent now
                auditor_report = self.auditor.inference({"signature": signature})
                security_score = auditor_report.get("score", 0.0)
            except Exception as e:
                logger.error(f"[Processor] Security sweep failed: {e}")
        t_auditor = time.time()

        # 5. Result Packaging
        total_time = t_auditor - start_time
        results = {
            "spatial_embedding": spatial_embedding,
            "forecast": forecast,
            "security_score": security_score,
            "processing_time": total_time
        }
        
        # --- Debug Log: CYCLE (per-stage neural performance) ---
        perf_logger.info(
            f"CYCLE | tensor={(t_tensor - start_time)*1000:.2f}ms "
            f"| coordinator={(t_coordinator - t_tensor)*1000:.2f}ms "
            f"| fuser={(t_fuser - t_coordinator)*1000:.2f}ms "
            f"| auditor={(t_auditor - t_fuser)*1000:.2f}ms "
            f"| total={total_time*1000:.2f}ms"
        )
        
        # 6. Alerting (Delegated to SecurityMonitor - SRP)
        alert = self.security_monitor.evaluate(security_score)
        
        return results, alert