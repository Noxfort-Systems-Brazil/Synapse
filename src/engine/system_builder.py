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
# File: src/engine/system_builder.py
# Author: Gabriel Moraes
# Date: 2025-12-24

import torch
from typing import Dict, Any

# Import Core Domain
from src.domain.app_state import AppState
from src.domain.entities import SourceStatus

# Import SRP Components
from src.engine.ingestion_pipeline import IngestionPipeline
from src.managers.graph_manager import GraphManager
from src.managers.node_manager import NodeManager
from src.managers.xai_manager import XAIManager
from src.managers.kse_manager import KSEManager
from src.managers.pbt_manager import PBTManager

# Import Infrastructure
from src.infrastructure.sensor_gateway import SensorGateway

# Import Workers
from src.workers.async_agents import AuditorWorker, LinguistWorker
from src.workers.xai_worker import XAIWorker

# Import Services
from src.services.historical_manager import HistoricalManager
from src.services.telemetry_service import TelemetryService
from src.factories.agent_factory import AgentFactory

class SystemBuilder:
    """
    Encapsulates the complexity of constructing and wiring the SYNAPSE system.
    
    Responsibility:
    - Instantiates all Managers, Agents, and Workers.
    - Resolves dependencies between components.
    - Returns a context dictionary with fully initialized objects.
    
    This cleans up the InferenceEngine, removing 'Construction' responsibility.
    """

    def __init__(self, app_state: AppState, device: torch.device, model_config: dict):
        self.app_state = app_state
        self.device = device
        self.config = model_config
        self.components: Dict[str, Any] = {}

    def build(self) -> Dict[str, Any]:
        """
        Main build sequence.
        Returns a dictionary containing all system components ready to run.
        """
        print("[SystemBuilder] 🔨 Starting System Construction...")

        # 1. Base Services
        telemetry = TelemetryService()
        historical_manager = HistoricalManager()
        
        # 2. Spatial Context (Graph)
        map_nodes = self.app_state.get_all_nodes()
        map_edges = self.app_state.get_all_edges()
        
        # Fallback: create dummy nodes for sensors if no map loaded
        if not map_nodes and self.app_state.get_all_data_sources():
            from src.domain.entities import MapNode
            for src in self.app_state.get_all_data_sources():
                map_nodes.append(MapNode(id=src.id, x=0, y=0, node_type="sensor"))

        graph_manager = GraphManager(
            nodes=map_nodes,
            edges=map_edges,
            embedding_dim=32, # Default dim
            device=self.device
        )

        # 3. Node & Memory Manager
        node_manager = NodeManager(
            config=self.config,
            device=self.device,
            historical_manager=historical_manager,
            graph_manager=graph_manager
        )

        # 4. Global Agents (Coordinator & Fuser)
        coordinator = None
        if map_nodes:
            coordinator = AgentFactory.create_coordinator(
                config=self.config,
                input_dim=32,
                hidden_dim=32,
                output_dim=32
            )
            coordinator.to(self.device)

        num_sources = max(1, len(self.app_state.get_all_data_sources()))
        fuser = AgentFactory.create_fuser(
            config=self.config,
            num_variates=num_sources,
            seq_len=60, # Standard Synapse Seq Len
            pred_len=10
        )
        fuser.to(self.device)

        # 5. Workers (Async Threads)
        auditor_worker = AuditorWorker(input_dim=num_sources)
        linguist_worker = LinguistWorker()
        xai_worker = XAIWorker(model_config=self.config)

        # 6. High-Level Managers (XAI, KSE, Pipeline)
        xai_manager = XAIManager(worker=xai_worker, app_state=self.app_state)
        
        ingestion_pipeline = IngestionPipeline(
            app_state=self.app_state,
            telemetry=telemetry
        )
        
        kse_manager = KSEManager(app_state=self.app_state)
        
        pbt_manager = PBTManager(exploit_ratio=0.2)

        # 7. Infrastructure (Gateway)
        sensor_gateway = SensorGateway()

        # Pack everything
        self.components = {
            "telemetry": telemetry,
            "historical_manager": historical_manager,
            "graph_manager": graph_manager,
            "node_manager": node_manager,
            "coordinator": coordinator,
            "fuser": fuser,
            "auditor_worker": auditor_worker,
            "linguist_worker": linguist_worker,
            "xai_worker": xai_worker,
            "xai_manager": xai_manager,
            "ingestion_pipeline": ingestion_pipeline,
            "kse_manager": kse_manager,
            "pbt_manager": pbt_manager,
            "sensor_gateway": sensor_gateway
        }
        
        print("[SystemBuilder] ✅ Construction Complete.")
        return self.components