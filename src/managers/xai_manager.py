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
# File: src/managers/xai_manager.py
# Author: Gabriel Moraes
# Date: 2025-12-03

import numpy as np
import torch
from typing import List, Dict, Optional
from datetime import datetime

# Import Domain Interfaces
from src.workers.xai_worker import XAIWorker
from src.engine.traffic_node import TrafficNode
from src.agents.fuser_agent import FuserAgent
from src.domain.app_state import AppState 

from src.services.semantic_enricher import SemanticEnricher
from src.engine.strategies.veto_medoid_strategy import VetoMedoidStrategy

class XAIManager:
    """
    Manages the strategies and state for Explainable AI.
    
    Refactored V3 (SOLID / Context-Aware / RAG):
    - [SRP] Removed internal Numpy Matrix Math (Delegated to VetoMedoidStrategy).
    - [SRP] Removed Sensor Classification logic (Delegated to SemanticEnricher).
    - [DIP] SemanticEnricher is injected into the manager.
    """

    def __init__(self, worker: XAIWorker, enricher: SemanticEnricher):
        """
        Args:
            worker: Reference to the XAI Worker thread.
            enricher: Service responsible for naming/translation context.
        """
        self.worker = worker
        self.enricher = enricher
        self.veto_buffer: List[Dict] = []
        self.max_buffer_size = 1000

    # --- Veto / Auditor Logic ---

    def buffer_veto(self, state_vector: list, error: float):
        entry = {
            'vector': state_vector,
            'error': error,
            'timestamp': datetime.now().isoformat()
        }
        self.veto_buffer.append(entry)
        if len(self.veto_buffer) > self.max_buffer_size:
            self.veto_buffer.pop(0)

    def process_buffer_strategy(self, available_nodes: List[str]):
        """Executes Medoid Strategy with Semantic Name Injection."""
        if not self.veto_buffer: return

        try:
            # 1. Math: Find Medoid (Delegated to Strategy object)
            medoid_evt = VetoMedoidStrategy.find_medoid(self.veto_buffer)
            
            # 2. Context: Resolve Names
            # available_nodes must be sorted alphabetically to match vector order (from Fuser)
            sorted_nodes = sorted(available_nodes)
            
            feature_names = []
            if len(sorted_nodes) == len(medoid_evt['vector']):
                feature_names = [self.enricher.resolve_semantic_name(nid) for nid in sorted_nodes]
            else:
                feature_names = [f"SENSOR_{i}" for i in range(len(medoid_evt['vector']))]

            print(f"[XAIManager] 📦 Aggregating {len(self.veto_buffer)} vetos. Medoid Error: {medoid_evt['error']:.4f}")

            # 3. Submit
            self.worker.submit_request(
                target_type="auditor",
                input_vector=medoid_evt['vector'],
                feature_names=feature_names,
                error=medoid_evt['error']
            )
            self.veto_buffer.clear()

        except Exception as e:
            print(f"[XAIManager] Error in Medoid Strategy: {e}")

    # --- TCN / Local Logic ---

    def explain_local(self, source_id: str, node: TrafficNode):
        if not node.is_ready: return

        history = node.memory.get_numpy().flatten().tolist()
        
        # Resolve base name for the sensor
        base_name = self.enricher.resolve_semantic_name(source_id)
        
        # Feature names are time-lagged versions of this semantic entity
        feature_names = [f"{base_name} [t-{i}]" for i in range(len(history), 0, -1)]
        
        print(f"[XAIManager] Dispatching TCN Analysis for {base_name}.")
        
        self.worker.submit_request(
            target_type="tcn",
            input_vector=history,
            feature_names=feature_names
        )

    # --- Fuser / Global Logic ---

    def explain_global(self, fuser: FuserAgent, nodes: Dict[str, TrafficNode], seq_len: int):
        if not fuser: return

        active_histories = []
        ordered_ids = sorted([nid for nid in nodes.keys()])
        
        for nid in ordered_ids:
            node = nodes[nid]
            if node.is_ready:
                active_histories.append(node.memory.get_numpy().flatten())
            else:
                print(f"[XAIManager] Global XAI aborted: Node {nid} not ready.")
                return

        global_vector = np.concatenate(active_histories).tolist()
        
        # Build Semantic Names
        feature_names = []
        for nid in ordered_ids:
            base_name = self.enricher.resolve_semantic_name(nid)
            feature_names.extend([f"{base_name} [t-{i}]" for i in range(seq_len, 0, -1)])

        print(f"[XAIManager] Dispatching Fuser Analysis with Semantic Context.")
        
        self.worker.submit_request(
            target_type="fuser",
            input_vector=global_vector,
            feature_names=feature_names
        )