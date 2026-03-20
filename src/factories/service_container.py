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
# File: src/factories/service_container.py
# Author: Gabriel Moraes
# Date: 2026-03-09

"""
Dependency Injection Container (SOLID: SRP + DIP).

Extracted from LifecycleOrchestrator to respect Single Responsibility.
The orchestrator should only manage state transitions, not object creation.

This container wires all subsystems and returns them as a typed dict,
allowing the orchestrator to remain a pure state machine.
"""

from typing import Any, Dict, Optional

from src.domain.app_state import AppState
from src.utils.logging_setup import logger


class ServiceContainer:
    """
    Centralizes all subsystem instantiation (Dependency Injection).
    
    Replaces the manual DI wiring that was inside LifecycleOrchestrator.__init__.
    Each subsystem is created once and injected into dependents.
    
    Usage:
        container = ServiceContainer(app_state)
        services = container.build()
        engine = services["inference_engine"]
    """

    def __init__(self, app_state: AppState):
        self.app_state = app_state

    def build(self) -> Dict[str, Any]:
        """
        Wires and returns all subsystems as a dictionary.
        
        Returns:
            Dict with keys: 'inference_engine', 'cartographer', 'fenix',
            'storage_manager', 'ingestion', 'optimizer_service' (None, lazy).
        """
        logger.info("[ServiceContainer] Wiring subsystems...")

        # --- 1. Storage ---
        storage_manager = self._build_storage()

        # --- 2. Core Workers & Managers ---
        from src.workers.ingestion_worker import IngestionWorker
        from src.managers.graph_manager import GraphManager
        from src.services.historical_manager import HistoricalManager

        ingestion = IngestionWorker(self.app_state)
        graph_manager = GraphManager(self.app_state)
        historical_manager = HistoricalManager(self.app_state)

        # --- 3. Agent Factory + Linguist ---
        from src.factories.agent_factory import AgentFactory
        from src.services.linguist_service import LinguistService

        config = getattr(self.app_state, 'config', {})
        agent_factory = AgentFactory(config)
        linguist_service = LinguistService(self.app_state, ingestion, agent_factory)

        # --- 4. XAI Subsystem ---
        from src.workers.xai_worker import XAIWorker
        from src.managers.xai_manager import XAIManager
        from src.services.semantic_enricher import SemanticEnricher

        xai_worker = XAIWorker(model_config={"feature_dim": 4})
        enricher = SemanticEnricher(self.app_state)
        xai_manager = XAIManager(xai_worker, enricher)

        # --- 5. Inference Engine ---
        from src.engine.inference_engine import InferenceEngine

        inference_engine = InferenceEngine(
            self.app_state,
            ingestion,
            graph_manager,
            historical_manager,
            linguist_service,
            xai_worker,
            xai_manager
        )

        # --- 6. Cartographer (Safe Import) ---
        cartographer = self._build_cartographer()

        # --- 7. Fenix (Reliability) ---
        fenix = self._build_fenix(storage_manager)

        logger.info("[ServiceContainer] ✅ All subsystems wired.")

        return {
            "inference_engine": inference_engine,
            "ingestion": ingestion,
            "cartographer": cartographer,
            "fenix": fenix,
            "storage_manager": storage_manager,
            "optimizer_service": None,  # Lazy: created on demand by orchestrator
        }

    # --- Private Builders (Safe Import Pattern) ---

    @staticmethod
    def _build_storage():
        """Builds StorageManager with fallback mock."""
        try:
            from src.managers.storage_manager import StorageManager
            return StorageManager(base_path="data/")
        except ImportError:
            logger.error("[ServiceContainer] StorageManager not found. Using mock.")
            class _MockStorage:
                def __init__(self): pass
                def save_json(self, f, d): return False
                def load_json(self, f): return None
            return _MockStorage()

    def _build_cartographer(self) -> Optional[Any]:
        """Builds CartographerAgent with safe import and deferred map injection."""
        try:
            from src.agents.cartographer_agent import CartographerAgent
            agent = CartographerAgent()
            
            # Attempt early map injection (may be deferred if no map loaded yet)
            map_nodes = self.app_state.get_all_nodes()
            map_edges = self.app_state.get_all_edges()
            if map_edges:
                agent.set_map_data(map_edges, map_nodes)
            else:
                logger.info("[ServiceContainer] 📋 Cartographer created, map injection deferred.")
            
            return agent
        except ImportError:
            logger.warning("[ServiceContainer] CartographerAgent not available.")
            return None
        except Exception as e:
            logger.warning(f"[ServiceContainer] ⚠️ Cartographer init failed: {e}")
            return None

    @staticmethod
    def _build_fenix(storage_manager):
        """Builds FenixService with fallback mock."""
        try:
            from src.services.fenix_service import FenixService
            return FenixService(storage_manager, None)
        except ImportError:
            logger.warning("[ServiceContainer] FenixService not found. Using mock.")
            class _MockFenix:
                def start_watchdog(self): pass
                def stop_watchdog(self): pass
                def report_health(self, status): pass
            return _MockFenix()
