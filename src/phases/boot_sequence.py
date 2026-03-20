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
# File: src/phases/boot_sequence.py
# Author: Gabriel Moraes
# Date: 2026-03-09

"""
Boot Sequence Phase (SOLID: SRP).

Extracted from LifecycleOrchestrator.boot_system() to respect
Single Responsibility. The orchestrator manages state transitions;
this class handles the actual power-on logic for each subsystem.
"""

import os
from typing import Any, Dict

from src.domain.app_state import AppState
from src.utils.logging_setup import logger


class BootSequence:
    """
    Executes the power-on sequence for all subsystems.
    
    Extracted from LifecycleOrchestrator to follow SRP.
    Each step is isolated and testable independently.
    
    Steps:
        1. Start Fenix Watchdog (reliability monitor)
        2. Inject map data into Cartographer (if available)
        3. Load Cartographer diploma (pre-trained weights)
        4. Initialize InferenceEngine
    """

    @staticmethod
    def execute(app_state: AppState, services: Dict[str, Any]) -> None:
        """
        Runs the full boot sequence.
        
        Args:
            app_state: Central application state.
            services: Dict from ServiceContainer.build().
            
        Raises:
            Exception: Propagated to orchestrator for emergency stop.
        """
        logger.info("[BootSequence] 🔌 Initiating power-on sequence...")

        # Step 1: Start Recovery Watchdog
        fenix = services.get("fenix")
        if fenix:
            fenix.start_watchdog()
            logger.info("[BootSequence] 🛡️ Fenix Watchdog started.")

        # Step 2: Cartographer Map Injection & Diploma
        BootSequence._boot_cartographer(app_state, services)

        # Step 3: Initialize Inference Engine
        engine = services.get("inference_engine")
        if engine:
            engine.initialize_system()
            logger.info("[BootSequence] 🧠 InferenceEngine initialized.")

        logger.info("[BootSequence] ✅ Power-on sequence complete.")

    @staticmethod
    def _boot_cartographer(app_state: AppState, services: Dict[str, Any]) -> None:
        """
        Handles Cartographer-specific boot logic:
        1. Injects map data if not yet loaded.
        2. Loads pre-trained diploma (safetensors) if available.
        """
        cartographer = services.get("cartographer")
        if cartographer is None:
            return

        # 2a. Inject map data if R-Tree not yet built
        if cartographer.fast_matcher is None:
            map_nodes = app_state.get_all_nodes()
            map_edges = app_state.get_all_edges()
            if map_edges:
                cartographer.set_map_data(map_edges, map_nodes)

        # 2b. Load pre-trained diploma (skip training if found)
        diploma_path = os.path.join("data", "weights", "cartographer.safetensors")
        if os.path.exists(diploma_path):
            try:
                cartographer.model.load_diploma(
                    diploma_path, device=str(cartographer.device)
                )
                logger.info("[BootSequence] 🎓 Cartographer diploma loaded.")
            except Exception as e:
                logger.warning(f"[BootSequence] ⚠️ Diploma load failed: {e}")

        # 2c. Log final status
        if cartographer.fast_matcher:
            logger.info("[BootSequence] ✅ Cartographer R-Tree ready. Map matching enabled.")
        else:
            logger.info("[BootSequence] ℹ️ Cartographer active but no map data loaded.")
