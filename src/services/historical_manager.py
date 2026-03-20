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
# File: src/services/historical_manager.py
# Author: Gabriel Moraes
# Date: 2026-02-28

from datetime import datetime
from typing import Optional, List, Dict, Any, TYPE_CHECKING
import numpy as np

from PyQt6.QtCore import QObject
from src.utils.logging_setup import logger

# Import the new MEH specialized engines
from src.meh.data_loader import HistoricalDataLoader
from src.meh.fallback_engine import FallbackEngine
from src.meh.playback_engine import PlaybackEngine

# Lazy Type Hinting to prevent circular imports
if TYPE_CHECKING:
    from src.domain.app_state import AppState

class HistoricalManager(QObject):
    """
    Facade/Orchestrator for the Historical and MEH (Historical State Module) subsystems.
    
    Responsibilities:
    - Delegates I/O and Analytics to HistoricalDataLoader.
    - Delegates Level 1/2 Temporal Fallback logic to FallbackEngine.
    - Delegates sequential Simulation to PlaybackEngine.
    - Formats the flat MEH payload into KSE Edge-specific payloads.
    """

    def __init__(self, app_state: 'AppState'):
        super().__init__()
        self.app_state = app_state
        
        # Instantiate the specialized MEH engines (SOLID Principles)
        self.loader = HistoricalDataLoader()
        self.fallback_engine = FallbackEngine(self.loader)
        self.playback_engine = PlaybackEngine(self.loader)
        
        # Attempt immediate load upon initialization
        self.load_data()

    def load_data(self) -> bool:
        """
        Orchestrates the loading of data and the building of MEH profiles.
        """
        logger.info("[HistoricalManager] Starting data load sequence...")
        success = self.loader.load()
        
        if success:
            self.fallback_engine.build_profiles()
            self.playback_engine.reset()
            if self.app_state:
                self.app_state.is_meh_ready = True
            return True
            
        return False

    # =========================================================================
    # KSE INTERFACE (The Connector)
    # =========================================================================

    def get_current_state_prediction(self) -> Dict[str, dict]:
        """
        Called by KSEManager when real-time sensors fail.
        Returns a structured dictionary {edge_id: {speed, density, queue}} 
        representing the expected traffic for the current moment.
        
        V2: Falls back to sensor profiles when no edges are configured.
        """
        now = datetime.now()
        
        # 1. Get Flat Data from the MEH Fallback Engine
        flat_data = self.get_meh_state(now)
        
        # 2. Structure Data for KSE (Map flat columns to Edges)
        result = {}
        
        if self.app_state:
            edges = self.app_state.get_all_edges()
            
            if edges:
                # Normal path: map data to edges
                for edge in edges:
                    # DEFAULT VALUES (Safety Baseline)
                    edge_stats = {
                        "speed": 40.0,
                        "density": 10.0,
                        "queue": 0.0
                    }
                    
                    # INTELLIGENT MAPPING
                    speed_key = f"{edge.id}_speed"
                    density_key = f"{edge.id}_density"
                    queue_key = f"{edge.id}_queue"
                    
                    if speed_key in flat_data:
                        edge_stats["speed"] = flat_data[speed_key]
                    if density_key in flat_data:
                        edge_stats["density"] = flat_data[density_key]
                    if queue_key in flat_data:
                        edge_stats["queue"] = flat_data[queue_key]
                    
                    result[edge.id] = edge_stats
            elif flat_data:
                # Fallback path: No edges configured, use flat sensor data directly
                # Structure each sensor's flat values as synthetic edge entries
                for sensor_id in self.fallback_engine.sensor_profiles.keys():
                    result[sensor_id] = {
                        "speed": flat_data.get(f"{sensor_id}_speed", flat_data.get("speed", 40.0)),
                        "density": flat_data.get(f"{sensor_id}_density", flat_data.get("density", 10.0)),
                        "queue": flat_data.get(f"{sensor_id}_queue", flat_data.get("queue", 0.0)),
                    }
                
                # If still empty, create at least one entry from the raw flat data
                if not result and flat_data:
                    result["synthetic_edge"] = {
                        "speed": flat_data.get("speed", 40.0),
                        "density": flat_data.get("density", 10.0),
                        "queue": flat_data.get("queue", 0.0),
                    }
                
        return result

    def get_exact_reading(self, sensor_id: str, target_timestamp: float, tolerance: float = 0.25) -> Optional[float]:
        """
        Per-sensor temporal lookup in the Golden Database.
        Delegated to the DataLoader for actual I/O.
        
        Called by TrafficNode.ghost_step() to find historical data
        for a specific sensor at the current time-of-day.
        
        Args:
            sensor_id: The sensor/node ID (matches parquet sensor_id column).
            target_timestamp: Unix epoch timestamp (time.time()).
            tolerance: Maximum time-of-day difference in seconds.
            
        Returns:
            Float value if a historical match exists, None otherwise.
        """
        return self.loader.get_exact_reading(sensor_id, target_timestamp, tolerance)

    # =========================================================================
    # MEH INTERFACE (Level 3 Fallback - Temporal Lookup)
    # =========================================================================

    def get_meh_state(self, target_time: datetime) -> Dict[str, Any]:
        """
        Delegates the temporal lookup to the Fallback Engine.
        If the Fallback Engine cannot resolve, degrades to Playback Engine.
        """
        state = self.fallback_engine.get_fallback_state(target_time)
        if not state:
            logger.warning("[HistoricalManager] MEH Fallback failed, degrading to Playback Simulation.")
            return self.get_next_playback_frame()
        return state

    # =========================================================================
    # SIMULATION / PLAYBACK INTERFACE (Sequential)
    # =========================================================================

    def get_next_playback_frame(self) -> Dict[str, float]:
        """Delegates to Playback Engine."""
        return self.playback_engine.get_next_frame()

    def reset_playback(self):
        """Delegates to Playback Engine."""
        self.playback_engine.reset()

    # =========================================================================
    # ANALYTICAL INTERFACE (Statistics)
    # =========================================================================

    def get_expected_value(self, metric_name: str) -> float:
        """Delegates to Data Loader."""
        return self.loader.get_expected_value(metric_name)

    def get_context_window(self, window_size: int = 60) -> np.ndarray:
        """Delegates to Data Loader."""
        return self.loader.get_context_window(window_size)

    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """Delegates to Data Loader."""
        return self.loader.stats_cache

    @property
    def columns(self) -> List[str]:
        return self.loader.columns

    @property
    def is_ready(self) -> bool:
        return self.loader.is_loaded