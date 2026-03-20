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
# File: src/domain/entities.py
# Author: Gabriel Moraes
# Date: 2026-02-16

from typing import List, Tuple, Optional, Any, Dict
from dataclasses import dataclass, field
from enum import Enum

# --- Enums for State Management ---

class SourceType(Enum):
    """
    Defines the supported types of data ingestion.
    Used by ProjectController to categorize inputs.
    """
    CSV = "CSV"
    PARQUET = "Parquet"
    API = "REST API"
    MQTT = "MQTT Stream"
    DATABASE = "SQL Database"
    SUMO_NET_XML = "SUMO Network"

class SourceStatus(Enum):
    """
    Defines the lifecycle states of a Data Source.
    Crucial for the 'Zero Trust' architecture.
    """
    QUARANTINE = "Quarantine"   # Newly connected. Buffering data for analysis.
    VALIDATING = "Validating"   # Data sent to Linguist Agent. Waiting for verdict.
    ACTIVE = "Active"           # Approved. TCN is running.
    
    # --- Resilience States (AFB/MEH Logic) ---
    FALLBACK = "Fallback"       # Sensor Failed. Using MEH/AFB.
    PROBATION = "Probation"     # Sensor returning from failure.
    
    REJECTED = "Rejected"       # Denied by Linguist.
    OFFLINE = "Offline"         # Connection lost.

# --- Data Structures for SUMO Map Elements ---

@dataclass
class MapNode:
    """
    Represents a traffic junction or node in the SUMO network.
    """
    id: str
    x: float
    y: float
    node_type: str
    real_name: Optional[str] = None
    
    # The ID of the Traffic Light Logic (tlLogic) controlling this node.
    tl_logic_id: Optional[str] = None

@dataclass
class MapEdge:
    """
    Represents a traffic street (edge) in the SUMO network.
    """
    id: str
    from_node: str
    to_node: str
    shape: List[Tuple[float, float]] # List of (x, y) points defining the geometry
    real_name: Optional[str] = None
    
    # Cost/Distance for Graph Algorithms
    weight: float = 1.0
    
    # NTCIP Signal Group (Phase) controlling this edge.
    signal_group_id: int = -1 

# --- Data Structures for External/Internal Sources ---

# Trust Hierarchy Constants (Critical Systems Engineering)
# Local sources (loops, cameras, radar) are ground truth ─ highest confidence.
# Global sources (Waze, TomTom) are crowdsourced ─ lower confidence.
LOCAL_TRUST_SCORE: float = 0.95
GLOBAL_TRUST_SCORE: float = 0.60

@dataclass
class DataSource:
    """
    Represents an external or internal data source (e.g., Camera, API, Loop).
    
    Updated V5 (Flexible Schema):
    - latest_value is now Any to support diverse payloads (JSON, Lists).
    - Added 'metadata' dict to store arbitrary sensor configs without strict schema.
    """
    id: str
    name: str
    source_type: SourceType = SourceType.API # Default
    connection_string: str = "" 
    is_local: bool = True       
    
    # Lifecycle State
    status: SourceStatus = SourceStatus.QUARANTINE
    
    # Spatial Info (Required for IngestionPipeline)
    lat: float = 0.0
    lon: float = 0.0
    
    # Runtime State (Flexible)
    latest_value: Any = None 
    last_update: float = 0.0
    
    # Semantic Metadata (Populated by Linguist Agent)
    semantic_type: Optional[str] = None # e.g., "speed", "flow"
    inferred_unit: Optional[str] = None # e.g., "km/h", "m/s"
    confidence_score: float = 0.0
    
    # NEW: Flexible Bag for Sensor-Specific Configs
    # e.g., {"camera_ip": "192.168.1.10", "loop_sensitivity": 0.8}
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Auto-set trust score based on source category if not explicitly set."""
        if self.confidence_score == 0.0:
            self.confidence_score = LOCAL_TRUST_SCORE if self.is_local else GLOBAL_TRUST_SCORE

    @property
    def display_status(self) -> str:
        """Returns a string representation of the status for UI."""
        return self.status.value

@dataclass
class DataAssociation:
    """
    Represents the logical link between a DataSource and a Map Element.
    """
    source_id: str
    element_id: str