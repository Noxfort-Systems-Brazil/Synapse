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
# File: src/domain/interfaces.py
# Author: Gabriel Moraes
# Date: 2025-12-24

from typing import Any, List, Dict, Protocol, runtime_checkable

# --- AGENT INTERFACES (DIP/OCP) ---

@runtime_checkable
class IAgent(Protocol):
    """Base contract for any AI Agent in the system."""
    def to(self, device: Any) -> 'IAgent':
        """Moves the agent's internal model to a computing device (CPU/GPU)."""
        ...

@runtime_checkable
class ISpatialAgent(IAgent, Protocol):
    """Contract for agents that understand space (e.g. GATv2)."""
    def process_region(self, node_features: Any, edge_index: Any) -> Any:
        """Processes a graph snapshot and returns node embeddings."""
        ...

@runtime_checkable
class ITemporalAgent(IAgent, Protocol):
    """Contract for agents that understand time (e.g. Transformers)."""
    def predict_state(self, history: Any) -> Any:
        """Predicts future states based on historical time-series."""
        ...

# --- INFRASTRUCTURE INTERFACES (DIP) ---

@runtime_checkable
class IPipeline(Protocol):
    """Contract for data ingestion pipelines."""
    def process_packet(self, source_id: str, payload: Any) -> bool:
        ...
    
    def is_ready_for_linguist(self, source_id: str) -> bool:
        ...

# --- STATE INTERFACES (ISP) ---

@runtime_checkable
class ITopologyProvider(Protocol):
    """Interface for components that only need to READ map data."""
    def get_all_nodes(self) -> List[Any]: ...
    def get_all_edges(self) -> List[Any]: ...

@runtime_checkable
class ISourceProvider(Protocol):
    """Interface for components that only need to READ sensor data."""
    def get_all_data_sources(self) -> List[Any]: ...
    def get_data_source(self, source_id: str) -> Any: ...