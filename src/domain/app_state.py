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
# File: src/domain/app_state.py
# Author: Gabriel Moraes
# Date: 2025-12-25

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot
from typing import Dict, List, Optional
from src.domain.entities import MapNode, MapEdge, DataSource
from src.domain.source_repository import SourceRepository

# --- SUB-MANAGERS (Internal Composition for SRP) ---

class TopologyRepository(QObject):
    """
    Responsibility: Manage Static Map Data (The 'World').
    """
    map_loaded = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._nodes: Dict[str, MapNode] = {}
        self._edges: Dict[str, MapEdge] = {}
        self._map_file_path: Optional[str] = None

    def load_data(self, nodes_data: List[dict], edges_data: List[dict]):
        self._nodes.clear()
        self._edges.clear()
        
        # Hydrate MapNode entities
        for n_data in nodes_data:
            node = MapNode(
                id=n_data['id'], 
                x=n_data['x'], 
                y=n_data['y'], 
                node_type=n_data['type'],
                real_name=n_data.get('name'),
                tl_logic_id=n_data.get('tl_logic_id')
            )
            self._nodes[n_data['id']] = node

        # Hydrate MapEdge entities
        for e_data in edges_data:
            src = e_data.get('from_node')
            dst = e_data.get('to_node')
            if not src or not dst:
                src = e_data.get('from', "N/A")
                dst = e_data.get('to', "N/A")

            edge = MapEdge(
                id=e_data['id'], 
                from_node=src, 
                to_node=dst, 
                shape=e_data['shape'],
                real_name=e_data.get('name') 
            )
            self._edges[e_data['id']] = edge
        
        self.map_loaded.emit()

    def get_all_nodes(self) -> List[MapNode]:
        return list(self._nodes.values())
    
    def get_all_edges(self) -> List[MapEdge]:
        return list(self._edges.values())

    def get_node(self, node_id: str) -> Optional[MapNode]:
        return self._nodes.get(node_id)

    def get_edge(self, edge_id: str) -> Optional[MapEdge]:
        return self._edges.get(edge_id)

class InteractionManager(QObject):
    """
    Responsibility: Manage UI State (Modes, Selections).
    """
    mode_changed = pyqtSignal(bool)

    def __init__(self):
        super().__init__()
        self._is_association_mode_active: bool = False
        self._selected_source_id: Optional[str] = None

    def enter_association_mode(self, source_id: str):
        self._is_association_mode_active = True
        self._selected_source_id = source_id
        self.mode_changed.emit(True)

    def exit_association_mode(self):
        self._is_association_mode_active = False
        self._selected_source_id = None
        self.mode_changed.emit(False)
    
    @property
    def is_active(self) -> bool:
        return self._is_association_mode_active
    
    @property
    def selected_id(self) -> Optional[str]:
        return self._selected_source_id


# --- MAIN APP STATE (FACADE) ---

class AppState(QObject):
    """
    Central Facade for Application State.
    Ensures strict Zero Trust initialization.
    """
    
    # Re-exposing Signals
    map_data_loaded = pyqtSignal()
    association_mode_changed = pyqtSignal(bool)
    data_association_changed = pyqtSignal(str, str)
    data_source_added = pyqtSignal(DataSource)
    data_source_removed = pyqtSignal(str)
    source_origin_toggled = pyqtSignal(str, bool)

    def __init__(self):
        super().__init__()
        
        # Instantiate Sub-Managers
        self.topology = TopologyRepository()
        self.sources = SourceRepository()
        self.interaction = InteractionManager()
        
        # Synchronization Flags
        self.is_meh_ready = False
        
        # Wire Signals
        self.topology.map_loaded.connect(self.map_data_loaded)
        self.sources.source_added.connect(self.data_source_added)
        self.sources.source_removed.connect(self.data_source_removed)
        self.sources.association_changed.connect(self.data_association_changed)
        self.sources.source_origin_toggled.connect(self.source_origin_toggled)
        self.interaction.mode_changed.connect(self.association_mode_changed)

    # --- DELEGATION METHODS ---

    @pyqtSlot(list, list)
    def set_map_data(self, nodes_data: List[dict], edges_data: List[dict]):
        self.topology.load_data(nodes_data, edges_data)

    def set_map_source_path(self, path: str):
        self.topology._map_file_path = path

    def get_map_source_path(self) -> Optional[str]:
        return self.topology._map_file_path

    def get_all_nodes(self) -> List[MapNode]:
        return self.topology.get_all_nodes()
    
    def get_all_edges(self) -> List[MapEdge]:
        return self.topology.get_all_edges()

    def get_node(self, node_id: str) -> Optional[MapNode]:
        return self.topology.get_node(node_id)

    def get_edge(self, edge_id: str) -> Optional[MapEdge]:
        return self.topology.get_edge(edge_id)

    # --- Source Delegation ---
    
    def add_data_source(self, source: DataSource):
        self.sources.add(source)

    def emit_restored_sources(self):
        """Emit source_added for all pre-loaded sources (call AFTER UI signal wiring)."""
        self.sources.emit_restored_sources()

    def remove_data_source(self, source_id: str):
        self.sources.remove(source_id)

    def get_data_source(self, source_id: str) -> Optional[DataSource]:
        return self.sources.get(source_id)
    
    def get_all_data_sources(self) -> List[DataSource]:
        return self.sources.get_all()
    
    def get_source_by_device_id(self, device_id: str) -> Optional[DataSource]:
        return self.sources.get(device_id)

    def register_source(self, source: DataSource):
        self.sources.add(source)
    
    def update_source_value(self, source_id: str, value: float):
        self.sources.update_value(source_id, value)

    def toggle_source_origin(self, source_id: str):
        """Toggle source between Local and Global scope."""
        self.sources.toggle_origin(source_id)

    # --- Interaction & Association Delegation ---

    @pyqtSlot(str)
    def enter_association_mode(self, source_id: str):
        self.interaction.enter_association_mode(source_id)
        
    @pyqtSlot()
    def exit_association_mode(self):
        self.interaction.exit_association_mode()

    def is_in_association_mode(self) -> bool:
        return self.interaction.is_active
    
    @pyqtSlot(str)
    def associate_selected_source_to_element(self, element_id: str):
        if not self.interaction.is_active or not self.interaction.selected_id:
            return
            
        source_id = self.interaction.selected_id
        self.sources.associate(source_id, element_id)
        self.interaction.exit_association_mode()

    def get_sources_associated_with_element(self, element_id: str) -> List[str]:
        return self.sources.get_associations(element_id)

    def get_element_for_source(self, source_id: str) -> Optional[str]:
        for element_id, sources in self.sources._associations.items():
            if source_id in sources:
                return element_id
        return None