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
# File: ui/controllers/map_controller.py
# Author: Gabriel Moraes
# Date: 2026-03-01

import os
from typing import Optional
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot, Qt

from ui.widgets.map_widget import MapWidget
from src.domain.app_state import AppState
from ui.workers.map_loader_thread import MapLoaderThread
from ui.renderers.map_renderer import MapRenderer

class MapController(QObject):
    """
    Bridge between MapWidget (View) and AppState (Model).
    Handles user interaction logic and orchestrates background loading and rendering.
    
    Refactored Version:
    - Adheres to SRP by delegating background loading to MapLoaderThread.
    - Adheres to SRP by delegating visual drawing logic to MapRenderer.
    """
    
    status_message = pyqtSignal(str)
    
    def __init__(self, map_widget: MapWidget, app_state: AppState):
        super().__init__()
        self._view = map_widget
        self._app_state = app_state 
        
        # Initialize the dedicated renderer
        self._renderer = MapRenderer(self._view.scene)
        
        # Thread reference storage
        self._loader: Optional[MapLoaderThread] = None
        
        # --- Connections ---
        self._view.nodeClicked.connect(self._on_node_clicked)
        self._view.edgeClicked.connect(self._on_edge_clicked)
        
        self._app_state.map_data_loaded.connect(self._draw_map_from_state)
        self._app_state.association_mode_changed.connect(self._on_association_mode_changed)

    def load_map(self, file_path: str):
        """Starts the background loading process with thread safety."""
        
        if self._loader is not None and self._loader.isRunning():
            self.status_message.emit("⚠️ Map loading already in progress. Please wait.")
            return

        self.status_message.emit(f"Parsing network file: {file_path}...")
        
        # Create new worker thread
        self._loader = MapLoaderThread(file_path)
        
        # Connect Logic Signals
        self._loader.data_loaded.connect(self._on_map_data_parsed)
        self._loader.error_occurred.connect(self._on_load_error)
        
        # Connect Lifecycle Signals
        self._loader.finished.connect(self._loader.deleteLater)
        self._loader.finished.connect(self._on_loader_finished)
        
        self._loader.start()

    @pyqtSlot()
    def _on_loader_finished(self):
        """Clean up reference when thread is done."""
        self._loader = None

    @pyqtSlot(list, list)
    def _on_map_data_parsed(self, nodes, edges):
        """Received raw data from thread. Updates the Model."""
        self.status_message.emit("Updating App State with map data...")
        self._app_state.set_map_data(nodes, edges)
        
        # Sync Raw Path for HFT Transfer
        sender = self.sender()
        if isinstance(sender, MapLoaderThread) and sender.file_path:
             abs_path = os.path.abspath(sender.file_path)
             self._app_state.set_map_source_path(abs_path)

    @pyqtSlot()
    def _draw_map_from_state(self):
        """Orchestrates the redrawing of the map using the Renderer."""
        nodes = self._app_state.get_all_nodes()
        edges = self._app_state.get_all_edges()
        
        self.status_message.emit(f"Drawing {len(nodes)} nodes and {len(edges)} edges from AppState...")
        
        self._renderer.clear()
        
        for edge in edges:
            self._renderer.draw_edge(edge)
        for node in nodes:
            self._renderer.draw_node(node)
            
        self._view.fit_map_in_view()
        self.status_message.emit("Map drawn successfully.")

    @pyqtSlot(str)
    def _on_load_error(self, err_msg):
        self.status_message.emit(err_msg)

    # --- Interaction Logic ---

    @pyqtSlot(bool)
    def _on_association_mode_changed(self, active: bool):
        """Updates the cursor style based on the current mode."""
        if active:
            self._view.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self._view.setCursor(Qt.CursorShape.ArrowCursor)

    @pyqtSlot(str)
    def _on_node_clicked(self, node_id):
        """Handles node selection from the UI."""
        if self._app_state.is_in_association_mode():
            self._app_state.associate_selected_source_to_element(node_id)
            self._renderer.highlight_elements([node_id], is_associated=True)
        else:
            self.status_message.emit(f"Selected Junction: {node_id}")
            self._renderer.highlight_elements([node_id], is_associated=False)

    @pyqtSlot(str)
    def _on_edge_clicked(self, edge_id):
        """Handles edge selection from the UI, supporting bidirectional streets."""
        base_id = edge_id.lstrip('-')
        neg_id = f"-{base_id}"
        
        ids_to_highlight = []
        if self._renderer.has_element(base_id):
            ids_to_highlight.append(base_id)
        if self._renderer.has_element(neg_id):
            ids_to_highlight.append(neg_id)

        if self._app_state.is_in_association_mode():
            self._app_state.associate_selected_source_to_element(base_id)
            self._renderer.highlight_elements(ids_to_highlight, is_associated=True)
        else:
            self.status_message.emit(f"Selected Street: {base_id}")
            self._renderer.highlight_elements(ids_to_highlight, is_associated=False)