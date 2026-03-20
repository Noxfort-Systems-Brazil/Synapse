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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
#
# File: ui/widgets/map_widget.py
# Author: Gabriel Moraes
# Date: 2025-11-17

# (Importing from PyQt6 as specified in requirements.txt)
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot, QPointF
from PyQt6.QtGui import QMouseEvent, QWheelEvent, QPainter
from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene

class MapWidget(QGraphicsView):
    """
    A specialized QGraphicsView for displaying the SUMO network map.
    Handles zoom (wheel) and pan (right-mouse drag).
    Emits signals when map elements (nodes, edges) or empty space are clicked.
    
    This implementation is based on the provided reference files 
    (map_controller.py, map_renderer.py) and adapted for PyQt6.
    """
    
    # Signals to notify the controller (inspired by map_controller.py)
    nodeClicked = pyqtSignal(str)
    edgeClicked = pyqtSignal(str)
    emptySpaceClicked = pyqtSignal()

    def __init__(self, parent=None):
        """
        Initializes the MapWidget.
        """
        super().__init__(parent)
        
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        
        # Use FullViewportUpdate for smooth updates
        self.setViewportUpdateMode(
            QGraphicsView.ViewportUpdateMode.FullViewportUpdate
        )
        
        # Antialiasing for smooth lines
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # We will handle panning manually (RightButton) instead of ScrollHandDrag
        # to keep LeftButton free for item selection.
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        
        self._min_zoom = 0.1
        self._max_zoom = 10.0
        
        self._is_panning = False
        self._last_pan_pos = QPointF()

    @property
    def scene(self) -> QGraphicsScene:
        """
        Provides public access to the internal scene, required by 
        the MapRenderer.
        """
        return self._scene

    @pyqtSlot(float, float)
    def set_zoom_limits(self, min_zoom: float, max_zoom: float):
        """
        Sets the zoom limits (inspired by map_renderer.py).
        """
        self._min_zoom = min_zoom
        self._max_zoom = max_zoom

    @pyqtSlot()
    def fit_map_in_view(self):
        """
        Adjusts the view to fit the entire scene content, keeping aspect ratio.
        """
        if not self._scene.itemsBoundingRect().isEmpty():
            self.fitInView(
                self._scene.itemsBoundingRect(), 
                Qt.AspectRatioMode.KeepAspectRatio
            )

    # --- Event Handlers for Interaction ---

    def wheelEvent(self, event: QWheelEvent):
        """
        Handles zooming with the mouse wheel.
        """
        
        # Check if wheel delta is vertical
        if event.angleDelta().y() == 0:
            event.ignore()
            return

        scale_factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        
        current_scale = self.transform().m11()
        
        # Check zoom limits
        if (current_scale * scale_factor < self._min_zoom) or \
           (current_scale * scale_factor > self._max_zoom):
            return
            
        self.setTransformationAnchor(
            QGraphicsView.ViewportAnchor.AnchorUnderMouse
        )
        self.scale(scale_factor, scale_factor)

    def mousePressEvent(self, event: QMouseEvent):
        """
        Handles clicks to select items (Left Button) or
        to start panning (Right Button).
        """
        
        # 1. Handle Panning (Right Mouse Button)
        if event.button() == Qt.MouseButton.RightButton:
            self._is_panning = True
            self._last_pan_pos = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return

        # 2. Handle Item Clicks (Left Mouse Button)
        if event.button() == Qt.MouseButton.LeftButton:
            # Get item at the click position
            item = self.itemAt(event.pos())
            
            # Check if item has ID data (data(1))
            if item and item.data(1) is not None: 
                item_type = item.data(0) # 'node' or 'edge'
                item_id = item.data(1)   # The element ID
                
                if item_type == "node":
                    self.nodeClicked.emit(item_id)
                elif item_type == "edge":
                    self.edgeClicked.emit(item_id)
                else:
                    # Clicked on something, but not a recognized element
                    self.emptySpaceClicked.emit() 
            else:
                # Clicked on empty background
                self.emptySpaceClicked.emit()
            
            event.accept()
            return

        # Fallback to default behavior if no button is handled
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        """
        Handles view panning when Right Mouse Button is pressed and dragged.
        """
        if self._is_panning:
            # Calculate the difference in scene coordinates
            scene_delta = self.mapToScene(event.pos()) - \
                          self.mapToScene(self._last_pan_pos)
            
            # Translate the view (which moves the scene)
            self.translate(scene_delta.x(), scene_delta.y())
            
            self._last_pan_pos = event.pos()
            event.accept()
            return
            
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        """
        Stops panning when the Right Mouse Button is released.
        """
        if event.button() == Qt.MouseButton.RightButton:
            self._is_panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
            return
            
        super().mouseReleaseEvent(event)