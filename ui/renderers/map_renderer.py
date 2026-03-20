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
# File: ui/renderers/map_renderer.py
# Author: Gabriel Moraes
# Date: 2026-03-01

from typing import Dict, List
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPen, QBrush, QColor, QPainterPath, QPainterPathStroker
from PyQt6.QtWidgets import QGraphicsScene, QGraphicsItem
from ui.styles.theme_manager import ThemeManager

class MapRenderer:
    """
    Handles all visual rendering logic for the map.
    Isolates drawing routines, colors, and item management from the controller.
    """

    def __init__(self, scene: QGraphicsScene):
        self._scene = scene
        
        # State tracking for graphical items
        self._drawable_items_by_id: Dict[str, List[QGraphicsItem]] = {}
        self._current_highlight: List[QGraphicsItem] = []
        
        # --- Visual Styles ---
        self._node_pen = QPen(ThemeManager.get_color("map_node"), 1)
        self._node_brush = QBrush(ThemeManager.get_color("map_node"))
        
        self._edge_brush = QBrush(ThemeManager.get_color("map_edge"))
        self._edge_pen = QPen(Qt.PenStyle.NoPen)
        
        self._sel_assoc_color = ThemeManager.get_color("map_select_assoc")
        self._sel_free_color = ThemeManager.get_color("map_select_free")

    def clear(self):
        """Clears the scene and resets tracking dictionaries."""
        self._scene.clear()
        self._drawable_items_by_id.clear()
        self._current_highlight.clear()

    def draw_node(self, node):
        """Draws a circular node on the scene."""
        r = 5.0
        x, y = node.x, -node.y
        ellipse = self._scene.addEllipse(-r, -r, 2*r, 2*r, self._node_pen, self._node_brush)
        ellipse.setPos(x, y)
        
        # Store metadata
        ellipse.setData(0, "node")
        ellipse.setData(1, node.id)
        ellipse.setZValue(10)
        
        self._drawable_items_by_id.setdefault(node.id, []).append(ellipse)

    def draw_edge(self, edge):
        """Draws a path representing a street/edge on the scene."""
        points = edge.shape
        if len(points) < 2: 
            return
        
        path = QPainterPath()
        path.moveTo(points[0][0], -points[0][1])
        for x, y in points[1:]: 
            path.lineTo(x, -y)
            
        stroker = QPainterPathStroker()
        stroker.setWidth(4.0)
        stroker.setCapStyle(Qt.PenCapStyle.RoundCap)
        
        item = self._scene.addPath(stroker.createStroke(path), self._edge_pen, self._edge_brush)
        
        # Store metadata
        item.setData(0, "edge")
        item.setData(1, edge.id)
        item.setZValue(0)
        
        self._drawable_items_by_id.setdefault(edge.id, []).append(item)

    def has_element(self, element_id: str) -> bool:
        """Checks if a graphical element exists in the current render."""
        return element_id in self._drawable_items_by_id

    def highlight_elements(self, element_ids: List[str], is_associated: bool):
        """Applies highlight styles to specific map elements."""
        
        # 1. Reset previously highlighted items
        for item in self._current_highlight:
            typ = item.data(0)
            if typ == "node":
                item.setPen(self._node_pen)
                item.setBrush(self._node_brush)
                item.setZValue(10)
            elif typ == "edge":
                item.setBrush(self._edge_brush)
                item.setZValue(0)
        
        self._current_highlight.clear()
        
        # 2. Apply new highlight
        color = self._sel_assoc_color if is_associated else self._sel_free_color
        brush = QBrush(color)
        pen = QPen(color, 2)
        
        for eid in element_ids:
            items = self._drawable_items_by_id.get(eid, [])
            for item in items:
                typ = item.data(0)
                if typ == "node":
                    item.setBrush(brush)
                    item.setPen(pen)
                    item.setZValue(12)
                elif typ == "edge":
                    item.setBrush(brush)
                    item.setZValue(2)
                
                self._current_highlight.append(item)