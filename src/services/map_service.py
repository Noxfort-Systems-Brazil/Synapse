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
# File: src/services/map_service.py
# Author: Gabriel Moraes
# Date: 2025-12-25

import os
import gzip
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple, Dict
from PyQt6.QtCore import QObject, pyqtSignal

from src.utils.logging_setup import logger
from src.domain.entities import MapNode, MapEdge

class MapService(QObject):
    """
    The 'Cartographer' of Synapse.
    
    Responsibility:
    - Parses SUMO Network files (.net.xml).
    - Extracts topological ground truth (Nodes & Edges).
    - Filters out internal simulation artifacts to build a clean graph.
    - Provides the 'Spatial Context' for the GATv2 Neural Network.
    
    Refactoring V3: Added support for GZIP compressed maps (.net.xml.gz).
    """
    
    # Signals
    map_loaded = pyqtSignal(int, int) # (num_nodes, num_edges)
    error_occurred = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.nodes: List[MapNode] = []
        self.edges: List[MapEdge] = []
        self._node_lookup: Dict[str, MapNode] = {}

    def load_network(self, file_path: str) -> bool:
        """
        Parses a .net.xml (or .net.xml.gz) file and populates the domain entities.
        """
        path = Path(file_path)
        if not path.exists():
            msg = f"Map file not found: {file_path}"
            logger.error(f"[MapService] ❌ {msg}")
            self.error_occurred.emit(msg)
            return False

        logger.info(f"[MapService] 🗺️ Loading Topology from {path.name}...")

        try:
            # FIX V3: Handle GZIP compression transparently
            if str(path).endswith('.gz'):
                # Open with gzip in text mode (rt) with utf-8 encoding
                source = gzip.open(path, 'rt', encoding='utf-8')
            else:
                # Open normally (let parse handle it or just path)
                source = str(path)

            tree = ET.parse(source)
            root = tree.getroot()
            
            # If we opened a file object (gzip), assume we should close it?
            # ET.parse doesn't close external file objects automatically in all Py versions,
            # but usually garbage collection handles it. 
            # Ideally use context manager, but ET.parse takes a stream.
            if hasattr(source, 'close'):
                source.close()

            self.nodes.clear()
            self.edges.clear()
            self._node_lookup.clear()

            # --- STEP 1: Parse Junctions (Nodes) ---
            for junction in root.findall('junction'):
                # SUMO has 'internal' junctions inside intersections. We skip them.
                j_type = junction.get('type')
                if j_type == 'internal':
                    continue

                j_id = junction.get('id')
                try:
                    x = float(junction.get('x'))
                    y = float(junction.get('y'))
                except (ValueError, TypeError):
                    continue # Skip if coords are missing

                # Create Entity
                node = MapNode(id=j_id, x=x, y=y, node_type=str(j_type))
                self.nodes.append(node)
                self._node_lookup[j_id] = node

            logger.info(f"[MapService] Extracted {len(self.nodes)} physical junctions (Nodes).")

            # --- STEP 2: Parse Streets (Edges) ---
            for edge in root.findall('edge'):
                # Skip internal edges (connections inside the intersection box)
                func = edge.get('function')
                if func == 'internal' or func == 'crossing' or func == 'walkingarea':
                    continue

                e_id = edge.get('id')
                from_id = edge.get('from')
                to_id = edge.get('to')
                
                # We only want edges connecting valid physical nodes
                if from_id in self._node_lookup and to_id in self._node_lookup:
                    
                    lanes = edge.findall('lane')
                    
                    # --- A. Extract Geometry (Shape) ---
                    # Priority: lane[0].shape > edge.shape > fallback to node coords
                    shape_points: List[Tuple[float, float]] = []
                    shape_str = None
                    
                    if lanes:
                        shape_str = lanes[0].get('shape')
                    if not shape_str:
                        shape_str = edge.get('shape')
                    
                    if shape_str:
                        try:
                            for point_str in shape_str.strip().split():
                                coords = point_str.split(',')
                                if len(coords) >= 2:
                                    shape_points.append((float(coords[0]), float(coords[1])))
                        except (ValueError, IndexError):
                            shape_points = []
                    
                    # Fallback: use from/to node coordinates if no shape parsed
                    if not shape_points:
                        fn = self._node_lookup[from_id]
                        tn = self._node_lookup[to_id]
                        shape_points = [(fn.x, fn.y), (tn.x, tn.y)]
                    
                    # --- B. Extract Length ---
                    try:
                        length = float(edge.get('length', 0.0))
                        if length == 0.0 and lanes:
                            length = float(lanes[0].get('length', 1.0))
                    except (ValueError, TypeError):
                        length = 1.0
                    
                    # --- C. Extract Speed & Lane Count (for Cartographer features) ---
                    speed = 13.89  # Default ~50 km/h
                    num_lanes = len(lanes) if lanes else 1
                    try:
                        if lanes:
                            speed = float(lanes[0].get('speed', 13.89))
                    except (ValueError, TypeError):
                        pass

                    # Create Entity
                    map_edge = MapEdge(
                        id=e_id,
                        from_node=from_id,
                        to_node=to_id,
                        shape=shape_points,
                        weight=length
                    )
                    # Store extra metadata for Cartographer (avoids changing dataclass)
                    map_edge._speed = speed
                    map_edge._num_lanes = num_lanes
                    self.edges.append(map_edge)

            logger.info(f"[MapService] Extracted {len(self.edges)} navigable streets (Edges).")
            
            # Emit success
            self.map_loaded.emit(len(self.nodes), len(self.edges))
            return True

        except ET.ParseError as e:
            msg = f"XML Parsing Error: {e}"
            logger.error(f"[MapService] ❌ {msg}")
            self.error_occurred.emit(msg)
            return False
        except Exception as e:
            msg = f"Critical Map Load Error: {e}"
            logger.error(f"[MapService] ❌ {msg}")
            import traceback
            traceback.print_exc()
            self.error_occurred.emit(msg)
            return False

    def get_topology(self) -> Tuple[List[MapNode], List[MapEdge]]:
        """Returns the processed graph data for the Optimizer."""
        return self.nodes, self.edges