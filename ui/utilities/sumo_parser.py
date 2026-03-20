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
# File: ui/utilities/sumo_parser.py
# Author: Gabriel Moraes
# Date: 2025-12-14

import gzip
import logging
from lxml import etree
from typing import List, Tuple, Dict, Any

class SumoNetworkParser:
    """
    Dedicated service for parsing SUMO network files (.net.xml).
    Handles file I/O, decompression, and XML extraction logic.
    
    Updated V8 (Traffic Light Support): 
    - Extracts the 'tl' attribute from junctions (Traffic Light Logic ID).
    - Populates 'tl_logic_id' in the node data dictionary.
    - Maintains V7.2 fixes (Geometry Backfill & Crash Safety).
    """

    @staticmethod
    def parse_file(file_path: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Parses the given .net.xml (or .net.xml.gz) file.

        Args:
            file_path: Path to the SUMO network file.

        Returns:
            A tuple containing:
            - List of node data dictionaries (now with 'tl_logic_id').
            - List of edge data dictionaries (including 'shape' for geometry).
        """
        
        open_func = gzip.open if file_path.endswith('.gz') else open
        
        nodes = []
        edges_dict: Dict[str, Dict[str, Any]] = {} 

        try:
            with open_func(file_path, 'rb') as f:
                # Iterate events end-element to have fully populated elements
                context = etree.iterparse(
                    f, 
                    events=('end',), 
                )

                for event, elem in context:
                    # Robust tag name extraction (ignores namespace {http://...}tag)
                    tag = etree.QName(elem).localname
                    
                    if tag == 'junction':
                        if elem.get('type') != 'internal':
                            # UPDATE V8: Extract Traffic Light Logic ID
                            tl_id = elem.get('tl') # None if not a TLS junction
                            
                            nodes.append({
                                'id': elem.get('id'),
                                'x': float(elem.get('x')),
                                'y': float(elem.get('y')),
                                'type': elem.get('type'),
                                'tl_logic_id': tl_id  # Store for AppState
                            })

                    elif tag == 'edge':
                        if elem.get('function') != 'internal':
                            edge_id = elem.get('id')
                            edge_name = elem.get('name')
                            
                            from_node = elem.get('from')
                            to_node = elem.get('to')
                            
                            # --- GEOMETRY EXTRACTION STRATEGY ---
                            # Goal: Find the most detailed shape available.
                            best_points = []
                            
                            # 1. Check Edge-level shape
                            if elem.get('shape'):
                                best_points = SumoNetworkParser._parse_shape_string(elem.get('shape'))
                            
                            # 2. Check All Lanes (prefer lane shape if it has more detail)
                            # Iterate children directly to avoid namespace issues with findall
                            for child in elem:
                                child_tag = etree.QName(child).localname
                                if child_tag == 'lane':
                                    if child.get('shape'):
                                        lane_points = SumoNetworkParser._parse_shape_string(child.get('shape'))
                                        # Heuristic: Use the shape with the most points (curves have more points)
                                        if len(lane_points) > len(best_points):
                                            best_points = lane_points

                            edges_dict[edge_id] = {
                                'id': edge_id,
                                'from_node': from_node, 
                                'to_node': to_node,     
                                'shape': best_points,   # List[(float, float)]
                                'name': edge_name,
                                'signal_group_id': -1 
                            }

                    elif tag == 'connection':
                        tl_id = elem.get('tl')
                        from_edge = elem.get('from')
                        link_index = elem.get('linkIndex')

                        if tl_id and link_index and from_edge in edges_dict:
                            try:
                                edges_dict[from_edge]['signal_group_id'] = int(link_index)
                            except ValueError:
                                pass

                    # Memory Management: Clear processed elements
                    elem.clear()
                    if elem.getparent() is not None:
                        while elem.getprevious() is not None:
                            del elem.getparent()[0]
            
            # --- POST-PROCESSING: GEOMETRY BACKFILL ---
            # Fix for "Missing Streets" (Implicit Geometry)
            node_map = {n['id']: (n['x'], n['y']) for n in nodes}
            
            for edge in edges_dict.values():
                if not edge['shape'] or len(edge['shape']) < 2:
                    p1 = node_map.get(edge['from_node'])
                    p2 = node_map.get(edge['to_node'])
                    
                    if p1 and p2:
                        edge['shape'] = [p1, p2]
            
            return nodes, list(edges_dict.values())

        except Exception as e:
            logging.error(f"SumoNetworkParser: Failed to parse {file_path}: {e}")
            raise e

    @staticmethod
    def _parse_shape_string(shape_str: str) -> List[Tuple[float, float]]:
        """
        Parses 'x1,y1 x2,y2' string into list of tuples.
        Handles optional Z coordinate (x,y,z) by ignoring Z.
        """
        points = []
        if not shape_str:
            return points
            
        try:
            parts = shape_str.strip().split(' ')
            for p in parts:
                if ',' in p:
                    coords = p.split(',')
                    # Ensure we have at least x,y
                    if len(coords) >= 2:
                        points.append((float(coords[0]), float(coords[1])))
        except Exception:
            pass
        return points