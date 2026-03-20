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
# File: src/infrastructure/hft_serializers.py
# Author: Gabriel Mateus
# Date: 2026-02-19

import os
import time
import sys
from typing import Dict, Any

# Ensure proto modules are reachable
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

from proto import synapse_hft_pb2

class HFTSerializer:
    """
    Responsible solely for converting Python Dictionaries into gRPC Protobuf Messages.
    
    Refactored V2 (SRP Extraction & File Name Support):
    - Extracted from grpc_connector.py to isolate data transformation logic.
    - Handles Scenario packing (Binary/Graph) and Traffic Frame packing.
    - Extracts and attaches the original file name for exact binary replication.
    """

    @staticmethod
    def pack_scenario(data: Dict[str, Any]) -> synapse_hft_pb2.ScenarioDefinition:
        """
        Converts a map definition dict into a Protobuf ScenarioDefinition.
        Handles both file-based binary read and logical graph parsing.
        """
        scenario = synapse_hft_pb2.ScenarioDefinition()
        scenario.map_hash = data.get("map_hash", "v1.0")
        
        # 1. Binary Upload (Raw Map File)
        file_path = data.get("map_file_path")
        if file_path and os.path.exists(file_path):
            try:
                # Extract the exact file name (e.g., osm.net.xml.gz)
                file_name = os.path.basename(file_path)
                scenario.map_file_name = file_name
                
                with open(file_path, "rb") as f:
                    file_bytes = f.read()
                    scenario.map_file_content = file_bytes
            except Exception as e:
                print(f"[HFT-Serializer] Warning: File Read Error ({file_path}): {e}")
                
        # 1.5 JSON Peak Schedule Transfer (Offline Context)
        peak_schedule = data.get("peak_schedule_json")
        if peak_schedule:
            scenario.peak_schedule_json = peak_schedule

        # 2. Logical Topology (Nodes & Edges)
        if "graph" in data:
            # Nodes
            for n_data in data["graph"].get("nodes", []):
                node = scenario.graph.nodes.add()
                node.id = str(n_data.get("id"))
                node.type = str(n_data.get("type", "priority"))
                node.x = float(n_data.get("x", 0.0))
                node.y = float(n_data.get("y", 0.0))
                
                # TLS Logic Mapping
                tl_id = n_data.get("tl_logic_id")
                if tl_id:
                    node.tl_logic_id = str(tl_id)
            
            # Edges
            for e_data in data["graph"].get("edges", []):
                edge = scenario.graph.edges.add()
                edge.id = str(e_data.get("id"))
                edge.source_node = str(e_data.get("source_node"))
                edge.target_node = str(e_data.get("target_node"))
                edge.length = float(e_data.get("length", 100.0))
                edge.lanes = int(e_data.get("lanes", 1))
                edge.max_speed = float(e_data.get("limit", 13.89))
                edge.signal_group_id = int(e_data.get("signal_group_id", -1))

        # 3. Geometric Shapes (Polygons/Lanes)
        if "geometry" in data:
            shapes_data = data["geometry"].get("shapes", [])
            for s_data in shapes_data:
                shape = scenario.geometry.shapes.add()
                shape.edge_id = str(s_data.get("edge_id"))
                shape.coords.extend(s_data.get("coords", []))
                
        return scenario

    @staticmethod
    def pack_traffic_frame(data: Dict[str, Any]) -> synapse_hft_pb2.TrafficFrame:
        """
        Converts a realtime traffic snapshot into a Protobuf TrafficFrame.
        """
        frame = synapse_hft_pb2.TrafficFrame()
        frame.timestamp = float(data.get("timestamp", time.time()))
        
        # Generate a unique microsecond-based sequence ID
        frame.sequence_id = int(time.time() * 1000)
        
        edges_data = data.get("edges", {})
        for edge_id, metrics in edges_data.items():
            state = synapse_hft_pb2.EdgeState()
            state.occupancy = float(metrics.get("occupancy", 0.0))
            state.mean_speed = float(metrics.get("speed", 0.0))
            state.queue_length = int(metrics.get("queue", 0))
            state.density = float(metrics.get("density", 0.0))
            
            # Protobuf map assignment
            frame.edges[edge_id].CopyFrom(state)
            
        return frame