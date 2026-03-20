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
# File: src/optimization/data_loader.py
# Author: Gabriel Moraes
# Date: 2026-02-15

import gzip
import torch
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from typing import Optional, Any, Union

# Try importing PyTorch Geometric Data object
try:
    from torch_geometric.data import Data
except ImportError:
    Data = None

from src.utils.logging_setup import logger

class DataLoader:
    """
    Handles data ingestion and sanitization for the Optimization Service.
    
    Responsibilities:
    - Loading raw Parquet files (Historical Traffic Data).
    - Parsing SUMO Network Maps (XML/GZ).
    - Data sanitization (NaN handling, Type coercion).
    """

    @staticmethod
    def load_parquet_data(path: str) -> Optional[np.ndarray]:
        """
        Loads and sanitizes historical traffic data from a Parquet file.
        
        Args:
            path: Absolute path to the .parquet file.
            
        Returns:
            np.ndarray: A float32 matrix [TimeSteps, Features] or None if failed.
        """
        try:
            # Read Raw Parquet
            df = pd.read_parquet(path)
            
            # Select only numeric columns (ignore timestamps/metadata)
            df_numeric = df.select_dtypes(include=[np.number])
            
            # Fallback: If no numeric types detected, attempt forced coercion
            if df_numeric.empty:
                logger.warning("[DataLoader] No numeric columns found. Attempting forced conversion...")
                for col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                df_numeric = df.select_dtypes(include=[np.number])

            # Final Sanitization: Fill NaNs/Infs with 0.0
            clean_matrix = df_numeric.fillna(0.0).values.astype(np.float32)
            clean_matrix = np.nan_to_num(clean_matrix, nan=0.0, posinf=0.0, neginf=0.0)
            
            return clean_matrix
            
        except Exception as e:
            logger.error(f"[DataLoader] Parquet Load Error: {e}")
            return None

    @staticmethod
    def load_sumo_map(path: str) -> Optional[Any]:
        """
        Parses a SUMO Network file (.net.xml or .net.xml.gz) into a PyTorch Geometric Data object.
        
        Args:
            path: Absolute path to the map file.
            
        Returns:
            torch_geometric.data.Data or Dict: Graph components ready for GATv2.
        """
        try:
            # Handle Gzipped XML
            if path.endswith('.gz'):
                with gzip.open(path, 'rb') as f:
                    tree = ET.parse(f)
            else:
                tree = ET.parse(path)
                
            root = tree.getroot()
            nodes = []
            edges = []
            node_map = {}

            # 1. Parse Junctions (Nodes)
            # Filter out internal SUMO junctions (starting with ':')
            for j in root.findall("junction"):
                jid = j.get("id")
                # We skip internal junctions to keep the graph semantic (only real intersections)
                if jid and not jid.startswith(":"):
                    node_map[jid] = len(nodes)
                    try:
                        x = float(j.get("x"))
                        y = float(j.get("y"))
                        # Normalize coordinates roughly to avoid massive values
                        nodes.append([x / 1000.0, y / 1000.0]) 
                    except (ValueError, TypeError):
                        nodes.append([0.0, 0.0])

            # 2. Parse Edges (Connections)
            for e in root.findall("edge"):
                # Skip internal edges
                eid = e.get("id")
                if eid and eid.startswith(":"):
                    continue
                    
                from_node = e.get("from")
                to_node = e.get("to")
                
                if from_node in node_map and to_node in node_map:
                    edges.append([node_map[from_node], node_map[to_node]])

            if not nodes:
                logger.warning("[DataLoader] Map parsed but no valid junctions found.")
                return None

            # 3. Construct Tensors
            x = torch.tensor(nodes, dtype=torch.float32)
            
            if edges:
                edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            else:
                # Fallback for disconnected graphs (single node map?)
                edge_index = torch.empty((2, 0), dtype=torch.long)

            # 4. Return Data Object (if PyG is available) or Dict
            if Data is not None:
                data = Data(x=x, edge_index=edge_index)
                data.num_nodes = len(nodes)
                return data
            else:
                # Fallback Dict
                return {
                    "x": x,
                    "edge_index": edge_index,
                    "num_nodes": len(nodes)
                }
            
        except Exception as e:
            logger.error(f"[DataLoader] Map Load Error: {e}")
            return None