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
# File: src/services/line_graph_builder.py
# Author: Gabriel Moraes
# Date: 2026-03-08

"""
Line Graph Builder — Data Transformation Service.

Transforms a road network into a Line Graph suitable for GNN processing.
    Original graph: Nodes = junctions, Edges = streets
    Line graph:     Nodes = streets,   Edges = shared junctions

This is a data utility, NOT a neural model.
"""

import math
import torch
import logging
from typing import List, Tuple, Dict, Optional

try:
    from torch_geometric.data import Data
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    Data = None

from src.domain.entities import MapEdge, MapNode

logger = logging.getLogger("Synapse.LineGraphBuilder")


class LineGraphBuilder:
    """
    Converts road network topology to PyG Line Graph Data objects.
    
    Produces feature vectors with 11 dimensions per node (street):
    [x_start, y_start, x_end, y_end, length, sin_θ, cos_θ, deg_from, deg_to, speed, lanes]
    """

    @staticmethod
    def build_from_edges(
        edges: List[MapEdge],
        nodes: List[MapNode],
        bbox_normalize: bool = True,
    ) -> Optional['Data']:
        """
        Build a PyG Data object representing the Line Graph.

        Args:
            edges: MapEdge entities with shape geometry.
            nodes: MapNode entities (for topology info).
            bbox_normalize: Normalize coordinates to [0, 1].

        Returns:
            PyG Data with x=[num_edges, 11], edge_index=[2, connections],
            plus edge_ids attribute for mapping back to original IDs.
        """
        if not PYG_AVAILABLE or not edges:
            return None

        # Node degree lookup
        node_degree: Dict[str, int] = {}
        for e in edges:
            node_degree[e.from_node] = node_degree.get(e.from_node, 0) + 1
            node_degree[e.to_node] = node_degree.get(e.to_node, 0) + 1

        # Bounding box
        all_x, all_y = [], []
        for e in edges:
            for pt in e.shape:
                all_x.append(pt[0])
                all_y.append(pt[1])

        if not all_x:
            return None

        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        range_x = max(max_x - min_x, 1e-6)
        range_y = max(max_y - min_y, 1e-6)

        # ── Feature Vectors ──
        features = []
        for e in edges:
            if not e.shape or len(e.shape) < 2:
                features.append([0.0] * 11)
                continue

            x_s, y_s = e.shape[0]
            x_e, y_e = e.shape[-1]

            if bbox_normalize:
                x_s = (x_s - min_x) / range_x
                y_s = (y_s - min_y) / range_y
                x_e = (x_e - min_x) / range_x
                y_e = (y_e - min_y) / range_y

            length = e.weight if e.weight > 0 else 1.0
            dx = e.shape[-1][0] - e.shape[0][0]
            dy = e.shape[-1][1] - e.shape[0][1]
            angle = math.atan2(dy, dx)

            features.append([
                x_s, y_s, x_e, y_e,
                length,
                math.sin(angle), math.cos(angle),
                float(node_degree.get(e.from_node, 1)),
                float(node_degree.get(e.to_node, 1)),
                float(getattr(e, '_speed', 13.89)),
                float(getattr(e, '_num_lanes', 1)),
            ])

        feat_tensor = torch.tensor(features, dtype=torch.float32)

        # Normalize columns: length(4), deg_from(7), deg_to(8), speed(9), lanes(10)
        for col in [4, 7, 8, 9, 10]:
            c = feat_tensor[:, col]
            c_max = c.max()
            if c_max > 1e-6:
                feat_tensor[:, col] = c / c_max

        # ── Line Graph Edges ──
        junction_edges: Dict[str, List[int]] = {}
        for idx, e in enumerate(edges):
            junction_edges.setdefault(e.from_node, []).append(idx)
            junction_edges.setdefault(e.to_node, []).append(idx)

        src_list, dst_list = [], []
        for connected in junction_edges.values():
            for i in range(len(connected)):
                for j in range(i + 1, len(connected)):
                    src_list.extend([connected[i], connected[j]])
                    dst_list.extend([connected[j], connected[i]])

        if src_list:
            edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        data = Data(x=feat_tensor, edge_index=edge_index)
        data.num_nodes = len(edges)
        data.edge_ids = [e.id for e in edges]
        return data

    @staticmethod
    def build_from_polyline(
        polyline: List[Tuple[float, float]],
        bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> Optional['Data']:
        """
        Build a Line Graph from a raw polyline (e.g., Waze/TomTom).

        Args:
            polyline: List of (x, y) coordinates.
            bbox: Optional (min_x, min_y, max_x, max_y) for normalization.
        """
        if not PYG_AVAILABLE or not polyline or len(polyline) < 2:
            return None

        n_seg = len(polyline) - 1

        if bbox is None:
            xs = [p[0] for p in polyline]
            ys = [p[1] for p in polyline]
            bbox = (min(xs), min(ys), max(xs), max(ys))

        min_x, min_y, max_x, max_y = bbox
        range_x = max(max_x - min_x, 1e-6)
        range_y = max(max_y - min_y, 1e-6)

        features = []
        for i in range(n_seg):
            x_s, y_s = polyline[i]
            x_e, y_e = polyline[i + 1]
            dx, dy = x_e - x_s, y_e - y_s
            angle = math.atan2(dy, dx)

            features.append([
                (x_s - min_x) / range_x, (y_s - min_y) / range_y,
                (x_e - min_x) / range_x, (y_e - min_y) / range_y,
                math.sqrt(dx * dx + dy * dy),
                math.sin(angle), math.cos(angle),
                1.0, 1.0,    # degree placeholders
                0.0, 0.0,    # speed/lanes unknown
            ])

        feat_tensor = torch.tensor(features, dtype=torch.float32)
        l_col = feat_tensor[:, 4]
        l_max = l_col.max()
        if l_max > 1e-6:
            feat_tensor[:, 4] = l_col / l_max

        # Sequential edges
        src = list(range(n_seg - 1)) + list(range(1, n_seg))
        dst = list(range(1, n_seg)) + list(range(n_seg - 1))

        if src:
            edge_index = torch.tensor([src, dst], dtype=torch.long)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        data = Data(x=feat_tensor, edge_index=edge_index)
        data.num_nodes = n_seg
        return data

    @staticmethod
    def crop_subgraph(
        all_edges: List[MapEdge],
        center_x: float,
        center_y: float,
        radius: float,
    ) -> List[MapEdge]:
        """
        Extract edges whose shape centroid falls within radius of (center_x, center_y).
        """
        result = []
        r_sq = radius * radius

        for e in all_edges:
            if not e.shape:
                continue
            cx = sum(p[0] for p in e.shape) / len(e.shape)
            cy = sum(p[1] for p in e.shape) / len(e.shape)
            if (cx - center_x) ** 2 + (cy - center_y) ** 2 <= r_sq:
                result.append(e)

        return result

    @staticmethod
    def get_context_edges(
        candidate_ids: set,
        all_edges: List[MapEdge],
        max_context: int = 20,
    ) -> List[MapEdge]:
        """Get 1-hop neighboring edges for topological context."""
        junctions = set()
        for e in all_edges:
            if e.id in candidate_ids:
                junctions.add(e.from_node)
                junctions.add(e.to_node)

        context = []
        for e in all_edges:
            if e.id not in candidate_ids:
                if e.from_node in junctions or e.to_node in junctions:
                    context.append(e)
                    if len(context) >= max_context:
                        break
        return context
