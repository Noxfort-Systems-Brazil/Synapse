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
# File: src/agents/cartographer_agent.py
# Author: Gabriel Moraes
# Date: 2026-03-08

"""
CartographerAgent — The Hybrid Map Matcher.

Orchestrates the cascading pipeline:
    1. FastMapMatcher (CPU)  → R-Tree + Fréchet + Shannon Entropy
    2. SinkhornCrossAttention (GPU) → GATv2 Siamesa + Sinkhorn (when ambiguous)

Follows the same pattern as FuserAgent, AuditorAgent, CoordinatorAgent:
    - __init__: instantiate model + optimizer + scaler
    - inference(): AMP forward with domain-specific logic
    - train_step(): AMP forward + loss + backward
"""

import torch
import torch.optim as optim
import logging
from typing import Any, Dict, List, Tuple, Optional
from torch.amp import autocast, GradScaler

from src.agents.base_agent import BaseAgent
from src.models.sinkhorn_cross_attention import SinkhornCrossAttention
from src.services.fast_map_matcher import FastMapMatcher
from src.services.line_graph_builder import LineGraphBuilder
from src.domain.entities import MapEdge, MapNode

logger = logging.getLogger("Synapse.CartographerAgent")


class CartographerAgent(BaseAgent):
    """
    The Cartographer Agent ('O Cartógrafo').
    
    Cascading Pipeline:
    - Low entropy  → Fast heuristic match (CPU, microseconds)
    - High entropy → Neural disambiguation (GPU, milliseconds)
    
    Domain-Specific Methods:
    - set_map_data(): Injects map topology (like CoordinatorAgent.set_topology)
    - match_point(): Convenience wrapper for single GPS point matching
    - match_polyline(): Convenience wrapper for polyline matching
    """

    def __init__(
        self,
        # Neural hyperparams (tunable by Optuna)
        raw_dim: int = 11,
        d_model: int = 64,
        n_heads: int = 4,
        n_gat_layers: int = 2,
        sinkhorn_iters: int = 10,
        dropout: float = 0.1,
        temperature: float = 0.1,
        learning_rate: float = 1e-3,
        entropy_threshold: float = 0.7,
        crop_radius: float = 200.0,
    ):
        """
        Initializes the agent.

        Args:
            raw_dim: Feature vector size per Line Graph node.
            d_model: Latent embedding dimension.
            n_heads: Multi-head attention heads.
            n_gat_layers: GATv2 layers (shared Siamese).
            sinkhorn_iters: Doubly-stochastic normalization passes.
            dropout: Regularization dropout.
            temperature: Sinkhorn temperature.
            learning_rate: Adam optimizer learning rate.
            entropy_threshold: Shannon Entropy above this → neural path.
            crop_radius: Radius (meters) for sub-graph cropping.
        """
        # 1. Create Model
        model = SinkhornCrossAttention(
            raw_dim=raw_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_gat_layers=n_gat_layers,
            sinkhorn_iters=sinkhorn_iters,
            dropout=dropout,
            temperature=temperature,
        )

        # 2. Initialize Base
        super().__init__(model=model, name="CartographerAgent")

        # 3. Optimizer & Scaler
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scaler = GradScaler('cuda' if torch.cuda.is_available() else 'cpu')

        # 4. Cascade thresholds
        self.entropy_threshold = entropy_threshold
        self.crop_radius = crop_radius

        # 5. Services (injected via set_map_data)
        self.fast_matcher: Optional[FastMapMatcher] = None
        self.edges: List[MapEdge] = []
        self.nodes: List[MapNode] = []

    def _get_current_device(self) -> torch.device:
        """Introspects the model to find its actual physical location."""
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    # ─── Domain-Specific: Map Injection ───────────────────────────────────

    def set_map_data(self, edges: List[MapEdge], nodes: List[MapNode]):
        """
        Inject map topology (equivalent to CoordinatorAgent.set_topology).
        
        Builds the FastMapMatcher R-Tree index from the provided edges.
        Called by LifecycleOrchestrator during boot.
        """
        self.edges = edges
        self.nodes = nodes
        self.fast_matcher = FastMapMatcher(edges, entropy_threshold=self.entropy_threshold)

        logger.info(
            f"[{self.name}] 🗺️ Map data set: {len(edges)} edges, "
            f"{len(nodes)} nodes. R-Tree index built."
        )

    # ─── BaseAgent Interface ──────────────────────────────────────────────

    def inference(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Primary map matching inference (cascading pipeline).

        Args:
            input_data: Dict with keys:
                - 'x', 'y': GPS coordinates (SUMO local coords)
                - 'polyline': Optional list of (x, y) points

        Returns:
            Dict with: 'edge_id', 'confidence', 'method', 'entropy'
        """
        if self.fast_matcher is None:
            return {'edge_id': None, 'confidence': 0.0, 'method': 'no_map', 'entropy': 0.0}

        x = input_data.get('x', 0.0)
        y = input_data.get('y', 0.0)
        polyline = input_data.get('polyline')

        # ── STEP 1: Fast Heuristic (CPU) ──
        if polyline and len(polyline) >= 2:
            fast_result = self.fast_matcher.match_polyline(polyline, self.crop_radius)
        else:
            fast_result = self.fast_matcher.match(x, y, self.crop_radius)

        if not fast_result.candidates:
            return {'edge_id': None, 'confidence': 0.0, 'method': 'none', 'entropy': 0.0}

        # ── STEP 2: Check Entropy ──
        if not fast_result.is_ambiguous:
            return {
                'edge_id': fast_result.best_edge_id,
                'confidence': fast_result.best_probability,
                'method': 'heuristic',
                'entropy': fast_result.entropy,
            }

        # ── STEP 3: Neural Disambiguation (GPU) ──
        return self._neural_match(x, y, fast_result, polyline)

    def train_step(self, batch_data: Any) -> float:
        """
        Standard Training Step (AMP).
        
        Args:
            batch_data: Tuple (source_graph, target_graph, ground_truth_alignment)
        """
        self.model.train()
        device = self._get_current_device()

        source_graph, target_graph, gt_alignment = batch_data
        source_graph = source_graph.to(device)
        target_graph = target_graph.to(device)
        gt_alignment = gt_alignment.to(device)

        self.optimizer.zero_grad()

        device_type = device.type if device.type != 'mps' else 'cpu'

        with autocast(device_type=device_type, enabled=(device.type == 'cuda')):
            predicted = self.model(source_graph, target_graph)
            loss = SinkhornCrossAttention.alignment_loss(predicted, gt_alignment)

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return loss.item()

    # ─── Domain-Specific: Convenience Wrappers ────────────────────────────

    def match_point(self, x: float, y: float) -> Dict[str, Any]:
        """Convenience: match a single GPS point."""
        return self.inference({'x': x, 'y': y})

    def match_polyline(self, polyline: List[Tuple[float, float]]) -> Dict[str, Any]:
        """Convenience: match a polyline from external sensor."""
        return self.inference({'polyline': polyline})

    # ─── Private: Neural Path ─────────────────────────────────────────────

    def _neural_match(
        self,
        x: float,
        y: float,
        fast_result,
        polyline: Optional[List[Tuple[float, float]]],
    ) -> Dict[str, Any]:
        """Neural disambiguation using SinkhornCrossAttention."""
        self.model.eval()
        device = self._get_current_device()

        # 1. Build SUMO sub-graph (candidates + context)
        candidate_ids = {c.edge_id for c in fast_result.candidates}
        candidate_edges = [c.edge for c in fast_result.candidates]
        context_edges = LineGraphBuilder.get_context_edges(candidate_ids, self.edges)
        all_edges = list({e.id: e for e in candidate_edges + context_edges}.values())

        sumo_graph = LineGraphBuilder.build_from_edges(all_edges, self.nodes)
        if sumo_graph is None:
            return self._fallback(fast_result)

        # 2. Build Sensor graph
        if polyline and len(polyline) >= 2:
            sensor_graph = LineGraphBuilder.build_from_polyline(polyline)
        else:
            sensor_graph = LineGraphBuilder.build_from_polyline([
                (x - 10, y - 10), (x, y), (x + 10, y + 10)
            ])

        if sensor_graph is None:
            return self._fallback(fast_result)

        # 3. Forward (AMP)
        sumo_graph = sumo_graph.to(device)
        sensor_graph = sensor_graph.to(device)

        device_type = device.type if device.type != 'mps' else 'cpu'

        with torch.no_grad():
            with autocast(device_type=device_type, enabled=(device.type == 'cuda')):
                alignment = self.model(sumo_graph, sensor_graph)

        # 4. Extract best match from candidates only
        matches = SinkhornCrossAttention.get_best_matches(alignment)
        edge_ids = getattr(sumo_graph, 'edge_ids', [])

        best_id = fast_result.best_edge_id
        best_conf = 0.0

        for src_idx, tgt_idx, conf in matches:
            if src_idx < len(edge_ids):
                eid = edge_ids[src_idx]
                if eid in candidate_ids and conf > best_conf:
                    best_id = eid
                    best_conf = conf

        logger.info(
            f"[{self.name}] 🧠 Neural: H={fast_result.entropy:.2f} → "
            f"{best_id} (conf={best_conf:.3f})"
        )

        return {
            'edge_id': best_id,
            'confidence': float(best_conf),
            'method': 'neural',
            'entropy': fast_result.entropy,
        }

    @staticmethod
    def _fallback(fast_result) -> Dict[str, Any]:
        """Fallback to heuristic when neural path fails."""
        return {
            'edge_id': fast_result.best_edge_id,
            'confidence': fast_result.best_probability,
            'method': 'heuristic_fallback',
            'entropy': fast_result.entropy,
        }
