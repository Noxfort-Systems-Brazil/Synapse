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
# File: src/optimization/strategies_spatial.py
# Author: Gabriel Moraes
# Date: 2026-03-08

"""
Optuna Strategies for the Cartographer Agent (Spatial Alignment).

Training Method: Self-supervised via Synthetic Mutations.
    1. Extract random sub-graphs from the real .net.xml.gz map
    2. Create "mutant" copies with noise (shifted coords, removed edges)
    3. Train the SinkhornCrossAttention to recover the original alignment

Follows the same pattern as strategies_flow.py.
"""

import random
import logging
import torch
import torch.nn as nn
from typing import Any, List, Tuple
from torch.amp import autocast, GradScaler

try:
    from torch_geometric.data import Data
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False

from src.models.sinkhorn_cross_attention import SinkhornCrossAttention
from src.services.line_graph_builder import LineGraphBuilder
from src.domain.entities import MapEdge, MapNode

logger = logging.getLogger("Synapse.Strategies.Spatial")


class SpatialStrategies:
    """Optimization strategies for the Cartographer Agent."""

    @staticmethod
    def cartographer_strategy(
        trial,
        graph_data: Any,
        device: torch.device,
        n_epochs: int = 30,
        n_subgraphs_per_epoch: int = 10,
    ) -> float:
        """
        Optuna trial for the SinkhornCrossAttention model.

        Args:
            trial: Optuna trial object.
            graph_data: Dict with 'edges' (List[MapEdge]) and 'nodes' (List[MapNode]).
            device: Torch device.
            n_epochs: Training epochs per trial.
            n_subgraphs_per_epoch: Random subgraphs per epoch.

        Returns:
            Final mean loss (lower is better).
        """
        if not PYG_AVAILABLE:
            logger.error("[Spatial] PyTorch Geometric not available.")
            return float('inf')

        edges = graph_data.get('edges', [])
        nodes = graph_data.get('nodes', [])

        if len(edges) < 10:
            logger.error(f"[Spatial] Not enough edges: {len(edges)}")
            return float('inf')

        # ── Hyperparameters (Optuna Search Space) ──
        d_model = trial.suggest_categorical("cart_d_model", [32, 64, 128])
        n_heads = trial.suggest_categorical("cart_n_heads", [2, 4, 8])
        n_gat_layers = trial.suggest_int("cart_n_gat_layers", 2, 4)
        dropout = trial.suggest_float("cart_dropout", 0.05, 0.3)
        sinkhorn_iters = trial.suggest_int("cart_sinkhorn_iters", 5, 20)
        lr = trial.suggest_float("cart_lr", 1e-4, 1e-2, log=True)
        noise_scale = trial.suggest_float("cart_noise_scale", 5.0, 50.0)
        temperature = trial.suggest_float("cart_temperature", 0.05, 0.5, log=True)

        logger.info(
            f"[Spatial] 🌐 Trial {trial.number} starting | "
            f"d_model={d_model}, n_heads={n_heads}, layers={n_gat_layers}, "
            f"sinkhorn={sinkhorn_iters}, lr={lr:.5f}"
        )

        # ── Model ──
        model = SinkhornCrossAttention(
            raw_dim=11,
            d_model=d_model,
            n_heads=n_heads,
            n_gat_layers=n_gat_layers,
            sinkhorn_iters=sinkhorn_iters,
            dropout=dropout,
            temperature=temperature,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scaler = GradScaler(device=str(device))

        # ── Training Loop ──
        epoch_losses: List[float] = []

        for epoch in range(n_epochs):
            model.train()
            batch_losses: List[float] = []

            for _ in range(n_subgraphs_per_epoch):
                try:
                    # 1. Random Sub-Graph
                    sub_edges, sub_nodes = SpatialStrategies._random_subgraph(
                        edges, nodes, min_edges=5, max_edges=30
                    )
                    if len(sub_edges) < 3:
                        continue

                    # 2. Build Source Line Graph
                    source = LineGraphBuilder.build_from_edges(sub_edges, sub_nodes)
                    if source is None:
                        continue

                    # 3. Mutant
                    mut_edges, perm = SpatialStrategies._create_mutation(
                        sub_edges, noise_scale=noise_scale
                    )
                    mutant = LineGraphBuilder.build_from_edges(mut_edges, sub_nodes)
                    if mutant is None:
                        continue

                    # 4. Ground Truth Permutation Matrix
                    n_s, n_m = source.num_nodes, mutant.num_nodes
                    gt = torch.zeros(n_s, n_m)
                    for src_idx, mut_idx in enumerate(perm):
                        if src_idx < n_s and mut_idx < n_m:
                            gt[src_idx, mut_idx] = 1.0

                    source = source.to(device)
                    mutant = mutant.to(device)
                    gt = gt.to(device)

                    # 5. Forward + Loss
                    optimizer.zero_grad()
                    with autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                        predicted = model(source, mutant)
                        loss = SinkhornCrossAttention.alignment_loss(predicted, gt)

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    batch_losses.append(loss.item())

                except Exception as e:
                    logger.debug(f"[Spatial] Subgraph error: {e}")
                    continue

            if batch_losses:
                mean_loss = sum(batch_losses) / len(batch_losses)
                epoch_losses.append(mean_loss)

                trial.report(mean_loss, epoch)
                if trial.should_prune():
                    import optuna
                    raise optuna.TrialPruned()

                if epoch % 10 == 0:
                    logger.info(
                        f"[Spatial] Trial {trial.number} | "
                        f"Epoch {epoch}/{n_epochs} | Loss: {mean_loss:.4f}"
                    )

        if not epoch_losses:
            return float('inf')

        final_loss = sum(epoch_losses[-5:]) / len(epoch_losses[-5:])

        logger.info(
            f"[Spatial] Trial {trial.number} finished with value: {final_loss:.6f} "
            f"and parameters: {trial.params}"
        )

        return final_loss

    # ─── Synthetic Data ───────────────────────────────────────────────────

    @staticmethod
    def _random_subgraph(
        edges: List[MapEdge],
        nodes: List[MapNode],
        min_edges: int = 5,
        max_edges: int = 30,
    ) -> Tuple[List[MapEdge], List[MapNode]]:
        """BFS expansion from a random seed edge."""
        if len(edges) <= min_edges:
            return edges, nodes

        junction_map: dict = {}
        for e in edges:
            junction_map.setdefault(e.from_node, []).append(e)
            junction_map.setdefault(e.to_node, []).append(e)

        target = random.randint(min_edges, min(max_edges, len(edges)))
        seed = random.choice(edges)

        visited = {seed.id}
        result = [seed]
        frontier = {seed.from_node, seed.to_node}

        while len(result) < target and frontier:
            junction = random.choice(list(frontier))
            frontier.discard(junction)

            for neighbor in junction_map.get(junction, []):
                if neighbor.id not in visited:
                    visited.add(neighbor.id)
                    result.append(neighbor)
                    frontier.add(neighbor.from_node)
                    frontier.add(neighbor.to_node)
                    if len(result) >= target:
                        break

        node_ids = set()
        for e in result:
            node_ids.add(e.from_node)
            node_ids.add(e.to_node)

        node_lookup = {n.id: n for n in nodes}
        result_nodes = [node_lookup[nid] for nid in node_ids if nid in node_lookup]

        return result, result_nodes

    @staticmethod
    def _create_mutation(
        edges: List[MapEdge],
        noise_scale: float = 20.0,
        drop_prob: float = 0.1,
    ) -> Tuple[List[MapEdge], List[int]]:
        """Create noisy mutant: jitter + deletion + shuffle."""
        indices = list(range(len(edges)))

        # Deletion
        surviving = []
        for idx in indices:
            if random.random() > drop_prob or len(surviving) < 3:
                surviving.append(idx)

        # Shuffle
        random.shuffle(surviving)

        # Noisy copies
        mutants = []
        for orig_idx in surviving:
            e = edges[orig_idx]
            noisy_shape = [
                (px + random.gauss(0, noise_scale), py + random.gauss(0, noise_scale))
                for px, py in e.shape
            ]
            m = MapEdge(
                id=f"mut_{e.id}",
                from_node=e.from_node,
                to_node=e.to_node,
                shape=noisy_shape,
                weight=e.weight,
            )
            m._speed = getattr(e, '_speed', 13.89)
            m._num_lanes = getattr(e, '_num_lanes', 1)
            mutants.append(m)

        return mutants, surviving
