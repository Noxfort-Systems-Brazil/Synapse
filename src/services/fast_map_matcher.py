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
# File: src/services/fast_map_matcher.py
# Author: Gabriel Moraes
# Date: 2026-03-08

"""
FastMapMatcher — The First Line of Defense (CPU Heuristic Layer).

Architecture:
    1. R-Tree Spatial Index  → Instant candidate filtering (µs)
    2. Fréchet / Hausdorff   → Exact geometric distance (ms)
    3. Shannon Entropy        → Ambiguity thermometer  

If entropy is low  → match is confident, return immediately.
If entropy is high → ambiguity detected, escalate to neural disambiguator.
"""

import math
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field

from rtree import index as rtree_index

from src.domain.entities import MapEdge
from src.utils.logging_setup import logger


# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MatchCandidate:
    """A single candidate edge with its matching score."""
    edge_id: str
    edge: MapEdge
    distance: float           # Geometric distance (meters)
    probability: float = 0.0  # Normalized probability [0, 1]


@dataclass
class MatchResult:
    """Complete result from the fast matching pipeline."""
    candidates: List[MatchCandidate]
    entropy: float              # Shannon Entropy H ∈ [0, log2(N)]
    best_edge_id: Optional[str] # Top candidate ID (may be None)
    best_probability: float     # Confidence of the best match [0, 1]
    is_ambiguous: bool          # True if entropy > threshold


# ─────────────────────────────────────────────────────────────────────────────
# FAST MAP MATCHER
# ─────────────────────────────────────────────────────────────────────────────

class FastMapMatcher:
    """
    CPU-only heuristic map matcher using R-Tree + geometric distance + entropy.
    
    Pipeline:
        match(lat, lon) → R-Tree filter → Fréchet distance → Probabilities → Entropy
    
    Usage:
        matcher = FastMapMatcher(edges)
        result = matcher.match(lat, lon, radius_m=200)
        if result.is_ambiguous:
            # Escalate to neural disambiguator
            ...
    """

    def __init__(self, edges: List[MapEdge], entropy_threshold: float = 0.7):
        """
        Args:
            edges: List of MapEdge entities with populated shape geometry.
            entropy_threshold: H above this → ambiguity (triggers neural).
        """
        self.edges = edges
        self.entropy_threshold = entropy_threshold
        self._edge_lookup: Dict[str, MapEdge] = {e.id: e for e in edges}
        
        # Build R-Tree spatial index
        self._rtree_idx = None
        self._rtree_id_map: Dict[int, str] = {}  # int_id → edge_id
        self._build_rtree()
        
        logger.info(
            f"[FastMapMatcher] ✅ R-Tree built: {len(edges)} edges indexed. "
            f"Entropy threshold: {entropy_threshold:.2f}"
        )

    # ─────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────────────────────────────

    def match(self, x: float, y: float, radius_m: float = 200.0) -> MatchResult:
        """
        Match a GPS coordinate to the nearest edge(s) using cascading heuristics.
        
        Args:
            x: X coordinate (SUMO local coords, not lat/lon).
            y: Y coordinate (SUMO local coords, not lat/lon).
            radius_m: Search radius in meters.
            
        Returns:
            MatchResult with candidates, entropy, and ambiguity flag.
        """
        # 1. R-Tree: Filter candidates within bounding radius
        candidate_ids = self._query_rtree(x, y, radius_m)
        
        if not candidate_ids:
            return MatchResult(
                candidates=[], entropy=0.0,
                best_edge_id=None, best_probability=0.0,
                is_ambiguous=False
            )
        
        # 2. Geometric Distance: Exact point-to-polyline distance
        candidates: List[MatchCandidate] = []
        for edge_id in candidate_ids:
            edge = self._edge_lookup[edge_id]
            dist = self._point_to_polyline_distance(x, y, edge.shape)
            candidates.append(MatchCandidate(
                edge_id=edge_id, edge=edge, distance=dist
            ))
        
        # Sort by distance (closest first)
        candidates.sort(key=lambda c: c.distance)
        
        # 3. Convert distances to probabilities (inverse softmax)
        self._compute_probabilities(candidates)
        
        # 4. Shannon Entropy
        entropy = self._shannon_entropy(candidates)
        
        # 5. Determine ambiguity
        is_ambiguous = entropy > self.entropy_threshold and len(candidates) > 1
        best = candidates[0] if candidates else None
        
        return MatchResult(
            candidates=candidates,
            entropy=entropy,
            best_edge_id=best.edge_id if best else None,
            best_probability=best.probability if best else 0.0,
            is_ambiguous=is_ambiguous
        )

    def match_polyline(self, polyline: List[Tuple[float, float]], radius_m: float = 200.0) -> MatchResult:
        """
        Match an entire polyline (e.g., from Waze/TomTom) against the map.
        Uses the centroid of the polyline for R-Tree query, then Fréchet for scoring.
        """
        if not polyline:
            return MatchResult([], 0.0, None, 0.0, False)
        
        # Centroid for R-Tree query
        cx = sum(p[0] for p in polyline) / len(polyline)
        cy = sum(p[1] for p in polyline) / len(polyline)
        
        candidate_ids = self._query_rtree(cx, cy, radius_m)
        if not candidate_ids:
            return MatchResult([], 0.0, None, 0.0, False)
        
        candidates: List[MatchCandidate] = []
        for edge_id in candidate_ids:
            edge = self._edge_lookup[edge_id]
            dist = self._frechet_distance(polyline, edge.shape)
            candidates.append(MatchCandidate(
                edge_id=edge_id, edge=edge, distance=dist
            ))
        
        candidates.sort(key=lambda c: c.distance)
        self._compute_probabilities(candidates)
        entropy = self._shannon_entropy(candidates)
        is_ambiguous = entropy > self.entropy_threshold and len(candidates) > 1
        best = candidates[0] if candidates else None
        
        return MatchResult(
            candidates=candidates,
            entropy=entropy,
            best_edge_id=best.edge_id if best else None,
            best_probability=best.probability if best else 0.0,
            is_ambiguous=is_ambiguous
        )

    # ─────────────────────────────────────────────────────────────────────
    # R-TREE INDEX
    # ─────────────────────────────────────────────────────────────────────

    def _build_rtree(self):
        """Build R-Tree spatial index from edge shapes."""
        p = rtree_index.Property()
        p.dimension = 2
        self._rtree_idx = rtree_index.Index(properties=p)
        
        for i, edge in enumerate(self.edges):
            if not edge.shape:
                continue
            
            # Compute bounding box of edge shape
            xs = [pt[0] for pt in edge.shape]
            ys = [pt[1] for pt in edge.shape]
            bbox = (min(xs), min(ys), max(xs), max(ys))
            
            self._rtree_idx.insert(i, bbox)
            self._rtree_id_map[i] = edge.id

    def _query_rtree(self, x: float, y: float, radius_m: float) -> List[str]:
        """Query R-Tree for edges within radius of point."""
        if self._rtree_idx is None:
            return []
        
        bbox = (x - radius_m, y - radius_m, x + radius_m, y + radius_m)
        hits = list(self._rtree_idx.intersection(bbox))
        return [self._rtree_id_map[h] for h in hits if h in self._rtree_id_map]

    # ─────────────────────────────────────────────────────────────────────
    # GEOMETRIC DISTANCE
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _point_to_polyline_distance(px: float, py: float, polyline: List[Tuple[float, float]]) -> float:
        """
        Minimum perpendicular distance from point (px, py) to a polyline.
        Uses segment-by-segment projection.
        """
        if not polyline:
            return float('inf')
        
        min_dist = float('inf')
        
        for i in range(len(polyline) - 1):
            ax, ay = polyline[i]
            bx, by = polyline[i + 1]
            
            # Vector AB
            abx = bx - ax
            aby = by - ay
            
            # Vector AP
            apx = px - ax
            apy = py - ay
            
            # Project AP onto AB, clamped to [0, 1]
            ab_sq = abx * abx + aby * aby
            if ab_sq < 1e-12:
                # Degenerate segment (zero length)
                dist = math.sqrt(apx * apx + apy * apy)
            else:
                t = max(0.0, min(1.0, (apx * abx + apy * aby) / ab_sq))
                # Closest point on segment
                cx = ax + t * abx
                cy = ay + t * aby
                dx = px - cx
                dy = py - cy
                dist = math.sqrt(dx * dx + dy * dy)
            
            min_dist = min(min_dist, dist)
        
        return min_dist

    @staticmethod
    def _frechet_distance(polyline_a: List[Tuple[float, float]], polyline_b: List[Tuple[float, float]]) -> float:
        """
        Discrete Fréchet distance between two polylines.
        Measures similarity between curves accounting for ordering.
        
        Complexity: O(n*m) where n,m are the polyline lengths.
        """
        n = len(polyline_a)
        m = len(polyline_b)
        
        if n == 0 or m == 0:
            return float('inf')
        
        # Dynamic programming table
        ca = np.full((n, m), -1.0)
        
        def _dist(i: int, j: int) -> float:
            dx = polyline_a[i][0] - polyline_b[j][0]
            dy = polyline_a[i][1] - polyline_b[j][1]
            return math.sqrt(dx * dx + dy * dy)
        
        def _frechet_rec(i: int, j: int) -> float:
            if ca[i, j] >= 0.0:
                return ca[i, j]
            
            d = _dist(i, j)
            
            if i == 0 and j == 0:
                ca[i, j] = d
            elif i == 0:
                ca[i, j] = max(_frechet_rec(0, j - 1), d)
            elif j == 0:
                ca[i, j] = max(_frechet_rec(i - 1, 0), d)
            else:
                ca[i, j] = max(
                    min(
                        _frechet_rec(i - 1, j),
                        _frechet_rec(i - 1, j - 1),
                        _frechet_rec(i, j - 1)
                    ),
                    d
                )
            return ca[i, j]
        
        # Use iterative approach for large polylines to avoid stack overflow
        if n * m > 10000:
            return FastMapMatcher._frechet_iterative(polyline_a, polyline_b)
        
        return _frechet_rec(n - 1, m - 1)

    @staticmethod
    def _frechet_iterative(polyline_a: List[Tuple[float, float]], polyline_b: List[Tuple[float, float]]) -> float:
        """Iterative Fréchet distance for large polylines."""
        n = len(polyline_a)
        m = len(polyline_b)
        ca = np.zeros((n, m))
        
        def _dist(i, j):
            dx = polyline_a[i][0] - polyline_b[j][0]
            dy = polyline_a[i][1] - polyline_b[j][1]
            return math.sqrt(dx * dx + dy * dy)
        
        ca[0, 0] = _dist(0, 0)
        for i in range(1, n):
            ca[i, 0] = max(ca[i - 1, 0], _dist(i, 0))
        for j in range(1, m):
            ca[0, j] = max(ca[0, j - 1], _dist(0, j))
        
        for i in range(1, n):
            for j in range(1, m):
                ca[i, j] = max(
                    min(ca[i - 1, j], ca[i - 1, j - 1], ca[i, j - 1]),
                    _dist(i, j)
                )
        
        return ca[n - 1, m - 1]

    # ─────────────────────────────────────────────────────────────────────
    # PROBABILITY & ENTROPY
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_probabilities(candidates: List[MatchCandidate]):
        """
        Convert distances to probabilities using inverse softmax.
        Closer distance → higher probability.
        """
        if not candidates:
            return
        
        # Inverse distances (add epsilon to avoid division by zero)
        eps = 1e-6
        inv_dists = np.array([1.0 / (c.distance + eps) for c in candidates])
        
        # Temperature-scaled softmax for sharper distributions
        temperature = 0.5
        scaled = inv_dists / temperature
        scaled -= scaled.max()  # Numerical stability
        exp_vals = np.exp(scaled)
        probs = exp_vals / exp_vals.sum()
        
        for i, c in enumerate(candidates):
            c.probability = float(probs[i])

    @staticmethod
    def _shannon_entropy(candidates: List[MatchCandidate]) -> float:
        """
        Shannon Entropy: H = -Σ p·log₂(p)
        
        Returns:
            H ∈ [0, log₂(N)]. Low = certainty, High = ambiguity.
        """
        if not candidates:
            return 0.0
        
        H = 0.0
        for c in candidates:
            if c.probability > 1e-10:
                H -= c.probability * math.log2(c.probability)
        
        return H
