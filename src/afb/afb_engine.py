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
# File: src/afb/afb_engine.py
# Author: Gabriel Moraes
# Date: 2026-03-09
#
# Refactored V2 (2026-03-09): Full SOLID Compliance
# - SRP: Engine only orchestrates — strategies own their logic and state.
# - OCP: New strategies are added via register(), no code modified.
# - ISP: Returns typed FusionResult, accepts typed SensorReading.
# - DIP: Depends on FusionStrategy Protocol, not concrete classes.

"""
AFB Engine — Level 2 Safety Fallback Orchestrator.

Activated when the AuditorAgent detects anomalies in FuserAgent output.
Uses live sensor data + simple math to produce fused estimates.

SOLID Architecture:
    - Strategies are registered at init (OCP: extensible without modification).
    - Engine iterates registered strategies and picks the first that can_handle().
    - Strategies are ordered by priority (most capable first).
"""

import logging
from typing import Dict, List, Optional

from src.afb.models import SensorReading, FusionResult, NO_DATA
from src.afb.sensor_guard import SensorGuard
from src.afb.strategies import (
    FusionStrategy,
    TrimmedMeanStrategy,
    KalmanLiteStrategy,
    LastKnownGoodStrategy,
)

logger = logging.getLogger("Synapse.AFB")


class AFBEngine:
    """
    Algoritmo de Fusão Baseline (AFB) — Level 2 Safety Fallback.
    
    SOLID Design:
        - SRP: Only orchestrates strategy selection. No fusion math here.
        - OCP: Register new strategies without modifying engine code.
        - DIP: Depends on FusionStrategy Protocol, not concrete classes.
        - ISP: Typed inputs (SensorReading) and outputs (FusionResult).
    
    Usage:
        afb = AFBEngine()
        result = afb.fuse([
            SensorReading("cam_01", 25.0, 0.95),
            SensorReading("radar_01", 23.0, 0.95),
            SensorReading("waze_api", 30.0, 0.60),
        ])
        print(result.value, result.strategy, result.confidence)
    """

    # Default trust score when source has no configured score
    DEFAULT_TRUST = 0.5

    def __init__(self, guard: Optional[SensorGuard] = None):
        # Pre-filter: rejects individual bad readings before fusion
        self._guard = guard or SensorGuard()
        
        # Strategy registry — ordered by priority (most capable first)
        self._strategies: List[FusionStrategy] = []
        
        # Observability counters
        self._total_fusions: int = 0
        self._strategy_hits: Dict[str, int] = {}
        
        # Reference to last_known_good strategy for cache updates
        self._lkg_strategy: Optional[LastKnownGoodStrategy] = None
        
        # Register default strategies (OCP: can add more via register())
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Registers the 3 built-in strategies in priority order."""
        self.register(TrimmedMeanStrategy())
        
        kalman = KalmanLiteStrategy()
        self.register(kalman)
        
        lkg = LastKnownGoodStrategy()
        self.register(lkg)
        self._lkg_strategy = lkg  # Keep reference for cache management

    def register(self, strategy: FusionStrategy) -> None:
        """
        Registers a new fusion strategy (OCP).
        
        Strategies are evaluated in registration order.
        Register more capable strategies first.
        
        Args:
            strategy: Any object implementing the FusionStrategy Protocol.
        """
        self._strategies.append(strategy)
        self._strategy_hits[strategy.name] = 0
        logger.info(f"[AFB] Registered strategy: {strategy.name}")

    def fuse(
        self,
        readings: List[SensorReading],
        meh_baseline: Optional[float] = None,
        metric_key: str = "_global",
    ) -> FusionResult:
        """
        Main fusion entry point. Auto-selects strategy via registry.
        
        Args:
            readings: List of typed SensorReading from active sensors.
            meh_baseline: Historical average from MEH (for last_known_good decay).
            metric_key: Identifier for stateful strategies (Kalman, LKG).
        
        Returns:
            Typed FusionResult with value, strategy name, confidence, and source count.
        """
        self._total_fusions += 1
        
        # Step 1: Filter out NaN/None readings
        valid = [r for r in readings if r.value == r.value]  # NaN != NaN
        
        # Step 2: SensorGuard pre-filter (z-score per sensor)
        clean = self._guard.filter(valid)
        n_sensors = len(clean)
        
        # Context passed to strategies
        context = {
            "meh_baseline": meh_baseline,
            "metric_key": metric_key,
        }
        
        # Iterate strategies in priority order, pick first that can handle
        for strategy in self._strategies:
            if strategy.can_handle(n_sensors):
                result = strategy.fuse(clean, **context)
                self._strategy_hits[strategy.name] = (
                    self._strategy_hits.get(strategy.name, 0) + 1
                )
                
                # Update last known good cache on successful fusion
                if not result.is_degraded and self._lkg_strategy is not None:
                    self._lkg_strategy.update_cache(metric_key, result.value)
                
                logger.debug(
                    f"[AFB] {result.strategy}: {result.value:.2f} "
                    f"(conf={result.confidence:.2f}, sensors={result.source_count})"
                )
                return result
        
        # No strategy could handle — full degradation
        logger.warning("[AFB] No strategy available. Yielding to MEH (Level 3).")
        return NO_DATA

    def get_diagnostics(self) -> Dict[str, object]:
        """Returns observability info for telemetry/debugging."""
        return {
            "total_fusions": self._total_fusions,
            "strategy_hits": dict(self._strategy_hits),
            "registered_strategies": [s.name for s in self._strategies],
            "guard": self._guard.get_diagnostics(),
        }

    def reset(self) -> None:
        """Clears all internal state across all strategies and guard."""
        self._total_fusions = 0
        self._strategy_hits = {k: 0 for k in self._strategy_hits}
        self._guard.reset()
        for strategy in self._strategies:
            if hasattr(strategy, 'reset'):
                strategy.reset()
