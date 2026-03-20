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
# File: src/afb/strategies.py
# Author: Gabriel Moraes
# Date: 2026-03-09
#
# Refactored V2 (2026-03-09): SOLID Compliance
# - OCP: Each strategy is a class implementing FusionStrategy Protocol.
# - ISP: Returns typed FusionResult instead of Dict[str, Any].
# - DIP: Engine depends on Protocol, not concrete classes.

"""
AFB Fusion Strategies — Pure Math, Zero Dependencies.

Each strategy implements the FusionStrategy Protocol, making the engine
extensible without modification (OCP).

Strategy Cascade:
    2a: TrimmedMeanStrategy  — ≥3 sensors  (remove outliers, weighted average)
    2b: KalmanLiteStrategy   — 1-2 sensors (simplified 1D Kalman filter)
    2c: LastKnownGoodStrategy — 0 sensors  (exponential decay to historical baseline)
"""

import math
from typing import List, Optional, Protocol, Tuple, Dict

from src.afb.models import SensorReading, FusionResult


# =============================================================================
# PROTOCOL (Contract for all strategies — DIP)
# =============================================================================

class FusionStrategy(Protocol):
    """
    Contract that all AFB strategies must satisfy.
    
    The engine depends on this Protocol, not on concrete classes (DIP).
    Adding a new strategy only requires implementing this interface (OCP).
    """

    @property
    def name(self) -> str:
        """Human-readable strategy identifier."""
        ...

    def can_handle(self, sensor_count: int) -> bool:
        """Returns True if this strategy is appropriate for the given sensor count."""
        ...

    def fuse(self, readings: List[SensorReading], **context) -> FusionResult:
        """Produces a fused estimate from the given sensor readings."""
        ...


# =============================================================================
# STRATEGY 2a: TRIMMED MEAN (≥3 sensors)
# =============================================================================

class TrimmedMeanStrategy:
    """
    Removes the highest and lowest values (outlier protection),
    then computes a trust-weighted average of the remaining readings.
    """

    @property
    def name(self) -> str:
        return "trimmed_mean"

    def can_handle(self, sensor_count: int) -> bool:
        return sensor_count >= 3

    def fuse(self, readings: List[SensorReading], **context) -> FusionResult:
        # Sort by value, trim top and bottom
        sorted_readings = sorted(readings, key=lambda r: r.value)
        trimmed = sorted_readings[1:-1]

        # Weighted average using trust scores
        total_weight = sum(r.trust_score for r in trimmed)
        if total_weight == 0:
            fused = sum(r.value for r in trimmed) / len(trimmed)
        else:
            fused = sum(r.value * r.trust_score for r in trimmed) / total_weight

        # Confidence: high when sensors agree (low variance in trimmed set)
        values = [r.value for r in trimmed]
        confidence = self._compute_confidence(values, fused)

        return FusionResult(
            value=fused,
            strategy=self.name,
            confidence=confidence,
            source_count=len(readings),
        )

    @staticmethod
    def _compute_confidence(values: List[float], mean_val: float) -> float:
        """Lower variance among trimmed sensors → higher confidence."""
        if len(values) <= 1:
            return 0.8
        variance = sum((v - mean_val) ** 2 for v in values) / len(values)
        return max(0.5, 1.0 - min(variance / (mean_val ** 2 + 1e-8), 0.5))


# =============================================================================
# STRATEGY 2b: KALMAN LITE (1-2 sensors)
# =============================================================================

class KalmanLiteStrategy:
    """
    Simplified 1D Kalman filter for temporal smoothing.
    Maintains internal state between calls for continuity.
    """

    def __init__(self, process_noise: float = 0.5):
        self._process_noise = process_noise
        # State per metric: {key: (estimate, uncertainty)}
        self._state: Dict[str, Tuple[float, float]] = {}

    @property
    def name(self) -> str:
        return "kalman_lite"

    def can_handle(self, sensor_count: int) -> bool:
        return 1 <= sensor_count < 3

    def fuse(self, readings: List[SensorReading], **context) -> FusionResult:
        # Weighted measurement from available readings
        total_trust = sum(r.trust_score for r in readings)
        if total_trust == 0:
            measurement = sum(r.value for r in readings) / len(readings)
        else:
            measurement = sum(r.value * r.trust_score for r in readings) / total_trust

        # Get or initialize Kalman state
        key = context.get("metric_key", "_global")
        prev_est, prev_unc = self._state.get(key, (measurement, 5.0))

        # Noise proportional to trust (lower trust → higher noise)
        avg_trust = total_trust / len(readings)
        measurement_noise = max(0.1, 2.0 * (1.0 - avg_trust))

        # Predict
        predicted_est = prev_est
        predicted_unc = prev_unc + self._process_noise

        # Update
        kalman_gain = predicted_unc / (predicted_unc + measurement_noise)
        new_est = predicted_est + kalman_gain * (measurement - predicted_est)
        new_unc = (1 - kalman_gain) * predicted_unc

        # Store state
        self._state[key] = (new_est, new_unc)

        # Confidence from uncertainty
        confidence = max(0.4, min(0.9, 1.0 - new_unc / 10.0))

        return FusionResult(
            value=new_est,
            strategy=self.name,
            confidence=confidence,
            source_count=len(readings),
        )

    def reset(self) -> None:
        """Clears Kalman state."""
        self._state.clear()


# =============================================================================
# STRATEGY 2c: LAST KNOWN GOOD (0 sensors)
# =============================================================================

class LastKnownGoodStrategy:
    """
    Exponential decay from last known value toward MEH baseline.
    Bridge between AFB and MEH — prevents sudden jumps.
    """

    def __init__(self, decay_rate: float = 0.01):
        self._decay_rate = decay_rate
        # Cache: {key: (value, timestamp)}
        self._cache: Dict[str, Tuple[float, float]] = {}

    @property
    def name(self) -> str:
        return "last_known_good"

    def can_handle(self, sensor_count: int) -> bool:
        return sensor_count == 0

    def fuse(self, readings: List[SensorReading], **context) -> FusionResult:
        import time

        key = context.get("metric_key", "_global")
        meh_baseline: Optional[float] = context.get("meh_baseline")
        now = time.time()

        cached = self._cache.get(key)
        if cached is None:
            # No data at all — signal degradation to MEH
            from src.afb.models import NO_DATA
            return NO_DATA

        last_value, last_time = cached
        elapsed = max(0.0, now - last_time)
        target = meh_baseline if meh_baseline is not None else 0.0

        # Exponential decay toward target
        decay_factor = math.exp(-self._decay_rate * elapsed)
        decayed = target + (last_value - target) * decay_factor

        # Confidence decays with time
        confidence = max(0.1, 0.7 * (0.99 ** elapsed))

        return FusionResult(
            value=decayed,
            strategy=self.name,
            confidence=confidence,
            source_count=0,
        )

    def update_cache(self, key: str, value: float) -> None:
        """Called by the engine after a successful fusion to store last known good."""
        import time
        self._cache[key] = (value, time.time())

    def reset(self) -> None:
        """Clears the cache."""
        self._cache.clear()
