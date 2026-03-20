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
# File: src/afb/sensor_guard.py
# Author: Gabriel Moraes
# Date: 2026-03-09

"""
SensorGuard — Z-Score Pre-Filter for AFB.

Filters out individual sensor readings that deviate significantly from
that sensor's own recent history BEFORE passing data to the AFB engine.

This solves the "1 bad sensor among many" problem:
    - Each sensor has its own rolling window of recent values.
    - A new reading is compared against that window using z-score.
    - If |z| > threshold, the reading is flagged as suspicious.

Design:
    - Pure statistics: mean + std of rolling window (no ML).
    - Per-sensor state: each sensor tracks independently.
    - Configurable sensitivity via z_threshold and window_size.
    - SOLID: implements a simple Protocol for future replacement.
"""

import math
import logging
from collections import deque
from typing import Dict, List, Tuple

from src.afb.models import SensorReading

logger = logging.getLogger("Synapse.AFB.Guard")


class SensorGuard:
    """
    Per-sensor Z-Score anomaly filter.
    
    Maintains a rolling window of recent values for each sensor.
    New readings that fall outside z_threshold standard deviations
    from the rolling mean are flagged and excluded from AFB fusion.
    
    Usage:
        guard = SensorGuard(window_size=30, z_threshold=3.0)
        clean = guard.filter(readings)
        # clean contains only trustworthy readings
    """

    def __init__(
        self,
        window_size: int = 30,
        z_threshold: float = 3.0,
        min_samples: int = 5,
    ):
        """
        Args:
            window_size: Number of recent readings to keep per sensor.
            z_threshold: Standard deviations before flagging (3.0 = 99.7% normal).
            min_samples: Minimum readings before filtering starts.
                         Below this, all readings pass (not enough history).
        """
        self._window_size = window_size
        self._z_threshold = z_threshold
        self._min_samples = min_samples
        
        # Rolling windows per sensor: {source_id: deque([values])}
        self._windows: Dict[str, deque] = {}
        
        # Counters for observability
        self.total_checked: int = 0
        self.total_rejected: int = 0

    def filter(self, readings: List[SensorReading]) -> List[SensorReading]:
        """
        Filters a batch of sensor readings, rejecting anomalous values.
        
        Returns only readings that pass the z-score test (or have
        insufficient history to be judged). Updates rolling windows
        for all readings, including rejected ones (to track drift).
        
        Args:
            readings: Raw sensor readings to evaluate.
            
        Returns:
            List of clean readings that passed the z-score check.
        """
        clean: List[SensorReading] = []
        
        for reading in readings:
            self.total_checked += 1
            is_trusted = self._evaluate(reading)
            
            # Always update the window (even rejected values contribute
            # to long-term drift tracking, just with lower weight)
            self._update_window(reading)
            
            if is_trusted:
                clean.append(reading)
            else:
                self.total_rejected += 1
                logger.warning(
                    f"[SensorGuard] ⚠️ Rejected '{reading.source_id}': "
                    f"value={reading.value:.2f} (exceeds {self._z_threshold}σ)"
                )
        
        return clean

    def _evaluate(self, reading: SensorReading) -> bool:
        """
        Checks if a reading is within acceptable range for its sensor.
        
        Returns:
            True if trusted, False if anomalous.
        """
        window = self._windows.get(reading.source_id)
        
        # Not enough history → pass through (can't judge yet)
        if window is None or len(window) < self._min_samples:
            return True
        
        mean, std = self._compute_stats(window)
        
        # Zero std means all values are identical → any deviation is suspicious
        if std < 1e-10:
            return abs(reading.value - mean) < 1e-10
        
        z_score = abs(reading.value - mean) / std
        return z_score <= self._z_threshold

    def _update_window(self, reading: SensorReading) -> None:
        """Appends the value to the sensor's rolling window."""
        if reading.source_id not in self._windows:
            self._windows[reading.source_id] = deque(maxlen=self._window_size)
        
        self._windows[reading.source_id].append(reading.value)

    @staticmethod
    def _compute_stats(window: deque) -> Tuple[float, float]:
        """Computes mean and standard deviation of the window."""
        n = len(window)
        if n == 0:
            return 0.0, 0.0
        
        mean = sum(window) / n
        
        if n < 2:
            return mean, 0.0
        
        variance = sum((x - mean) ** 2 for x in window) / (n - 1)  # Bessel's correction
        return mean, math.sqrt(variance)

    def get_diagnostics(self) -> Dict[str, object]:
        """Returns observability info."""
        return {
            "total_checked": self.total_checked,
            "total_rejected": self.total_rejected,
            "rejection_rate": (
                f"{(self.total_rejected / self.total_checked * 100):.1f}%"
                if self.total_checked > 0 else "0%"
            ),
            "tracked_sensors": len(self._windows),
            "window_sizes": {
                sid: len(w) for sid, w in self._windows.items()
            },
        }

    def reset(self) -> None:
        """Clears all sensor history."""
        self._windows.clear()
        self.total_checked = 0
        self.total_rejected = 0
