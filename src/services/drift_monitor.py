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
# File: src/services/drift_monitor.py
# Author: Gabriel Moraes
# Date: 2025-11-26

import numpy as np
from collections import deque
from typing import Optional, Tuple, Dict

class DriftMonitor:
    """
    Service responsible for calculating statistical drift (Data Drift).
    
    Metrics implemented:
    1. PSI (Population Stability Index): Measures the shift in distribution stability.
    2. KL Divergence (Kullback-Leibler): Measures relative entropy.
    
    Architecture:
    - Sliding Window vs Reference Distribution.
    - UPDATED: Dynamic Binning (Auto-Scale) to prevent zero-PSI on tight data ranges.
    - UPDATED: Verbose Logging for debugging.
    """

    def __init__(self, window_size: int = 1000, num_buckets: int = 20):
        """
        Initializes the Drift Monitor.
        
        Args:
            window_size: Sliding window size.
            num_buckets: Number of histogram bins. Increased to 20 for finer granularity.
        """
        self.window_size = window_size
        self.num_buckets = num_buckets
        
        # If None, they will be calculated dynamically during calibration
        self.bucket_range: Optional[Tuple[float, float]] = None
        
        # Sliding window for real-time data
        self.window = deque(maxlen=window_size)
        self.total_samples = 0
        
        # Baselines
        self.ref_probs: Optional[np.ndarray] = None
        
        # State
        self.is_ready = False
        self._epsilon = 1e-5 
        self.min_calibration_samples = 50

    def set_reference_data(self, data: np.ndarray):
        """
        Calculates the baseline distribution and DYNAMICALLY sets the bucket range.
        """
        if len(data) == 0: return

        # 1. Auto-Scale: Find the range of the data + Margin
        # This ensures our "ruler" matches the data scale (e.g., 35-45 instead of 0-120)
        data_min = float(np.min(data))
        data_max = float(np.max(data))
        margin = (data_max - data_min) * 0.2 # 20% margin
        if margin == 0: margin = 1.0 # Prevent singular range
        
        self.bucket_range = (data_min - margin, data_max + margin)
        
        print(f"[DriftMonitor] 📏 Auto-Scaling Bins: Range=[{self.bucket_range[0]:.2f}, {self.bucket_range[1]:.2f}] for {len(data)} samples.")

        # 2. Compute Reference Histogram
        counts, _ = np.histogram(data, bins=self.num_buckets, range=self.bucket_range)
        
        # Normalize
        counts = counts.astype(np.float32) + self._epsilon
        self.ref_probs = counts / np.sum(counts)
        
        self.is_ready = True

    def add_sample(self, value: float):
        self.window.append(value)
        self.total_samples += 1
        
        # Auto-Calibration Trigger
        if not self.is_ready and len(self.window) >= self.min_calibration_samples:
            self.set_reference_data(np.array(self.window))

    def compute_metrics(self) -> Dict[str, float]:
        base_metrics = {"sample_count": self.total_samples}

        if not self.is_ready:
            return {
                **base_metrics, "psi": 0.0, "kl_div": 0.0, "status_code": 0,
                "message": f"Warming Up ({len(self.window)}/{self.min_calibration_samples})"
            }

        # 1. Compute Current Distribution
        # CRITICAL: Must use the SAME bucket_range as Reference
        current_data = np.array(self.window)
        curr_counts, _ = np.histogram(current_data, bins=self.num_buckets, range=self.bucket_range)
        
        curr_counts = curr_counts.astype(np.float32) + self._epsilon
        curr_probs = curr_counts / np.sum(curr_counts)

        # 2. Calculate PSI
        psi_values = (curr_probs - self.ref_probs) * np.log(curr_probs / self.ref_probs)
        psi = np.sum(psi_values)

        # 3. Calculate KL
        kl_div = np.sum(curr_probs * np.log(curr_probs / self.ref_probs))
        
        # --- DEBUG LOGGING ---
        # This will print to terminal so you can see WHY it is 0.0 or moving
        # We only print if there's a change or periodically to avoid spam
        if self.total_samples % 50 == 0 or psi > 0.01:
            # Show the top 3 buckets to see movement
            top_ref_idx = np.argmax(self.ref_probs)
            top_curr_idx = np.argmax(curr_probs)
            print(f"[DriftMonitor] 🔍 Stats #{self.total_samples}: PSI={psi:.6f} | "
                  f"Peak Bucket: Ref={top_ref_idx}->Curr={top_curr_idx}")

        return {
            **base_metrics,
            "psi": float(psi),
            "kl_div": float(kl_div),
            "status_code": self._get_status_code(psi),
            "message": self._get_status_message(psi)
        }

    def _get_status_code(self, psi: float) -> int:
        if psi < 0.1: return 0
        elif psi < 0.25: return 1
        else: return 2

    def _get_status_message(self, psi: float) -> str:
        if psi < 0.1: return "Stable"
        elif psi < 0.25: return "Minor Drift"
        else: return "CRITICAL DRIFT"