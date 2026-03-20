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
# File: src/utils/frequency_finder.py
# Author: Gabriel Moraes
# Date: 2025-11-27

import numpy as np
from typing import Dict, Optional, Tuple

class FrequencyFinder:
    """
    Mathematical Detective for Time-Series Analysis.
    
    Responsibility:
    - Analyzes raw numerical sequences (without timestamps).
    - Detects periodicity using Autocorrelation (ACF) and FFT.
    - Estimates the likely sampling rate based on signal volatility (Zero-Crossing).
    """

    # Standard Traffic Cycle (24 hours in minutes)
    DAY_MINUTES = 1440

    @staticmethod
    def analyze_signal(data: np.ndarray) -> Dict[str, any]:
        """
        Main entry point. Runs multiple heuristics to guess the data frequency.
        
        Args:
            data: 1D Numpy array of traffic values (flow/speed).
            
        Returns:
            Dictionary with 'estimated_interval_min', 'confidence', 'type'.
        """
        # 1. Clean Data (Handle NaNs)
        clean_data = data[~np.isnan(data)]
        if len(clean_data) < 100:
            return {"type": "insufficient_data", "confidence": 0.0}

        # 2. Check for Daily Seasonality (Strongest Signal)
        period, confidence = FrequencyFinder._find_periodicity(clean_data)
        
        if confidence > 0.6:
            # If we found a cycle (period = N steps per day)
            # Interval = 1440 minutes / N steps
            interval = FrequencyFinder.DAY_MINUTES / period
            return {
                "type": "periodic",
                "estimated_interval_min": round(interval, 2),
                "confidence": confidence,
                "detected_period_steps": period
            }

        # 3. If no cycle, check Volatility (High Freq vs Low Freq)
        is_high_freq = FrequencyFinder._check_volatility(clean_data)
        
        if is_high_freq:
            return {
                "type": "high_frequency_noise",
                "estimated_interval_min": 0.016, # ~1 second (1/60 min)
                "confidence": 0.4,
                "note": "Highly volatile signal (likely raw sensor)"
            }
        else:
            return {
                "type": "low_frequency_step",
                "estimated_interval_min": 5.0, # Guess 5 min for stable API data
                "confidence": 0.3,
                "note": "Stable/Stepped signal (likely API)"
            }

    @staticmethod
    def _find_periodicity(data: np.ndarray) -> Tuple[int, float]:
        """
        Uses Autocorrelation to find the dominant lag (cycle).
        """
        # Safety Check: If signal is constant (variance ~ 0), no periodicity exists.
        if np.std(data) < 1e-6:
            return 0, 0.0

        # Normalize
        data = (data - np.mean(data)) / (np.std(data) + 1e-5)
        
        # Compute Autocorrelation
        n = len(data)
        variance = np.var(data)
        
        # Double check variance again after normalization (should be ~1, but safety first)
        if variance < 1e-6:
            return 0, 0.0

        data = data - np.mean(data)
        r = np.correlate(data, data, mode='full')[-n:]
        
        # Avoid division by zero in the formula
        denominator = variance * (np.arange(n, 0, -1))
        denominator[denominator == 0] = 1e-5 # Patch zeros
        
        result = r / denominator
        
        # Find peaks in the autocorrelation
        # We ignore the first few lags (noise) and look for the first major peak
        # Heuristic: Min cycle length for traffic is usually ~1 hour (so at least 12 steps if 5min)
        min_lag = 12 
        if len(result) <= min_lag: return 0, 0.0
        
        peaks = []
        for i in range(min_lag, len(result) - 1):
            if result[i] > result[i-1] and result[i] > result[i+1]:
                peaks.append((i, result[i]))
        
        if not peaks:
            return 0, 0.0
            
        # Sort by correlation strength
        peaks.sort(key=lambda x: x[1], reverse=True)
        best_lag, strength = peaks[0]
        
        return int(best_lag), float(strength)

    @staticmethod
    def _check_volatility(data: np.ndarray) -> bool:
        """
        Determines if signal is 'nervous' (High Freq) or 'calm' (Low Freq).
        Uses Zero-Crossing Rate on the derivative.
        """
        # Calculate changes (derivative)
        diff = np.diff(data)
        
        # Count how many times the change switches sign (up/down/up)
        # High freq noise switches sign constantly.
        zero_crossings = np.where(np.diff(np.sign(diff)))[0]
        rate = len(zero_crossings) / len(data)
        
        # Threshold: If it changes direction more than 30% of the time, it's likely raw noise
        return rate > 0.3