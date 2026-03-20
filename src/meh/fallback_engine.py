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
# File: src/meh/fallback_engine.py
# Author: Gabriel Moraes
# Date: 2026-02-28

import gc
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, TYPE_CHECKING
from src.utils.logging_setup import logger

# Lazy import for type hinting to avoid circular dependencies
if TYPE_CHECKING:
    from src.meh.data_loader import HistoricalDataLoader

class FallbackEngine:
    """
    Single Responsibility: Fallback Logic & Digital Twin Simulation.
    Calculates dynamic sensor frequencies, builds historical profiles,
    and resolves the Level 1/Level 2 degradation cascade for MEH state.
    """
    
    # Maximum time bins per sensor to cap memory
    MAX_BINS_PER_SENSOR = 10_000
    
    def __init__(self, data_loader: 'HistoricalDataLoader'):
        self.loader = data_loader
        # Lightweight dict profiles: {sensor_id: {(day, bin): {col: value}}}
        self.sensor_profiles: Dict[str, Dict[tuple, Dict[str, float]]] = {}
        self.sensor_frequencies: Dict[str, float] = {}
        self.is_ready = False

    def build_profiles(self):
        """
        Analyzes historical data to find sensor frequencies and build aggregated time bins.
        Must be called after the DataLoader has successfully loaded the data.
        
        Memory-Optimized V3:
        - Uses DataLoader's sensor catalog instead of re-detecting.
        - Converts profiles to lightweight dict-of-dicts (drops DataFrames).
        - Caps bins per sensor to prevent memory explosion.
        - Explicit gc.collect() after each sensor to free intermediate memory.
        """
        if not self.loader.is_loaded or self.loader.data is None or self.loader.data.empty:
            logger.warning("[FallbackEngine] Cannot build profiles: DataLoader is empty or not loaded.")
            self.is_ready = False
            return

        df = self.loader.data  # Reference, NOT copy
        
        # Use the DataLoader's detected group column
        group_col = self.loader.group_column
        if not group_col:
            # Fallback: add synthetic column
            df['sensor_id'] = 'global_sensor'
            group_col = 'sensor_id'

        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns.tolist()
                        if c not in ['_day_of_week', '_seconds_since_midnight']]

        # Pre-compute helper columns in-place (once for all sensors)
        _added_cols = []
        if '_seconds_since_midnight' not in df.columns:
            df['_seconds_since_midnight'] = (
                df['timestamp'].dt.hour * 3600 +
                df['timestamp'].dt.minute * 60 +
                df['timestamp'].dt.second
            )
            _added_cols.append('_seconds_since_midnight')
        if '_day_of_week' not in df.columns:
            df['_day_of_week'] = df['timestamp'].dt.dayofweek
            _added_cols.append('_day_of_week')

        logger.info(f"[FallbackEngine] Building profiles for {len(self.loader.sensor_ids)} sensors ({len(df)} records)...")

        grouped = df.groupby(group_col)
        
        for sensor_id, group_df in grouped:
            # Calculate dynamic frequency (Digital Twin heartbeat)
            if len(group_df) > 1:
                deltas = group_df['timestamp'].diff().dt.total_seconds().dropna()
                freq_sec = float(deltas.median())
                if freq_sec <= 0 or pd.isna(freq_sec): 
                    freq_sec = 1.0
            else:
                freq_sec = 1.0
                
            self.sensor_frequencies[sensor_id] = freq_sec
            
            # Compute time bins using the pre-calculated seconds column
            time_bins = (group_df['_seconds_since_midnight'] // freq_sec) * freq_sec
            
            # Build aggregated profile — numeric columns only (no text copy overhead)
            agg_dict = {col: 'mean' for col in numeric_cols 
                        if col in group_df.columns and col not in ['_day_of_week', '_seconds_since_midnight']}

            if not agg_dict:
                continue

            # Minimal temporary DataFrame (only needed columns)
            cols_needed = list(agg_dict.keys()) + ['_day_of_week']
            cols_needed = [c for c in cols_needed if c in group_df.columns]
            temp = group_df[cols_needed].copy()
            temp['time_bin'] = time_bins.values
            temp['day_of_week'] = temp['_day_of_week']
            if '_day_of_week' in temp.columns:
                temp.drop(columns=['_day_of_week'], inplace=True)

            profile_df = temp.groupby(['day_of_week', 'time_bin']).agg(agg_dict).reset_index()
            
            # Cap bins to prevent memory explosion
            if len(profile_df) > self.MAX_BINS_PER_SENSOR:
                profile_df = profile_df.sample(n=self.MAX_BINS_PER_SENSOR, random_state=42)
            
            # Convert to lightweight dict-of-dicts: {(day, bin): {col: val}}
            profile_dict = {}
            value_cols = [c for c in profile_df.columns if c not in ['day_of_week', 'time_bin']]
            for _, row in profile_df.iterrows():
                key = (int(row['day_of_week']), float(row['time_bin']))
                profile_dict[key] = {col: float(row[col]) for col in value_cols if pd.notna(row[col])}
            
            self.sensor_profiles[sensor_id] = profile_dict
            
            # Free intermediate memory immediately
            del temp, profile_df
            gc.collect()
            
            logger.debug(f"[FallbackEngine] Sensor '{sensor_id}' mapped: Freq={freq_sec:.2f}s, Bins={len(profile_dict)}.")

        # Cleanup: remove helper columns from the source DataFrame
        for col in _added_cols:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

        gc.collect()
        self.is_ready = True
        logger.info(f"[FallbackEngine] Historical profiles built successfully. {len(self.sensor_profiles)} sensors mapped. Ready for fallback.")

    def get_fallback_state(self, target_time: datetime) -> Dict[str, Any]:
        """
        Resolves the temporal lookup matching the Level 1 and Level 2 cascade logic.
        Uses lightweight dict-based profiles (V3).
        Returns a flat dictionary mapping expected values for the target time.
        """
        if not self.is_ready:
            return {}

        target_day = target_time.weekday()
        seconds_since_midnight = target_time.hour * 3600 + target_time.minute * 60 + target_time.second
        
        flat_payload = {}

        for sensor_id, profile_dict in self.sensor_profiles.items():
            freq_sec = self.sensor_frequencies.get(sensor_id, 1.0)
            target_bin = (seconds_since_midnight // freq_sec) * freq_sec
            
            # Attempt Level 1: Exact Match (Same Day + Same Bin)
            key_exact = (target_day, target_bin)
            match_data = profile_dict.get(key_exact)
            
            if match_data is None:
                # Attempt Level 2: Broad Match (Average across all days for same bin)
                same_bin_entries = [
                    vals for (day, tbin), vals in profile_dict.items()
                    if tbin == target_bin
                ]
                if same_bin_entries:
                    # Average across matched entries
                    match_data = {}
                    all_keys = same_bin_entries[0].keys()
                    for k in all_keys:
                        values = [e[k] for e in same_bin_entries if k in e]
                        match_data[k] = sum(values) / len(values) if values else 0.0
                else:
                    # Missing bin entirely, skip this sensor for this tick
                    continue
            
            # Semantic Translation step
            translated_data = self._apply_semantic_translation(match_data)
            flat_payload.update(translated_data)

        return flat_payload

    def _apply_semantic_translation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Appends semantic metadata if an ontology exists in the DataLoader."""
        if not self.loader.ontology:
            return data
            
        processed_data = data.copy()
        for key, value in data.items():
            if isinstance(value, str) and value.strip():
                # Flag for downstream NLP / Neuro-symbolic integration
                processed_data[f"{key}_semantic"] = "translated_from_history"
                
        return processed_data