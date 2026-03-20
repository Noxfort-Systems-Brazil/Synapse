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
# File: src/meh/data_loader.py
# Author: Gabriel Moraes
# Date: 2026-02-28

import os
import gc
import pandas as pd
import numpy as np
import torch
from safetensors.torch import load_file
from typing import Optional, Dict, List

from src.managers.storage_manager import StorageManager
from src.utils.logging_setup import logger

class HistoricalDataLoader:
    """
    Single Responsibility: I/O Operations and basic statistical extraction.
    Loads the Golden Parquet dataset and Safetensors ontology into memory.
    """
    # Maximum number of rows to load to prevent OOM on huge datasets
    MAX_ROWS = 500_000
    
    # Possible sensor ID column names (priority order)
    SENSOR_ID_CANDIDATES = ['sensor_id', 'device_id', 'camera_id', 'source_id', 'id']

    def __init__(self):
        self.storage = StorageManager()
        
        # Path resolution
        self.gold_path = os.path.join(self.storage.get_datalake_golden_path(), "golden_v1.parquet")
        
        # Resolve config path for Safetensors ontology
        base_dir = os.path.dirname(os.path.dirname(self.storage.get_datalake_golden_path()))
        self.config_path = os.path.join(base_dir, "data", "config", "ontology.safetensors")
        
        self.data: Optional[pd.DataFrame] = None
        self.ontology: Dict[str, torch.Tensor] = {}
        self.stats_cache: Dict[str, Dict[str, float]] = {}
        self.is_loaded = False
        
        # Sensor Catalog (built during load)
        self.group_column: Optional[str] = None
        self.sensor_ids: List[str] = []

    def load(self) -> bool:
        """Loads data from disk into memory."""
        if not os.path.exists(self.gold_path):
            logger.warning(f"[HistoricalDataLoader] ⚠️ Golden Dataset not found at: {self.gold_path}")
            logger.info(f"[HistoricalDataLoader] 💡 Tip: Run Phase 1 (Offline Bootstrap) first.")
            self.is_loaded = False
            return False

        try:
            # 1. Load Ontology (Safetensors)
            if os.path.exists(self.config_path):
                self.ontology = load_file(self.config_path)
                logger.info(f"[HistoricalDataLoader] 📚 Loaded ontology with {len(self.ontology)} semantic concepts.")
            else:
                logger.warning("[HistoricalDataLoader] ⚠️ Ontology safetensors not found. Semantic translation bypassed.")

            # 2. Load Golden Data (Memory-Optimized)
            self.data = pd.read_parquet(self.gold_path)
            
            # 2a. Cap rows to prevent OOM on huge datasets
            original_len = len(self.data)
            if original_len > self.MAX_ROWS:
                logger.warning(
                    f"[HistoricalDataLoader] ⚠️ Dataset has {original_len} rows, "
                    f"capping to {self.MAX_ROWS} most recent rows to save memory."
                )
                self.data = self.data.tail(self.MAX_ROWS).reset_index(drop=True)
                gc.collect()
            
            # 2b. Downcast numeric columns to float32 (halves RAM usage)
            float_cols = self.data.select_dtypes(include=['float64']).columns
            for col in float_cols:
                self.data[col] = self.data[col].astype(np.float32)
            
            int_cols = self.data.select_dtypes(include=['int64']).columns
            for col in int_cols:
                self.data[col] = pd.to_numeric(self.data[col], downcast='integer')
            
            # Auto-detect time column
            time_cols = ['timestamp', 'event_timestamp', 'time', 'datetime', 'date']
            time_col = next((col for col in time_cols if col in self.data.columns), None)
            
            # Ensure proper datetime indexing
            if time_col:
                if time_col != 'timestamp':
                    self.data.rename(columns={time_col: 'timestamp'}, inplace=True)
                
                self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
                self.data.set_index('timestamp', inplace=True, drop=False)
                self.data.index.name = 'time_idx'
                self.data.sort_index(inplace=True)
            
            # 2c. Build Sensor Catalog
            self._build_sensor_catalog()
            
            self.is_loaded = True
            self._calculate_statistics()
            
            mem_mb = self.data.memory_usage(deep=True).sum() / (1024 * 1024)
            logger.info(
                f"[HistoricalDataLoader] ✅ Loaded Golden History: {len(self.data)} records "
                f"({mem_mb:.1f} MB RAM). Sensors detected: {len(self.sensor_ids)}."
            )
            return True
            
        except Exception as e:
            logger.error(f"[HistoricalDataLoader] ❌ Failed to load dataset: {str(e)}")
            import traceback
            traceback.print_exc()
            self.is_loaded = False
            return False

    def _calculate_statistics(self):
        """Calculates Mean/Std for all numeric columns to serve KSE expected values."""
        if not self.is_loaded or self.data is None: 
            return
            
        numeric_df = self.data.select_dtypes(include=[np.number])
        for col in numeric_df.columns:
            self.stats_cache[col] = {
                "mean": float(numeric_df[col].mean()),
                "std": float(numeric_df[col].std()),
                "min": float(numeric_df[col].min()),
                "max": float(numeric_df[col].max())
            }

    def get_context_window(self, window_size: int = 60) -> np.ndarray:
        """Returns the last N records from history to serve as a 'warm start' buffer."""
        if not self.is_loaded or self.data is None:
            return np.zeros((window_size, 1))

        numeric_df = self.data.select_dtypes(include=[np.number])
        data_values = numeric_df.values
        
        current_len = len(data_values)
        if current_len < window_size:
            padding = np.zeros((window_size - current_len, data_values.shape[1]))
            return np.vstack([padding, data_values])
        
        return data_values[-window_size:]

    def get_expected_value(self, metric_name: str) -> float:
        """Returns the historical mean for a specific metric."""
        if not self.is_loaded: 
            return 0.0
        if metric_name in self.stats_cache:
            return self.stats_cache[metric_name].get("mean", 0.0)
        return 0.0

    def _build_sensor_catalog(self):
        """Detects the sensor grouping column and builds a catalog of sensor IDs."""
        if self.data is None or self.data.empty:
            return
        
        self.group_column = next(
            (col for col in self.SENSOR_ID_CANDIDATES if col in self.data.columns), 
            None
        )
        
        if self.group_column:
            self.sensor_ids = self.data[self.group_column].unique().tolist()
            logger.info(f"[HistoricalDataLoader] 🔍 Sensor column: '{self.group_column}' → {len(self.sensor_ids)} sensors found.")
        else:
            # No sensor column → treat entire dataset as one global sensor
            self.sensor_ids = ['global_sensor']
            logger.info("[HistoricalDataLoader] 🔍 No sensor column detected. Treating as single global sensor.")

    def get_sensor_data(self, sensor_id: str) -> Optional[pd.DataFrame]:
        """Returns only the rows for a specific sensor. Efficient filtered access."""
        if not self.is_loaded or self.data is None:
            return None
        
        if not self.group_column or sensor_id == 'global_sensor':
            return self.data
        
        mask = self.data[self.group_column] == sensor_id
        return self.data.loc[mask]

    def get_exact_reading(self, sensor_id: str, target_timestamp: float, tolerance: float = 0.25) -> Optional[float]:
        """
        Per-sensor temporal lookup in the Golden Database.
        
        Searches for a historical reading that matches the current time-of-day
        for a specific sensor, within a tolerance window.
        
        Args:
            sensor_id: The sensor/node ID to query.
            target_timestamp: Unix epoch timestamp (time.time()).
            tolerance: Maximum allowed time difference in seconds.
            
        Returns:
            Float value if a match is found, None otherwise.
        """
        if not self.is_loaded or self.data is None:
            return None
        
        from datetime import datetime
        target_dt = datetime.fromtimestamp(target_timestamp)
        target_seconds = target_dt.hour * 3600 + target_dt.minute * 60 + target_dt.second
        
        # Get sensor-specific data
        sensor_df = self.get_sensor_data(sensor_id)
        if sensor_df is None or sensor_df.empty:
            return None
        
        # Calculate seconds-of-day for each row
        if 'timestamp' in sensor_df.columns:
            ts_col = sensor_df['timestamp']
            row_seconds = ts_col.dt.hour * 3600 + ts_col.dt.minute * 60 + ts_col.dt.second
        else:
            return None
        
        # Find closest match within tolerance
        diff = (row_seconds - target_seconds).abs()
        min_diff = diff.min()
        
        if min_diff <= tolerance:
            best_idx = diff.idxmin()
            row = sensor_df.loc[best_idx]
            
            # Return the first numeric value column (typically speed/flow/density)
            numeric_cols = [c for c in sensor_df.select_dtypes(include=[np.number]).columns
                          if c not in ['_day_of_week', '_seconds_since_midnight']]
            if numeric_cols:
                return float(row[numeric_cols[0]])
        
        return None

    @property
    def columns(self) -> List[str]:
        if self.data is not None: 
            return list(self.data.columns)
        return []