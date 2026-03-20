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
# File: src/services/database_importer.py
# Author: Gabriel Moraes
# Date: 2026-02-27

import os
import shutil
import json
import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from PyQt6.QtCore import QObject, pyqtSignal

# Utils
from src.utils.frequency_finder import FrequencyFinder
from src.utils.parquet_validator import ParquetValidator

# Managers
from src.managers.storage_manager import StorageManager

class DatabaseImporter(QObject):
    """
    ETL Engine responsible for ingesting Raw Data.
    
    Refactored V3 (Corrected Path + Preserved Logic + Safety Locks):
    - Maintains FULL legacy SQLite support.
    - Maintains Direct Parquet Copy logic.
    - FIX: Saves output to 'datalake/base/base_v1.parquet' instead of 'data/db'.
    - ADDED: Minimum timespan validation for .parquet ingestion.
    """
    
    # Signals
    log_message = pyqtSignal(str)
    progress_update = pyqtSignal(int)
    import_finished = pyqtSignal(bool, str) # Success, Message

    def __init__(self, storage_manager: Optional[StorageManager] = None):
        super().__init__()
        # Allow injection, but fallback to new instance if not provided (for UI compatibility)
        if storage_manager:
            self.storage = storage_manager
        else:
            self.storage = StorageManager()
            
        self.target_freq_min = 1.0 
        self.min_required_points = 50 

    def execute_import(self, source_path_str: str, target_freq_min: float = 1.0):
        """
        Main entry point.
        """
        self.target_freq_min = target_freq_min
        
        abs_path = Path(os.path.abspath(source_path_str))
        self.log_message.emit(f"[ETL] Starting Import from: {abs_path}")
        self.progress_update.emit(5)

        if not abs_path.exists():
            self.import_finished.emit(False, f"File not found: {abs_path}")
            return

        try:
            # --- FIX: USE DATALAKE BASE PATH ---
            target_path = Path(self.storage.get_datalake_base_path()) / "base_v1.parquet"
            
            # Ensure directory exists (Safety net)
            target_path.parent.mkdir(parents=True, exist_ok=True)

            # --- BRANCH 1: DIRECT COPY (PARQUET) ---
            # If the user provides a Parquet, we assume it's already Golden/Formatted.
            if abs_path.suffix.lower() == '.parquet':
                self.log_message.emit("[ETL] Parquet detected. Executing Direct Faithful Copy...")
                
                try:
                    # 1. Verify readability (Sanity Check)
                    pd.read_parquet(abs_path) 
                    
                    # 2. Validate historical span (Minimum 7 days)
                    self.log_message.emit("[ETL] Validating historical span (minimum 7 days requirement)...")
                    try:
                        is_valid = ParquetValidator.validate_minimum_timespan(str(abs_path))
                        if not is_valid:
                            self.import_finished.emit(False, "Dataset rejected: Must contain at least 7 days of historical data.")
                            return
                    except Exception as ve:
                        self.import_finished.emit(False, f"Validation failed due to dataset structure: {ve}")
                        return
                    
                    # 3. Direct Binary Copy (shutil)
                    # This ensures the file is bitwise identical (100% faithful)
                    shutil.copy2(abs_path, target_path)
                    
                    self.log_message.emit(f"[ETL] Success. File copied to: {target_path}")
                    self.progress_update.emit(100)
                    self.import_finished.emit(True, "Import Successful (Direct Copy).")
                    return # STOP HERE. Do not resample.
                    
                except Exception as e:
                    self.import_finished.emit(False, f"Failed to verify/copy Parquet: {e}")
                    return

            # --- BRANCH 2: LEGACY ETL (SQLite/.db) ---
            # Only used for old Registry Mode or non-standard DBs
            elif abs_path.suffix.lower() == '.db':
                self.log_message.emit("[ETL] SQLite detected. Starting Processing Pipeline...")
                self._process_legacy_sqlite(abs_path, target_path)
                
            else:
                self.import_finished.emit(False, "Unsupported file format. Use .parquet or .db")

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.import_finished.emit(False, f"Critical ETL Error: {str(e)}")

    def _process_legacy_sqlite(self, abs_path: Path, target_path: Path):
        """
        Legacy logic for extracting, resampling and merging .db files.
        Only runs if input is NOT parquet.
        """
        raw_sources_map = self._extract_from_sqlite(str(abs_path))
        
        if not raw_sources_map:
            self.import_finished.emit(False, "No valid data found in SQLite.")
            return

        processed_series = {} 
        global_min_len = float('inf')
        source_names = list(raw_sources_map.keys())

        for i, name in enumerate(source_names):
            self.log_message.emit(f"[ETL] Processing table: '{name}'...")
            
            try:
                raw_df = raw_sources_map[name]
                time_col = self._find_time_column(raw_df)
                value_col = self._find_value_column(raw_df)
                
                if not time_col or not value_col:
                    continue

                # Parse & Index
                raw_df[time_col] = pd.to_datetime(raw_df[time_col], errors='coerce')
                raw_df = raw_df.dropna(subset=[time_col])
                raw_df.set_index(time_col, inplace=True)
                
                # Resample
                series_raw = pd.to_numeric(raw_df[value_col], errors='coerce').dropna()
                resampled = self._resample_series(series_raw, self.target_freq_min)
                
                if len(resampled) >= self.min_required_points:
                    processed_series[name] = resampled
                    global_min_len = min(global_min_len, len(resampled))
                
            except Exception as e:
                self.log_message.emit(f"   -> Error on '{name}': {e}")
                continue
            
            self.progress_update.emit(int(10 + (i / len(source_names)) * 80))

        if not processed_series:
            self.import_finished.emit(False, "ETL Failed: No valid series produced.")
            return

        # Synchronize & Save
        final_data = {name: s.iloc[:global_min_len].values for name, s in processed_series.items()}
        
        # Save Generated Parquet
        df_export = pd.DataFrame(final_data)
        df_export.insert(0, "step_index", range(global_min_len))
        df_export.to_parquet(target_path, engine='pyarrow', compression='snappy')
        
        self.progress_update.emit(100)
        self.import_finished.emit(True, "Import Successful (Processed SQLite).")

    # --- Helpers: SQLite Extraction ---

    def _extract_from_sqlite(self, db_path: str) -> Dict[str, pd.DataFrame]:
        sources = {}
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'view');")
            tables = [r[0] for r in cursor.fetchall() if not self._is_metadata_table(r[0])]
            
            for t in tables:
                df = pd.read_sql_query(f"SELECT * FROM {t}", conn)
                if not df.empty: sources[t] = df
            conn.close()
        except Exception as e:
            self.log_message.emit(f"[ETL] SQLite error: {e}")
        return sources

    def _is_metadata_table(self, table_name: str) -> bool:
        keywords = ['metadata', 'schema', 'sqlite', 'migration', 'config']
        return any(k in table_name.lower() for k in keywords)

    # --- Helpers: Analysis (Legacy) ---

    def _find_time_column(self, df: pd.DataFrame) -> Optional[str]:
        candidates = ['timestamp', 'time', 'date', 'datetime', 'ts']
        for col in df.columns:
            if str(col).lower() in candidates: return col
            if pd.api.types.is_datetime64_any_dtype(df[col]): return col
        return None

    def _find_value_column(self, df: pd.DataFrame) -> Optional[str]:
        for col in df.columns:
            if pd.api.types.is_float_dtype(df[col]): return col
        return None

    def _resample_series(self, series: pd.Series, target_freq_min: float) -> pd.Series:
        rule = f"{target_freq_min}min"
        return series.resample(rule).mean().interpolate(method='time', limit_direction='both')