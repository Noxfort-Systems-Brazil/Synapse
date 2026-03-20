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
# File: src/stages/sanitization_stage.py
# Author: Gabriel Moraes
# Date: 2026-03-02

import os
import pandas as pd
import numpy as np
from typing import Dict, Any

from src.stages.base_stage import BaseStage
from src.agents.corrector_agent import CorrectorAgent
from src.agents.imputer_agent import ImputerAgent

class SanitizationStage(BaseStage):
    """
    Sanitization Stage (Step 0, 1, and 2).
    
    Responsible solely for:
    - Finding the raw input data.
    - Creating the base backup.
    - Running the CorrectorAgent (VAE-TCN) to remove outliers/noise.
    - Running the ImputerAgent (TimeGAN) to fill temporal gaps.
    - Generating the optimized 'golden_v1.parquet' dataset.
    """

    def execute(self, shared_context: Dict[str, Any]) -> bool:
        """Executes the data cleaning and imputation pipeline."""
        golden_path = shared_context.get("golden_path")
        base_dir = shared_context.get("base_dir")
        
        # Check if we can intelligently skip this heavy phase
        golden_valid = os.path.exists(golden_path) and os.path.getsize(golden_path) > 0
        if golden_valid:
            self.log(f"[SanitizationStage] 🏆 Golden dataset validated: {golden_path}")
            self.log("[SanitizationStage] ⏩ Skipping sanitization (VAE-TCN and TimeGAN).")
            self.progress(60)
            return True

        if self.check_interruption(): return False

        # Step 0: Locate Input Data (STRICT)
        input_file = self._find_input_parquet()
        if not input_file:
            self.log("[SanitizationStage] ❌ CRITICAL: No input .parquet file found in Synapse folder.")
            raise FileNotFoundError("No input .parquet file found. Please import a file first.")
        
        self.log(f"[SanitizationStage] 📂 Found Input: {os.path.basename(input_file)}")
        
        # Step 0.5: Load and Create Base Copy (Exact Duplicate)
        df_raw = pd.read_parquet(input_file)
        base_path = os.path.join(base_dir, "base_v1.parquet")
        df_raw.to_parquet(base_path) 
        
        self.log(f"[SanitizationStage] 💾 BASE Created (Exact Copy): {base_path}")
        self.log(f"[SanitizationStage] 📊 Input Shape: {df_raw.shape}")
        self.progress(25)

        if self.check_interruption(): return False

        # Step 1: Pipeline - Correction (VAE-TCN)
        numeric_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            raise ValueError("Dataset has no numeric columns for processing.")

        data_matrix = df_raw[numeric_cols].values.astype(np.float32)
        
        self.log("[SanitizationStage] 🧹 Step 1/4: Running Corrector Agent (VAE-TCN)...")
        processed_matrix = self._apply_corrector(data_matrix)
        self.progress(40)

        if self.check_interruption(): return False

        # Step 2: Pipeline - Imputation (TimeGAN)
        nan_count = np.isnan(processed_matrix).sum()
        if nan_count > 0:
            self.log(f"[SanitizationStage] 🧬 Step 2/4: Found {nan_count} gaps/outliers. Running Imputer Agent (TimeGAN)...")
            processed_matrix = self._apply_imputer(processed_matrix)
        else:
            self.log("[SanitizationStage] ⏩ Step 2/4: Data is clean. Skipping Imputer.")
        
        self.progress(55)

        if self.check_interruption(): return False

        # Reassemble and Save Golden
        df_golden = df_raw.copy()
        df_golden[numeric_cols] = processed_matrix
        
        if 'timestamp' in df_golden.columns:
            df_golden = self._enrich_time_features(df_golden)

        df_golden.to_parquet(golden_path)
        
        self.log(f"[SanitizationStage] 🏆 GOLDEN dataset generated: {golden_path}")
        self.progress(60)
        
        return True

    def _find_input_parquet(self) -> str:
        """
        Searches for a .parquet file to use as source.
        Prioritizes files NOT in 'base' or 'golden' to avoid recursion.
        """
        candidates = []
        for root, _, files in os.walk(self.synapse_root):
            for file in files:
                if file.endswith(".parquet"):
                    full_path = os.path.join(root, file)
                    if "golden_v1.parquet" in file:
                        continue 
                    candidates.append(full_path)
        
        if not candidates:
            return None
            
        return max(candidates, key=os.path.getmtime)

    def _apply_corrector(self, data: np.ndarray) -> np.ndarray:
        """Uses CorrectorAgent to smooth data and inject NaNs on anomalies."""
        features = data.shape[1]
        agent = CorrectorAgent(input_dim=features)
        
        corrected = agent.inference(data)
        
        deviation = np.abs(data - corrected)
        threshold = np.nanstd(data, axis=0) * 3  # 3 Sigma
        
        threshold = np.where(np.isnan(threshold) | (threshold == 0), 1e-6, threshold)
        
        mask = deviation > threshold
        output = data.copy()
        output[mask] = np.nan # Create hole for Imputer
        
        return output

    def _apply_imputer(self, data: np.ndarray) -> np.ndarray:
        """Uses ImputerAgent to fill NaNs created by the Corrector."""
        features = data.shape[1]
        agent = ImputerAgent(feature_dim=features, seq_len=24)
        return agent.impute(data)

    def _enrich_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extracts standard temporal markers needed for downstream analysis."""
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        return df