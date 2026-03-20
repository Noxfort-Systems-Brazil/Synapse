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
# File: src/agents/translation_agent.py
# Author: Gabriel Moraes
# Date: 2026-02-28

import os
import torch
import logging
import pandas as pd
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Tuple
from safetensors.torch import save_file

# Enable Tensor Cores globally for matrix multiplications and cuDNN operations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from src.models.distilroberta import DistilRobertaSemanticExtractor


class TranslationAgent:
    """
    Orchestrator Agent for Phase 1 semantic extraction.
    Interfaces with the OfflineService and implements smart sampling per sensor.
    """
    def __init__(self, datalake_path: str):
        self.datalake_path = datalake_path
        self.extractor = DistilRobertaSemanticExtractor()
        self.logger = logging.getLogger(self.__class__.__name__)

    def run_translation_phase(self, filename: str) -> bool:
        """
        Reads the processed dataset, groups by sensor to save compute,
        extracts text data in smart batches, learns semantics, and saves the ontology.
        """
        try:
            file_path = os.path.join(self.datalake_path, filename)
            if not os.path.exists(file_path):
                self.logger.error(f"Golden dataset not found: {file_path}")
                return False

            self.logger.info(f"Starting semantic translation for: {filename}")
            df = pd.read_parquet(file_path)

            text_columns = df.select_dtypes(include=['object', 'string']).columns.tolist()
            if not text_columns:
                self.logger.warning("No textual columns found. Generating empty ontology.")
                self._save_empty_ontology()
                return True

            # Attempt to find a grouping column representing the sensor/device
            possible_id_cols = ['sensor_id', 'device_id', 'camera_id', 'source_id', 'id']
            group_col = next((col for col in possible_id_cols if col in df.columns), None)

            if group_col and group_col in text_columns:
                text_columns.remove(group_col)
                
            if not text_columns:
                self.logger.warning("Only ID textual columns found. Generating empty ontology.")
                self._save_empty_ontology()
                return True

            BATCH_SIZE = 30
            MAX_SAMPLES_PER_SENSOR = 150

            if group_col:
                self.logger.info(f"Group column '{group_col}' found. Processing by sensor in batches of {BATCH_SIZE}...")
                grouped = df.groupby(group_col)
                
                for sensor_id, group_df in grouped:
                    self.logger.info(f"--- Analyzing vocabulary for Sensor: {sensor_id} ---")
                    
                    sensor_texts = []
                    for col in text_columns:
                        sensor_texts.extend(group_df[col].dropna().astype(str).unique().tolist())
                    sensor_texts = list(set(sensor_texts))
                    
                    processed_count = 0
                    for i in range(0, len(sensor_texts), BATCH_SIZE):
                        if processed_count >= MAX_SAMPLES_PER_SENSOR:
                            self.logger.info(f"Reached safety limit of {MAX_SAMPLES_PER_SENSOR} samples for sensor '{sensor_id}'. Moving on.")
                            break
                            
                        batch = sensor_texts[i : i + BATCH_SIZE]
                        new_concepts_in_batch = 0
                        
                        for text in batch:
                            _, is_new = self.extractor.learn_and_map_semantics([text])
                            if is_new:
                                new_concepts_in_batch += 1
                                
                        processed_count += len(batch)
                        
                        # Smart Skip: If we processed a batch and no new concepts were created, 
                        # the model has mastered this sensor's language.
                        if new_concepts_in_batch == 0 and processed_count >= BATCH_SIZE:
                            self.logger.info(f"Sensor '{sensor_id}' language mastered after {processed_count} samples. Skipping remaining logs.")
                            break
            else:
                self.logger.warning("No sensor grouping column found. Analyzing globally...")
                global_texts = []
                for col in text_columns:
                    global_texts.extend(df[col].dropna().astype(str).unique().tolist())
                global_texts = list(set(global_texts))
                
                self.logger.info(f"Processing {len(global_texts)} unique global semantic patterns...")
                for text in global_texts:
                    self.extractor.learn_and_map_semantics([text])

            # Output path logic
            synapse_root = os.path.dirname(os.path.dirname(self.datalake_path))
            config_dir = os.path.join(synapse_root, "data", "config")
            os.makedirs(config_dir, exist_ok=True)
            
            out_path = os.path.join(config_dir, "ontology.safetensors")
            self.extractor.save_ontology(out_path)
            
            return True

        except Exception as e:
            self.logger.error(f"Translation phase failed: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    def _save_empty_ontology(self):
        """Creates a dummy safetensors file to satisfy the warm-start requirement next time."""
        synapse_root = os.path.dirname(os.path.dirname(self.datalake_path))
        config_dir = os.path.join(synapse_root, "data", "config")
        os.makedirs(config_dir, exist_ok=True)
        out_path = os.path.join(config_dir, "ontology.safetensors")
        
        try:
            dummy_tensor = {"empty_ontology": torch.zeros(1)}
            save_file(dummy_tensor, out_path)
            self.logger.info(f"Empty ontology saved to {out_path}")
        except Exception as e:
            self.logger.error(f"Failed to save empty ontology: {e}")