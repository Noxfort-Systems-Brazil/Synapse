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
# File: src/stages/classification_stage.py
# Author: Gabriel Moraes
# Date: 2026-03-02

import os
import pandas as pd
from typing import Dict, Any

from src.stages.base_stage import BaseStage
from src.agents.peak_classifier_agent import PeakClassifierAgent

class ClassificationStage(BaseStage):
    """
    Classification Stage (Step 4).
    
    Responsible solely for:
    - Checking if the peak schedule JSON already exists (Warm Start).
    - Loading the golden dataset.
    - Dynamically configuring the sequence length for the neural networks.
    - Initializing the PeakClassifierAgent (which now handles Semantic Discovery internally).
    - Executing the pipeline to generate peak_schedule.json.
    """

    def execute(self, shared_context: Dict[str, Any]) -> bool:
        """Executes the peak classification pipeline."""
        golden_path = shared_context.get("golden_path")
        json_path = shared_context.get("json_path")
        
        # Check if we can intelligently skip this phase (Warm Start)
        json_valid = os.path.exists(json_path) and os.path.getsize(json_path) > 0
        
        if json_valid:
            self.log("[ClassificationStage] ⚡ Warm Start: Peak schedule JSON found and valid. Skipping TimesNet Classifier.")
            self.progress(100)
            return True

        if self.check_interruption(): 
            return False

        self.log("[ClassificationStage] 📈 Step 4/4: Running Peak Classifier Agent (Semantic Discovery + TimesNet + iTransformer)...")
        
        try:
            # Load the validated Golden Dataset
            golden_df = pd.read_parquet(golden_path)
            
            # Dynamic sequence length based on the dataset size
            seq_len = len(golden_df)
            
            # Define Neural Network Configurations for ADAGIO Phase 1
            itrans_cfg = {
                'num_variates': 2, 
                'seq_len': seq_len, 
                'pred_len': seq_len, 
                'd_model': 64
            }
            tnet_cfg = {
                'enc_in': 1, 
                'seq_len': seq_len, 
                'pred_len': seq_len, 
                'd_model': 64
            }
            
            # Initialize and execute the updated Peak Classifier Agent
            peak_classifier = PeakClassifierAgent(
                itransformer_config=itrans_cfg, 
                timesnet_config=tnet_cfg,
                output_path=json_path
            )
            
            peak_classifier.process_and_classify(golden_df)
            self.log("[ClassificationStage] 🕒 Peak schedule JSON generated successfully.")
            
        except Exception as e:
            self.log(f"[ClassificationStage] ❌ Error during classification: {str(e)}")
            return False
            
        self.progress(100)
        return True