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
# File: src/stages/translation_stage.py
# Author: Gabriel Moraes
# Date: 2026-03-02

import os
from typing import Dict, Any

from src.stages.base_stage import BaseStage
from src.agents.translation_agent import TranslationAgent

class TranslationStage(BaseStage):
    """
    Translation Stage (Step 3).
    
    Responsible solely for:
    - Checking if the semantic ontology already exists.
    - Initializing the TranslationAgent (DistilRoBERTa).
    - Reading the golden dataset to extract semantic logs.
    - Saving the resulting ontology.safetensors.
    """

    def execute(self, shared_context: Dict[str, Any]) -> bool:
        """Executes the NLP semantic translation pipeline."""
        ontology_path = shared_context.get("ontology_path")
        golden_dir = shared_context.get("golden_dir")
        
        # Check if we can intelligently skip this phase (Warm Start)
        ontology_valid = os.path.exists(ontology_path) and os.path.getsize(ontology_path) > 0
        
        if ontology_valid:
            self.log("[TranslationStage] ⚡ Warm Start: Ontology .safetensors found. Skipping Translation Agent.")
            self.progress(80)
            return True

        if self.check_interruption(): 
            return False

        self.log("[TranslationStage] 🧠 Step 3/4: Running Translation Agent (DistilRoBERTa)...")
        
        try:
            translator = TranslationAgent(datalake_path=golden_dir)
            success = translator.run_translation_phase("golden_v1.parquet")
            
            if success:
                self.log("[TranslationStage] 📚 Translation complete. Ontology saved to Safetensors.")
            else:
                self.log("[TranslationStage] ⚠️ Translation failed or skipped.")
        except Exception as e:
            self.log(f"[TranslationStage] ❌ Error during translation: {str(e)}")
            return False
            
        self.progress(80)
        return True