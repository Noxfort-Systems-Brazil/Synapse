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
# File: src/services/offline_service.py
# Author: Gabriel Moraes
# Date: 2026-03-02

import os
import traceback
from PyQt6.QtCore import QThread, pyqtSignal

# Import Domain
from src.domain.app_state import AppState

# Import Pipeline Stages
from src.stages.sanitization_stage import SanitizationStage
from src.stages.translation_stage import TranslationStage
from src.stages.classification_stage import ClassificationStage

class OfflineService(QThread):
    """
    Service responsible for Phase 1: Offline Bootstrap.
    
    Refactored Architecture (SOLID - SRP & OCP):
    - Acts exclusively as an Orchestrator / UI Thread Manager.
    - Delegates the heavy lifting to specialized Stage classes.
    - Shares state through a common context dictionary.
    """

    # Signals for UI Updates
    log_message = pyqtSignal(str)
    bootstrap_finished = pyqtSignal()
    progress_update = pyqtSignal(int)

    def __init__(self, app_state: AppState = None):
        """
        Args:
            app_state: The central application state injected from BootstrapPhase.
        """
        super().__init__()
        self.app_state = app_state
        self.running = True
        
        # Setup Paths for Ubuntu
        self.home_dir = os.path.expanduser("~")
        self.synapse_root = os.path.join(self.home_dir, "Documentos", "Synapse")
        
        # Fallback if Documents is used instead of Documentos
        if not os.path.exists(self.synapse_root):
            self.synapse_root = os.path.join(self.home_dir, "Documents", "Synapse")
            
        self.datalake_dir = os.path.join(self.synapse_root, "datalake")
        self.base_dir = os.path.join(self.datalake_dir, "base")
        self.golden_dir = os.path.join(self.datalake_dir, "golden")
        self.config_dir = os.path.join(self.synapse_root, "data", "config")

    def run(self):
        """Main execution pipeline."""
        self.log_message.emit("[OfflineService] 🟢 Starting Offline Service Thread...")
        self.progress_update.emit(0)
        
        try:
            self._ensure_directories()
            
            # Setup Shared Context (Shared memory for the stages)
            shared_context = {
                "base_dir": self.base_dir,
                "golden_dir": self.golden_dir,
                "config_dir": self.config_dir,
                "golden_path": os.path.join(self.golden_dir, "golden_v1.parquet"),
                "ontology_path": os.path.join(self.config_dir, "ontology.safetensors"),
                "json_path": os.path.join(self.config_dir, "peak_schedule.json")
            }
            
            # --- MASTER SKIP LOCK ---
            if self._check_master_lock(shared_context):
                self.log_message.emit("[OfflineService] 🏆 All Phase 1 artifacts found and validated (Golden, Ontology, JSON).")
                self.log_message.emit("[OfflineService] ⏩ Skipping Offline Phase entirely.")
                self.progress_update.emit(100)
                self.bootstrap_finished.emit()
                return

            # Initialize Pipeline Stages
            stages = [
                SanitizationStage(self.synapse_root, self._handle_log, self._handle_progress, self._check_stop),
                TranslationStage(self.synapse_root, self._handle_log, self._handle_progress, self._check_stop),
                ClassificationStage(self.synapse_root, self._handle_log, self._handle_progress, self._check_stop)
            ]
            
            # Execute Pipeline sequentially
            for stage in stages:
                success = stage.execute(shared_context)
                if not success:
                    # Break the pipeline if a stage fails or is interrupted by the user
                    break

        except Exception as e:
            self.log_message.emit(f"[OfflineService] ❌ CRITICAL ERROR: {str(e)}")
            traceback.print_exc()
        finally:
            self.bootstrap_finished.emit()

    def _ensure_directories(self):
        """Creates the necessary folder structure if it does not exist."""
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(self.golden_dir, exist_ok=True)
        os.makedirs(self.config_dir, exist_ok=True)

    def _check_master_lock(self, ctx: dict) -> bool:
        """Verifies if all target artifacts are already generated and valid."""
        golden_valid = os.path.exists(ctx["golden_path"]) and os.path.getsize(ctx["golden_path"]) > 0
        ontology_valid = os.path.exists(ctx["ontology_path"]) and os.path.getsize(ctx["ontology_path"]) > 0
        json_valid = os.path.exists(ctx["json_path"]) and os.path.getsize(ctx["json_path"]) > 0
        return golden_valid and ontology_valid and json_valid

    def _handle_log(self, message: str):
        """Callback to emit logs from the stages to the UI."""
        self.log_message.emit(message)

    def _handle_progress(self, value: int):
        """Callback to emit progress from the stages to the UI."""
        self.progress_update.emit(value)

    def _check_stop(self) -> bool:
        """Callback to let stages know if the thread was stopped."""
        return not self.running

    def stop(self):
        """Safely flags the thread to stop."""
        self.running = False