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
# File: src/services/linguist_service.py
# Author: Gabriel Moraes
# Date: 2026-02-16

import torch
import numpy as np
from typing import Dict, Optional, List
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

# --- Domain ---
from src.domain.app_state import AppState
from src.domain.entities import SourceStatus, DataSource
from src.workers.ingestion_worker import IngestionWorker

# --- Agents & Factories ---
from src.factories.agent_factory import AgentFactory
from src.agents.linguist_agent import LinguistAgent
from src.agents.specialist_agent import SpecialistAgent

# --- Utils ---
from src.utils.logging_setup import logger

class LinguistService(QObject):
    """
    The 'Neuro-Symbolic' Gatekeeper Service.
    
    Refactored V2 (Standby Thread Architecture):
    - Lives in its own dedicated QThread.
    - `run_check()` is a @pyqtSlot invoked via cross-thread signal from the Neural thread.
    - When no quarantined sources exist, naturally returns and the thread idles (standby).
    - When triggered, wakes up, processes all quarantine sources, then returns to standby.
    
    Orchestration Logic:
    1.  **Collection Loop:** Accumulates sensor samples (chunks of 60).
    2.  **Grammar Acquisition:** Attempts to train the LinguistAgent (TCN-AE) to reconstruct the signal.
    3.  **Physics Validation:** Once the grammar is learned, the 'Symbolic' layer validates physical laws.
    4.  **Knowledge Transfer:** If valid, 'teaches' the SpecialistAgent (TCN) how to read this source.
    """
    
    # Signal emitted when a source is successfully analyzed and promoted
    update_signal = pyqtSignal(str, str, float)

    def __init__(self, app_state: AppState, ingestion_worker: IngestionWorker, agent_factory: AgentFactory):
        super().__init__()
        self.app_state = app_state
        self.ingestion = ingestion_worker
        self.agent_factory = agent_factory
        
        # Memory to track training attempts per source
        # Key: source_id, Value: attempt_count
        self.learning_attempts: Dict[str, int] = {}
        
        # Thresholds
        self.SAMPLE_CHUNK_SIZE = 60
        self.MAX_ATTEMPTS = 5 # Allows up to 300 samples total
        self.GRAMMAR_LOSS_THRESHOLD = 0.05 # MSE limit to consider "Learned"

    @pyqtSlot()
    def run_check(self):
        """
        Invoked via signal from the Neural thread.
        Scans quarantined sources and progresses their state machine.
        If no quarantined sources exist, returns immediately (standby).
        """
        pipeline = self.ingestion.get_pipeline()
        sources = self.app_state.get_all_data_sources()
        
        for source in sources:
            if source.status == SourceStatus.QUARANTINE:
                self._process_quarantine_source(source, pipeline)

    def _process_quarantine_source(self, source: DataSource, pipeline):
        """
        Executes the logic: Collect -> Learn -> Validate -> Teach.
        """
        # 1. Check Data Availability
        # We need at least 60 samples (or multiples thereof based on attempts)
        required_samples = self.SAMPLE_CHUNK_SIZE
        
        if not pipeline.has_enough_data(source.id, required_samples):
            return # Wait for ingestion

        data_chunk = pipeline.get_quarantine_data(source.id)
        current_attempt = self.learning_attempts.get(source.id, 0)
        
        logger.info(f"[Linguist] 🔄 Attempt {current_attempt+1}: Analyzing {len(data_chunk)} samples from '{source.name}'...")

        # 2. Summon the Linguist Agent (The Learner)
        # We verify if we can learn the "Grammar" of this signal
        linguist = self.agent_factory.get_or_create_linguist(source.id)
        
        # Train on the current chunk
        # Note: We assume the agent can handle the raw numerical data for this "signal grammar" task
        loss = linguist.train_step(data_chunk)
        
        # 3. Decision Gate: Did we learn the grammar?
        if loss < self.GRAMMAR_LOSS_THRESHOLD:
            logger.info(f"[Linguist] 🧠 Grammar Learned! (Loss: {loss:.4f} < {self.GRAMMAR_LOSS_THRESHOLD})")
            
            # 4. Physics Validation (The Symbolic Check)
            if self._validate_physics(source, data_chunk, linguist):
                # 5. Teach the Specialist (Knowledge Transfer)
                self._teach_specialist(source, linguist)
                
                # 6. Promote
                self._promote_source(source, pipeline)
            else:
                # Physics failed despite good grammar -> Probable Spoofing/Attack
                self._reject_source(source, reason="Physics Violation (Possible Injection Attack)")
        
        else:
            # Grammar not learned yet
            self._handle_learning_failure(source, pipeline, loss)

    def _handle_learning_failure(self, source: DataSource, pipeline, loss: float):
        """
        Logic for when the agent fails to understand the signal pattern.
        """
        attempts = self.learning_attempts.get(source.id, 0) + 1
        self.learning_attempts[source.id] = attempts
        
        if attempts >= self.MAX_ATTEMPTS:
            logger.warning(f"[Linguist] ❌ Failed to learn grammar after {attempts} attempts. Signal too chaotic.")
            self._reject_source(source, reason="Unlearnable Pattern (High Entropy)")
        else:
            logger.info(f"[Linguist] ⏳ Grammar unclear (Loss: {loss:.4f}). Requesting +{self.SAMPLE_CHUNK_SIZE} samples.")
            # Signal the pipeline to keep buffering and NOT clear the quarantine buffer yet
            # effectively "collecting more samples" for the next pass
            pipeline.extend_quarantine_buffer(source.id, self.SAMPLE_CHUNK_SIZE)

    def _validate_physics(self, source: DataSource, data: List[float], agent: LinguistAgent) -> bool:
        """
        Uses the trained Agent + Symbolic Rules to validate physical feasibility.
        """
        # A. Symbolic Sanity Checks (Rule-based)
        data_np = np.array(data)
        if np.min(data_np) < 0:
            logger.warning(f"[Physics] Violation: Negative value detected in '{source.name}'.")
            return False
            
        if np.mean(data_np) == 0 and np.var(data_np) == 0:
             logger.warning(f"[Physics] Violation: Dead signal (Flatline).")
             return False

        # B. Neuro-Validation (Anomaly Detection)
        # We ask the agent to score the very data it just learned. 
        # If it finds high anomaly scores within its own training set, the data is self-contradictory.
        analysis = agent.inference(data)
        if analysis.get('is_anomaly', False):
             logger.warning(f"[Physics] Violation: Semantic Inconsistency detected by TCN-AE.")
             return False
             
        return True

    def _teach_specialist(self, source: DataSource, linguist: LinguistAgent):
        """
        Transfers the learned features (Encoder) from Linguist to Specialist.
        This prepares the Specialist to 'read' the sensor immediately.
        """
        logger.info(f"[Linguist] 🎓 Teaching Specialist Agent how to read '{source.name}'...")
        
        # 1. Spawn/Get the Specialist for this node
        specialist = self.agent_factory.get_or_create_specialist(source.id)
        
        # 2. Transfer Weights (The "Teaching")
        # We copy the TCN Encoder weights from Linguist (Teacher) to Specialist (Student)
        # Assuming both share a compatible TCN backbone structure
        try:
            # Extract encoder state
            encoder_state = linguist.model.ae.encoder.state_dict()
            
            # Load into Specialist's TCN
            # strict=False allows ignoring heads/decoders that differ
            specialist.tcn.load_state_dict(encoder_state, strict=False)
            
            logger.info(f"[System] ✅ Knowledge Transfer Complete. Specialist is ready.")
        except Exception as e:
            logger.error(f"[System] ⚠️ Failed to transfer weights: {e}. Specialist will learn from scratch.")

    def _promote_source(self, source: DataSource, pipeline):
        """Final promotion to ACTIVE state."""
        source.status = SourceStatus.ACTIVE
        source.semantic_type = "Traffic Flow" # Determined by Linguist
        source.confidence_score = 1.0
        
        pipeline.promote_to_active(source.id)
        self.update_signal.emit(source.name, "Traffic Flow", 1.0)
        
        # Cleanup memory
        if source.id in self.learning_attempts:
            del self.learning_attempts[source.id]

    def _reject_source(self, source: DataSource, reason: str):
        """Rejects the source, keeping it in Quarantine or banning it."""
        logger.error(f"[System] ⛔ Source '{source.name}' REJECTED. Reason: {reason}")
        # Reset attempts to allow future retry if the sensor is fixed
        self.learning_attempts[source.id] = 0 
        # Note: In a real system, we might set status to ERROR or BANNED.
        # Here we leave in QUARANTINE but log the error.