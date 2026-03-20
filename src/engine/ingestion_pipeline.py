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
# File: src/engine/ingestion_pipeline.py
# Author: Gabriel Moraes
# Date: 2026-02-16

import json
from typing import Dict, List, Any, Optional

# Import Domain Entities
from src.domain.app_state import AppState
from src.domain.entities import SourceStatus, DataSource
from src.services.telemetry_service import TelemetryService
from src.utils.logging_setup import logger

class IngestionPipeline:
    """
    Handles the 'Data Customs' logic (Zero Trust Ingestion Layer).
    
    Refactored V26 (Type Safety):
    - Responsibility 1: Zero Trust Authentication (Block unknown IDs).
    - Responsibility 2: Raw Buffering (Store payloads exactly as received).
    - Fixed: AttributeError when 'connection_string' is bool (defensive coding).
    """

    def __init__(self, app_state: AppState, telemetry: Optional[TelemetryService] = None):
        """
        Args:
            app_state: The central application state.
            telemetry: Shared telemetry instance.
        """
        self.app_state = app_state
        self.telemetry = telemetry if telemetry else TelemetryService()
        
        # Buffer for 'Quarantined' sources.
        self.quarantine_buffers: Dict[str, List[Any]] = {}
        
        # Default sample size required for the Linguist to make a decision
        self.default_quarantine_size = 60 
        
        # Dynamic limits per source (allows Linguist to request more data)
        self.custom_buffer_limits: Dict[str, int] = {}
        
        # Cache to Map external device IDs to internal Source IDs
        self.device_id_map: Dict[str, str] = {}

    def process_packet(self, source_id: str, payload: Any) -> bool:
        """
        Receives data from Workers, authenticates source, and buffers/updates.
        """
        
        # --- 1. IDENTIFICATION & AUTHENTICATION (Zero Trust) ---
        
        # Check Cache first
        internal_id = self.device_id_map.get(source_id)
        source = None
        if internal_id:
            source = self.app_state.get_source_by_device_id(internal_id)
            
        # Check Direct Lookup
        if not source:
            source = self.app_state.get_source_by_device_id(source_id)

        # Smart Lookup (Fuzzy/IP match) if direct failed
        if not source:
            existing_match = self._find_existing_match(source_id)
            if existing_match:
                logger.info(f"[Pipeline] 🔗 Authenticated '{source_id}' as '{existing_match.id}'")
                self.device_id_map[source_id] = existing_match.id
                source = existing_match
            else:
                # BLOCK UNKNOWN SOURCES
                logger.warning(f"[Pipeline] 🛡️ BLOCKED: Unknown device '{source_id}' tried to connect.")
                return False

        # --- 2. TELEMETRY ---
        self.telemetry.record_ingestion(source.id)

        # --- 3. BUFFERING LOGIC (Pure Collection) ---
        
        # A. QUARANTINE PATH: Collect samples for the Linguist
        if source.status == SourceStatus.QUARANTINE:
            if source.id not in self.quarantine_buffers:
                self.quarantine_buffers[source.id] = []
            
            # STORE RAW PAYLOAD
            self.quarantine_buffers[source.id].append(payload)
            
            count = len(self.quarantine_buffers[source.id])
            
            # Determine dynamic limit (Default 60 or Custom if requested)
            base_size = self.custom_buffer_limits.get(source.id, self.default_quarantine_size)
            max_buffer = base_size * 2 
            
            # Log progress visually
            if count % 10 == 0 or count == 1:
                logger.info(f"[Pipeline] 🧪 Collecting Raw Sample '{source.name}': {count}/{base_size}")

            # Maintain FIFO buffer size
            if count > max_buffer:
                 self.quarantine_buffers[source.id].pop(0)
        
        # B. ACTIVE PATH: Forward to System
        elif source.status == SourceStatus.ACTIVE:
            self.app_state.update_source_value(source.id, payload)

        return True

    # --- Linguist Support Methods ---

    def has_enough_data(self, source_id: str, required_count: int) -> bool:
        """Checks if buffer reached the requested count."""
        if source_id not in self.quarantine_buffers:
            return False
        return len(self.quarantine_buffers[source_id]) >= required_count

    def extend_quarantine_buffer(self, source_id: str, extra_samples: int):
        """Increases buffer capacity for iterative learning."""
        current_base = self.custom_buffer_limits.get(source_id, self.default_quarantine_size)
        new_base = current_base + extra_samples
        self.custom_buffer_limits[source_id] = new_base
        logger.info(f"[Pipeline] 📈 Extended buffer requirement for '{source_id}' to {new_base} samples.")

    def is_ready_for_linguist(self, source_id: str) -> bool:
        """Legacy check."""
        return self.has_enough_data(source_id, self.default_quarantine_size)

    def get_quarantine_data(self, source_id: str) -> List[Any]:
        """Returns the list of RAW payloads."""
        if source_id in self.quarantine_buffers:
            return list(self.quarantine_buffers[source_id])
        return []

    # --- Helpers ---

    def _find_existing_match(self, external_id: str) -> Optional[DataSource]:
        """Smart matching logic with Type Safety."""
        all_sources = self.app_state.get_all_data_sources()
        ext_clean = self._normalize_string(external_id)
        
        for src in all_sources:
            # 1. Exact Match
            if src.id == external_id or src.name == external_id: 
                return src
            
            # 2. Connection String / IP Match
            # FIXED: Check if connection_string is str before normalizing
            if isinstance(src.connection_string, str) and src.connection_string:
                conn_clean = self._normalize_string(src.connection_string)
                if len(conn_clean) > 5 and (conn_clean in ext_clean or ext_clean in conn_clean):
                    return src
            
            # 3. Fuzzy Name Match
            # FIXED: Check if name is str
            if isinstance(src.name, str):
                src_clean = self._normalize_string(src.name)
                if (len(src_clean) > 2 and src_clean in ext_clean) or \
                   (len(ext_clean) > 2 and ext_clean in src_clean):
                     return src
                 
        return None

    def _normalize_string(self, text: str) -> str:
        """Helper to clean strings. Safe against non-string inputs."""
        if not isinstance(text, str):
            return ""
            
        text = text.lower().replace("_", "").replace("-", "").replace(".", "").replace(":", "")
        text = text.replace("device", "").replace("sensor", "").replace("camera", "cam")
        text = text.replace("http", "").replace("src", "").replace("https", "")
        return text

    def remove_source_context(self, source_id: str):
        """Cleans up buffers when a source is deleted."""
        if source_id in self.quarantine_buffers:
            del self.quarantine_buffers[source_id]
        
        if source_id in self.custom_buffer_limits:
            del self.custom_buffer_limits[source_id]
        
        keys = [k for k, v in self.device_id_map.items() if v == source_id]
        for k in keys:
            del self.device_id_map[k]

    def promote_to_active(self, source_id: str):
        """Called by Linguist Service when a source is validated."""
        if source_id in self.quarantine_buffers:
            del self.quarantine_buffers[source_id]
        if source_id in self.custom_buffer_limits:
            del self.custom_buffer_limits[source_id]