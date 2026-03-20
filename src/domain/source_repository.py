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
# File: src/domain/source_repository.py
# Author: Gabriel Moraes
# Date: 2026-03-09

import json
import os
import logging

from PyQt6.QtCore import QObject, pyqtSignal
from typing import Dict, List, Optional
from datetime import datetime
from src.domain.entities import DataSource, SourceType, SourceStatus

logger = logging.getLogger("Synapse.SourceRepository")


class SourceRepository(QObject):
    """
    Responsibility: Manage Dynamic Sensors and their Associations.
    
    Persistence: Sources and associations are saved to a JSON file on disk.
    On startup, previously registered sources are restored so that Push
    sensors can reconnect automatically without re-registration.
    
    Rules:
    - Previously ACTIVE sources reload as QUARANTINE (Linguist re-validates).
    - REJECTED sources are NOT restored.
    - Runtime state (latest_value, last_update) is NOT persisted.
    """
    source_added = pyqtSignal(DataSource)
    source_removed = pyqtSignal(str)
    association_changed = pyqtSignal(str, str)
    source_origin_toggled = pyqtSignal(str, bool)  # (source_id, new_is_local)

    # Persistence file path
    _PERSIST_PATH = os.path.join(
        os.path.expanduser("~"), "Documentos", "Synapse", "config", "sources.json"
    )

    def __init__(self):
        super().__init__()
        self._data_sources: Dict[str, DataSource] = {}
        self._associations: Dict[str, List[str]] = {}
        # Load previously saved sources from disk
        self._load_from_disk()

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def add(self, source: DataSource):
        self._data_sources[source.id] = source
        self.source_added.emit(source)
        self._save_to_disk()

    def remove(self, source_id: str):
        if source_id in self._data_sources:
            del self._data_sources[source_id]
            for element_id in list(self._associations.keys()):
                if source_id in self._associations[element_id]:
                    self._associations[element_id].remove(source_id)
            self.source_removed.emit(source_id)
            self._save_to_disk()

    def get(self, source_id: str) -> Optional[DataSource]:
        return self._data_sources.get(source_id)

    def get_all(self) -> List[DataSource]:
        return list(self._data_sources.values())
    
    def update_value(self, source_id: str, value: float):
        source = self.get(source_id)
        if source:
            source.latest_value = value
            source.last_update = datetime.now().timestamp()

    def associate(self, source_id: str, element_id: str):
        if element_id not in self._associations:
            self._associations[element_id] = []
        if source_id not in self._associations[element_id]:
            self._associations[element_id].append(source_id)
            self.association_changed.emit(source_id, element_id)
            self._save_to_disk()

    def get_associations(self, element_id: str) -> List[str]:
        return self._associations.get(element_id, [])

    def toggle_origin(self, source_id: str):
        """Toggle a source between Local and Global scope, persist and emit signal."""
        source = self.get(source_id)
        if source:
            source.is_local = not source.is_local
            self._save_to_disk()
            self.source_origin_toggled.emit(source_id, source.is_local)

    def emit_restored_sources(self):
        """Emit source_added for all pre-loaded sources (call AFTER UI signal wiring)."""
        for source in self._data_sources.values():
            self.source_added.emit(source)

    # =========================================================================
    # PERSISTENCE
    # =========================================================================

    def _serialize_source(self, src: DataSource) -> dict:
        """Convert a DataSource to a JSON-safe dict."""
        return {
            "id": src.id,
            "name": src.name,
            "source_type": src.source_type.value if isinstance(src.source_type, SourceType) else str(src.source_type),
            "connection_string": src.connection_string,
            "is_local": src.is_local,
            "status": src.status.value if isinstance(src.status, SourceStatus) else str(src.status),
            "lat": src.lat,
            "lon": src.lon,
            "semantic_type": src.semantic_type,
            "inferred_unit": src.inferred_unit,
            "confidence_score": src.confidence_score,
            "metadata": src.metadata if isinstance(src.metadata, dict) else {},
        }

    def _deserialize_source(self, data: dict) -> DataSource:
        """Reconstruct a DataSource from a saved dict."""
        # Resolve SourceType enum
        source_type = SourceType.API
        for st in SourceType:
            if st.value == data.get("source_type"):
                source_type = st
                break

        # Resolve SourceStatus — ACTIVE sources come back as QUARANTINE
        raw_status = data.get("status", "Quarantine")
        status = SourceStatus.QUARANTINE  # Default: re-validate
        if raw_status == SourceStatus.QUARANTINE.value:
            status = SourceStatus.QUARANTINE
        elif raw_status == SourceStatus.ACTIVE.value:
            # Was active → needs Linguist re-validation
            status = SourceStatus.QUARANTINE
        # REJECTED / FALLBACK / PROBATION → skip (handled by caller filter)

        return DataSource(
            id=data["id"],
            name=data["name"],
            source_type=source_type,
            connection_string=data.get("connection_string", ""),
            is_local=data.get("is_local", True),
            status=status,
            lat=data.get("lat", 0.0),
            lon=data.get("lon", 0.0),
            semantic_type=data.get("semantic_type"),
            inferred_unit=data.get("inferred_unit"),
            confidence_score=data.get("confidence_score", 0.0),
            metadata=data.get("metadata", {}),
        )

    def _save_to_disk(self):
        """Persist current sources and associations to JSON."""
        try:
            os.makedirs(os.path.dirname(self._PERSIST_PATH), exist_ok=True)

            payload = {
                "sources": [
                    self._serialize_source(s) for s in self._data_sources.values()
                    if "Historical" not in s.name
                ],
                "associations": self._associations,
            }

            with open(self._PERSIST_PATH, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.warning(f"[SourceRepository] Failed to save sources: {e}")

    def _load_from_disk(self):
        """Restore sources and associations from JSON (if file exists)."""
        if not os.path.exists(self._PERSIST_PATH):
            return

        try:
            with open(self._PERSIST_PATH, "r", encoding="utf-8") as f:
                payload = json.load(f)

            loaded_count = 0
            for src_data in payload.get("sources", []):
                # Skip REJECTED sources — they should not come back
                raw_status = src_data.get("status", "")
                if raw_status in ("Rejected", "rejected"):
                    continue

                source = self._deserialize_source(src_data)
                self._data_sources[source.id] = source
                loaded_count += 1

            # Restore associations
            self._associations = payload.get("associations", {})

            if loaded_count > 0:
                logger.info(f"[SourceRepository] ♻️ Restored {loaded_count} source(s) from disk.")

        except Exception as e:
            logger.warning(f"[SourceRepository] Failed to load sources: {e}")
