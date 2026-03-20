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
# File: src/controllers/project_controller.py
# Author: Gabriel Moraes
# Date: 2025-12-27

import os
from typing import Optional, List

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

# --- Domain ---
from src.domain.app_state import AppState
from src.domain.entities import DataSource, SourceType, SourceStatus

# --- Managers ---
from src.managers.storage_manager import StorageManager

# --- Utils ---
from src.utils.logging_setup import logger

class ProjectController(QObject):
    """
    The Project & Data Manager (Project Controller).
    
    Responsibilities (SRP):
    - Manages the Project Lifecycle (New, Load, Save).
    - Acts as the Write-Controller for AppState (Adding/Removing Sources).
    - Validates inputs before updating the Domain Model.
    """

    # --- SIGNALS ---
    project_loaded = pyqtSignal(str)        # Emits project name/path
    project_saved = pyqtSignal(str)
    
    source_added = pyqtSignal(str)          # Emits source ID
    source_removed = pyqtSignal(str)
    map_updated = pyqtSignal(str)           # Emits map path
    
    error_occurred = pyqtSignal(str)
    log_message = pyqtSignal(str)

    def __init__(self, app_state: AppState, storage_manager: StorageManager):
        super().__init__()
        self.app_state = app_state
        self.storage = storage_manager

    # =========================================================================
    # PROJECT FILE OPERATIONS
    # =========================================================================

    @pyqtSlot()
    def create_new_project(self):
        """Resets the application state to default."""
        logger.info("[ProjectController] Creating New Project...")
        self.app_state.clear()
        self.log_message.emit("New Project Created.")

    @pyqtSlot(str)
    def save_project(self, file_path: str):
        """Persists the current AppState to a file."""
        logger.info(f"[ProjectController] Saving Project to {file_path}...")
        try:
            # We delegate the actual serialization to StorageManager
            success = self.storage.save_state(self.app_state, file_path)
            if success:
                self.project_saved.emit(file_path)
                self.log_message.emit(f"Project saved: {os.path.basename(file_path)}")
            else:
                self.error_occurred.emit("Failed to write project file.")
        except Exception as e:
            logger.error(f"[ProjectController] Save Error: {e}")
            self.error_occurred.emit(f"Save Error: {str(e)}")

    @pyqtSlot(str)
    def load_project(self, file_path: str):
        """Loads AppState from a file."""
        logger.info(f"[ProjectController] Loading Project from {file_path}...")
        try:
            # StorageManager reads file and populates AppState
            success = self.storage.load_state(self.app_state, file_path)
            if success:
                self.project_loaded.emit(file_path)
                self.log_message.emit(f"Project loaded: {os.path.basename(file_path)}")
            else:
                self.error_occurred.emit("Failed to read project file or invalid format.")
        except Exception as e:
            logger.error(f"[ProjectController] Load Error: {e}")
            self.error_occurred.emit(f"Load Error: {str(e)}")

    # =========================================================================
    # DATA SOURCE MANAGEMENT
    # =========================================================================

    @pyqtSlot(str, str, str)
    def add_data_source(self, name: str, source_type_str: str, connection_string: str):
        """
        Validates and adds a new Data Source (Sensor, API, etc) to the domain.
        """
        logger.info(f"[ProjectController] Adding Source: {name} ({source_type_str})")
        
        # 1. Validation
        if not name or not connection_string:
            self.error_occurred.emit("Invalid Source Data: Name and Connection required.")
            return

        # 2. Type Conversion
        try:
            sType = SourceType[source_type_str.upper()]
        except KeyError:
            self.error_occurred.emit(f"Unknown Source Type: {source_type_str}")
            return

        # 3. Creation
        # Assuming ID generation happens inside AppState or we generate a UUID here.
        # For this refactor, let's assume AppState.add_source handles ID gen if None.
        new_source = DataSource(
            id=f"src_{len(self.app_state.get_all_data_sources()) + 1}",
            name=name,
            source_type=sType,
            connection_string=connection_string,
            status=SourceStatus.ACTIVE
        )

        # 4. Domain Update
        self.app_state.add_data_source(new_source)
        self.source_added.emit(new_source.id)
        self.log_message.emit(f"Source '{name}' added successfully.")

    @pyqtSlot(str)
    def remove_data_source(self, source_id: str):
        """Removes a data source by ID."""
        logger.info(f"[ProjectController] Removing Source ID: {source_id}")
        self.app_state.remove_data_source(source_id)
        self.source_removed.emit(source_id)

    @pyqtSlot(str)
    def set_map_file(self, file_path: str):
        """Specific handler for the SUMO Network Map."""
        if not os.path.exists(file_path):
            self.error_occurred.emit("Map file not found.")
            return

        # Create a special DataSource for the map
        # Or use a dedicated field in AppState if implemented.
        # Based on previous context, AppState treats map as a special source or dedicated logic.
        
        # Logic: Update AppState dedicated map path
        # (Assuming AppState has a specialized method or we find the map source)
        
        # Strategy: Remove old map source if exists
        current_map = self.app_state.get_map_source_path()
        if current_map:
            # Logic to find and remove old map entry if stored as DataSource
            pass

        # Add new Map Source
        map_source = DataSource(
            id="map_main",
            name="City Map",
            source_type=SourceType.SUMO_NET_XML,
            connection_string=file_path,
            status=SourceStatus.ACTIVE
        )
        self.app_state.add_data_source(map_source)
        
        self.map_updated.emit(file_path)
        self.log_message.emit(f"Map set to: {os.path.basename(file_path)}")