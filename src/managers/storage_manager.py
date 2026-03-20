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
# File: src/managers/storage_manager.py
# Author: Gabriel Moraes
# Date: 2026-02-16

import os
import logging
from pathlib import Path
from typing import Optional
from PyQt6.QtCore import pyqtSignal

class StorageManager:
    """
    Manages external configuration files (JSON).
    Handles OS-level folder scaffolding (SRP).
    """
    # Refactored V4 (Strict Folder Structure):
    # - REMOVED 'data' folder creation.
    # - REMOVED SQLite database logic (synapse_metadata.sqlite).
    # - ONLY creates:
    #     1. ~/Documentos/Synapse/Checkpoint
    #     2. ~/Documentos/Synapse/datalake/base
    #     3. ~/Documentos/Synapse/datalake/golden
    """
    Central Persistence Manager for SYNAPSE.
    
    Refactored V4 (Strict Folder Structure):
    - REMOVED 'data' folder creation.
    - REMOVED SQLite database logic (synapse_metadata.sqlite).
    - ONLY creates:
        1. ~/Documentos/Synapse/Checkpoint
        2. ~/Documentos/Synapse/datalake/base
        3. ~/Documentos/Synapse/datalake/golden
    """

    def __init__(self):
        """
        Initializes the storage manager with strict directory rules.
        """
        # 1. Resolve User Home Directory
        self.home = Path.home()
        
        # 2. Locate 'Documents' folder (Handles PT-BR 'Documentos' vs EN 'Documents')
        self.documents_dir = self.home / "Documentos"
        if not self.documents_dir.exists():
            self.documents_dir = self.home / "Documents"
            
        # 3. Define Project Root
        self.project_root = self.documents_dir / "Synapse"
        
        # 4. Define ONLY The Allowed Folders
        self.checkpoint_dir = self.project_root / "Checkpoint"
        self.datalake_dir = self.project_root / "datalake"
        self.base_dir = self.datalake_dir / "base"
        self.golden_dir = self.datalake_dir / "golden"
        
        # Internal State
        self.is_connected = True # Mocked as true since we removed DB
        
        # Ensure directories exist immediately upon instantiation
        self._ensure_structure()
        
        logging.info(f"[StorageManager] Storage initialized at: {self.project_root}")

    def _ensure_structure(self):
        """Creates ONLY the necessary folder structure."""
        paths = [
            self.project_root,
            self.checkpoint_dir,    # Pasta 1
            self.datalake_dir,      # Pasta 2 (Pai)
            self.base_dir,          # Pasta 2.1 (Base)
            self.golden_dir         # Pasta 2.2 (Golden)
        ]
        
        for path in paths:
            if not path.exists():
                try:
                    path.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    logging.critical(f"[StorageManager] Failed to create directory {path}: {e}")

    # =========================================================================
    #  PATH ACCESSORS (Getters)
    # =========================================================================

    def get_datalake_base_path(self) -> str:
        """Returns path to where raw .parquet files should be saved."""
        return str(self.base_dir)

    def get_datalake_golden_path(self) -> str:
        """Returns path to where processed files are saved."""
        return str(self.golden_dir)
    
    def get_checkpoint_path(self) -> str:
        """Returns path for best_hparams.pth."""
        return str(self.checkpoint_dir)
    # =========================================================================
    #  CONNECTION LIFECYCLE (Simplified)
    # =========================================================================

    def _ensure_directories(self):
        """
        Creates the mandatory folder structure requested.
        Crucial for preventing errors when files are deleted.
        Moved from SystemController to adhere to SRP.
        """
        try:
            home = Path.home()
            docs = home / "Documentos"
            if not docs.exists():
                docs = home / "Documents"
            
            synapse_root = docs / "Synapse"
            
            paths_to_check = [
                synapse_root / "Checkpoint",          # For best_hparams.pth
                synapse_root / "datalake" / "base",   # For raw .parquet copy
                synapse_root / "datalake" / "golden", # For processed .parquet
                synapse_root / "data" / "db",         # For metadata
                synapse_root / "data" / "config"      # For JSON configs
            ]
            
            for p in paths_to_check:
                os.makedirs(p, exist_ok=True)
                
        except Exception as e:
            print(f"[StorageManager] ⚠️ Directory creation failed: {e}")

    def connect(self):
        """
        Ensures physical path integrity on boot.
        """
        self._ensure_directories()

    def close(self):
        """No-op."""
        pass