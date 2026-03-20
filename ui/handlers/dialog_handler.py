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
# File: ui/handlers/dialog_handler.py
# Author: Gabriel Moraes
# Date: 2026-03-01

import os
import time
from PyQt6.QtCore import pyqtSignal, QObject, QSettings
from PyQt6.QtWidgets import QFileDialog

from src.domain.entities import DataSource, SourceStatus, SourceType
from ui.dialogs.settings_dialog import SettingsDialog
from ui.dialogs.add_source_dialog import AddSourceDialog
from ui.wizards.import_wizard import ImportWizard

class DialogHandler(QObject):
    """
    Handles the creation and business logic of application dialogs and wizards.
    Offloads this responsibility from the MainWindow to adhere to the Single Responsibility Principle.
    """
    
    # Signals to communicate results back to the MainWindow orchestrator
    log_requested = pyqtSignal(str)
    status_requested = pyqtSignal(str)
    language_changed = pyqtSignal(str)
    theme_changed = pyqtSignal(str)
    log_level_changed = pyqtSignal(str)
    source_added = pyqtSignal(object) 

    def __init__(self, main_window, app_state, current_language: str):
        super().__init__(main_window)
        self.main_window = main_window 
        self.app_state = app_state
        self.current_language = current_language

    def open_settings(self):
        """Opens the settings dialog and handles configuration changes."""
        
        settings = QSettings("Noxfort Systems", "SYNAPSE")
        
        db_config = {
            "host": settings.value("Database/host", "localhost", type=str),
            "port": settings.value("Database/port", 5432, type=int),
            "dbname": settings.value("Database/dbname", "synapse_db", type=str),
            "user": settings.value("Database/user", "synapse_user", type=str),
            "password": settings.value("Database/password", "synapse123", type=str)
        }
        
        dialog = SettingsDialog(
            current_language=self.main_window.current_language, 
            current_theme=self.main_window.current_theme,
            current_log_level=self.main_window.current_log_level,
            monitor_enabled=self.main_window.monitor_enabled,
            monitor_host=self.main_window.monitor_host,
            monitor_port=self.main_window.monitor_port,
            db_config=db_config,
            parent=self.main_window
        )
        
        if dialog.exec():
            # Extract
            selected_code = dialog.tab_general.get_selected_language_code()
            config = dialog.tab_general.get_config_data()
            new_theme = config["theme"]
            new_log = config["log_level"]
            
            mon_config = dialog.tab_monitor.get_config_data()
            new_mon_enabled = mon_config["monitor_enabled"]
            new_mon_host = mon_config["monitor_host"]
            new_mon_port = mon_config["monitor_port"]
            
            new_db_config = dialog.tab_database.get_config_data()
            
            # Save to Disk via QSettings
            settings.setValue("General/language", selected_code)
            settings.setValue("General/theme", new_theme)
            settings.setValue("General/log_level", new_log)
            settings.setValue("Monitor/enabled", new_mon_enabled)
            settings.setValue("Monitor/host", new_mon_host)
            settings.setValue("Monitor/port", new_mon_port)
            
            settings.setValue("Database/host", new_db_config.get("host", "localhost"))
            settings.setValue("Database/port", new_db_config.get("port", 5432))
            settings.setValue("Database/dbname", new_db_config.get("dbname", "synapse_db"))
            settings.setValue("Database/user", new_db_config.get("user", "synapse_user"))
            settings.setValue("Database/password", new_db_config.get("password", "synapse123"))
            
            settings.sync()
            
            # Handle Language Change (Requires Restart for Stability)
            if selected_code != self.main_window.current_language:
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.information(
                    self.main_window, 
                    self.tr("Restart Required"), 
                    self.tr("Please restart the application for the language changes to take effect.")
                )
                # We intentionally DO NOT emit language_changed to prevent UI layout breakage.
                # The new language is already saved to QSettings and will load on next boot.
                
            self.main_window.current_theme = new_theme
            self.theme_changed.emit(str(new_theme))
            
            self.main_window.current_log_level = new_log
            self.log_level_changed.emit(new_log)
            
            # Hot-Swap telemetry logic if updated
            self.main_window.monitor_enabled = new_mon_enabled
            self.main_window.monitor_host = new_mon_host
            self.main_window.monitor_port = new_mon_port
            self.main_window.controller.reconfigure_telemetry(new_mon_enabled, new_mon_host, new_mon_port)
            
        dialog.deleteLater()

    def _on_language_changed(self, lang_code: str):
        self.current_language = lang_code
        self.language_changed.emit(lang_code)
        self.log_requested.emit(f"[System] UI Language updated to {lang_code}")

    def open_add_source_dialog(self, current_source_count: int):
        """Opens the dialog to add a new data source and registers it."""
        d = AddSourceDialog(self.main_window)
        if d.exec():
            data = d.get_source_data()
            sid = f"src_{data['name'].lower().replace(' ', '_')}_{current_source_count}"
            
            src = DataSource(
                id=sid,
                name=data['name'],
                connection_string=data['connection'],
                is_local=data['is_local']
            )
            self.app_state.add_data_source(src)
            self.source_added.emit(src)
            
            if src.is_local: 
                self.app_state.enter_association_mode(src.id)


    def open_import_wizard(self):
        """Opens the historical data import wizard and registers the resulting Parquet file."""
        wizard = ImportWizard(self.main_window)
        if wizard.exec():
            base_parquet_path = os.path.join(
                os.path.expanduser("~"), 
                "Documentos", "Synapse", "datalake", "base", "base_v1.parquet"
            )
            
            if os.path.exists(base_parquet_path):
                self.log_requested.emit(f"[System] Historical Data imported: {base_parquet_path}")
                
                # Register a SINGLE historical marker — NOT individual sensors.
                # Sensor categorization is handled internally by Phase 1 (MEH pipeline).
                new_source = DataSource(
                    id=f"historical_{int(time.time())}",
                    name="Historical Base (Imported)",
                    source_type=SourceType.PARQUET,
                    connection_string=base_parquet_path,
                    is_local=True,
                    status=SourceStatus.ACTIVE,
                )
                self.app_state.add_data_source(new_source)
                self.source_added.emit(new_source)
            else:
                self.log_requested.emit(f"[System] Warning: Import finished but file not found at {base_parquet_path}")

            self.log_requested.emit("[System] Data Import Complete. Golden Source Ready.")
            self.status_requested.emit(self.tr("Ready for Phase 0 (Optimization)"))