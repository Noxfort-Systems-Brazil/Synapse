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
# File: ui/handlers/locale_manager.py
# Author: Gabriel Moraes
# Date: 2026-03-03

import os
from PyQt6.QtCore import QObject, QTranslator, QCoreApplication, QSettings

class LocaleManager(QObject):
    """
    Manages application localization, loading .qm files and persisting 
    the user's language preference in config/settings.ini.
    """
    
    def __init__(self, app: QCoreApplication):
        super().__init__()
        self._app = app
        self._translator = QTranslator()
        
        # Define root directory (assuming this file is in ui/handlers/)
        self._base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self._config_dir = os.path.join(self._base_dir, 'config')
        
        # Ensure config directory exists
        if not os.path.exists(self._config_dir):
            os.makedirs(self._config_dir)
            
        # Initialize QSettings targeting config/settings.ini
        self._settings_path = os.path.join(self._config_dir, 'settings.ini')
        self._settings = QSettings(self._settings_path, QSettings.Format.IniFormat)
        
        # Define locales directory mapping to ui/locales/
        self._locales_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'locales')
        
        # Load persisted language or default to English
        self._current_language = self._settings.value("General/language", "en_us")
        self.apply_language(self._current_language)

    def apply_language(self, language_code: str) -> bool:
        """
        Loads the .qm translation file and applies it to the application.
        Persists the choice in settings.ini.
        
        Args:
            language_code (str): The language code (e.g., 'en_us', 'pt_br').
            
        Returns:
            bool: True if the translation was successfully loaded, False otherwise.
        """
        qm_file_path = os.path.join(self._locales_dir, f"{language_code}.qm")
        
        # Remove previous translator if any
        self._app.removeTranslator(self._translator)
        
        if os.path.exists(qm_file_path):
            if self._translator.load(qm_file_path):
                self._app.installTranslator(self._translator)
                self._current_language = language_code
                
                # Persist the new language choice
                self._settings.setValue("General/language", self._current_language)
                self._settings.sync()
                return True
                
        return False

    def get_current_language(self) -> str:
        """
        Returns the currently active language code.
        """
        return self._current_language