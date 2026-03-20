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
# File: ui/utilities/translation_manager.py
# Author: Gabriel Moraes
# Date: 2026-02-28

import os
from PyQt6.QtCore import QTranslator, QCoreApplication, QLocale, QLibraryInfo

class TranslationManager:
    """
    Manages the application's internationalization (i18n) by dynamically loading
    and applying .qm translation files to the QCoreApplication instance.
    """

    def __init__(self, app_instance: QCoreApplication, translations_dir: str = "ui/translations"):
        """
        Initializes the TranslationManager.

        Args:
            app_instance (QCoreApplication): The main application instance.
            translations_dir (str): Relative or absolute path to the directory containing .qm files.
        """
        self._app = app_instance
        self._translations_dir = os.path.abspath(translations_dir)
        self._app_translator = QTranslator()
        self._system_translator = QTranslator()

        # Try to install system defaults (e.g., standard dialog buttons like 'Ok', 'Cancel')
        self._load_system_defaults()

    def load_language(self, language_code: str) -> bool:
        """
        Loads and applies a specific language based on its code (e.g., 'pt_BR', 'en_US').

        Args:
            language_code (str): The language code matching the .qm file suffix.

        Returns:
            bool: True if the translation was successfully loaded, False otherwise.
        """
        # Remove previously installed translator to avoid stacking
        self._app.removeTranslator(self._app_translator)

        if language_code == "en_US":
            # English is usually the base source code language; no translation file needed
            return True

        qm_file_name = f"synapse_{language_code}.qm"
        file_path = os.path.join(self._translations_dir, qm_file_name)

        if self._app_translator.load(file_path):
            self._app.installTranslator(self._app_translator)
            return True
        else:
            print(f"[WARNING] Failed to load translation file: {file_path}")
            return False

    def _load_system_defaults(self):
        """
        Loads standard Qt framework translations to ensure base UI elements
        respect the OS or default application locale.
        """
        locale = QLocale.system().name()
        qt_translations_path = QLibraryInfo.path(QLibraryInfo.LibraryPath.TranslationsPath)
        
        if self._system_translator.load(f"qtbase_{locale}", qt_translations_path):
            self._app.installTranslator(self._system_translator)