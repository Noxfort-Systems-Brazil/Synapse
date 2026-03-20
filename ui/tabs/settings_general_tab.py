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
# File: ui/tabs/settings_general_tab.py
# Author: Gabriel Moraes
# Date: 2026-03-01

from PyQt6.QtWidgets import QWidget, QFormLayout, QComboBox

class SettingsGeneralTab(QWidget):
    """
    Component responsible for General settings UI (Theme, Language, Log Level).
    Adheres to SRP by isolating the configuration layout for global application behaviors.
    """
    def __init__(self, current_language: str = "pt_BR", current_theme: int = 0, current_log_level: str = "INFO", parent=None):
        super().__init__(parent)
        self.current_language = current_language
        self.current_theme = current_theme
        self.current_log_level = current_log_level
        self._setup_ui()

    def _setup_ui(self):
        """Initializes the layout and widgets for the tab."""
        layout = QFormLayout(self)
        
        # Theme Settings
        self.theme_combo = QComboBox()
        self.theme_combo.addItems([self.tr("System Default"), self.tr("Light Mode"), self.tr("Dark Mode")])
        self.theme_combo.setCurrentIndex(self.current_theme)
        layout.addRow(self.tr("Theme:"), self.theme_combo)
        
        # Language Settings
        self.lang_combo = QComboBox()
        self.lang_options = {
            "Português (BR)": "pt_BR",
            "English": "en_US",
            "Français": "fr_FR",
            "Español": "es_ES",
            "Русский": "ru_RU",
            "中文 (Mandarin)": "zh_CN"
        }
        self.lang_combo.addItems(list(self.lang_options.keys()))
        
        # Set current active language
        for text, code in self.lang_options.items():
            if code == self.current_language:
                self.lang_combo.setCurrentText(text)
                break
                
        layout.addRow(self.tr("Language (UI):"), self.lang_combo)
        
        # Logging Configuration
        self.log_level = QComboBox()
        self.log_level.addItems(["DEBUG", "INFO", "WARNING", "ERROR"])
        self.log_level.setCurrentText(self.current_log_level)
        layout.addRow(self.tr("Log Level:"), self.log_level)

    def get_selected_language_code(self) -> str:
        """Returns the internal code of the selected language (e.g., 'pt_BR')."""
        selected_display = self.lang_combo.currentText()
        return self.lang_options.get(selected_display, "en_US")
        
    def get_config_data(self) -> dict:
        """Retrieves the general settings input by the user."""
        return {
            "theme": self.theme_combo.currentIndex(),
            "language": self.get_selected_language_code(),
            "log_level": self.log_level.currentText()
        }