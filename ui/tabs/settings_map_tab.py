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
# File: ui/tabs/settings_map_tab.py
# Author: Gabriel Moraes
# Date: 2026-03-01

from PyQt6.QtWidgets import QWidget, QFormLayout, QCheckBox, QSpinBox

class SettingsMapTab(QWidget):
    """
    Component responsible for the Map and Visuals settings UI.
    Adheres to SRP by isolating the configuration layout for UI refresh rates and map debug overlays.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        """Initializes the layout and widgets for the tab."""
        layout = QFormLayout(self)
        
        # Map Debug Visuals Configuration
        self.show_ids = QCheckBox(self.tr("Show Node/Edge IDs on Map"))
        layout.addRow(self.tr("Debug Visuals:"), self.show_ids)
        
        # Dashboard Refresh Rate Configuration
        self.refresh_rate = QSpinBox()
        self.refresh_rate.setRange(100, 5000)
        self.refresh_rate.setValue(1000)
        self.refresh_rate.setSuffix(" ms")
        layout.addRow(self.tr("Dashboard Refresh Rate:"), self.refresh_rate)

    def get_config_data(self) -> dict:
        """
        Retrieves the selected visual and map settings.
        """
        return {
            "show_map_ids": self.show_ids.isChecked(),
            "refresh_rate_ms": self.refresh_rate.value()
        }