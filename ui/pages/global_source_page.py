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
# File: ui/pages/global_source_page.py
# Author: Gabriel Moraes
# Date: 2026-03-01

from PyQt6.QtWidgets import QFrame, QVBoxLayout, QLabel, QLineEdit
from PyQt6.QtGui import QFont

class GlobalSourcePage(QFrame):
    """
    Page 2A of the Add Source Wizard.
    Collects configuration data for a Global (Cloud API) data source.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        """Initializes the visual components of the page."""
        layout = QVBoxLayout(self)
        
        title = QLabel("Configure Global Source")
        title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        layout.addWidget(title)
        layout.addSpacing(20)
        
        layout.addWidget(QLabel("Source Name:"))
        self.inp_global_name = QLineEdit()
        self.inp_global_name.setPlaceholderText("e.g. Waze API North")
        layout.addWidget(self.inp_global_name)
        
        layout.addWidget(QLabel("Endpoint URL:"))
        self.inp_global_url = QLineEdit()
        self.inp_global_url.setPlaceholderText("https://api.provider.com/v1/traffic")
        layout.addWidget(self.inp_global_url)
        
        layout.addStretch()

    def focus_first_input(self):
        """Sets focus to the first input field for better UX."""
        self.inp_global_name.setFocus()

    def is_valid(self) -> bool:
        """
        Validates if the required fields are filled.
        
        Returns:
            bool: True if valid, False otherwise.
        """
        name = self.inp_global_name.text().strip()
        url = self.inp_global_url.text().strip()
        return bool(name and url)

    def get_source_data(self) -> dict:
        """
        Retrieves the formulated data dictionary for the global source.
        
        Returns:
            dict: The source configuration data.
        """
        name = self.inp_global_name.text().strip()
        url = self.inp_global_url.text().strip()
        
        return {
            "name": name,
            "connection": url,
            "is_local": False,
            "requires_location": False
        }