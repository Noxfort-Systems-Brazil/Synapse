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
# File: ui/pages/local_source_page.py
# Author: Gabriel Moraes
# Date: 2026-03-01

from PyQt6.QtWidgets import (
    QFrame, QVBoxLayout, QHBoxLayout, QLabel, 
    QLineEdit, QPushButton, QGroupBox, QApplication
)
from PyQt6.QtGui import QFont
from ui.utilities.network_utils import NetworkUtils
from ui.styles.theme_manager import ThemeManager

class LocalSourcePage(QFrame):
    """
    Page 2B of the Add Source Wizard.
    Collects configuration data for a Local (Push) data source and 
    provides the generated endpoint details to the user.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.local_ip = NetworkUtils.get_local_ip()
        self.generated_id = ""
        self._init_ui()

    def _init_ui(self):
        """Initializes the visual components of the page."""
        layout = QVBoxLayout(self)
        
        title = QLabel(self.tr("Configure Local Device"))
        title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        layout.addWidget(title)
        
        # 1. Name Input
        layout.addWidget(QLabel(self.tr("1. Name this Source:")))
        self.inp_local_name = QLineEdit()
        self.inp_local_name.setPlaceholderText(self.tr("e.g. Camera 01"))
        self.inp_local_name.textChanged.connect(self._update_local_fields)
        layout.addWidget(self.inp_local_name)
        
        layout.addSpacing(20)
        
        # 2. Connection Details Group
        group = QGroupBox(self.tr("Device Configuration Details"))
        group_layout = QVBoxLayout()
        
        # URL Field
        group_layout.addWidget(QLabel(self.tr("Server Endpoint (URL):")))
        url_layout = QHBoxLayout()
        self.out_url = QLineEdit()
        self.out_url.setReadOnly(True)
        self.out_url.setStyleSheet(f"background-color: {ThemeManager.get_hex('background_light')}; color: {ThemeManager.get_hex('text_main')}; border: 1px solid {ThemeManager.get_hex('card_border')}; padding: 2px;")
        
        btn_copy_url = QPushButton(self.tr("Copy"))
        btn_copy_url.setFixedWidth(60)
        btn_copy_url.clicked.connect(lambda: self._copy_text(self.out_url.text()))
        
        url_layout.addWidget(self.out_url)
        url_layout.addWidget(btn_copy_url)
        group_layout.addLayout(url_layout)
        
        # ID Field
        group_layout.addWidget(QLabel(self.tr("Source ID (Must be in payload):")))
        id_layout = QHBoxLayout()
        self.out_id = QLineEdit()
        self.out_id.setReadOnly(True)
        self.out_id.setStyleSheet(f"background-color: {ThemeManager.get_hex('background_light')}; color: {ThemeManager.get_hex('text_main')}; font-weight: bold; border: 1px solid {ThemeManager.get_hex('card_border')}; padding: 2px;")
        
        btn_copy_id = QPushButton(self.tr("Copy"))
        btn_copy_id.setFixedWidth(60)
        btn_copy_id.clicked.connect(lambda: self._copy_text(self.out_id.text()))
        
        id_layout.addWidget(self.out_id)
        id_layout.addWidget(btn_copy_id)
        group_layout.addLayout(id_layout)
        
        group.setLayout(group_layout)
        layout.addWidget(group)

        # Warning Note
        layout.addSpacing(10)
        note = QLabel(
            self.tr("ℹ️ Configure your device (CSV/JSON/XML) to POST data to the URL above.\nEnsure the payload includes the Source ID.")
        )
        note.setStyleSheet(f"color: {ThemeManager.get_hex('text_muted')}; font-style: italic;")
        note.setWordWrap(True)
        layout.addWidget(note)
        
        layout.addStretch()

        # Initialize default fields
        self._update_local_fields()

    def _update_local_fields(self):
        """Updates the read-only URL and ID fields based on the provided name."""
        name = self.inp_local_name.text()
        if not name: 
            name = "device"
        
        # Generate ID: Clean snake_case
        self.generated_id = f"src_{name.lower().replace(' ', '_').replace('-', '_')}"
        
        # Generate URL
        url = f"http://{self.local_ip}:8080"
        
        self.out_url.setText(url)
        self.out_id.setText(self.generated_id)

    def _copy_text(self, text: str):
        """Copies the given text to the system clipboard."""
        QApplication.clipboard().setText(text)

    def focus_first_input(self):
        """Sets focus to the first input field for better UX."""
        self.inp_local_name.setFocus()

    def is_valid(self) -> bool:
        """
        Validates if the required fields are filled.
        
        Returns:
            bool: True if valid, False otherwise.
        """
        name = self.inp_local_name.text().strip()
        return bool(name)

    def get_source_data(self) -> dict:
        """
        Retrieves the formulated data dictionary for the local source.
        
        Returns:
            dict: The source configuration data.
        """
        name = self.inp_local_name.text().strip()
        
        return {
            "name": name,
            "connection": "Listen on Port 8080",  # Just a label for UI
            "is_local": True,
            "requires_location": True
        }