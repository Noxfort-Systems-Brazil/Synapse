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
# File: ui/tabs/settings_security_tab.py
# Author: Gabriel Moraes
# Date: 2026-03-01

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QLineEdit, 
    QPushButton, QFileDialog, QHBoxLayout, QLabel
)

class SettingsSecurityTab(QWidget):
    """
    Component responsible for the Security (mTLS) settings UI.
    Adheres to SRP by isolating the layout and file picking logic for certificates.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        """Initializes the layout and widgets for the tab."""
        layout = QVBoxLayout(self)
        
        info_lbl = QLabel(self.tr("Select the certificates for mTLS authentication with the Core."))
        info_lbl.setWordWrap(True)
        layout.addWidget(info_lbl)
        
        # Use a container layout for the form to align properly
        form_container = QWidget()
        form = QFormLayout(form_container)
        form.setContentsMargins(0, 0, 0, 0)
        
        self.ca_path = self._create_file_picker(form, self.tr("CA Certificate (.crt):"))
        self.client_cert = self._create_file_picker(form, self.tr("Client Certificate (.crt):"))
        self.client_key = self._create_file_picker(form, self.tr("Client Key (.key):"))
        
        layout.addWidget(form_container)
        layout.addStretch()

    def _create_file_picker(self, parent_layout: QFormLayout, label_text: str) -> QLineEdit:
        """Helper to create a row with a LineEdit and a Browse Button."""
        container = QWidget()
        h_layout = QHBoxLayout(container)
        h_layout.setContentsMargins(0, 0, 0, 0)
        
        line_edit = QLineEdit()
        btn = QPushButton(self.tr("Browse..."))
        
        btn.clicked.connect(lambda: self._open_file_dialog(line_edit))
        
        h_layout.addWidget(line_edit)
        h_layout.addWidget(btn)
        
        parent_layout.addRow(label_text, container)
        
        return line_edit

    def _open_file_dialog(self, target_line_edit: QLineEdit):
        """Opens the system file dialog and updates the target line edit."""
        fname, _ = QFileDialog.getOpenFileName(self, self.tr("Select File"))
        if fname:
            target_line_edit.setText(fname)

    def get_config_data(self) -> dict:
        """
        Retrieves the selected certificate paths.
        """
        return {
            "ca_path": self.ca_path.text(),
            "client_cert": self.client_cert.text(),
            "client_key": self.client_key.text()
        }