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
# File: ui/panels/database_setup_panel.py
# Author: Gabriel Moraes
# Date: 2026-02-28

from PyQt6.QtCore import Qt, pyqtSignal, QEvent, pyqtSlot
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QLineEdit, 
    QPushButton, QLabel, QGroupBox, QMessageBox
)
from ui.styles.theme_manager import ThemeManager

class DatabaseSetupPanel(QWidget):
    """
    Panel responsible for the first-time database installation UI.
    Emits the requested root password to the orchestrator.
    """
    
    # Signal emitted when the user requests initialization
    # Payload: (root_password)
    request_initialize = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        lbl_title = QLabel(self.tr("⚠️ Database Not Configured"))
        lbl_title.setStyleSheet(f"font-size: 16pt; font-weight: bold; color: {ThemeManager.get_hex('warning')};")
        layout.addWidget(lbl_title, alignment=Qt.AlignmentFlag.AlignCenter)
        
        lbl_desc = QLabel(
            self.tr("The SYNAPSE infrastructure (PostgreSQL) is not initialized.\nWe can set it up automatically for you.")
        )
        lbl_desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(lbl_desc)
        
        group = QGroupBox(self.tr("Installation Credentials"))
        form = QFormLayout(group)
        
        self.setup_pg_pass = QLineEdit()
        self.setup_pg_pass.setEchoMode(QLineEdit.EchoMode.Password)
        self.setup_pg_pass.setPlaceholderText(self.tr("Enter the 'postgres' user password"))
        
        self.setup_pg_pass.installEventFilter(self)
        
        form.addRow(self.tr("Root Password:"), self.setup_pg_pass)
        layout.addWidget(group)
        
        self.btn_install = QPushButton(self.tr("🚀 Initialize Infrastructure"))
        self.btn_install.setMinimumHeight(45)
        self.btn_install.setStyleSheet(f"background-color: {ThemeManager.get_hex('status_online')}; color: white; font-weight: bold; border-radius: 4px;")
        self.btn_install.clicked.connect(self._on_click_initialize)
        layout.addWidget(self.btn_install)
        
        layout.addStretch()

    def eventFilter(self, obj, event):
        """
        Intercepts key presses on the password field to prevent the Enter key 
        from propagating to the parent QDialog.
        """
        if obj == self.setup_pg_pass and event.type() == QEvent.Type.KeyPress:
            if event.key() == Qt.Key.Key_Return or event.key() == Qt.Key.Key_Enter:
                self._on_click_initialize()
                return True
        return super().eventFilter(obj, event)

    @pyqtSlot()
    def _on_click_initialize(self):
        root_pass = self.setup_pg_pass.text()
        if not root_pass:
            QMessageBox.warning(self, self.tr("Input Required"), self.tr("Please enter the root password for PostgreSQL."))
            return
            
        self.set_ui_state(is_processing=True)
        self.request_initialize.emit(root_pass)

    def set_ui_state(self, is_processing: bool):
        """Disables or enables the UI based on the processing state."""
        if is_processing:
            self.btn_install.setEnabled(False)
            self.btn_install.setText(self.tr("Installing... Please Wait"))
        else:
            self.btn_install.setEnabled(True)
            self.btn_install.setText(self.tr("🚀 Initialize Infrastructure"))

    def clear_password(self):
        """Clears the password field for security."""
        self.setup_pg_pass.clear()