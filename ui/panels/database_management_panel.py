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
# File: ui/panels/database_management_panel.py
# Author: Gabriel Moraes
# Date: 2026-02-28

from PyQt6.QtCore import pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QLineEdit, 
    QPushButton, QLabel, QGroupBox, QSpinBox, QMessageBox, QInputDialog
)
from ui.styles.theme_manager import ThemeManager

class DatabaseManagementPanel(QWidget):
    """
    Panel responsible for the day-to-day database management UI.
    Emits signals for testing connection and factory resets.
    """
    
    # Signals to request actions from the orchestrator
    request_check_connection = pyqtSignal(dict)
    request_reset_db = pyqtSignal(str, dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        
        self.lbl_status = QLabel(self.tr("✅ System Online"))
        self.lbl_status.setStyleSheet(f"font-size: 12pt; font-weight: bold; color: {ThemeManager.get_hex('status_online')}; margin-bottom: 10px;")
        layout.addWidget(self.lbl_status)
        
        grp_conn = QGroupBox("Connection Details (synapse_user)")
        form = QFormLayout(grp_conn)
        
        self.edit_host = QLineEdit("localhost")
        self.edit_port = QSpinBox()
        self.edit_port.setRange(1, 65535)
        self.edit_port.setValue(5432)
        self.edit_db = QLineEdit("synapse_db")
        self.edit_user = QLineEdit("synapse_user")
        self.edit_pass = QLineEdit("synapse123")
        self.edit_pass.setEchoMode(QLineEdit.EchoMode.Password)
        
        form.addRow(self.tr("Host:"), self.edit_host)
        form.addRow(self.tr("Port:"), self.edit_port)
        form.addRow(self.tr("Database:"), self.edit_db)
        form.addRow(self.tr("User:"), self.edit_user)
        form.addRow(self.tr("Password:"), self.edit_pass)
        
        layout.addWidget(grp_conn)
        
        grp_maint = QGroupBox(self.tr("Maintenance Actions"))
        vbox = QVBoxLayout(grp_maint)
        
        btn_check = QPushButton(self.tr("Test Connection"))
        btn_check.clicked.connect(self._on_click_check)
        vbox.addWidget(btn_check)
        
        lbl_danger = QLabel(self.tr("Danger Zone"))
        lbl_danger.setStyleSheet(ThemeManager.get_style('danger_text'))
        vbox.addWidget(lbl_danger)
        
        btn_reset = QPushButton(self.tr("⚠️ Factory Reset (Recreate Schema)"))
        btn_reset.setStyleSheet(f"background-color: {ThemeManager.get_hex('danger')}; color: white;")
        btn_reset.setToolTip(self.tr("Deletes all traffic history and resets the database structure."))
        btn_reset.clicked.connect(self._on_click_reset)
        vbox.addWidget(btn_reset)
        
        layout.addWidget(grp_maint)
        layout.addStretch()

    def _get_current_config(self) -> dict:
        """Helper to extract config dict from UI fields."""
        return {
            "host": self.edit_host.text().strip(),
            "port": self.edit_port.value(),
            "dbname": self.edit_db.text().strip(),
            "user": self.edit_user.text().strip(),
            "password": self.edit_pass.text()
        }

    def set_config(self, config: dict):
        """Pre-fills the UI fields with the existing configuration."""
        if "host" in config: self.edit_host.setText(config["host"])
        if "port" in config: self.edit_port.setValue(int(config["port"]))
        if "dbname" in config: self.edit_db.setText(config["dbname"])
        if "user" in config: self.edit_user.setText(config["user"])
        if "password" in config: self.edit_pass.setText(config["password"])

    @pyqtSlot()
    def _on_click_check(self):
        self.request_check_connection.emit(self._get_current_config())

    @pyqtSlot()
    def _on_click_reset(self):
        confirm = QMessageBox.warning(
            self, 
            self.tr("CRITICAL WARNING"),
            self.tr("This will DELETE ALL DATA in 'synapse_db'.\nAre you sure?"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return

        root_pass, ok = QInputDialog.getText(
            self, self.tr("Authentication"), self.tr("Enter PostgreSQL ROOT password to authorize reset:"), 
            QLineEdit.EchoMode.Password
        )
        
        if ok and root_pass:
            self.request_reset_db.emit(root_pass, self._get_current_config())