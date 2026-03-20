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
# File: ui/tabs/settings_monitor_tab.py
# Author: Gabriel Moraes
# Date: 2026-03-03

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QFormLayout, QLineEdit, QPushButton, QHBoxLayout, QLabel
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from src.infrastructure.monitor_client import MonitorClient
from ui.styles.theme_manager import ThemeManager

class SettingsMonitorTab(QWidget):
    """
    Component responsible for the external Telemetry/Monitor Server settings UI.
    Simplified: Accepts only a Server IP and a Connect button.
    """
    def __init__(self, monitor_enabled: bool = False, monitor_host: str = "localhost", monitor_port: int = 1883, parent=None):
        super().__init__(parent)
        # We still accept legacy enabled/port args from the Dialog to avoid breaking the signature pipeline,
        # but we only represent HOST visually.
        if monitor_enabled and monitor_port != 1883:
            self.monitor_host_val = f"{monitor_host}:{monitor_port}"
        else:
            self.monitor_host_val = monitor_host if monitor_enabled else ""
        self.is_connected = monitor_enabled
        self._setup_ui()

    def _setup_ui(self):
        """Initializes the layout and widgets for the tab."""
        layout = QVBoxLayout(self)
        
        # Info Label
        info = QLabel(self.tr("Enter the Monitor IP to enable telemetry streaming."))
        info.setStyleSheet(f"color: {ThemeManager.get_hex('text_muted')}; margin-bottom: 10px;")
        layout.addWidget(info)
        
        # Input Form
        form = QFormLayout()
        self.mon_host = QLineEdit(self.monitor_host_val)
        self.mon_host.setPlaceholderText("192.168.1.100")
        form.addRow(self.tr("Server IP:"), self.mon_host)
        layout.addLayout(form)
        
        # Action Button Row
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        
        self.connect_btn = QPushButton(self.tr("Connect"))
        self.connect_btn.setFixedWidth(120)
        self.connect_btn.clicked.connect(self._on_connect_clicked)
        btn_layout.addWidget(self.connect_btn)
        
        layout.addLayout(btn_layout)
        layout.addStretch()
        
        # Apply initial visual state based on existing configuration
        if self.mon_host.text().strip():
            self._set_btn_success()

    def _on_connect_clicked(self):
        """Visual feedback slot for the Connect button."""
        if self.is_connected:
            self._handle_disconnect()
            return
            
        ip_raw = self.mon_host.text().strip()
        if not ip_raw:
            self._set_btn_normal()
            return
            
        host = ip_raw
        port = 1883
        if ":" in ip_raw:
            parts = ip_raw.split(":")
            host = parts[0]
            try:
                port = int(parts[1])
            except ValueError:
                pass
            
        # Give immediate feedback that we are trying
        self.connect_btn.setText(self.tr("Connecting..."))
        self.connect_btn.setStyleSheet(f"background-color: {ThemeManager.get_hex('warning')}; color: white;")
        self.connect_btn.setEnabled(False)
        self.mon_host.setEnabled(False) # Prevent editing while connecting
        
        # Fire a quick connection test in the background so UI doesn't freeze
        self._test_thread = HeartbeatTestThread(host, port)
        self._test_thread.finished.connect(self._on_test_finished)
        self._test_thread.start()
            
    def _on_test_finished(self, success: bool):
        self.connect_btn.setEnabled(True)
        if success:
            self.is_connected = True
            self._set_btn_success()
        else:
            self.is_connected = False
            self.mon_host.setEnabled(True)
            self._set_btn_error()

    def _handle_disconnect(self):
        self.is_connected = False
        self.mon_host.setEnabled(True)
        self.mon_host.clear()
        self._set_btn_normal()

    def _set_btn_success(self):
        self.connect_btn.setText(self.tr("Disconnect"))
        self.connect_btn.setStyleSheet(f"background-color: {ThemeManager.get_hex('status_online')}; color: white;")
        
    def _set_btn_error(self):
        self.connect_btn.setText(self.tr("Connection Failed"))
        self.connect_btn.setStyleSheet(f"background-color: {ThemeManager.get_hex('danger')}; color: white;")

    def _set_btn_normal(self):
        self.connect_btn.setText(self.tr("Connect"))
        self.connect_btn.setStyleSheet("")

    def get_config_data(self) -> dict:
        """
        Retrieves the simplified configuration.
        Relies on the `is_connected` state.
        """
        ip_raw = self.mon_host.text().strip()
        
        host = ip_raw
        port = 1883
        if ":" in ip_raw:
            parts = ip_raw.split(":")
            host = parts[0]
            try:
                port = int(parts[1])
            except ValueError:
                pass
                
        return {
            "monitor_enabled": self.is_connected and bool(host),
            "monitor_host": host if host else "localhost",
            "monitor_port": port
        }

class HeartbeatTestThread(QThread):
    finished = pyqtSignal(bool)
    
    def __init__(self, host: str, port: int):
        super().__init__()
        self.host = host
        self.port = port
        
    def run(self):
        success = MonitorClient.test_connection_and_send_heartbeat(self.host, self.port)
        self.finished.emit(success)
