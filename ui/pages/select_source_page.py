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
# File: ui/pages/select_source_page.py
# Author: Gabriel Moraes
# Date: 2026-03-01

from PyQt6.QtWidgets import QFrame, QVBoxLayout, QLabel, QPushButton
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
from ui.styles.theme_manager import ThemeManager

class SelectSourcePage(QFrame):
    """
    Page 1 of the Add Source Wizard.
    Allows the user to select between a Local (Push) or Global (Pull) source strategy.
    """
    
    # Signal emitted when a mode is selected by the user. 
    # Emits "LOCAL" or "GLOBAL".
    mode_selected = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        """Initializes the visual components of the page."""
        layout = QVBoxLayout(self)
        
        title = QLabel(self.tr("Select Source Strategy"))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        layout.addWidget(title)
        
        layout.addSpacing(20)

        # Button: LOCAL (PUSH)
        btn_local = QPushButton(
            self.tr("📡 LOCAL (Push Mode)\n\n"
            "Use this for Sensors, Cameras, or Loops inside your network.\n"
            "Synapse will LISTEN for incoming data.")
        )
        btn_local.setStyleSheet(f"""
            QPushButton {{ 
                text-align: left; 
                padding: 15px; 
                font-size: 14px; 
                background-color: {ThemeManager.get_hex('background_light')};
                color: {ThemeManager.get_hex('text_main')};
                border-radius: 8px; 
                border: 1px solid transparent;
            }}
            QPushButton:hover {{ 
                background-color: {ThemeManager.get_hex('background_hover')}; 
                border: 2px solid {ThemeManager.get_hex('primary')}; 
            }}
        """)
        btn_local.clicked.connect(lambda: self.mode_selected.emit("LOCAL"))
        layout.addWidget(btn_local)
        
        layout.addSpacing(10)

        # Button: GLOBAL (REQUEST)
        btn_global = QPushButton(
            self.tr("☁️ GLOBAL (Request Mode)\n\n"
            "Use this for Cloud APIs (Waze, TomTom, Weather).\n"
            "Synapse will REQUEST data from a URL.")
        )
        btn_global.setStyleSheet(f"""
            QPushButton {{ 
                text-align: left; 
                padding: 15px; 
                font-size: 14px; 
                background-color: {ThemeManager.get_hex('background_light')};
                color: {ThemeManager.get_hex('text_main')};
                border-radius: 8px;
                border: 1px solid transparent;
            }}
            QPushButton:hover {{ 
                background-color: {ThemeManager.get_hex('background_hover')}; 
                border: 2px solid {ThemeManager.get_hex('success')}; 
            }}
        """)
        btn_global.clicked.connect(lambda: self.mode_selected.emit("GLOBAL"))
        layout.addWidget(btn_global)
        
        layout.addStretch()