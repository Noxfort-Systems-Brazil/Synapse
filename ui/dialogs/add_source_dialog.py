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
# File: ui/dialogs/add_source_dialog.py
# Author: Gabriel Moraes
# Date: 2026-03-01

from typing import Optional
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QStackedWidget, QMessageBox
from ui.pages.select_source_page import SelectSourcePage
from ui.pages.global_source_page import GlobalSourcePage
from ui.pages.local_source_page import LocalSourcePage

class AddSourceDialog(QDialog):
    """
    Wizard-style dialog to register a new Data Source.
    
    Refactored Version:
    - Acts solely as an orchestrator/controller (SRP).
    - Delegates UI rendering and validation to specific Page classes.
    - Utilizes signals for navigation.
    """

    def __init__(self, parent=None):
        super().__init__(parent) # type: ignore
        self.setWindowTitle(self.tr("Add Data Source Wizard"))
        self.resize(600, 400)
        self.setModal(True)
        
        # Result Container
        self.source_data: dict = {}
        self.current_mode: Optional[str] = None
        
        # Main Layout
        self.layout = QVBoxLayout(self)
        
        # Stacked Widget for Wizard Pages
        self.pages = QStackedWidget()
        self.layout.addWidget(self.pages)
        
        # Initialize Pages
        self.page_select = SelectSourcePage()
        self.page_global = GlobalSourcePage()
        self.page_local = LocalSourcePage()
        
        self.pages.addWidget(self.page_select)
        self.pages.addWidget(self.page_global)
        self.pages.addWidget(self.page_local)

        # Navigation Buttons (Bottom)
        self.nav_layout = QHBoxLayout()
        self.btn_back = QPushButton("Back")
        self.btn_back.clicked.connect(self._go_back)
        self.btn_back.hide() 
        
        self.btn_next = QPushButton("Next") 
        self.btn_next.clicked.connect(self._on_submit)
        self.btn_next.hide() 
        
        self.nav_layout.addWidget(self.btn_back)
        self.nav_layout.addStretch()
        self.nav_layout.addWidget(self.btn_next)
        self.layout.addLayout(self.nav_layout)

        # Connect Signals
        self.page_select.mode_selected.connect(self._handle_mode_selection)

    def _handle_mode_selection(self, mode: str):
        """Routes the user to the correct page based on their selection."""
        self.current_mode = mode
        
        if mode == "GLOBAL":
            self.pages.setCurrentWidget(self.page_global)
            self.btn_next.setText("Finish & Register")
            self.page_global.focus_first_input()
        elif mode == "LOCAL":
            self.pages.setCurrentWidget(self.page_local)
            self.btn_next.setText("Finish & Listen")
            self.page_local.focus_first_input()

        self.btn_back.show()
        self.btn_next.show()

    def _go_back(self):
        """Returns to the first page and hides navigation buttons."""
        self.current_mode = None
        self.pages.setCurrentWidget(self.page_select)
        self.btn_back.hide()
        self.btn_next.hide()

    def _on_submit(self):
        """Validates the current page and extracts the source data."""
        active_page = self.pages.currentWidget()
        
        # Validate data via the specific page's method
        if not active_page.is_valid():
            QMessageBox.warning(self, "Missing Data", "Please fill in all required fields.")
            return
            
        # Extract formulated data
        self.source_data = active_page.get_source_data()
        self.accept()

    def get_source_data(self) -> dict:
        """
        Returns the configuration data collected from the wizard.
        To be called by the main window after the dialog is accepted.
        """
        return self.source_data