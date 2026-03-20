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
# File: ui/components/main_menu.py
# Author: Gabriel Moraes
# Date: 2026-03-01

from PyQt6.QtWidgets import QMenuBar, QMenu
from PyQt6.QtGui import QAction

class MainMenu(QMenuBar):
    """
    Component responsible for managing the application's top menu bar.
    Encapsulates the creation and translation of menus and their respective actions.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_menus()

    def _setup_menus(self):
        """Initializes the menus and their default empty actions (text set by translation)."""
        # --- File Menu ---
        self.file_menu = self.addMenu("")
        
        self.import_db_act = QAction("", self)
        self.file_menu.addAction(self.import_db_act)
        
        self.file_menu.addSeparator()
        
        self.settings_act = QAction("", self)
        self.file_menu.addAction(self.settings_act)
        
        self.file_menu.addSeparator()
        
        self.exit_act = QAction("", self)
        self.file_menu.addAction(self.exit_act)

        # --- View Menu ---
        self.view_menu = self.addMenu("")

    def add_view_action(self, action: QAction):
        """
        Allows the main orchestrator to inject dock toggle actions 
        into the View menu without tightly coupling the components.
        """
        self.view_menu.addAction(action)

    def retranslate_ui(self):
        """Updates the menu titles and actions when the language changes."""
        self.file_menu.setTitle(self.tr("&File"))
        self.import_db_act.setText(self.tr("Import Raw Data (.db)..."))
        self.settings_act.setText(self.tr("Settings..."))
        self.exit_act.setText(self.tr("Exit"))
        
        self.view_menu.setTitle(self.tr("&View"))