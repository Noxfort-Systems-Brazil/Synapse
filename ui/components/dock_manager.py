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
# File: ui/components/dock_manager.py
# Author: Gabriel Moraes
# Date: 2026-03-01

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction
from typing import List

from ui.widgets.source_list_dock import SourceListDock
from ui.widgets.control_panel_dock import ControlPanelDock

class DockManager:
    """
    Component responsible for instantiating and placing dock widgets in the main window.
    Adheres to SRP by removing layout management logic from the main window.
    """
    def __init__(self, main_window):
        self.main_window = main_window
        
        # Instantiate Docks
        self.source_dock = SourceListDock(self.main_window)
        self.control_dock = ControlPanelDock(self.main_window)
        
        self._setup_docks()

    def _setup_docks(self):
        """Adds the initialized docks to their respective areas in the main window."""
        self.main_window.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.source_dock)
        self.main_window.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.control_dock)

    def get_view_actions(self) -> List[QAction]:
        """
        Returns the toggle view actions for the menus to use,
        maintaining encapsulation of the dock instances.
        """
        return [
            self.source_dock.toggleViewAction(),
            self.control_dock.toggleViewAction()
        ]