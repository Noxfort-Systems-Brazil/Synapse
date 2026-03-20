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
# File: ui/tabs/central_tabs.py
# Author: Gabriel Moraes
# Date: 2026-03-01

from PyQt6.QtWidgets import QTabWidget

from ui.widgets.tab_dashboard import DashboardTab
from ui.widgets.tab_xai import XAITab

class CentralTabs(QTabWidget):
    """
    Component responsible for managing the central tabbed view of the application.
    Encapsulates the instantiation and translation of the main views (Map, Dashboard, XAI).
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_tabs()
        self.retranslate_ui()

    def _setup_tabs(self):
        """Initializes and adds the sub-widgets to the tabs."""
        # Tab 1: Dashboard (MLOps)
        self.dashboard_tab = DashboardTab()
        self.addTab(self.dashboard_tab, "")
        
        # Tab 3: XAI (Explainability)
        self.xai_tab = XAITab()
        self.addTab(self.xai_tab, "")

    def retranslate_ui(self):
        """Updates the tab titles when the application language changes."""
        self.setTabText(0, self.tr("Monitoring / MLOps"))
        self.setTabText(1, self.tr("Explainable AI (XAI)"))