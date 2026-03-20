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
# File: ui/widgets/tab_dashboard.py
# Author: Gabriel Moraes
# Date: 2026-02-28

from PyQt6.QtCore import Qt, pyqtSlot, QPointF
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QSplitter

from src.domain.entities import DataSource

# Import the extracted UI Components (SOLID - Single Responsibility Principle)
from ui.panels.dashboard_header_panel import DashboardHeaderPanel
from ui.panels.dashboard_table_panel import DashboardTablePanel
from ui.panels.dashboard_charts_panel import DashboardChartsPanel

class DashboardTab(QWidget):
    """
    The 'Monitoring Dashboard' tab (Tab 2) - Orchestrator.
    
    Responsibility (SRP):
    - Acts as a Mediator/Controller between the individual UI panels.
    - Maintains the internal data buffers for the charts.
    - Routes incoming system signals to the appropriate visual components.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(10, 10, 10, 10)
        self._layout.setSpacing(10)
        
        # Internal State Management
        self._selected_source_id = None
        self._loss_history = {} 
        self._psi_history = {}
        self._x_counters = {} 

        self._init_ui()
        self._connect_signals()

    def _init_ui(self):
        # 1. Header Stats Panel
        self.header_panel = DashboardHeaderPanel()
        self._layout.addWidget(self.header_panel)

        # Main Splitter for Table and Charts
        splitter = QSplitter(Qt.Orientation.Vertical)
        self._layout.addWidget(splitter)

        # 2. Table Panel (Top)
        self.table_panel = DashboardTablePanel()
        splitter.addWidget(self.table_panel)

        # 3. Charts Panel (Bottom)
        self.charts_panel = DashboardChartsPanel()
        splitter.addWidget(self.charts_panel)
        
        # Set initial sizes
        splitter.setSizes([300, 400])

    def _connect_signals(self):
        """Wires the internal components together."""
        self.table_panel.source_selected.connect(self._on_source_selected)

    def _on_source_selected(self, source_id: str):
        """Switches the active charts when a user clicks a different table row."""
        if source_id != self._selected_source_id:
            self._selected_source_id = source_id
            self.charts_panel.update_loss_data(self._loss_history.get(source_id, []))
            self.charts_panel.update_drift_data(self._psi_history.get(source_id, []))

    # --- Public API (Slots called by MainWindow) ---

    @pyqtSlot(DataSource)
    def add_source_row(self, source: DataSource):
        """Registers a new source in the UI and initializes its buffers."""
        if "Historical" in source.name:
            return  # Hide the Historical Base from the MLOps grids

        # Initialize internal buffers
        if source.id not in self._loss_history:
            self._loss_history[source.id] = []
            self._psi_history[source.id] = []
            self._x_counters[source.id] = 0

            # Route to table
            self.table_panel.add_source_row(source)
            
            # Auto-select if it's the first one
            if self._selected_source_id is None:
                self._selected_source_id = source.id

            # Update Header
            active_count = len(self._loss_history)
            self.header_panel.update_active_sources(active_count)

    @pyqtSlot(str)
    def remove_source(self, source_id: str):
        """Wipes a source from the dashboard history and table geometry."""
        # Guard: source may have been removed before any data arrived
        self._loss_history.pop(source_id, None)
        self._psi_history.pop(source_id, None)
        self._x_counters.pop(source_id, None)
        
        # Tell the visual table to drop the row
        self.table_panel.remove_source_row(source_id)
        
        # Reselect top of history if we were viewing the deleted source
        if self._selected_source_id == source_id:
            if self._loss_history:
                first_remaining = list(self._loss_history.keys())[0]
                self._on_source_selected(first_remaining)
            else:
                self._selected_source_id = None
                self.charts_panel.update_loss_data([])
                self.charts_panel.update_drift_data([])
        
        # Update Header
        active_count = len(self._loss_history)
        self.header_panel.update_active_sources(active_count)

    @pyqtSlot(dict)
    def update_realtime_data(self, data: dict):
        """Processes high-frequency kinetic updates."""
        sid = data.get('source_id')
        if not sid or sid not in self._loss_history: 
            return
        
        if 'value' in data:
            self.table_panel.update_value(sid, data['value'])
        
        if 'status' in data:
            # Reconstruct the enum or just pass the string logic
            # Assuming 'status' string maps back to logic in the table panel
            buffer_count = data.get('buffer', 0)
            self.table_panel.update_status(sid, data['status'], buffer_count)
            
        if 'loss' in data:
            loss = float(data['loss'])
            self.table_panel.update_quality_bar(sid, loss)
            
            # Update History Buffer
            self._x_counters[sid] += 1
            x_val = self._x_counters[sid]
            self._loss_history[sid].append(QPointF(float(x_val), loss))
            
            if len(self._loss_history[sid]) > 100: 
                self._loss_history[sid].pop(0)
            
            # Route to charts if active
            if sid == self._selected_source_id:
                self.charts_panel.update_loss_data(self._loss_history[sid])

    @pyqtSlot(str, dict)
    def update_drift_metrics(self, source_id: str, metrics: dict):
        """Processes low-frequency statistical drift updates."""
        if source_id not in self._psi_history:
            return

        psi = float(metrics.get('psi', 0.0))
        sample_count = int(metrics.get('sample_count', 0))
        
        # Fallback to local counter if sample_count is missing or 0
        x_val = float(sample_count) if sample_count > 0 else float(self._x_counters.get(source_id, 0))
             
        self._psi_history[source_id].append(QPointF(x_val, psi))
        
        if len(self._psi_history[source_id]) > 100:
            self._psi_history[source_id].pop(0)
            
        if source_id == self._selected_source_id:
            self.charts_panel.update_drift_data(self._psi_history[source_id])

    @pyqtSlot(str, str, float)
    def update_semantic_info(self, source_id: str, inferred_type: str, confidence: float):
        """Updates the semantic type text in the table."""
        self.table_panel.update_semantic_info(source_id, inferred_type, confidence)

    # Note: Header updates like System Status, Global Avg PSI, and Latency 
    # would be connected here via additional slots if needed.