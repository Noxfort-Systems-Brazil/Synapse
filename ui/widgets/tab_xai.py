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
# File: ui/widgets/tab_xai.py
# Author: Gabriel Moraes
# Date: 2026-02-28

from datetime import datetime
from PyQt6.QtCore import Qt, pyqtSlot, pyqtSignal
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QSplitter

from src.domain.entities import DataSource

# Import the extracted UI Components (SOLID - Single Responsibility Principle)
from ui.panels.xai_history_panel import XAIHistoryPanel
from ui.panels.xai_control_panel import XAIControlPanel
from ui.panels.xai_semantic_panel import XAISemanticPanel
from ui.widgets.xai_attribution_chart import XAIAttributionChart

class XAITab(QWidget):
    """
    The 'Explainable AI' Tab (Tab 3) - Orchestrator.
    
    Responsibility (SRP):
    - Acts as a Mediator/Controller between the individual UI panels.
    - Manages the result cache.
    - Routes data from the Main Window to the specific visual components.
    """

    # Signals to request explanations (Caught by MainWindow / MainController)
    buffer_explanation_requested = pyqtSignal()      
    global_explanation_requested = pyqtSignal()      
    local_explanation_requested = pyqtSignal(str)    

    def __init__(self, parent=None):
        super().__init__(parent)
        
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(10, 10, 10, 10)
        
        # Internal State Cache
        self._buffer_count = 0
        self._results_cache = {}
        
        # --- UI Initialization ---
        self._init_ui()
        self._connect_signals()

    def _init_ui(self):
        """Builds the layout using the imported components."""
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self._layout.addWidget(splitter)

        # 1. Left Panel (History & Buffer)
        self.history_panel = XAIHistoryPanel()
        splitter.addWidget(self.history_panel)

        # 2. Right Panel (Controls, Semantics, Math)
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(10, 0, 0, 0)
        
        self.control_panel = XAIControlPanel()
        right_layout.addWidget(self.control_panel)
        
        self.semantic_panel = XAISemanticPanel()
        right_layout.addWidget(self.semantic_panel)
        
        self.chart_panel = XAIAttributionChart()
        right_layout.addWidget(self.chart_panel)
        
        splitter.addWidget(right_container)
        splitter.setSizes([250, 750])

    def _connect_signals(self):
        """Wires the internal components together."""
        # Control Panel -> Tab Signals
        self.control_panel.explain_auditor_requested.connect(self._on_explain_auditor)
        self.control_panel.explain_fuser_requested.connect(self._on_explain_fuser)
        self.control_panel.explain_tcn_requested.connect(self._on_explain_tcn)
        
        # History Panel -> Display Logic
        self.history_panel.event_selected.connect(self._on_history_event_selected)

    # --- Internal Signal Handlers (Routing) ---

    def _on_explain_auditor(self):
        self.control_panel.set_loading_state()
        self.semantic_panel.set_loading("Processing Vetos...")
        self.buffer_explanation_requested.emit()

    def _on_explain_fuser(self):
        self.control_panel.set_loading_state()
        self.semantic_panel.set_loading("Analyzing Global Network State...")
        self.global_explanation_requested.emit()

    def _on_explain_tcn(self, source_id: str):
        self.control_panel.set_loading_state()
        self.semantic_panel.set_loading(f"Analyzing {source_id}...")
        self.local_explanation_requested.emit(source_id)

    def _on_history_event_selected(self, data_id: str):
        if data_id == "buffer_root":
            self.semantic_panel.set_instruction("Pending Vetos waiting for consolidation. Select 'Auditor' and click Analyze.")
            self.chart_panel.clear_data()
        elif data_id in self._results_cache:
            self._display_result(data_id)

    def _display_result(self, req_id: str):
        """Passes cached data to the Semantic and Chart panels."""
        data = self._results_cache.get(req_id)
        if not data: return
        
        target = data.get("target", "Unknown")
        delta = data.get("convergence_delta", 0.0)
        semantic_text = data.get("semantic_text", "")
        attributions = data.get("attributions", [])
        
        self.semantic_panel.display_report(target, delta, semantic_text)
        self.chart_panel.update_data(attributions)
        
        self.control_panel.reset_state()

    # --- Public API (Slots called by MainWindow) ---

    @pyqtSlot(int)
    def set_current_phase(self, phase_id: int):
        self.control_panel.set_current_phase(phase_id)

    @pyqtSlot(DataSource)
    def add_source_item(self, source: DataSource):
        self.control_panel.add_source(source.id, source.name)

    @pyqtSlot(str)
    def remove_source_item(self, source_id: str):
        self.control_panel.remove_source(source_id)

    @pyqtSlot(bool, float, float, list)
    def add_pending_event(self, is_safe: bool, error: float, thresh: float, state_vector: list):
        if is_safe: return
        
        self._buffer_count += 1
        self.history_panel.set_buffer_count(self._buffer_count)
        self.control_panel.set_buffer_state(self._buffer_count)

    @pyqtSlot(dict)
    def add_xai_result(self, result_data: dict):
        target = result_data.get("target", "unknown").upper()
        req_id = result_data.get("request_id", "unknown")
        ts = datetime.now().strftime("%H:%M:%S")
        
        # Clear buffer if Auditor just processed it
        if target == "AUDITOR" and self._buffer_count > 0:
            self._buffer_count = 0
            self.history_panel.clear_buffer()
            self.control_panel.set_buffer_state(0)

        # Cache the result
        self._results_cache[req_id] = result_data
        
        # Add to history UI
        self.history_panel.add_history_record(req_id, target, ts)
        
        # Automatically display the new result
        self._display_result(req_id)