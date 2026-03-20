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
# File: ui/panels/xai_control_panel.py
# Author: Gabriel Moraes
# Date: 2026-02-28

from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QComboBox, QPushButton
)
from ui.styles.theme_manager import ThemeManager

class XAIControlPanel(QFrame):
    """
    Component responsible for the XAI target selection and action triggers.
    
    Responsibility (SRP):
    - Manages the ComboBoxes for Model and Source selection.
    - Handles the 'Explain Now' button state and phase locks.
    - Emits specific signals when an explanation is requested.
    """
    
    explain_auditor_requested = pyqtSignal()
    explain_fuser_requested = pyqtSignal()
    explain_tcn_requested = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background-color: palette(window); border-radius: 5px; padding: 5px; border: 1px solid palette(button);")
        
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(5, 5, 5, 5)

        self._current_phase = 0
        self._buffer_count = 0

        self._init_ui()

    def _init_ui(self):
        self.lbl_target = QLabel(self.tr("Target Model:"))
        self.lbl_target.setStyleSheet(f"font-weight: bold; color: {ThemeManager.get_hex('text_main')};")
        self._layout.addWidget(self.lbl_target)
        
        self.combo_model = QComboBox()
        self.combo_model.addItems([
            self.tr("Auditor (Safety Vetos)"), 
            self.tr("Global Fuser (Network State)"),
            self.tr("Local Specialist (TCN)")
        ])
        self.combo_model.currentIndexChanged.connect(self._on_model_changed)
        self._layout.addWidget(self.combo_model)
        
        # Source Selector (Only visible for TCN)
        self.combo_source_id = QComboBox()
        self.combo_source_id.setPlaceholderText(self.tr("Select Sensor..."))
        self.combo_source_id.setVisible(False)
        self.combo_source_id.setFixedWidth(200)
        self.combo_source_id.setStyleSheet(ThemeManager.get_style("combo_box"))
        self._layout.addWidget(self.combo_source_id)
        
        self._layout.addStretch()
        
        self.btn_explain = QPushButton(self.tr("⚡ Explain Now"))
        self.btn_explain.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_explain.setFixedSize(160, 35)
        self.btn_explain.setStyleSheet(f"""
            QPushButton {{
                background-color: {ThemeManager.get_hex('latency')}; 
                color: white; 
                font-weight: bold; 
                border-radius: 4px;
            }}
            QPushButton:hover {{
                background-color: {ThemeManager.get_hex('primary')};
            }}
            QPushButton:disabled {{
                background-color: {ThemeManager.get_hex('card_border')};
                color: {ThemeManager.get_hex('text_muted')};
            }}
        """)
        self.btn_explain.clicked.connect(self._on_click_explain)
        self._layout.addWidget(self.btn_explain)
        
        # Initial State Update
        self._on_model_changed(0)

    def set_current_phase(self, phase_id: int):
        """Updates the internal phase state and refreshes UI if needed."""
        self._current_phase = phase_id
        if self.combo_model.currentIndex() == 1:
            self._on_model_changed(1)

    def set_buffer_state(self, count: int):
        """Updates the known buffer count to adjust the button text."""
        self._buffer_count = count
        if self.combo_model.currentIndex() == 0:
            self._on_model_changed(0)

    def add_source(self, source_id: str, source_name: str):
        """Adds a new source option to the selector."""
        if "Historical" in source_name:
            return
            
        self.combo_source_id.addItem(source_name, source_id)
        if self.combo_model.currentIndex() == 2:
            self._on_model_changed(2)

    def remove_source(self, source_id: str):
        """Removes a source from the selector by ID."""
        for i in range(self.combo_source_id.count()):
            if self.combo_source_id.itemData(i) == source_id:
                self.combo_source_id.removeItem(i)
                break
        if self.combo_model.currentIndex() == 2:
            self._on_model_changed(2)

    def set_loading_state(self):
        """Disables the button while an explanation is being generated."""
        self.btn_explain.setEnabled(False)

    def reset_state(self):
        """Re-evaluates the button state after an action is completed."""
        self._on_model_changed(self.combo_model.currentIndex())

    def _on_model_changed(self, index: int):
        """Updates UI based on selected model."""
        is_tcn = (index == 2)
        self.combo_source_id.setVisible(is_tcn)
        
        if index == 0: # Auditor
            self.btn_explain.setText(self.tr("Analyze Buffer ({count})").format(count=self._buffer_count))
            self.btn_explain.setEnabled(self._buffer_count > 0)
            self.btn_explain.setToolTip("")
            
        elif index == 1: # Fuser
            self.btn_explain.setText(self.tr("Explain Forecast"))
            if self._current_phase >= 2:
                self.btn_explain.setEnabled(True)
                self.btn_explain.setToolTip("")
            else:
                self.btn_explain.setEnabled(False)
                self.btn_explain.setToolTip(self.tr("Global Fuser explanation is only available in Phase 2 (Runtime)."))
                
        else: # TCN
            self.btn_explain.setText(self.tr("Explain Source"))
            self.btn_explain.setEnabled(self.combo_source_id.count() > 0)
            self.btn_explain.setToolTip("")

    def _on_click_explain(self):
        """Routes the click to the correct signal based on selection."""
        idx = self.combo_model.currentIndex()
        
        if idx == 0:
            if self._buffer_count > 0:
                self.explain_auditor_requested.emit()
                
        elif idx == 1:
            self.explain_fuser_requested.emit()
            
        elif idx == 2:
            sid = self.combo_source_id.currentData()
            if sid:
                self.explain_tcn_requested.emit(str(sid))

    def retranslate_ui(self):
        if hasattr(self, 'lbl_target'):
            self.lbl_target.setText(self.tr("Target Model:"))
        if hasattr(self, 'combo_model'):
            self.combo_model.setItemText(0, self.tr("Auditor (Safety Vetos)"))
            self.combo_model.setItemText(1, self.tr("Global Fuser (Network State)"))
            self.combo_model.setItemText(2, self.tr("Local Specialist (TCN)"))
        self._on_model_changed(self.combo_model.currentIndex())
        if hasattr(self, 'combo_source_id'):
            self.combo_source_id.setPlaceholderText(self.tr("Select Sensor..."))

    def changeEvent(self, event):
        from PyQt6.QtCore import QEvent
        if event.type() == QEvent.Type.LanguageChange:
            self.retranslate_ui()
        super().changeEvent(event)