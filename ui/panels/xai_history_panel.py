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
# File: ui/panels/xai_history_panel.py
# Author: Gabriel Moraes
# Date: 2026-02-28

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QListWidget, QListWidgetItem
)
from PyQt6.QtGui import QColor, QBrush
from ui.styles.theme_manager import ThemeManager

class XAIHistoryPanel(QWidget):
    """
    Component responsible for displaying the XAI Request History and Veto Buffer.
    
    Responsibility (SRP):
    - Manages the QListWidget UI and styling.
    - Displays pending buffer counts and past analysis records.
    - Emits a signal when a user selects a specific historical event.
    """
    
    # Emits the 'request_id' of the clicked item, or 'buffer_root'
    event_selected = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        
        self._buffer_item = None
        
        self._init_ui()

    def _init_ui(self):
        self.lbl_list = QLabel(self.tr("XAI Request History"))
        self.lbl_list.setStyleSheet(f"font-weight: bold; color: {ThemeManager.get_hex('text_muted')};")
        self._layout.addWidget(self.lbl_list)
        
        self.event_list = QListWidget()
        self.event_list.setStyleSheet("""
            QListWidget {
                background-color: palette(window);
                border: 1px solid palette(button);
                color: palette(text);
                font-size: 13px;
            }
            QListWidget::item {
                padding: 10px;
            }
            QListWidget::item:selected {
                background-color: palette(highlight);
                color: palette(highlighted-text);
            }
        """)
        self.event_list.itemClicked.connect(self._on_item_clicked)
        self._layout.addWidget(self.event_list)

    def set_buffer_count(self, count: int):
        """Updates or creates the visual buffer item."""
        if count <= 0:
            self.clear_buffer()
            return
            
        label_text = self.tr("📦 VETO BUFFER ({count} pending)").format(count=count)
        
        if self._buffer_item is None:
            self._buffer_item = QListWidgetItem(label_text)
            self._buffer_item.setData(Qt.ItemDataRole.UserRole, "buffer_root")
            self._buffer_item.setForeground(QBrush(ThemeManager.get_color('warning')))
            font = self._buffer_item.font()
            font.setBold(True)
            self._buffer_item.setFont(font)
            self.event_list.insertItem(0, self._buffer_item)
        else:
            self._buffer_item.setText(label_text)

    def clear_buffer(self):
        """Removes the visual buffer item from the list."""
        if self._buffer_item:
            row = self.event_list.row(self._buffer_item)
            self.event_list.takeItem(row)
            self._buffer_item = None

    def add_history_record(self, req_id: str, target: str, timestamp: str):
        """Adds a completed XAI request to the top of the history list."""
        label = self.tr("[{time}] 🟢 {m_target} ANALYSIS").format(time=timestamp, m_target=target)
        item = QListWidgetItem(label)
        item.setData(Qt.ItemDataRole.UserRole, req_id)
        
        # Color coding based on target model
        if target == "AUDITOR": 
            item.setForeground(QBrush(ThemeManager.get_color('danger')))
        elif target == "FUSER": 
            item.setForeground(QBrush(ThemeManager.get_color('primary')))
        else: 
            item.setForeground(QBrush(ThemeManager.get_color('warning')))
            
        self.event_list.insertItem(0, item)
        self.event_list.setCurrentRow(0)

    def _on_item_clicked(self, item: QListWidgetItem):
        """Emits the custom signal containing the selected data ID."""
        data_id = item.data(Qt.ItemDataRole.UserRole)
        if data_id:
            self.event_selected.emit(str(data_id))

    def retranslate_ui(self):
        if hasattr(self, 'lbl_list'):
            self.lbl_list.setText(self.tr("XAI Request History"))

    def changeEvent(self, event):
        from PyQt6.QtCore import QEvent
        if event.type() == QEvent.Type.LanguageChange:
            self.retranslate_ui()
        super().changeEvent(event)