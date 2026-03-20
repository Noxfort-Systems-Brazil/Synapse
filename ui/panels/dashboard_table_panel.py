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
# File: ui/panels/dashboard_table_panel.py
# Author: Gabriel Moraes
# Date: 2026-03-01

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QTableWidget, 
    QTableWidgetItem, QHeaderView, QProgressBar
)
from PyQt6.QtGui import QColor, QBrush

from src.domain.entities import DataSource, SourceStatus
from ui.styles.theme_manager import ThemeManager

class DashboardTablePanel(QWidget):
    """
    Component responsible for the Data Sources Health Monitor table.
    
    Responsibility (SRP):
    - Manages the QTableWidget initialization, styling, and formatting.
    - Updates individual cells (Value, Status, Semantic Type, Quality Bar) cleanly.
    - Emits a signal when the user selects a source to view its specific charts.
    - Uses ThemeManager to enforce centralized styling and remove hardcoded CSS/colors.
    """

    # Signal emitted when a table row is clicked, passing the source_id
    source_selected = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        
        # Map Source ID -> Table Row Index for O(1) lookups
        self._row_map = {}
        
        self._init_ui()

    def _init_ui(self):
        self.lbl_table = QLabel(self.tr("Data Sources Health (Zero Trust Monitor)"))
        
        # Request the section title style from the ThemeManager
        self.lbl_table.setStyleSheet(ThemeManager.get_style("section_title"))
        self._layout.addWidget(self.lbl_table)
        
        self.table = QTableWidget()
        self._setup_table()
        self._layout.addWidget(self.table)

    def _setup_table(self):
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels([
            self.tr("Source Name"), 
            self.tr("Semantic Type"), 
            self.tr("Status"), 
            self.tr("Value"), 
            self.tr("Quality")
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.itemClicked.connect(self._on_table_row_clicked)

    def add_source_row(self, source: DataSource):
        """Adds a new row to the table for a given data source."""
        if source.id in self._row_map: 
            return

        row_idx = self.table.rowCount()
        self.table.insertRow(row_idx)
        self._row_map[source.id] = row_idx
        
        # Column 0: Name (and store ID internally)
        name_item = QTableWidgetItem(source.name)
        name_item.setData(Qt.ItemDataRole.UserRole, source.id)
        self.table.setItem(row_idx, 0, name_item)
        
        # Column 1: Semantic
        self.table.setItem(row_idx, 1, QTableWidgetItem(self.tr("Scanning...")))
        
        # Column 2: Status
        self.update_status(source.id, source.status)
        
        # Column 3: Value
        self.table.setItem(row_idx, 3, QTableWidgetItem("--"))
        
        # Column 4: Quality (Progress Bar)
        self.update_quality_bar(source.id, 0.0)

    def remove_source_row(self, source_id: str):
        """Removes a source row from the table by ID."""
        if source_id not in self._row_map:
            return
        
        row_idx = self._row_map.pop(source_id)
        self.table.removeRow(row_idx)
        
        # Rebuild row map — indices shift after removal
        self._row_map = {
            sid: (r if r < row_idx else r - 1)
            for sid, r in self._row_map.items()
        }

    def update_value(self, source_id: str, value: float):
        if source_id in self._row_map:
            row = self._row_map[source_id]
            self.table.item(row, 3).setText(f"{value:.2f}")

    def update_status(self, source_id: str, status: SourceStatus, buffer_count: int = 0):
        if source_id not in self._row_map: return
        
        row = self._row_map[source_id]
        text = status.value
        if buffer_count > 0:
            text += f" ({buffer_count})"

        # Determine color dynamically via ThemeManager
        color_hex = ThemeManager.get_hex("text_muted")
        
        if status == SourceStatus.ACTIVE: 
            color_hex = ThemeManager.get_hex("status_online")
        elif status == SourceStatus.QUARANTINE: 
            color_hex = ThemeManager.get_hex("warning") # using warning for quarantine
        elif status == SourceStatus.VALIDATING: 
            color_hex = ThemeManager.get_hex("primary")
        elif status == SourceStatus.REJECTED: 
            color_hex = ThemeManager.get_hex("danger")
            
        item = QTableWidgetItem(text)
        item.setForeground(QBrush(QColor(color_hex)))
        self.table.setItem(row, 2, item)

    def update_quality_bar(self, source_id: str, loss_val: float):
        if source_id not in self._row_map: return
        
        row = self._row_map[source_id]
        
        bar = QProgressBar()
        bar.setRange(0, 100)
        
        # Convert error (loss) to a quality percentage
        quality_pct = max(0, min(100, int((1.0 - (loss_val * 10)) * 100)))
        bar.setValue(quality_pct)
        bar.setFormat(self.tr("Err: {0:.4f}").format(loss_val))
        
        # Determine bar color dynamically
        if loss_val < 0.02: 
            bar_color = ThemeManager.get_hex("status_online")
        elif loss_val < 0.05: 
            bar_color = ThemeManager.get_hex("warning")
        else: 
            bar_color = ThemeManager.get_hex("danger")
            
        bar.setStyleSheet(f"QProgressBar::chunk {{ background-color: {bar_color}; }}")
        self.table.setCellWidget(row, 4, bar)

    def update_semantic_info(self, source_id: str, inferred_type: str, confidence: float):
        if source_id in self._row_map:
            row = self._row_map[source_id]
            txt = f"{inferred_type} ({int(confidence*100)}%)"
            self.table.item(row, 1).setText(txt)
            
            # Semantic validation usually means the source is active
            item_status = self.table.item(row, 2)
            if item_status and "ACTIVE" in item_status.text():
                 color_hex = ThemeManager.get_hex("status_online")
                 item_status.setForeground(QBrush(QColor(color_hex)))

    def _on_table_row_clicked(self, item: QTableWidgetItem):
        """Emits the ID of the clicked source."""
        row = item.row()
        name_item = self.table.item(row, 0)
        source_id = name_item.data(Qt.ItemDataRole.UserRole)
        
        if source_id:
            self.source_selected.emit(str(source_id))

    def get_first_source_id(self) -> str:
        """Helper to get the first source ID, useful for initial selection."""
        if self._row_map:
            # Return the first key in the map
            return next(iter(self._row_map))
        return None

    def retranslate_ui(self):
        if hasattr(self, 'lbl_table'):
            self.lbl_table.setText(self.tr("Data Sources Health (Zero Trust Monitor)"))
        
        self.table.setHorizontalHeaderLabels([
            self.tr("Source Name"), 
            self.tr("Semantic Type"), 
            self.tr("Status"), 
            self.tr("Value"), 
            self.tr("Quality")
        ])

    def changeEvent(self, event):
        from PyQt6.QtCore import QEvent
        if event.type() == QEvent.Type.LanguageChange:
            self.retranslate_ui()
        super().changeEvent(event)