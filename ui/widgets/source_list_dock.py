# SYNAPSE - A Gateway of Intelligent Perception for Traffic Management
# Copyright (C) 2025 Noxfort Systems
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
# File: ui/widgets/source_list_dock.py
# Author: Gabriel Moraes
# Date: 2025-12-24

from PyQt6.QtCore import Qt, pyqtSignal, QPoint, pyqtSlot
from PyQt6.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QPushButton, QListWidget,
    QListWidgetItem, QFrame, QMenu, QMessageBox
)
from enum import Enum, auto

class ButtonPhase(Enum):
    PHASE_DB = auto()
    PHASE_READY = auto()

class SourceListDock(QDockWidget):
    """
    Implements the dock widget for the "Data Sources" list (left panel).
    
    Features:
    - Add Source Button (Enabled only when Map is loaded).
    - List of Sources with Context Menu (Right-Click).
    - Support for Deletion, Origin Toggling, and Re-association.
    
    Updated V2: Added logic to disable 'Add Source' if no map is present.
    """
    
    # --- Core Signals ---
    add_db_clicked = pyqtSignal()
    add_source_clicked = pyqtSignal()
    source_selected = pyqtSignal(str) # Emits source_id
    
    # --- Context Menu Signals ---
    remove_requested = pyqtSignal(str)          # Emits source_id
    toggle_origin_requested = pyqtSignal(str)   # Emits source_id
    reassociate_requested = pyqtSignal(str)     # Emits source_id (Only for Local sources)

    def __init__(self, parent=None):
        """Initializes the Data Sources dock widget."""
        super().__init__(self.tr("Data Sources"), parent)
        self.setObjectName("SourceListDock")
        self.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | 
            Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        
        # --- Main Layout ---
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)

        # 1. Main Action Button
        self.add_button = QPushButton()
        self.add_button.clicked.connect(self._on_main_button_clicked)
        main_layout.addWidget(self.add_button)
        
        # Initial State: Waiting for DB
        self._current_phase = ButtonPhase.PHASE_DB
        self.set_button_phase(ButtonPhase.PHASE_DB)

        # 2. Separator
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        main_layout.addWidget(line)

        # 3. List of Sources
        self.source_list_widget = QListWidget()
        self.source_list_widget.setObjectName("SourceList")
        self.source_list_widget.itemClicked.connect(self._on_item_clicked)
        
        # Enable Right-Click Context Menu
        self.source_list_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.source_list_widget.customContextMenuRequested.connect(self._on_context_menu)
        
        main_layout.addWidget(self.source_list_widget)

        self.setWidget(main_widget)
        
        # NOTE: Mock data (Historical Base, etc.) has been removed.
        # The list starts empty and waits for the Controller to add items.

    def _on_item_clicked(self, item: QListWidgetItem):
        """Internal slot to handle item clicks."""
        source_id = item.data(Qt.ItemDataRole.UserRole)
        if source_id:
            self.source_selected.emit(source_id)

    def _on_context_menu(self, pos: QPoint):
        """Builds and displays the right-click menu."""
        item = self.source_list_widget.itemAt(pos)
        if not item: return
        
        source_id = item.data(Qt.ItemDataRole.UserRole)
        is_local = item.data(Qt.ItemDataRole.UserRole + 1) # Retrieved from storage
        source_name = item.text()
        
        menu = QMenu(self)
        
        # Option 1: Toggle Origin
        origin_text = "Switch to Global" if is_local else "Switch to Local"
        action_origin = menu.addAction(origin_text)
        action_origin.triggered.connect(lambda: self.toggle_origin_requested.emit(source_id))
        
        # Option 2: Reassociate (Only if Local)
        if is_local:
            action_reassoc = menu.addAction("Change Map Location...")
            action_reassoc.triggered.connect(lambda: self.reassociate_requested.emit(source_id))
            
        menu.addSeparator()
        
        # Option 3: Delete
        action_delete = menu.addAction("Delete Source")
        # We use a lambda to capture the ID and Name for the confirmation dialog
        action_delete.triggered.connect(lambda: self._confirm_delete(source_id, source_name))
        
        # Show menu at global position
        menu.exec(self.source_list_widget.mapToGlobal(pos))

    def _confirm_delete(self, source_id: str, name: str):
        """Shows confirmation dialog before emitting remove signal."""
        reply = QMessageBox.question(
            self, 
            'Confirm Deletion',
            f"Are you sure you want to delete the source:\n'{name}'?\n\n"
            "This will disconnect the sensor and stop any active inference.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.remove_requested.emit(source_id)

    def _on_main_button_clicked(self):
        """Routes the button click to the appropriate signal based on phase."""
        if self._current_phase == ButtonPhase.PHASE_DB:
            self.add_db_clicked.emit()
        elif self._current_phase == ButtonPhase.PHASE_READY:
            self.add_source_clicked.emit()

    # --- Public Methods ---
    
    @pyqtSlot(ButtonPhase)
    def set_button_phase(self, new_phase: ButtonPhase):
        """Updates the button's appearance and role based on the application setup phase."""
        self._current_phase = new_phase
        self.retranslate_ui() # Applies text and tooltips for the new phase

    
    def add_source_to_list(self, name: str, source_id: str, is_local: bool):
        """
        Adds a new source to the visual list.
        Stores 'is_local' in UserRole+1 to control menu options.
        """
        item = QListWidgetItem(name)
        item.setData(Qt.ItemDataRole.UserRole, source_id)
        item.setData(Qt.ItemDataRole.UserRole + 1, is_local)
        
        self.source_list_widget.addItem(item)

    def remove_source_by_id(self, source_id: str):
        """Removes the item from the list matching the given ID."""
        for i in range(self.source_list_widget.count()):
            item = self.source_list_widget.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == source_id:
                self.source_list_widget.takeItem(i)
                break

    def clear_sources(self):
        """Clears all sources from the list."""
        self.source_list_widget.clear()

    def retranslate_ui(self):
        self.setWindowTitle(self.tr("Data Sources"))
        if hasattr(self, 'add_button'):
            if self._current_phase == ButtonPhase.PHASE_DB:
                self.add_button.setText(self.tr("+ Add Database"))
                self.add_button.setToolTip(self.tr("Step 1: Import Historical Data."))
                self.add_button.setStyleSheet("font-weight: bold; color: #FFA500;")
            elif self._current_phase == ButtonPhase.PHASE_READY:
                self.add_button.setText(self.tr("+ Add Data Source"))
                self.add_button.setToolTip(self.tr("Add a new sensor or API."))
                self.add_button.setStyleSheet("") # default styling

    def changeEvent(self, event):
        from PyQt6.QtCore import QEvent
        if event.type() == QEvent.Type.LanguageChange:
            self.retranslate_ui()
        super().changeEvent(event)