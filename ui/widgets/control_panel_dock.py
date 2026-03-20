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
# File: ui/widgets/control_panel_dock.py
# Author: Gabriel Moraes
# Date: 2025-11-23

from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QPushButton, QPlainTextEdit,
    QSizePolicy, QScrollBar
)
from enum import Enum, auto
from ui.styles.theme_manager import ThemeManager

class ControlState(Enum):
    """
    Defines the explicit states for the application lifecycle.
    Enforces the order: Optimization -> Offline Bootstrap -> Online Operation.
    """
    IDLE_OPTIMIZATION = auto()    # Phase 0: Ready to optimize hyperparameters
    RUNNING_OPTIMIZATION = auto() # Phase 0: Optimizing...
    IDLE_OFFLINE = auto()         # Phase 1: Ready to train models
    RUNNING_OFFLINE = auto()      # Phase 1: Generating data / Training
    IDLE_ONLINE = auto()          # Phase 2: Ready: Models trained, waiting to start Live Mode
    RUNNING_ONLINE = auto()       # Phase 2: Busy: Real-time inference running

class ControlPanelDock(QDockWidget):
    """
    The Control Panel Dock (Right Side).
    
    Responsibility:
    - specific View for the System State Machine.
    - Updates the main Action Button based on the current ControlState.
    - Displays logs from background services.
    """
    
    # Signal emitted when the main action button is clicked.
    # Carries the *current* state so the Controller knows what transition to trigger.
    action_button_clicked = pyqtSignal(ControlState)

    def __init__(self, parent=None):
        """
        Initializes the Control Panel UI.
        """
        super().__init__(self.tr("Control Panel"), parent)
        self.setObjectName("ControlPanelDock")
        self.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | 
            Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        
        # Initial State (Starts at Optimization now)
        self._current_state = ControlState.IDLE_OPTIMIZATION

        # --- Main Widget Container ---
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(10)

        # --- 1. Main Action Button ---
        # This single button changes function based on context
        self.action_button = QPushButton()
        self.action_button.setObjectName("ActionButton")
        self.action_button.setMinimumHeight(45)
        self.action_button.setCursor(Qt.CursorShape.PointingHandCursor)
        
        # Basic styling (can be moved to qss later)
        # We set a default style here, but _get_btn_style handles colors
        self.action_button.setStyleSheet(self._get_btn_style(ThemeManager.get_hex('latency'))) 
        
        self.action_button.clicked.connect(self._on_action_clicked)
        main_layout.addWidget(self.action_button)

        # --- 2. Log Output Area ---
        self.log_output = QPlainTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setStyleSheet(ThemeManager.get_style("console_log"))
        
        # Fixed height policy for the log to not eat up all space
        log_policy = QSizePolicy(
            QSizePolicy.Policy.Expanding, 
            QSizePolicy.Policy.Fixed
        )
        log_policy.setHeightForWidth(True)
        log_policy.setHeightForWidth(200)
        self.log_output.setSizePolicy(log_policy)
        self.log_output.setMinimumHeight(150)
        
        main_layout.addWidget(self.log_output)
        
        # Spacer to push content up
        main_layout.addStretch()

        self.setWidget(main_widget)
        
        # Apply initial visual state
        self.set_state(ControlState.IDLE_OPTIMIZATION)

    def _on_action_clicked(self):
        """Emits the signal to the Controller (MainWindow)."""
        self.action_button_clicked.emit(self._current_state)

    def retranslate_ui(self):
        """Updates the dock title and button when language changes."""
        self.setWindowTitle(self.tr("Control Panel"))
        # The button text is dynamic based on state, so we re-apply the state
        self.set_state(self._current_state)

    def changeEvent(self, event):
        from PyQt6.QtCore import QEvent
        if event.type() == QEvent.Type.LanguageChange:
            self.retranslate_ui()
        super().changeEvent(event)

    @pyqtSlot(ControlState)
    def set_state(self, new_state: ControlState):
        """
        Updates the UI to reflect the application state.
        This is the visual enforcement of the "Optimization -> Offline -> Online" flow.
        """
        self._current_state = new_state
        self.action_button.setEnabled(True)

        # Phase 0: Optimization
        if new_state == ControlState.IDLE_OPTIMIZATION:
            self.action_button.setText(self.tr("Start Phase 0: Optimizer"))
            self.action_button.setStyleSheet(self._get_btn_style(ThemeManager.get_hex('latency'))) # Purple
            self.action_button.setToolTip(self.tr("Run Optuna to find best hyperparameters."))
            
        elif new_state == ControlState.RUNNING_OPTIMIZATION:
            self.action_button.setText(self.tr("Stop Optimization"))
            self.action_button.setStyleSheet(self._get_btn_style(ThemeManager.get_hex('danger'))) # Red
            self.action_button.setToolTip(self.tr("Abort the optimization process."))

        # Phase 1: Offline Bootstrap
        elif new_state == ControlState.IDLE_OFFLINE:
            self.action_button.setText(self.tr("Start Phase 1: Bootstrap"))
            self.action_button.setStyleSheet(self._get_btn_style(ThemeManager.get_hex('primary'))) # Blue
            self.action_button.setToolTip(self.tr("Generates synthetic data and trains models using optimized params."))
            
        elif new_state == ControlState.RUNNING_OFFLINE:
            self.action_button.setText(self.tr("Stop Bootstrap"))
            self.action_button.setStyleSheet(self._get_btn_style(ThemeManager.get_hex('danger'))) # Red
            self.action_button.setToolTip(self.tr("Abort the training process."))
            
        # Phase 2: Online Operation
        elif new_state == ControlState.IDLE_ONLINE:
            self.action_button.setText(self.tr("Start Phase 2: Live Operation"))
            self.action_button.setStyleSheet(self._get_btn_style(ThemeManager.get_hex('success'))) # Green
            self.action_button.setToolTip(self.tr("Start real-time inference using trained models."))
            
        elif new_state == ControlState.RUNNING_ONLINE:
            self.action_button.setText(self.tr("Stop System"))
            self.action_button.setStyleSheet(self._get_btn_style(ThemeManager.get_hex('danger'))) # Red
            self.action_button.setToolTip(self.tr("Halt real-time processing."))

    def _get_btn_style(self, color_hex: str) -> str:
        """Helper to generate the button stylesheet dynamically."""
        return f"""
            QPushButton {{
                background-color: {color_hex};
                color: white;
                font-weight: bold;
                font-size: 14px;
                border-radius: 4px;
                padding: 5px;
            }}
            QPushButton:hover {{
                background-color: {color_hex}DD; /* Slight transparency on hover */
            }}
            QPushButton:pressed {{
                background-color: {color_hex}AA;
            }}
        """

    def update_theme(self):
        """Re-evaluates hardcoded HEX parameters based on the new active theme."""
        self.set_state(self._current_state)

    @pyqtSlot(str)
    def add_log_message(self, message: str):
        """Appends a log message with auto-scroll."""
        self.log_output.appendPlainText(message)
        # Auto scroll to bottom
        scrollbar = self.log_output.verticalScrollBar()
        if scrollbar:
            scrollbar.setValue(scrollbar.maximum())