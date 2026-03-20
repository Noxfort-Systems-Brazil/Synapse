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
# File: ui/handlers/signal_router.py
# Author: Gabriel Moraes
# Date: 2026-03-01

from PyQt6.QtCore import QObject, pyqtSlot
from PyQt6.QtWidgets import QMessageBox
from PyQt6.QtGui import QAction

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.main_controller import MainController
    from ui.main_window import MainWindow

from ui.widgets.control_panel_dock import ControlState
from ui.widgets.source_list_dock import ButtonPhase

class SignalRouter(QObject):
    """
    Acts as a Mediator to route signals between the UI components and the business logic.
    Adheres to the Single Responsibility Principle by decoupling the MainWindow from event handling.
    
    Refactored V2.0 (SOLID Compliance):
    - Theme logic delegated to ThemeManager.apply_theme()
    - Log level logic delegated to logging_setup.set_global_level()
    - Source origin toggle delegated to AppState.toggle_source_origin()
    - Law of Demeter: uses MainWindow.log() instead of deep chains.
    """
    def __init__(self, main_window: 'MainWindow', controller: 'MainController'):
        super().__init__(main_window)
        self.main_window = main_window
        self.controller = controller
        self.app_state = controller.app_state

    def setup_routing(self):
        """Initializes all signal connections."""
        self._bind_handler_signals()
        self._bind_controller_signals()
        self._bind_model_signals()
        self._bind_ui_actions()

    def _bind_handler_signals(self):
        """Connects signals emitted by the DialogHandler to the UI."""
        dh = self.main_window.dialog_handler
        control_dock = self.main_window.dock_manager.control_dock
        source_dock = self.main_window.dock_manager.source_dock
        
        dh.log_requested.connect(control_dock.add_log_message)
        dh.status_requested.connect(self.main_window.status_bar.showMessage)
        dh.language_changed.connect(self._on_language_changed)
        dh.theme_changed.connect(self._on_theme_changed)
        dh.log_level_changed.connect(self._on_log_level_changed)
        

        
        # Advance button wizard when Historical Base is imported
        dh.source_added.connect(self._on_historical_base_loaded)


    def _bind_controller_signals(self):
        """Connects backend Controller events to UI updates."""
        c = self.controller
        mw = self.main_window
        cdock = mw.dock_manager.control_dock
        tabs = mw.tabs
        
        c.log_message.connect(cdock.add_log_message)
        c.status_message.connect(mw.status_bar.showMessage)
        c.error_occurred.connect(lambda msg: QMessageBox.critical(mw, mw.tr("Error"), msg))
        c.progress_update.connect(lambda val: mw.status_bar.showMessage(f"{mw.tr('Progress:')} {val}%"))

        # Lifecycle Transitions
        c.optimization_finished.connect(lambda: cdock.set_state(ControlState.IDLE_OFFLINE))
        c.bootstrap_finished.connect(lambda: cdock.set_state(ControlState.IDLE_ONLINE))
        c.online_system_started.connect(lambda: cdock.set_state(ControlState.RUNNING_ONLINE))
        c.online_system_stopped.connect(lambda: cdock.set_state(ControlState.IDLE_ONLINE))
        
        # Real-time Data Streams
        c.engine_data_processed.connect(tabs.dashboard_tab.update_realtime_data)
        c.linguist_update.connect(tabs.dashboard_tab.update_semantic_info)
        c.engine_global_results.connect(self._on_engine_global_results)
        c.linguist_update.connect(self._on_linguist_log)
        c.drift_update.connect(tabs.dashboard_tab.update_drift_metrics)
        
        # XAI Wiring
        tabs.xai_tab.buffer_explanation_requested.connect(c.request_buffer_explanation)
        tabs.xai_tab.local_explanation_requested.connect(c.request_local_explanation)
        tabs.xai_tab.global_explanation_requested.connect(c.request_global_explanation)
        
        c.xai_result_received.connect(tabs.xai_tab.add_xai_result)
        c.audit_update.connect(tabs.xai_tab.add_pending_event)
        c.audit_update.connect(self._on_audit_result_relay)

    def _bind_model_signals(self):
        """Connects application AppState changes to UI reflections."""
        tabs = self.main_window.tabs
        source_dock = self.main_window.dock_manager.source_dock
        
        self.app_state.data_association_changed.connect(self._on_association_confirmed)
        
        self.app_state.data_source_added.connect(tabs.dashboard_tab.add_source_row)
        self.app_state.data_source_added.connect(tabs.xai_tab.add_source_item)
        self.app_state.data_source_removed.connect(tabs.dashboard_tab.remove_source)
        self.app_state.data_source_removed.connect(tabs.xai_tab.remove_source_item)
        self.app_state.data_source_removed.connect(source_dock.remove_source_by_id)
        
        # Populate source_dock for restored (and newly added) sources
        def _handle_model_source_add(src):
            if "Historical" not in src.name:
                source_dock.add_source_to_list(src.name, src.id, src.is_local)
        
        self.app_state.data_source_added.connect(_handle_model_source_add)
        
        # React to origin toggle from the domain layer
        self.app_state.source_origin_toggled.connect(self._on_source_origin_toggled)

    def _bind_ui_actions(self):
        """Connects UI-triggered actions back to the Controller, DialogHandler, or state mutations."""
        source_dock = self.main_window.dock_manager.source_dock
        control_dock = self.main_window.dock_manager.control_dock
        dh = self.main_window.dialog_handler
        menu = self.main_window.main_menu

        # --- Menu Actions Connection ---
        menu.import_db_act.triggered.connect(dh.open_import_wizard)
        menu.settings_act.triggered.connect(dh.open_settings)
        menu.exit_act.triggered.connect(self.main_window.close)

        # --- Dock Actions Connection ---
        source_dock.add_db_clicked.connect(dh.open_import_wizard)
        source_dock.add_source_clicked.connect(
            lambda: dh.open_add_source_dialog(len(source_dock.findChildren(QAction)))
        )
        
        source_dock.source_selected.connect(self._on_source_selected)
        source_dock.remove_requested.connect(self._on_remove_source_requested)
        source_dock.toggle_origin_requested.connect(self._on_toggle_origin_requested)
        source_dock.reassociate_requested.connect(self._on_reassociate_requested)
        
        control_dock.action_button_clicked.connect(self._handle_control_action)

    # --- Event Handlers (Slots) ---

    @pyqtSlot(object)
    def _on_historical_base_loaded(self, src):
        """Intersects the data source loading to see if it's the offline DB, advancing the wizard."""
        if src.name == "Historical Base (Imported)":
            dock = self.main_window.dock_manager.source_dock
            dock.set_button_phase(ButtonPhase.PHASE_READY)

    @pyqtSlot(str)
    def _on_language_changed(self, lang_code: str):
        if self.main_window.translation_manager.load_language(lang_code):
            self.main_window.current_language = lang_code

    @pyqtSlot(str)
    def _on_theme_changed(self, theme_index_str: str):
        """Delegates theme switching entirely to ThemeManager."""
        from ui.styles.theme_manager import ThemeManager
        
        try:
            theme_index = int(theme_index_str)
        except ValueError:
            theme_index = 0
        
        ThemeManager.apply_theme(theme_index)
        
        theme_map = {0: "dark", 1: "light", 2: "dark"}
        theme_name = theme_map.get(theme_index, "dark")
        self.main_window.log(f"[System] Theme dynamically mapped to: {theme_name.upper()} Mode")

    @pyqtSlot(str)
    def _on_log_level_changed(self, level_str: str):
        """Delegates log level switching to logging_setup."""
        from src.utils.logging_setup import set_global_level
        set_global_level(level_str)
        self.main_window.log(f"[System] Global Log Level set to {level_str}")

    @pyqtSlot(dict)
    def _on_engine_global_results(self, results):
        log_parts = []
        if "gat_output_shape" in results:
            log_parts.append(f"GATv2: Reg {results['gat_output_shape']}")
        if "fuser_forecast_shape" in results:
            log_parts.append(f"iTransformer: OK")
        if log_parts:
            self.main_window.log(f"[GLOBAL] {' | '.join(log_parts)}")

    @pyqtSlot(bool, float, float, list)
    def _on_audit_result_relay(self, is_safe, error, thresh, vec):
        verdict = "SAFE" if is_safe else "VETO"
        self.main_window.log(f"[AUDITOR] Verdict: {verdict} (Err: {error:.4f})")

    @pyqtSlot(str, str, float)
    def _on_linguist_log(self, source_name, type_inf, conf):
        self.main_window.log(f"[LINGUIST] '{source_name}' -> {type_inf} ({conf*100:.1f}%)")

    @pyqtSlot(str)
    def _on_remove_source_requested(self, source_id):
        self.app_state.remove_data_source(source_id)
        self.main_window.log(f"[Manager] Removed source: {source_id}")

    @pyqtSlot(str)
    def _on_reassociate_requested(self, source_id):
        self.app_state.enter_association_mode(source_id)
        self.main_window.log(f"[Manager] Select a node on map for: {source_id}")

    @pyqtSlot(str)
    def _on_toggle_origin_requested(self, source_id):
        """Delegates state mutation to AppState (SRP)."""
        self.app_state.toggle_source_origin(source_id)

    @pyqtSlot(str, bool)
    def _on_source_origin_toggled(self, source_id: str, new_is_local: bool):
        """Reacts to AppState signal by updating the UI."""
        source = self.app_state.get_data_source(source_id)
        if not source:
            return
        
        new_type = "Local" if new_is_local else "Global"
        dock = self.main_window.dock_manager.source_dock
        dock.remove_source_by_id(source_id)
        dock.add_source_to_list(source.name, source.id, source.is_local)
        self.main_window.log(f"[Manager] Source '{source.name}' switched to {new_type}.")

    @pyqtSlot(str, str)
    def _on_association_confirmed(self, sid, eid):
        self.main_window.log(f"Linked {sid} -> {eid}")

    @pyqtSlot(str)
    def _on_source_selected(self, sid):
        self.main_window.status_bar.showMessage(f"{self.main_window.tr('Selected:')} {sid}")

    @pyqtSlot(ControlState)
    def _handle_control_action(self, current_state: ControlState):
        actions = {
            ControlState.IDLE_OPTIMIZATION: self.controller.start_optimization,
            ControlState.RUNNING_OPTIMIZATION: self.controller.stop_optimization,
            ControlState.IDLE_OFFLINE: self.controller.start_offline_bootstrap,
            ControlState.RUNNING_OFFLINE: self.controller.stop_offline_bootstrap,
            ControlState.IDLE_ONLINE: self.controller.start_online_operation,
            ControlState.RUNNING_ONLINE: self.controller.stop_online_operation
        }
        action = actions.get(current_state)
        if action:
            action()