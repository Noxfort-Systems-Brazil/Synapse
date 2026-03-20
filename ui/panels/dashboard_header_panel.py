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
# File: ui/panels/dashboard_header_panel.py
# Author: Gabriel Moraes
# Date: 2026-03-01

from PyQt6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QLabel, QFrame
from ui.styles.theme_manager import ThemeManager

class DashboardHeaderPanel(QWidget):
    """
    Component responsible for the top summary statistics cards on the Dashboard.
    
    Refactored Version (Absolute Clean Architecture):
    - ZERO hardcoded colors, sizes, borders, or CSS strings.
    - Entire stylesheet blocks are requested directly from the ThemeManager.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        
        # References to update values later
        self._val_labels = {}

        self._init_ui()

    def _init_ui(self):
        # We request the raw hex colors without providing hardcoded fallbacks here.
        # The ThemeManager and theme.json are solely responsible for defaults.
        color_success = ThemeManager.get_hex("status_online")
        color_primary = ThemeManager.get_hex("active_sources")
        color_warning = ThemeManager.get_hex("psi_score")
        color_accent  = ThemeManager.get_hex("latency")

        self.card_status = self._create_card(self.tr("System Status"), self.tr("ONLINE"), color_success, "status")
        self.card_sources = self._create_card(self.tr("Active Sources"), "0", color_primary, "sources")
        self.card_psi = self._create_card(self.tr("Avg PSI Score"), "0.00", color_warning, "psi")
        self.card_latency = self._create_card(self.tr("Avg Latency"), self.tr("0 ms"), color_accent, "latency")
        
        self._layout.addWidget(self.card_status)
        self._layout.addWidget(self.card_sources)
        self._layout.addWidget(self.card_psi)
        self._layout.addWidget(self.card_latency)

    def _create_card(self, title: str, initial_value: str, color: str, key: str) -> QFrame:
        """Helper to construct a uniform statistics card strictly using theme styles."""
        card = QFrame()
        
        # We request the complete CSS template for the card frame
        card.setStyleSheet(ThemeManager.get_style("card_frame"))
        
        vbox = QVBoxLayout(card)
        
        lbl_title = QLabel(title)
        # We request the complete CSS template for the card title
        lbl_title.setStyleSheet(ThemeManager.get_style("card_title"))
        
        lbl_val = QLabel(initial_value)
        # We request the CSS template for the value, injecting the dynamic color
        lbl_val.setStyleSheet(ThemeManager.get_style("card_value", color=color))
        
        vbox.addWidget(lbl_title)
        vbox.addWidget(lbl_val)
        
        self._val_labels[key] = lbl_val
        if not hasattr(self, '_title_labels'): self._title_labels = {}
        self._title_labels[key] = lbl_title
        
        return card

    # --- Public API for Updating Stats ---

    def update_system_status(self, status_text: str, color: str = None):
        """Updates the main system status card."""
        if color is None:
            color = ThemeManager.get_hex("status_online")
            
        self._val_labels["status"].setText(status_text)
        self._val_labels["status"].setStyleSheet(ThemeManager.get_style("card_value", color=color))

    def update_active_sources(self, count: int):
        """Updates the active sources count."""
        self._val_labels["sources"].setText(str(count))

    def update_avg_psi(self, psi_value: float):
        """Updates the average statistical drift score."""
        self._val_labels["psi"].setText(f"{psi_value:.2f}")

    def update_avg_latency(self, latency_ms: float):
        """Updates the average processing latency."""
        self._val_labels["latency"].setText(f"{int(latency_ms)} ms")

    def update_theme(self):
        """Forces string-evaluated themes to re-calculate during an active theme change."""
        color_success = ThemeManager.get_hex("status_online")
        color_primary = ThemeManager.get_hex("active_sources")
        color_warning = ThemeManager.get_hex("psi_score")
        color_accent  = ThemeManager.get_hex("latency")
        
        for card in [self.card_status, self.card_sources, self.card_psi, self.card_latency]:
            card.setStyleSheet(ThemeManager.get_style("card_frame"))
            title_lbl = card.findChildren(QLabel)[0]
            title_lbl.setStyleSheet(ThemeManager.get_style("card_title"))
            
        self._val_labels["status"].setStyleSheet(ThemeManager.get_style("card_value", color=color_success))
        self._val_labels["sources"].setStyleSheet(ThemeManager.get_style("card_value", color=color_primary))
        self._val_labels["psi"].setStyleSheet(ThemeManager.get_style("card_value", color=color_warning))
        self._val_labels["latency"].setStyleSheet(ThemeManager.get_style("card_value", color=color_accent))

    def retranslate_ui(self):
        if hasattr(self, '_title_labels'):
            if "status" in self._title_labels: self._title_labels["status"].setText(self.tr("System Status"))
            if "sources" in self._title_labels: self._title_labels["sources"].setText(self.tr("Active Sources"))
            if "psi" in self._title_labels: self._title_labels["psi"].setText(self.tr("Avg PSI Score"))
            if "latency" in self._title_labels: self._title_labels["latency"].setText(self.tr("Avg Latency"))

    def changeEvent(self, event):
        from PyQt6.QtCore import QEvent
        if event.type() == QEvent.Type.LanguageChange:
            self.retranslate_ui()
        super().changeEvent(event)