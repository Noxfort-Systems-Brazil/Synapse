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
# File: ui/panels/dashboard_charts_panel.py
# Author: Gabriel Moraes
# Date: 2026-03-01

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel
from PyQt6.QtCore import QPointF

# Import our generic chart component
from ui.widgets.live_metric_chart import LiveMetricChart
from ui.styles.theme_manager import ThemeManager

class DashboardChartsPanel(QWidget):
    """
    Component responsible for the MLOps Observability charts section.
    
    Responsibility (SRP):
    - Instantiates and arranges the specific LiveMetricChart instances (Loss and Drift).
    - Acts as a local router for updating these specific charts.
    - Uses ThemeManager to enforce centralized styling and remove hardcoded CSS.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        
        self._init_ui()

    def _init_ui(self):
        self.lbl_charts = QLabel(self.tr("MLOps Observability (Anomaly & Drift)"))
        
        # Request the section title style from the ThemeManager
        self.lbl_charts.setStyleSheet(ThemeManager.get_style("section_title", "font-weight: bold;"))
        self._layout.addWidget(self.lbl_charts)
        
        # Chart Views Layout
        charts_layout = QHBoxLayout()
        
        self.chart_loss = LiveMetricChart(
            title=self.tr("Reconstruction Error (Anomaly Score)"), 
            y_label=self.tr("MSE Loss"), 
            color_hex=ThemeManager.get_hex("danger")
        )
        
        self.chart_drift = LiveMetricChart(
            title=self.tr("Statistical Drift (PSI)"), 
            y_label=self.tr("PSI Value"), 
            color_hex=ThemeManager.get_hex("primary")
        )
        
        charts_layout.addWidget(self.chart_loss)
        charts_layout.addWidget(self.chart_drift)
        
        self._layout.addLayout(charts_layout)

    def update_loss_data(self, data_points: list[QPointF]):
        """Routes the data to the Anomaly (Loss) chart."""
        self.chart_loss.update_series(data_points)

    def update_drift_data(self, data_points: list[QPointF]):
        """Routes the data to the Statistical Drift (PSI) chart."""
        self.chart_drift.update_series(data_points)

    def retranslate_ui(self):
        if hasattr(self, 'lbl_charts'):
            self.lbl_charts.setText(self.tr("MLOps Observability (Anomaly & Drift)"))
        if hasattr(self, 'chart_loss'):
            self.chart_loss.set_title(self.tr("Reconstruction Error (Anomaly Score)"))
            self.chart_loss.set_y_label(self.tr("MSE Loss"))
        if hasattr(self, 'chart_drift'):
            self.chart_drift.set_title(self.tr("Statistical Drift (PSI)"))
            self.chart_drift.set_y_label(self.tr("PSI Value"))

    def changeEvent(self, event):
        from PyQt6.QtCore import QEvent
        if event.type() == QEvent.Type.LanguageChange:
            self.retranslate_ui()
        super().changeEvent(event)