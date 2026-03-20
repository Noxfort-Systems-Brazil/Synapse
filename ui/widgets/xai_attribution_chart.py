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
# File: ui/widgets/xai_attribution_chart.py
# Author: Gabriel Moraes
# Date: 2026-02-28

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPainter
from PyQt6.QtCharts import (
    QChart, QChartView, QBarSeries, QBarSet, 
    QBarCategoryAxis, QValueAxis
)
from ui.styles.theme_manager import ThemeManager

class XAIAttributionChart(QChartView):
    """
    Dedicated widget for rendering Feature Attribution (Integrated Gradients).
    
    Responsibility (SRP):
    - Encapsulates all QChart logic, themes, axes, and series management.
    - Exposes a clean API to update or clear the mathematical chart.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        
        self._chart = QChart()
        self._chart.setTheme(QChart.ChartTheme.ChartThemeDark)
        self._chart.setBackgroundVisible(False)
        self._chart.legend().setVisible(True)
        self._chart.legend().setAlignment(Qt.AlignmentFlag.AlignBottom)
        
        # Initial empty axes to maintain layout structure
        self._axis_x = QBarCategoryAxis()
        self._chart.addAxis(self._axis_x, Qt.AlignmentFlag.AlignBottom)
        
        self._axis_y = QValueAxis()
        self._axis_y.setTitleText("Impact Score")
        self._chart.addAxis(self._axis_y, Qt.AlignmentFlag.AlignLeft)
        
        self.setChart(self._chart)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)

    def clear_data(self):
        """Removes all series and resets the chart to a blank state."""
        self._chart.removeAllSeries()
        for axis in self._chart.axes():
            self._chart.removeAxis(axis)

    def update_data(self, attributions: list):
        """
        Builds the bar chart based on the provided attribution values.
        Red-ish colors mean positive impact, Blue-ish mean negative impact.
        """
        self.clear_data()
        
        if not attributions:
            return

        bar_set = QBarSet("Impact")
        categories = []
        
        # Limit to top 20 features for UI clarity
        display_limit = min(len(attributions), 20)
        
        for i in range(display_limit):
            val = attributions[i]
            bar_set.append(val)
            categories.append(f"F{i}")
            
            # Color coding based on attribution polarity and magnitude
            if abs(val) > 0.05: 
                if val > 0: 
                    bar_set.setColor(ThemeManager.get_color("danger")) # Red (Positive impact)
                else: 
                    bar_set.setColor(ThemeManager.get_color("primary")) # Blue (Negative impact)

        series = QBarSeries()
        series.append(bar_set)
        self._chart.addSeries(series)
        
        # Recreate and attach X Axis
        self._axis_x = QBarCategoryAxis()
        self._axis_x.append(categories)
        self._chart.addAxis(self._axis_x, Qt.AlignmentFlag.AlignBottom)
        
        # Recreate and attach Y Axis
        self._axis_y = QValueAxis()
        self._axis_y.setTitleText("Attribution")
        self._chart.addAxis(self._axis_y, Qt.AlignmentFlag.AlignLeft)
        
        series.attachAxis(self._axis_x)
        series.attachAxis(self._axis_y)