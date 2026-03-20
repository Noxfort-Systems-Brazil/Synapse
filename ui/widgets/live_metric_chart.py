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
# File: ui/widgets/live_metric_chart.py
# Author: Gabriel Moraes
# Date: 2026-03-01

import math
from PyQt6.QtCore import Qt, QPointF
from PyQt6.QtGui import QColor, QPainter
from PyQt6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis

from ui.styles.theme_manager import ThemeManager

class LiveMetricChart(QChartView):
    """
    Generic reusable QChartView component for plotting live time-series data.
    
    Responsibility (SRP):
    - Encapsulates QChart, QLineSeries, and QValueAxis configuration.
    - Manages dynamic scaling of the X and Y axes based on incoming data.
    - Sanitizes data (filtering out NaN or Infinity values).
    - Uses ThemeManager to enforce centralized typography and sizes.
    """

    def __init__(self, title: str, y_label: str, color_hex: str, parent=None):
        super().__init__(parent)
        
        self._chart = QChart()
        self._chart.setTitle(title)
        
        # Apply centralized fonts
        self._chart.setTitleFont(ThemeManager.get_font("subtitle_size"))
        
        self._chart.setTheme(QChart.ChartTheme.ChartThemeDark)
        self._chart.setBackgroundVisible(False)
        self._chart.layout().setContentsMargins(0, 0, 0, 0)
        self._chart.legend().setVisible(False)
        
        # Series Setup
        self._series = QLineSeries()
        self._series.setName("Live Metric")
        self._series.setPointsVisible(True) 
        self._series.setPointLabelsVisible(False)
        
        pen = self._series.pen()
        # Fetch dynamic line width (fallback to 2 if not defined in theme)
        pen.setWidth(ThemeManager.get_size("chart_line_width", 2))
        pen.setColor(QColor(color_hex))
        self._series.setPen(pen)
        self._chart.addSeries(self._series)
        
        body_font = ThemeManager.get_font("body_size")
        
        # X Axis Setup
        self._axis_x = QValueAxis()
        self._axis_x.setLabelFormat("%d")
        self._axis_x.setTitleText("Samples Processed")
        self._axis_x.setTitleFont(body_font)
        self._axis_x.setLabelsFont(body_font)
        self._axis_x.setRange(0, 100) 
        self._chart.addAxis(self._axis_x, Qt.AlignmentFlag.AlignBottom)
        self._series.attachAxis(self._axis_x)
        
        # Y Axis Setup
        self._axis_y = QValueAxis()
        self._axis_y.setTitleText(y_label)
        self._axis_y.setTitleFont(body_font)
        self._axis_y.setLabelsFont(body_font)
        self._axis_y.setRange(0, 1) 
        self._chart.addAxis(self._axis_y, Qt.AlignmentFlag.AlignLeft)
        self._series.attachAxis(self._axis_y)
        
        self.setChart(self._chart)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)

    def update_series(self, data_points: list[QPointF]):
        """
        Safely updates the chart series and dynamically adjusts axis ranges.
        Filters out any invalid floating point numbers.
        """
        if not data_points: 
            self._series.clear()
            return
        
        # Sanitize and prepare points
        valid_points = []
        ys = []
        for p in data_points:
            if not (math.isnan(p.x()) or math.isnan(p.y()) or math.isinf(p.x()) or math.isinf(p.y())):
                valid_points.append(p)
                ys.append(p.y())
        
        if not valid_points: 
            self._series.clear()
            return

        # Use clear + append for stability in PyQt6 charts
        self._series.clear()
        self._series.append(valid_points)
        
        # Dynamic Scaling
        min_x = valid_points[0].x()
        max_x = valid_points[-1].x()
        
        max_y = max(ys) if ys else 1.0
        min_y = min(ys) if ys else 0.0
        
        # X-Axis Range: keep a trailing window effect
        self._axis_x.setRange(min_x, max(min_x + 10, max_x))
        
        # Y-Axis Range: Intelligent padding
        # If values are tiny (e.g. 0.02), zoom in (max = ~0.03)
        # If values are flat 0.0, show 0 to 0.1
        ceiling = max(max_y * 1.1, 0.1)
        floor = min(min_y, 0.0)
        
        if floor == 0.0 and ceiling > 0.0:
            floor = - (ceiling * 0.05) # Tiny negative margin to clearly see the 0.0 baseline
            
        self._axis_y.setRange(floor, ceiling)
        
        
        # Force a visual update
        self.update()

    def set_title(self, title: str):
        self._chart.setTitle(title)

    def set_y_label(self, label: str):
        self._axis_y.setTitleText(label)