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
# File: ui/panels/xai_semantic_panel.py
# Author: Gabriel Moraes
# Date: 2026-02-28

from PyQt6.QtWidgets import QGroupBox, QVBoxLayout, QTextEdit
from ui.styles.theme_manager import ThemeManager

class XAISemanticPanel(QGroupBox):
    """
    Component responsible for displaying the Jurist Agent's semantic interpretation.
    
    Responsibility (SRP):
    - Manages the QTextEdit UI, styling, and HTML formatting.
    - Exposes a clean API to update the text or show loading states.
    """

    def __init__(self, parent=None):
        super().__init__(self.tr("Semantic Interpretation (Jurist Agent)"), parent)
        self.setStyleSheet(ThemeManager.get_style('group_box_title'))
        
        self._layout = QVBoxLayout(self)
        
        self.text_explanation = QTextEdit()
        self.text_explanation.setReadOnly(True)
        self.text_explanation.setPlaceholderText(self.tr("Select a model and click Explain to generate a report..."))
        self.text_explanation.setStyleSheet("""
            QTextEdit {
                background-color: palette(window);
                color: palette(text);
                font-family: 'Segoe UI', sans-serif;
                font-size: 14px;
                border: none;
            }
        """)
        self._layout.addWidget(self.text_explanation)

    def set_loading(self, message: str):
        """Displays a loading message while waiting for the Jurist Agent."""
        self.text_explanation.setHtml(self.tr("<b>⏳ {message}</b><br><i>Waiting for Jurist Agent...</i>").format(message=message))

    def set_instruction(self, message: str):
        """Displays an instructional or error message."""
        self.text_explanation.setHtml(f"<i>{message}</i>")

    def display_report(self, target: str, convergence_delta: float, semantic_text: str):
        """Formats and displays the final HTML report."""
        color = ThemeManager.get_hex("status_online")
        if target.lower() == "auditor": 
            color = ThemeManager.get_hex("danger")
            
        self.text_explanation.setHtml(f"""
            <h3 style='color: {color};'>{target.upper()} Report</h3>
            <p><b>Math Convergence:</b> {convergence_delta:.6f}</p>
            <hr>
            <div style='font-size: 14px; line-height: 1.4;'>{semantic_text}</div>
        """)

    def retranslate_ui(self):
        self.setTitle(self.tr("Semantic Interpretation (Jurist Agent)"))
        if hasattr(self, 'text_explanation'):
            # Only update placeholder if the text is empty, to not erase existing reports
            if self.text_explanation.toPlainText() == "":
                self.text_explanation.setPlaceholderText(self.tr("Select a model and click Explain to generate a report..."))

    def changeEvent(self, event):
        from PyQt6.QtCore import QEvent
        if event.type() == QEvent.Type.LanguageChange:
            self.retranslate_ui()
        super().changeEvent(event)