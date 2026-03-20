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
# File: ui/workers/map_loader_thread.py
# Author: Gabriel Moraes
# Date: 2026-03-01

from PyQt6.QtCore import QThread, pyqtSignal
from ui.utilities.sumo_parser import SumoNetworkParser

class MapLoaderThread(QThread):
    """
    Background worker thread that delegates parsing to the SumoNetworkParser service.
    Isolates heavy file I/O and XML parsing from the main UI thread.
    """
    
    # Signals to communicate back to the controller
    data_loaded = pyqtSignal(list, list)  # Emits (nodes, edges)
    error_occurred = pyqtSignal(str)      # Emits error message

    def __init__(self, file_path: str, parent=None):
        super().__init__(parent)
        self.file_path = file_path

    def run(self):
        """Executes the parsing operation in the background."""
        try:
            # Delegate the heavy lifting to the Parser in utilities
            nodes, edges = SumoNetworkParser.parse_file(self.file_path)
            self.data_loaded.emit(nodes, edges)
            
        except Exception as e:
            self.error_occurred.emit(f"Error parsing map data: {str(e)}")