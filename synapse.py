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
# File: synapse.py
# Author: Gabriel Moraes
# Date: 2026-02-28

import sys
print("[BOOT] Loading SYNAPSE modules...")

try:
    import click
    from PyQt6.QtWidgets import QApplication
    
    # Import the Orchestrator (Backend)
    from src.main_controller import MainController
    
    # Import the View (Frontend)
    from ui.main_window import MainWindow
    
except ImportError as e:
    print(f"[CRITICAL] Failed to import core modules: {e}")
    sys.exit(1)

def start_ui():
    """
    Launches the SYNAPSE Graphical User Interface (Qt).
    This entry point wires the Model-View-Controller architecture.
    """
    print("[Launcher] Initializing PyQt6 Application...")
    
    try:
        # 1. Initialize the Qt Application
        app = QApplication(sys.argv)
        app.setApplicationName("SYNAPSE")
        app.setOrganizationName("Noxfort Systems")

        # 3. Initialize the Main Controller (The Brain / Backend)
        # This initializes the AppState and prepares the Service Layer (Optimizer, Engine, etc.)
        controller = MainController()

        # 4. Initialize the Main Window (The Face / Frontend)
        # We inject the controller into the view so the UI can delegate user intents
        window = MainWindow(controller)
        window.show()
        
        print("[Launcher] UI Started. Event loop running.")

        # 5. Start the Event Loop
        sys.exit(app.exec())
        
    except Exception as e:
        print(f"[CRITICAL] Runtime Error during startup: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# Add the command to the group

if __name__ == "__main__":
    start_ui()