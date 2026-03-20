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
# File: src/infrastructure/postgres_manager.py
# Author: Gabriel Moraes
# Date: 2025-11-26

import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot, QThread

class PostgresWorker(QObject):
    """
    Worker responsible for executing blocking SQL operations in a background thread.
    """
    
    # Signals
    connection_checked = pyqtSignal(bool, str)  # (Success, Message)
    setup_finished = pyqtSignal(bool, str)      # (Success, Message)
    
    def __init__(self):
        super().__init__()

    @pyqtSlot(dict)
    def do_check_connection(self, config: dict):
        """
        Attempts to connect to the specific application database.
        """
        try:
            conn = psycopg2.connect(
                host=config.get("host", "localhost"),
                port=config.get("port", 5432),
                database=config.get("dbname", "synapse_db"),
                user=config.get("user", "synapse_user"),
                password=config.get("password", "synapse123"),
                connect_timeout=3
            )
            conn.close()
            self.connection_checked.emit(True, "Connection successful.")
        except Exception as e:
            # Clean up error message
            msg = str(e).split('\n')[0]
            self.connection_checked.emit(False, f"Connection failed: {msg}")

    @pyqtSlot(str, dict)
    def do_setup_database(self, root_password: str, target_config: dict):
        """
        Connects as 'postgres' (root) to create the application User and Database.
        This automates the "Step 3" of installation.
        """
        conn = None
        try:
            # 1. Connect to maintenance DB 'postgres' as root
            conn = psycopg2.connect(
                host=target_config.get("host", "localhost"),
                port=target_config.get("port", 5432),
                database="postgres",
                user="postgres",
                password=root_password,
                connect_timeout=5
            )
            
            # Necessary for CREATE DATABASE
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cursor = conn.cursor()
            
            target_user = target_config.get("user", "synapse_user")
            target_pass = target_config.get("password", "synapse123")
            target_db = target_config.get("dbname", "synapse_db")

            # 2. Check if User exists, create if not
            cursor.execute("SELECT 1 FROM pg_roles WHERE rolname=%s", (target_user,))
            if not cursor.fetchone():
                # Use SQL composition for safe identifier insertion
                cmd_user = sql.SQL("CREATE USER {} WITH ENCRYPTED PASSWORD %s").format(
                    sql.Identifier(target_user)
                )
                cursor.execute(cmd_user, (target_pass,))
                print(f"[PostgresManager] User '{target_user}' created.")
            else:
                print(f"[PostgresManager] User '{target_user}' already exists.")

            # 3. Check if Database exists, create if not
            cursor.execute("SELECT 1 FROM pg_database WHERE datname=%s", (target_db,))
            if not cursor.fetchone():
                cmd_db = sql.SQL("CREATE DATABASE {} OWNER {}").format(
                    sql.Identifier(target_db),
                    sql.Identifier(target_user)
                )
                cursor.execute(cmd_db)
                print(f"[PostgresManager] Database '{target_db}' created.")
            else:
                print(f"[PostgresManager] Database '{target_db}' already exists.")

            # 4. Grant Privileges (Redundant if Owner, but good practice)
            # Note: GRANT ALL ON DATABASE does not grant schema usage in newer PG versions, 
            # but ownership usually suffices for setup.
            
            cursor.close()
            conn.close()
            
            self.setup_finished.emit(True, "Database infrastructure initialized successfully.")

        except Exception as e:
            if conn: conn.close()
            self.setup_finished.emit(False, f"Setup failed: {str(e)}")


class PostgresManager(QObject):
    """
    Controller interface for PostgreSQL operations.
    Manages the worker thread lifecycle.
    """
    
    # Proxy signals for UI
    status_received = pyqtSignal(bool, str)
    setup_complete = pyqtSignal(bool, str)
    
    # Internal signals to talk to worker
    _cmd_check = pyqtSignal(dict)
    _cmd_setup = pyqtSignal(str, dict)

    def __init__(self):
        super().__init__()
        self.thread = QThread()
        self.worker = PostgresWorker()
        self.worker.moveToThread(self.thread)
        
        # Wiring
        self._cmd_check.connect(self.worker.do_check_connection)
        self._cmd_setup.connect(self.worker.do_setup_database)
        
        self.worker.connection_checked.connect(self.status_received)
        self.worker.setup_finished.connect(self.setup_complete)
        
        self.thread.start()

    def check_connection(self, config: dict):
        """Non-blocking check."""
        self._cmd_check.emit(config)

    def initialize_database(self, root_pass: str, target_config: dict):
        """Non-blocking setup (Create Role/DB)."""
        self._cmd_setup.emit(root_pass, target_config)

    def stop(self):
        self.thread.quit()
        self.thread.wait()