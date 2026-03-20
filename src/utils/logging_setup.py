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
# File: src/utils/logging_setup.py
# Author: Gabriel Moraes
# Date: 2026-03-02

import sys
import os
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime

class StreamToLogger:
    """
    Custom stream object that redirects standard system outputs (stdout/stderr)
    to the logger safely, avoiding recursion crashes with a thread-safe guard.
    """
    def __init__(self, logger_instance: logging.Logger, log_level: int):
        self.logger = logger_instance
        self.log_level = log_level
        self._is_logging = False  # The Recursion Guard

    def write(self, buf: str):
        """Intercepts the text and sends it to the log file, avoiding loops."""
        # If we are already inside a logging operation, bypass to prevent infinite recursion
        if self._is_logging:
            return
            
        self._is_logging = True
        try:
            for line in buf.rstrip().splitlines():
                if line.strip():
                    self.logger.log(self.log_level, line.strip())
        finally:
            self._is_logging = False

    def flush(self):
        """Pass-through for compatibility with sys.stdout/stderr."""
        pass

def setup_logger(logger_name: str = "SYNAPSE") -> logging.Logger:
    """
    Configures a logger for the SYNAPSE system.
    
    Refactored V4 (Full Capture Mode with Recursion Guard):
    - Safely redirects sys.stdout and sys.stderr via StreamToLogger.
    - Captures raw Python errors, stack traces, and standard prints.
    - Protected against recursion depth crashes.
    """
    
    # 1. Define Log Directory (Golden Source / Global System Path)
    base_dir = Path.home() / "Documentos" / "Synapse"
    log_dir = base_dir / "logs"
    
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"CRITICAL: Could not create log directory at {log_dir}: {e}")
    
    # Generate filename with Timestamp for Session History
    session_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = log_dir / f"session_{session_id}.log"

    # 2. Create the Logger Object
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # Prevent propagating to root logger to avoid double logging
    
    # Clean previous handlers to avoid duplication
    if logger.hasHandlers():
        logger.handlers.clear()

    # 3. Define Formatters
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-7s | %(filename)s:%(lineno)d | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    console_formatter = logging.Formatter(
        '[%(levelname)s] %(message)s'
    )

    # 4. Handler: File (Session Log)
    try:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"CRITICAL: Could not attach file handler: {e}")

    # 5. Handler: Console (Stream)
    # Use sys.__stdout__ explicitly to avoid looking like stderr errors
    console_handler = logging.StreamHandler(sys.__stdout__) 
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # 5.5 Configure Root Logger to avoid third-party library clutter in stderr
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] %(name)s: %(message)s',
        handlers=[logging.StreamHandler(sys.__stdout__)],
        force=True
    )

    # 6. Global Redirection of Terminal Outputs
    # This guarantees that everything in the VS Code terminal goes to the .log safely
    sys.stdout = StreamToLogger(logger, logging.INFO)
    sys.stderr = StreamToLogger(logger, logging.ERROR)
    
    logger.info(f"--- SESSION STARTED: {log_file} ---")

    return logger

def set_global_level(level_str: str):
    """
    Sets the global log level for the SYNAPSE logger and its file handlers.
    
    Args:
        level_str: One of 'DEBUG', 'INFO', 'WARNING', 'ERROR'.
    """
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR
    }
    target_level = level_map.get(level_str, logging.INFO)
    logger.setLevel(target_level)
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.setLevel(target_level)


# --- EXPORT THE DEFAULT SINGLETON INSTANCE ---
logger = setup_logger()