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
# File: src/utils/debug_logger.py
# Author: Gabriel Moraes
# Date: 2026-03-10
#
# Debug Logging Module for SYNAPSE Pre-Depuration.
# Provides 2 independent loggers for:
#   1. carina_logger  → ~/Documentos/Synapse/logs/carina/   (KSE + gRPC data flow)
#   2. perf_logger    → ~/Documentos/Synapse/logs/performance/ (neural network timing)

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime


def _create_debug_logger(name: str, subfolder: str, prefix: str) -> logging.Logger:
    """
    Creates an independent rotating file logger under ~/Documentos/Synapse/logs/<subfolder>/.
    
    Each logger:
    - Writes to a session-based file: <prefix>_YYYY-MM-DD_HH-MM-SS.log
    - Uses RotatingFileHandler (max 10MB, 3 backups)
    - Has its own isolated hierarchy (no propagation to root)
    - Includes high-resolution timestamps for cross-log correlation
    """
    base_dir = Path.home() / "Documentos" / "Synapse" / "logs" / subfolder
    
    try:
        base_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"[DEBUG_LOGGER] CRITICAL: Could not create {base_dir}: {e}")
    
    session_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = base_dir / f"{prefix}_{session_id}.log"
    
    dbg_logger = logging.getLogger(f"SYNAPSE.debug.{name}")
    dbg_logger.setLevel(logging.DEBUG)
    dbg_logger.propagate = False  # Isolate from root/SYNAPSE logger
    
    # Avoid duplicate handlers on reimport
    if dbg_logger.hasHandlers():
        dbg_logger.handlers.clear()
    
    # High-resolution formatter with milliseconds for cross-log correlation
    formatter = logging.Formatter(
        '%(asctime)s.%(msecs)03d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=3,
        encoding='utf-8'
    )
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    dbg_logger.addHandler(handler)
    
    dbg_logger.info(f"--- SESSION STARTED: {log_file} ---")
    
    return dbg_logger


# ============================================================================
# EXPORTED SINGLETON LOGGERS
# ============================================================================

# Log 1: Carina Data Flow (KSE packet build + gRPC serialization + transmission)
# Records: KSE_BUILD, GRPC_FRAME, GRPC_SCENARIO, GRPC_COMMAND
carina_logger = _create_debug_logger("carina", "carina", "carina")

# Log 2: Neural Network Performance Metrics
# Records: CYCLE (per-stage timing), GLOBAL_CYCLE (total engine cycle)
perf_logger = _create_debug_logger("performance", "performance", "perf")
