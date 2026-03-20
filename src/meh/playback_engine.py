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
# File: src/meh/playback_engine.py
# Author: Gabriel Moraes
# Date: 2026-02-28

import numpy as np
from typing import Dict, TYPE_CHECKING
from src.utils.logging_setup import logger

# Lazy import for type hinting to avoid circular dependencies
if TYPE_CHECKING:
    from src.meh.data_loader import HistoricalDataLoader

class PlaybackEngine:
    """
    Single Responsibility: Sequential Simulation / Playback.
    Manages the cursor and returns the next frame of data sequentially
    for system testing or as an absolute fallback.
    """
    
    def __init__(self, data_loader: 'HistoricalDataLoader'):
        self.loader = data_loader
        self._playback_cursor = 0

    def get_next_frame(self) -> Dict[str, float]:
        """
        Returns the next row of data sequentially in an infinite loop.
        """
        if not self.loader.is_loaded or self.loader.data is None or self.loader.data.empty:
            return {}

        try:
            row = self.loader.data.iloc[self._playback_cursor].to_dict()
        except IndexError:
            # Safety catch in case data changes underneath
            self._playback_cursor = 0
            row = self.loader.data.iloc[0].to_dict()

        # Advance Cursor
        self._playback_cursor += 1
        
        # Rewind if at the end
        if self._playback_cursor >= len(self.loader.data):
            self._playback_cursor = 0
            logger.debug("[PlaybackEngine] 🔄 Simulation Loop: Rewinding to start.")

        # Filter only numeric values for the flat payload
        clean_row = {k: float(v) for k, v in row.items() if isinstance(v, (int, float, np.number))}
        
        return clean_row

    def reset(self):
        """Resets the simulation cursor to the beginning."""
        self._playback_cursor = 0