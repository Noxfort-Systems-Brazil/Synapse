# SYNAPSE - A Gateway of Intelligent Perception for Traffic Management
# Copyright (C) 2026 Noxfort Systems

# File: src/engine/strategies/veto_medoid_strategy.py

import numpy as np
from typing import List, Dict

class VetoMedoidStrategy:
    """
    Mathematical Strategy (Strategy Pattern)
    Finds the medoid (most representative anomaly) in a list of events.
    Extracted from XAIManager (SRP).
    """

    @staticmethod
    def find_medoid(veto_buffer: List[Dict]) -> Dict:
        """
        Returns the event that represents the centroid of the vector cluster.
        """
        if not veto_buffer:
            raise ValueError("Cannot calculate medoid on empty buffer.")

        vectors = [item['vector'] for item in veto_buffer]
        matrix = np.array(vectors)
        centroid = np.mean(matrix, axis=0)
        distances = np.linalg.norm(matrix - centroid, axis=1)
        medoid_index = np.argmin(distances)
        
        return veto_buffer[medoid_index]
