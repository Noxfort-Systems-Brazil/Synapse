# SYNAPSE - A Gateway of Intelligent Perception for Traffic Management
# Copyright (C) 2026 Noxfort Systems

# File: src/services/semantic_enricher.py

from typing import Optional
from src.domain.app_state import AppState

class SemanticEnricher:
    """
    Extracts purely semantic/linguistic responsibilities from the Core Logic.
    Resolves Source IDs into human-readable strings for SLMs.
    """
    def __init__(self, app_state: AppState):
        self.app_state = app_state

    def infer_sensor_type(self, source_id: str) -> str:
        """Infers physical type to prevent hallucination."""
        lower_id = source_id.lower()
        if "cam" in lower_id:
            return "CÂMERA"
        elif "loop" in lower_id or "ind" in lower_id:
            return "LAÇO INDUTIVO"
        elif "rad" in lower_id:
            return "RADAR"
        elif "waze" in lower_id or "api" in lower_id:
            return "DADOS API"
        else:
            return "SENSOR"

    def resolve_semantic_name(self, source_id: str) -> str:
        """Resolves Node ID to localized Street Name if mapped."""
        sensor_type = self.infer_sensor_type(source_id)
        
        # 1. Direct Name
        source = self.app_state.get_data_source(source_id)
        if source and source.name and source.name != source_id:
            return f"[{sensor_type}] {source.name} ({source_id})"

        # 2. Map Name
        element_id = self.app_state.get_element_for_source(source_id)
        location_name = None
        
        if element_id:
            edge = self.app_state.get_edge(element_id)
            if edge and edge.real_name and edge.real_name != str(element_id):
                location_name = edge.real_name
            if not location_name:
                node = self.app_state.get_node(element_id)
                if node and node.real_name and node.real_name != str(element_id):
                    location_name = node.real_name

        if location_name:
            return f"[{sensor_type}] {location_name} ({source_id})"
            
        # 3. Fallback
        return f"[{sensor_type}] {source_id}"
