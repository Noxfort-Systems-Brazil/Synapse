# SYNAPSE - A Gateway of Intelligent Perception for Traffic Management
# Copyright (C) 2026 Noxfort Systems
#
# File: src/engine/security_monitor.py

import time
from typing import Dict, Any, Optional
from src.utils.logging_setup import logger

class SecurityMonitor:
    """
    Analyzes the outputs of the Auditor Autoencoder.
    Adheres to the Single Responsibility Principle (SRP) by keeping thresholding
    and alert payload generation away from the cycle math logic.
    """
    
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        
    def evaluate(self, security_score: float) -> Optional[Dict[str, Any]]:
        """
        Evaluates the Security Score against defined thresholds and returns an alert payload if triggered.
        """
        if security_score > self.threshold:
            logger.warning(f"[Auditor] 🛡️ Security Violation / Anomaly Detected! Score: {security_score:.4f}")
            
            return {
                "title": "Integrity Violation",
                "payload": {
                    "loss": security_score,
                    "status": "ATTACK",
                    "timestamp": time.time()
                }
            }
            
        return None
