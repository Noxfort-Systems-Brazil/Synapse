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
# File: src/kse/definitions.py
# Author: Gabriel Moraes
# Date: 2026-01-09

from dataclasses import dataclass

# --- CONSTANTS & CONFIGURATION ---

GRAVITY = 9.81
MAX_ACCEL = 4.5   # m/s^2 (Typical car max acceleration)
MAX_DECEL = -9.0  # m/s^2 (Emergency braking)
FRICTION_DECAY = 0.95 # Damping factor for Dead Reckoning (Simulates air resistance/friction)

@dataclass
class SensorProfile:
    """
    Defines noise characteristics for different data sources.
    Crucial for tuning the Kalman Matrix R (Measurement Noise).
    """
    name: str
    r_val: float      # Measurement Noise (Higher = Trust Sensor Less)
    q_val: float      # Process Noise (Higher = Expect More Erratic Movement)
    gating_threshold: float # Mahalanobis Distance for outlier rejection

# Pre-defined profiles
PROFILES = {
    "DEFAULT": SensorProfile("Generic", r_val=5.0, q_val=0.1, gating_threshold=3.0),
    "CAMERA": SensorProfile("Vision", r_val=2.0, q_val=0.2, gating_threshold=2.5),
    "INDUCTIVE": SensorProfile("Loop", r_val=0.5, q_val=0.05, gating_threshold=4.0),
    "HISTORY": SensorProfile("Simulation", r_val=0.1, q_val=0.01, gating_threshold=10.0)
}

@dataclass
class KineticState:
    """Immutable snapshot of a physics entity."""
    p: float # Position
    v: float # Velocity
    a: float # Acceleration
    uncertainty: float # Trace of Covariance Matrix
    confidence: float  # Normalized 0.0 - 1.0