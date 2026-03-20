import pytest
import numpy as np
from src.kse.filter import RobustKalmanFilter
from src.kse.definitions import PROFILES

def test_kalman_predict_physics_limits():
    """Test the Kalman filter prediction adheres to physical acceleration boundaries."""
    profile = PROFILES["DEFAULT"]
    # Initialize at position 0
    kf = RobustKalmanFilter(node_id="test_node", initial_val=0.0, profile=profile)
    
    # Inject an extreme acceleration state (e.g., 100 m/s^2)
    kf.x[2, 0] = 100.0
    kf.x[1, 0] = 50.0 # Extreme velocity
    
    # Predict 1 second into future
    kf.predict(dt=1.0)
    
    # Acceleration should be clamped to MAX_ACCEL (4.5)
    assert kf.x[2, 0] <= 4.5
    
    # Velocity should be clamped to MAX_VEL (40.0) from the _enforce_physics_limits code
    assert kf.x[1, 0] <= 40.0

def test_kalman_outlier_rejection():
    """Test Mahalanobis distance gating correctly rejects impossible values."""
    profile = PROFILES["DEFAULT"]
    kf = RobustKalmanFilter(node_id="test_node", initial_val=10.0, profile=profile)
    
    # Allow Filter to settle by giving it consistent good readings
    for i in range(10):
        kf.predict(dt=1.0)
        kf.update(measurement=10.0)
        
    state_before = float(kf.x[0, 0])
    
    # Inject a mathematically impossible reading (e.g., jump from 10 to 1,000,000 in 1s)
    kf.predict(dt=1.0)
    accepted = kf.update(measurement=1000000.0)
    
    # Should be rejected
    assert accepted is False
    assert kf.consecutive_misses == 1
    
    # State position should remain close to prediction (dead reckoning), not the outlier
    state_after = float(kf.x[0, 0])
    assert abs(state_after - state_before) < 5.0 # Dead reckoning should barely move it

def test_kalman_confidence_decay():
    """Test confidence metric decreases as uncertainty grows (Dead Reckoning without updates)."""
    profile = PROFILES["DEFAULT"]
    kf = RobustKalmanFilter(node_id="test_node", initial_val=0.0, profile=profile)
    
    # Do 1 good update to settle P matrix
    kf.predict(dt=1.0)
    kf.update(measurement=0.0)
    
    initial_conf = kf.get_kinetic_snapshot().confidence
    
    # Predict 10 seconds into the future blindly (No measurements)
    for _ in range(10):
        kf.predict(dt=1.0)
        
    final_conf = kf.get_kinetic_snapshot().confidence
    
    # Confidence should drop
    assert final_conf < initial_conf
