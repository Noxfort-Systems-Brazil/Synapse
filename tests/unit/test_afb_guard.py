import pytest
from src.afb.sensor_guard import SensorGuard
from src.afb.models import SensorReading

def test_sensor_guard_sudden_drop_rejection():
    """Test SensorGuard correctly rejects sudden anomalous drops."""
    guard = SensorGuard(window_size=10, z_threshold=3.0, min_samples=5)
    
    # Send 5 normal readings for 'S1' to establish a baseline of 100
    for _ in range(5):
        clean = guard.filter([SensorReading(source_id='S1', value=100.0, trust_score=0.9)])
        assert len(clean) == 1
        
    # Send a sudden drop reading for 'S1'
    # A value of 10 is WAY outside the std deviation of 0 (given [100, 100, 100, 100, 100])
    drop = guard.filter([SensorReading(source_id='S1', value=10.0, trust_score=0.9)])
    
    # Should be rejected
    assert len(drop) == 0
    assert guard.total_rejected == 1
    
    # Send normal reading again, should pass
    clean2 = guard.filter([SensorReading(source_id='S1', value=100.0, trust_score=0.9)])
    assert len(clean2) == 1

def test_sensor_guard_false_positive_prevention():
    """Test SensorGuard allows natural variations within the z_threshold."""
    guard = SensorGuard(window_size=10, z_threshold=3.0, min_samples=5)
    
    # Establish baseline with some variance
    # Values: 100, 102, 98, 101, 99 (Mean: 100.0, StdDev: ~1.58)
    history = [100.0, 102.0, 98.0, 101.0, 99.0]
    for val in history:
        guard.filter([SensorReading(source_id='S1', value=val, trust_score=0.9)])
        
    # An acceptable peak within 3 std devs (~+/- 4.74). Allowed range ~ [95.26, 104.74]
    acceptable_peak = guard.filter([SensorReading(source_id='S1', value=104.0, trust_score=0.9)])
    assert len(acceptable_peak) == 1
    
    # A peak outside 3 std devs (e.g., 110)
    unacceptable_peak = guard.filter([SensorReading(source_id='S1', value=110.0, trust_score=0.9)])
    assert len(unacceptable_peak) == 0

def test_sensor_guard_insufficient_history():
    """Test SensorGuard passes values when history < min_samples."""
    guard = SensorGuard(window_size=10, z_threshold=3.0, min_samples=5)
    
    # Only 3 samples
    history = [100.0, 100.0, 100.0]
    for val in history:
        res = guard.filter([SensorReading(source_id='S1', value=val, trust_score=0.9)])
        assert len(res) == 1
        
    # 4th sample is an anomaly, but we don't have enough history to judge, so it MUST pass
    anomaly = guard.filter([SensorReading(source_id='S1', value=10.0, trust_score=0.9)])
    assert len(anomaly) == 1
