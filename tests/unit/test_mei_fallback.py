import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock

from src.meh.fallback_engine import FallbackEngine
from src.meh.playback_engine import PlaybackEngine

@pytest.fixture
def mock_data_loader():
    loader = MagicMock()
    loader.is_loaded = True
    loader.group_column = 'sensor_id'
    
    # Create a simple valid dataframe
    times = pd.date_range(start='2026-03-01 10:00:00', periods=5, freq='1min')
    df = pd.DataFrame({
        'timestamp': times,
        'sensor_id': ['S1'] * 5,
        'value': [10, 20, 30, 40, 50],
        'corrupted_col': ['A', 'B', 'C', 'D', 'E'] # Should be ignored by Fallback Engine
    })
    
    loader.data = df
    loader.sensor_ids = ['S1']
    loader.ontology = None
    return loader

def test_fallback_engine_builds_profiles_safely(mock_data_loader):
    """Test that FallbackEngine ignores string columns and builds profile successfully."""
    engine = FallbackEngine(mock_data_loader)
    engine.build_profiles()
    
    assert engine.is_ready is True
    assert 'S1' in engine.sensor_profiles
    assert 'S1' in engine.sensor_frequencies
    
    # Assert frequency is 60.0 seconds (1 minute diff in our mock data)
    assert engine.sensor_frequencies['S1'] == 60.0

def test_fallback_engine_empty_data_rejection():
    """Test that FallbackEngine safely rejects empty data."""
    empty_loader = MagicMock()
    empty_loader.is_loaded = True
    empty_loader.data = pd.DataFrame()
    
    engine = FallbackEngine(empty_loader)
    engine.build_profiles()
    
    assert engine.is_ready is False

def test_playback_engine_sequential_loop(mock_data_loader):
    """Test Playback Engine returns frames and rewinds correctly."""
    engine = PlaybackEngine(mock_data_loader)
    
    # Pull frames 1 to 5
    for i in range(1, 6):
        frame = engine.get_next_frame()
        expected_value = i * 10
        assert frame['value'] == expected_value
        assert 'corrupted_col' not in frame  # Should filter out strings
        
    # Frame 6 should rewind to Frame 1
    rewind_frame = engine.get_next_frame()
    assert rewind_frame['value'] == 10

def test_playback_engine_handles_underlying_data_corruption(mock_data_loader):
    """Test Playback Engine doesn't crash if rows are suddenly dropped from data."""
    engine = PlaybackEngine(mock_data_loader)
    
    # Advance to cursor 2
    engine.get_next_frame()
    engine.get_next_frame()
    
    # Suddenly data becomes shorter than cursor
    mock_data_loader.data = mock_data_loader.data.iloc[:1]
    
    # Should catch IndexError and reset to 0 safely
    recovered_frame = engine.get_next_frame()
    assert recovered_frame['value'] == 10
    assert engine._playback_cursor == 0 # Advanced after reset, then rewound to 0 due to len=1
