# tests/conftest.py
import pytest
from unittest.mock import MagicMock
from src.domain.app_state import AppState
from src.managers.storage_manager import StorageManager

@pytest.fixture
def mock_storage_manager():
    """
    Returns a mocked StorageManager that avoids creating real directories.
    """
    sm = MagicMock()
    sm.get_datalake_base_path.return_value = "/tmp/mock/datalake/base"
    sm.get_datalake_golden_path.return_value = "/tmp/mock/datalake/golden"
    sm.get_checkpoint_path.return_value = "/tmp/mock/Checkpoint"
    sm.save_model_checkpoint.return_value = "/tmp/mock/Checkpoint/model.pth"
    sm.is_connected = True
    return sm

@pytest.fixture
def mock_app_state():
    """
    Returns a mocked AppState to avoid complex UI/Topology setups during unit tests.
    """
    app_state = MagicMock(spec=AppState)
    
    # Mocking essential signals to allow .connect() checks
    class MockSignal:
        def connect(self, *args, **kwargs): pass
        def emit(self, *args, **kwargs): pass
        
    app_state.map_data_loaded = MockSignal()
    app_state.association_mode_changed = MockSignal()
    app_state.data_association_changed = MockSignal()
    app_state.data_source_added = MockSignal()
    app_state.data_source_removed = MockSignal()
    app_state.source_origin_toggled = MockSignal()
    
    app_state.get_all_data_sources.return_value = []
    return app_state
