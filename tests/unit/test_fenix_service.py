import time
import pytest
from unittest.mock import patch, MagicMock
from src.services.fenix_service import FenixService, TriggerType

def test_fenix_emergency_trigger(qtbot, mock_storage_manager, mock_app_state):
    """Test FenixService triggers fallback cycle when reconstruction error exceeds threshold."""
    service = FenixService(storage_manager=mock_storage_manager, app_state=mock_app_state)
    service.is_monitoring_active = True
    
    # We want to test the signals emitted by the service
    with qtbot.waitSignal(service.request_fallback_activation, timeout=2000) as blocker:
        # Patching sleep to make the thread run instantly in test
        with patch('time.sleep', return_value=None):
            # 0.35 is > self.reconstruction_threshold of 0.30
            service.check_health_metrics(current_reconstruction_error=0.35)
            
    # The signal should be emitted with True to activate fallback
    assert blocker.args == [True]
    assert service.is_running is True

def test_fenix_ignores_normal_error(qtbot, mock_storage_manager, mock_app_state):
    """Test FenixService does NOT trigger on normal errors."""
    service = FenixService(storage_manager=mock_storage_manager, app_state=mock_app_state)
    service.is_monitoring_active = True
    
    # Normal error (0.10 < 0.30)
    service.check_health_metrics(current_reconstruction_error=0.10)
    
    assert service.is_running is False

def test_fenix_cycle_aborts_on_stop_request(qtbot, mock_storage_manager, mock_app_state):
    """Test the Fenix cycle aborts safely if stop requested mid-flight."""
    service = FenixService(storage_manager=mock_storage_manager, app_state=mock_app_state)
    
    # Trigger opportunistically
    with qtbot.waitSignal(service.cycle_finished, timeout=2000) as blocker:
        # Patch sleep to a tiny amount so the thread doesn't finish before we call stop()
        with patch('time.sleep', side_effect=lambda x: None if x < 1 else time.sleep(0.05)):
            service.start_fenix_cycle(TriggerType.OPPORTUNISTIC)
            # Immediately request a stop
            service.stop_cycle()
            
    success, message = blocker.args
    assert success is False
    assert "Cycle Aborted by User" in message
    assert service.is_running is False
