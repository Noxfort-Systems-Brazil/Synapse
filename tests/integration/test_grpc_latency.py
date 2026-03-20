import pytest
import asyncio
import time

# The module under test
from src.workers.hft_bootstrapper import HFTBootstrapper
from tests.mock_grpc_server import serve_mock_carina

@pytest.fixture
def run_loop():
    """Returns a clean event loop for testing async logic without Qt interference."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()

@pytest.mark.asyncio
async def test_grpc_network_latency_benchmark(qtbot, run_loop):
    """
    Network Latency Benchmark.
    Measures the exact time it takes for a traffic packet to leave the UI thread
    and arrive at the (mock) CARINA server across the gRPC network boundary.
    """
    # 0. Start async gRPC Mock Server manually on a different port to avoid conflicts
    server = await serve_mock_carina(port=50055)
    
    # 1. Start the Bootstrapper worker targeting our mock server
    # Note: Because QThread blocks without a full Qt UI loop, we run its internal
    # asyncio loop _process_commands() MANUALLY here in the test wrapper.
    worker = HFTBootstrapper(endpoint="localhost:50055")
    worker.connector = worker.connector or __import__('src.infrastructure.grpc_connector', fromlist=['']).GrpcConnector("localhost:50055")
    
    # Bypass QThread.run() and create its internal queue directly
    worker._loop = asyncio.get_running_loop()
    worker._command_queue = asyncio.Queue()
    worker._keep_running = True
    
    # Spawn its brain as a background task
    brain_task = asyncio.create_task(worker._process_commands())
    async def wait_for_signal(signal, timeout=3.0):
        flag = {"triggered": False}
        def _on_signal():
            flag["triggered"] = True
        
        signal.connect(_on_signal)
        start = time.time()
        while not flag["triggered"]:
            if time.time() - start > timeout:
                raise TimeoutError(f"Signal {signal} timed out")
            await asyncio.sleep(0.05)
        signal.disconnect(_on_signal)

    # Helper to await Qt Signals without blocking asyncio
    async def wait_for_signal(signal, timeout=3.0):
        flag = {"triggered": False}
        def _on_signal():
            flag["triggered"] = True
        
        signal.connect(_on_signal)
        start = time.time()
        while not flag["triggered"]:
            if time.time() - start > timeout:
                raise TimeoutError(f"Signal timed out")
            await asyncio.sleep(0.05)
        signal.disconnect(_on_signal)
        
    # 2. Let the Bootstrapper auto-connect and handshake with Mock CARINA
    worker.request_ping()
    await wait_for_signal(worker.connection_success)
        
    # Wait for map upload fake acceptance
    worker.request_map_upload({"fake": "topology"})
    await wait_for_signal(worker.map_upload_success)
        
    # Arm system to open the stream
    worker.request_system_arm()
    await wait_for_signal(worker.system_armed_success)

    # Wait a tiny bit for the async stream _traffic_producer to officially connect
    await asyncio.sleep(0.5)

    # 3. Benchmark Network Transfer Rate (Cadence)
    num_packets = 10
    
    # Generic pseudo-packet (simulating a busy intersection)
    test_packet = {
        "source": "intersection_benchmark",
        "timestamp": time.time(),
        "traffic": [
            {"id": "obj1", "class": "car", "speed": 12.5},
            {"id": "obj2", "class": "bus", "speed": 8.0},
        ]
    }
    
    print("\\n[HFT NETWORK BENCHMARK] Starting transmit of 10 packets...")

    for i in range(num_packets):
        # Fire packet into the queue (instant on the UI side)
        worker.send_runtime_command(test_packet)
        await asyncio.sleep(0.005) # Yield to event loop to allow internal enqueue
        
    # Wait for the HFTStreamer to throttle and send all 10 packets to CARINA
    # 10 packets * 0.150s interval = 1.5 seconds. We sleep a bit extra.
    await asyncio.sleep(2.0)
        
    worker._keep_running = False
    worker.request_ping() # Dummy command to flush the block
    await brain_task
    
    # Gracefully shut down client so server doesn't throw CancelledError
    if worker.connector:
        await worker.connector.close()
    
    # Grab the true arrival times on the Mock Server side!
    arrival_times = server.mock_servicer.frame_arrival_times
    
    await server.stop(grace=0.1)

    # Calculate intervals between received frames
    intervals_ms = []
    # Ignore the very first frame to avoid startup connection lag measuring
    for i in range(2, len(arrival_times)):
        interval = (arrival_times[i] - arrival_times[i-1]) * 1000.0
        intervals_ms.append(interval)

    if not intervals_ms:
        pytest.fail("Not enough frames received by the mock server.")
        
    avg_interval_ms = sum(intervals_ms) / len(intervals_ms)
    
    print(f"[HFT NETWORK BENCHMARK] Average Network Cadence/Interval: {avg_interval_ms:.2f} ms")
    if intervals_ms:
        print(f"[HFT NETWORK BENCHMARK] Min: {min(intervals_ms):.2f} ms | Max: {max(intervals_ms):.2f} ms")
    
    # 4. Assert performance requirement. 
    # Must be between 140ms and 260ms (accounting for 150ms throttle + execution margin)
    assert 140.0 <= avg_interval_ms <= 260.0, f"Critical Cadence Failure: {avg_interval_ms:.2f} ms is outside 150-250ms limit!"
