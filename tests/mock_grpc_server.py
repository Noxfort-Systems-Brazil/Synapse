import grpc
import asyncio
from concurrent import futures
from typing import Any

from proto import synapse_hft_pb2
from proto import synapse_hft_pb2_grpc

class MockHFTLinkServicer(synapse_hft_pb2_grpc.HFTLinkServicer):
    """
    A lightweight Mock of the CARINA gRPC Server.
    Designed specifically to benchmark network latency without needing the heavy CARINA backend.
    """
    
    def __init__(self):
        super().__init__()
        self.received_frames = 0
        self.frame_arrival_times = []
    
    async def Ping(self, request, context):
        """Answers the connection check."""
        return synapse_hft_pb2.SystemState(
            active=True,
            state="ONLINE",
            server_time=0
        )

    async def LoadScenario(self, request, context):
        """Accepts any valid protobuf geometry map."""
        return synapse_hft_pb2.ScenarioStatus(
            accepted=True,
            message="Mock Map Accepted"
        )

    async def SystemControl(self, request, context):
        """Acknowledges Arming."""
        return synapse_hft_pb2.CommandResponse(
            success=True,
            new_state="ONLINE"
        )

    async def StreamTraffic(self, request_iterator, context):
        """
        The critical path for HFT streaming.
        Reads incoming protobuf frames as fast as possible.
        """
        try:
            async for frame in request_iterator:
                import time
                self.frame_arrival_times.append(time.time())
                self.received_frames += 1
                # Simulating minimal parsing overhead
                _ = len(frame.edges) if hasattr(frame, 'edges') else 0
                
            return synapse_hft_pb2.SystemState(
                active=True,
                state="ONLINE",
                server_time=0
            )
        except Exception as e:
            print(f"[Mock Server] Stream error: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            raise

async def serve_mock_carina(port: int = 50051) -> grpc.aio.Server:
    """Spins up the Mock Server asynchronously."""
    server = grpc.aio.server()
    servicer = MockHFTLinkServicer()
    synapse_hft_pb2_grpc.add_HFTLinkServicer_to_server(servicer, server)
    server.mock_servicer = servicer # Attach for test inspections
    server.add_insecure_port(f'[::]:{port}')
    await server.start()
    return server
