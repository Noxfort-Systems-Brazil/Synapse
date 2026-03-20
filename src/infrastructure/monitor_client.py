import json
import logging
import datetime
import threading
import time
from typing import Optional
import paho.mqtt.client as mqtt

logger = logging.getLogger(__name__)

class MonitorClient:
    """
    Background Telemetry Client to report status and incidents to the external Monitor via MQTT.
    Supports strict JSON payload structures with ISO8601 formatting.
    """
    def __init__(self, host: str = "localhost", port: int = 1883, enabled: bool = False, topic: str = "noxfort/telemetry/"):
        self.host = host
        self.port = port
        self.enabled = enabled
        self.topic = topic
        self._running: bool = False
        self._thread: Optional[threading.Thread] = None
        self._client: Optional[mqtt.Client] = None
        
        if self.enabled:
            self.start()
            
    def _create_payload(self, origin: str, category: str, level: str, message: str) -> str:
        """Formats the strict JSON payload required by the monitoring server."""
        # Generating local machine timestamp with timezone offset (e.g., 2026-03-04T01:53:19-03:00)
        now_local = datetime.datetime.now().astimezone()
        iso_timestamp = now_local.isoformat()
        
        payload = {
            "category": category,
            "origin": origin,
            "level": level,
            "message": message,
            "occurred_at": iso_timestamp
        }
        return json.dumps(payload)

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        if reason_code == 0:
            logger.info(f"[Monitor] Successfully connected to Broker!")
        else:
            logger.error(f"[Monitor] Failed to connect. Code: {reason_code}")

    def _on_publish(self, client, userdata, mid, reason_code, properties):
        pass

    def _connect(self):
        if self._client:
            return
            
        try:
            import uuid
            main_client_id = f"synapse_{uuid.uuid4().hex[:8]}"
            self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=main_client_id)
            self._client.on_connect = self._on_connect
            self._client.on_publish = self._on_publish
            self._client.connect(self.host, self.port, keepalive=60)
            self._client.loop_start()  # Launch background network thread for paho
            logger.info(f"[Monitor] Connected to MQTT broker at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"[Monitor] Failed to connect to MQTT broker: {e}")
            self._client = None

    def start(self):
        if not self.enabled or self._running:
            return
            
        self._running = True
        self._connect()
        self._thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._thread.start()
        
    def stop(self):
        self._running = False
        if self._client:
            self._client.loop_stop()
            self._client.disconnect()
            self._client = None
            
        if self._thread is not None and self._thread.is_alive(): # type: ignore
            self._thread.join(timeout=2.0)

    def _heartbeat_loop(self):
        """Sends a Carina Heartbeat every 30 seconds using the 5-field schema."""
        while self._running:
            if self._client:
                # origin, category, level, message
                payload = self._create_payload("synapse", "SOFTWARE", "INFO", "heartbeat")
                try:
                    self._client.publish(self.topic, payload, qos=0)
                except Exception as e:
                    logger.warning(f"[Monitor] Failed to publish heartbeat: {e}")
                    
            # Sleep in chunks to allow responsive shutdown
            for _ in range(30):
                if not self._running:
                    break
                time.sleep(1)

    def report_incident(self, category: str, level: str, message: str):
        """
        Immediately dispatches an incident report.
        level must be 'INFO', 'WARNING', or 'CRITICAL'.
        category must be 'SOFTWARE' or 'HARDWARE'.
        """
        if not self.enabled:
            return
            
        if not self._client:
            self._connect()
            
        if self._client:
            # We enforce uppercase level for the server dashboard
            payload = self._create_payload("synapse", category, level.upper(), message)
            try:
                self._client.publish(self.topic, payload, qos=1)
                logger.debug(f"[Monitor] Incident reported: {level} - {message}")
            except Exception as e:
                logger.error(f"[Monitor] Failed to publish incident: {e}")

    @staticmethod
    def test_connection_and_send_heartbeat(host: str, port: int = 1883, topic: str = "noxfort/telemetry/") -> bool:
        """
        Static method to instantly test a connection and dispatch the Carina Heartbeat.
        Used by the UI before saving configurations to ensure the server is reachable.
        """
        try:
            # Use a unique client_id for the test to prevent disconnecting the main active client
            import uuid
            test_client_id = f"synapse_test_{uuid.uuid4().hex[:8]}"
            client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=test_client_id)
            client.connect(host, port, keepalive=5)
            client.loop_start()
            
            # Temporary instance just to use the serializer logic mapping dates
            temp = MonitorClient(enabled=False)
            payload = temp._create_payload("synapse", "SOFTWARE", "INFO", "heartbeat")
            
            msg_info = client.publish(topic, payload, qos=1)
            msg_info.wait_for_publish(timeout=2.0)
            
            client.loop_stop()
            client.disconnect()
            return True
        except Exception as e:
            logger.error(f"[Monitor] Instant heartbeat failed: {e}")
            return False
