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
# File: src/infrastructure/sensor_gateway.py
# Author: Gabriel Moraes
# Date: 2025-12-25

import json
import logging
import csv
import io
import xml.etree.ElementTree as ET
from http.server import BaseHTTPRequestHandler, HTTPServer
from PyQt6.QtCore import QThread, pyqtSignal

class IngestionHandler(BaseHTTPRequestHandler):
    """
    Handles incoming HTTP POST requests from sensors (Ingestion Layer).
    
    Refactored V2 (Polyglot Support):
    - Now accepts JSON, CSV, and XML.
    - Normalizes all formats into a Python Dictionary.
    - Ensures compatibility with legacy hardware (Inductive Loops, Radars).
    """
    
    # Injected by SensorGateway instance
    signal_emitter = None 

    def do_POST(self):
        """Receives data packets via POST on ANY endpoint."""
        
        try:
            # 1. Read Raw Data
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length == 0:
                self.send_error(400, "Empty Request Body")
                return
                
            raw_body = self.rfile.read(content_length)
            decoded_body = raw_body.decode('utf-8', errors='ignore').strip()
            content_type = self.headers.get('Content-Type', '').lower()
            
            # 2. Universal Parsing (The Polyglot Logic)
            payload = {}
            
            # A. JSON Strategy
            if 'application/json' in content_type or decoded_body.startswith('{'):
                try:
                    payload = json.loads(decoded_body)
                except json.JSONDecodeError:
                    pass # Fallback to others if headers lied

            # B. XML Strategy (Common in DATEX II / Legacy SOAP)
            if not payload and ('xml' in content_type or decoded_body.startswith('<')):
                payload = self._parse_xml(decoded_body)

            # C. CSV Strategy (Common in embedded Radars/Loops)
            # If not JSON/XML and contains commas or semicolons
            if not payload and (',' in decoded_body or ';' in decoded_body):
                payload = self._parse_csv(decoded_body)

            # D. Fallback (Raw Wrapper)
            if not payload:
                payload = {"raw_content": decoded_body, "format": "unknown"}

            # 3. Identify Source (Heuristic for Routing)
            source_id = self._extract_source_id(payload)
            
            # Fallback: Use IP if no ID found in payload
            if not source_id:
                client_ip = self.client_address[0]
                clean_ip = client_ip.replace('.', '_').replace(':', '_')
                source_id = f"device_{clean_ip}"

            # 4. Emit RAW Payload to Engine (Thread-Safe)
            if self.signal_emitter:
                # Signal is connected to IngestionPipeline via InferenceEngine
                self.signal_emitter.data_received.emit(str(source_id), payload)
                
            # 5. Response
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(b'{"status": "queued"}')
            
        except Exception as e:
            logging.error(f"[SensorGateway] Ingestion Error: {e}")
            self.send_error(500, str(e))

    def _parse_xml(self, xml_string):
        """Flattens basic XML to a dictionary."""
        try:
            root = ET.fromstring(xml_string)
            data = {}
            # Basic Strategy: Tag Name -> Value
            # Does not handle deep nesting well, but sufficient for sensor telemetry
            for child in root:
                data[child.tag] = child.text
            
            # Also capture attributes of the root (often contains ID)
            data.update(root.attrib)
            return data
        except ET.ParseError:
            return {}

    def _parse_csv(self, csv_string):
        """Parses single-line CSV or Key-Value pairs."""
        try:
            data = {}
            # Normalize separators
            line = csv_string.replace(';', ',')
            
            # Scenario 1: Key=Value pairs (e.g., id=cam1,speed=50)
            if '=' in line:
                parts = line.split(',')
                for p in parts:
                    if '=' in p:
                        k, v = p.split('=', 1)
                        data[k.strip()] = v.strip()
            
            # Scenario 2: Pure Values (e.g., cam1,50,1200)
            # We map to generic fields so Pipeline can hunt for numbers
            else:
                reader = csv.reader(io.StringIO(line))
                for row in reader:
                    for idx, val in enumerate(row):
                        # Try to guess key based on position or value type
                        data[f"field_{idx}"] = val
            
            return data
        except Exception:
            return {}

    def _extract_source_id(self, data):
        """Attempts to find an ID field in the dictionary."""
        if not isinstance(data, dict):
            return None
            
        # Common keys for ID (Expanded for XML/CSV usual headers)
        candidates = [
            'source_id', 'sensorId', 'id', 'camera_id', 'deviceId', 'uuid', 'ip', 'sensor_id',
            'UnitID', 'StationID', 'DetectorID' # Common in XML/Traffic standards
        ]
        
        # 1. Top level search
        for k in candidates:
            # Case-insensitive search
            for data_k in data.keys():
                if data_k.lower() == k.lower():
                     return str(data[data_k])
        
        # 2. Nested 'metadata' or 'header' search (if JSON)
        for sub in ['metadata', 'header', 'info', 'device']:
            if sub in data and isinstance(data[sub], dict):
                for k in candidates:
                    if k in data[sub]:
                        return str(data[sub][k])
                        
        return None

    def log_message(self, format, *args):
        """Suppress default HTTP logging to keep console clean."""
        pass


class SensorGateway(QThread):
    """
    The Inbound Gateway.
    Responsibility: Listens on port 8080 (default) for Sensor Data.
    Technique: Runs a Blocking HTTP Server in a dedicated QThread.
    """
    
    # Inbound Signal (Sensor -> Synapse)
    data_received = pyqtSignal(str, object)
    
    server_started = pyqtSignal(int)
    server_error = pyqtSignal(str)

    def __init__(self, host='0.0.0.0', port=8080):
        super().__init__()
        self.host = host
        self.port = port
        self.httpd = None
        self.is_running = True

    def run(self):
        """
        Main Thread Loop: Runs the Blocking HTTP Server for Ingestion.
        """
        try:
            # Inject self into handler to access signals
            IngestionHandler.signal_emitter = self
            
            self.httpd = HTTPServer((self.host, self.port), IngestionHandler)
            self.server_started.emit(self.port)
            print(f"[SensorGateway] 📥 Listening for sensors on {self.host}:{self.port} (JSON/CSV/XML supported)")
            
            while self.is_running:
                self.httpd.handle_request()
                
        except Exception as e:
            self.server_error.emit(str(e))
            print(f"[SensorGateway] Server Crash: {e}")

    def stop(self):
        """Stops the server safely."""
        self.is_running = False
        if self.httpd:
            self.httpd.server_close()
        self.wait()
        print("[SensorGateway] Stopped.")