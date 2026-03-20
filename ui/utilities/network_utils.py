# SYNAPSE - A Gateway of Intelligent Perception for Traffic Management
# Copyright (C) 2026 Noxfort Systems
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
# File: ui/utilities/network_utils.py
# Author: Gabriel Moraes
# Date: 2026-03-01

import socket

class NetworkUtils:
    """
    Utility class for network-related operations.
    Abstracts low-level socket interactions away from the UI.
    """

    @staticmethod
    def get_local_ip() -> str:
        """
        Attempts to determine the machine's LAN IP address.
        
        Returns:
            str: The local IP address (e.g., '192.168.1.10') or 'localhost' 
                 if the network is unreachable.
        """
        try:
            # We use a dummy UDP connection to a public DNS to force the OS 
            # to resolve the local IP address used for outbound routing.
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                ip = s.getsockname()[0]
            return ip
        except Exception:
            return "localhost"