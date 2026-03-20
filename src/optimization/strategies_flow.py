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
# File: src/optimization/strategies_flow.py
# Author: Gabriel Moraes
# Date: 2026-03-02

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Any

# Import PyG Data safely
try:
    from torch_geometric.data import Data
    from torch_geometric.utils import degree
except ImportError:
    Data = None

# Import Agents
from src.agents.coordinator_agent import CoordinatorAgent
from src.agents.fuser_agent import FuserAgent
from src.agents.specialist_agent import SpecialistAgent

# Logger
logger = logging.getLogger("Synapse.Strategies.Flow")

class FlowStrategies:
    """
    Optimization Strategies for Flow-Control Agents.
    Contains logic for Coordinator (Spatial), Fuser (Temporal), and Specialist (Pattern).
    """

    @staticmethod
    def coordinator_strategy(trial, graph_data: Any, device: torch.device) -> float:
        """
        Optimizes the Coordinator Agent (GATv2) purely on Topology (.net.xml).
        
        METHOD: Structural Self-Supervision.
        Since we don't use traffic flow data for this calibration, we train the GATv2
        to predict the 'Node Degree' (number of connections).
        
        This forces the Attention Mechanism to learn the map structure and
        generate valid spatial embeddings for the iTransformer downstream.
        """
        # 1. Hyperparameters
        lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
        hidden_channels = trial.suggest_categorical("hidden_channels", [32, 64, 128])
        heads = trial.suggest_categorical("heads", [2, 4, 8])
        dropout = trial.suggest_float("dropout", 0.1, 0.5)

        # 2. Data Sanitization & Target Synthesis (Topology Only)
        try:
            if Data is None:
                return float('inf')

            # A. Extract/Fix Edge Index (The Structure)
            if hasattr(graph_data, 'edge_index') and graph_data.edge_index is not None:
                edge_index = graph_data.edge_index
                if edge_index.dtype != torch.long:
                    edge_index = edge_index.long()
            else:
                return float('inf')

            # B. Extract/Synthesize Features (x)
            # If the map has no pre-calculated features, we use a constant signal.
            # This forces the GAT to learn PURELY from the edges (structure).
            if hasattr(graph_data, 'x') and graph_data.x is not None:
                x = graph_data.x
                num_nodes = x.shape[0]
            else:
                # Infer nodes from edges
                num_nodes = edge_index.max().item() + 1
                x = torch.ones((num_nodes, 1), dtype=torch.float32)

            # C. SYNTHESIZE TOPOLOGICAL TARGET (y)
            # Task: "Predict how connected I am"
            # This generates a valid 'y' so the model can calculate Loss and learn.
            
            # degree() calculates the number of edges pointing to each node
            node_degrees = degree(edge_index[1], num_nodes=num_nodes, dtype=torch.float32)
            
            # Normalize (0 to 1 range) to stabilize the Neural Network
            max_degree = node_degrees.max()
            if max_degree == 0: max_degree = 1.0
            y = node_degrees.view(-1, 1) / max_degree

            # D. Reconstruct Clean Data Object
            clean_data = Data(x=x, edge_index=edge_index, y=y)
            clean_data = clean_data.to(device)

        except Exception as e:
            logger.error(f"[Coordinator] Topology data prep failed: {e}")
            return float('inf')

        # 3. Instantiate Agent
        try:
            in_channels = clean_data.x.shape[1]
            out_channels = clean_data.y.shape[1] # Extracting out_channels from target shape
            
            agent = CoordinatorAgent(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                out_channels=out_channels,       # Passing required parameter
                heads=heads,
                dropout=dropout,
                learning_rate=lr
            )
            agent.to(device)
            
        except Exception as e:
            logger.error(f"[Coordinator] Init Error: {e}")
            return float('inf')

        # 4. Training Loop
        try:
            total_loss = 0.0
            steps = 5  # Fast epochs for HPO
            
            for _ in range(steps):
                loss = agent.train_step(clean_data)
                
                if not np.isfinite(loss):
                    return float('inf')
                    
                total_loss += loss
            
            avg_loss = total_loss / steps
            return avg_loss

        except Exception as e:
            logger.warning(f"[Coordinator] Training failed: {e}")
            return float('inf')
        finally:
            del agent
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @staticmethod
    def fuser_strategy(trial, data: np.ndarray, device: torch.device) -> float:
        """
        Optimizes the Fuser Agent (iTransformer).
        """
        # 1. Hyperparameters
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        d_model = trial.suggest_categorical("d_model", [256, 512])
        layers = trial.suggest_int("layers", 1, 3)
        
        # 2. Data Shape
        if data.ndim == 2:
            seq_len = 24
            if len(data) > seq_len:
                input_seq = data[-seq_len:] 
                input_seq = torch.FloatTensor(input_seq).unsqueeze(0).to(device) 
            else:
                return float('inf')
        else:
             return float('inf')

        num_variates = input_seq.shape[2]

        # 3. Instantiate
        try:
            agent = FuserAgent(
                num_variates=num_variates,
                seq_len=24,
                d_model=d_model,
                layers=layers,
                learning_rate=lr
            )
            agent.to(device)
        except Exception as e:
            return float('inf')

        # 4. Training Loop
        try:
            total_loss = 0.0
            steps = 3
            
            for _ in range(steps):
                # FIXED: Properly delegating to the agent's internal logic.
                # Passing tuple (inputs, targets) to trigger shape locks correctly.
                loss = agent.train_step((input_seq, input_seq))
                
                if not np.isfinite(loss):
                    return float('inf')
                    
                total_loss += loss
                
            return total_loss / steps

        except Exception as e:
            return float('inf')
        finally:
            del agent

    @staticmethod
    def specialist_strategy(trial, data: np.ndarray, device: torch.device) -> float:
        """
        Optimizes the Specialist Agent (TCN).
        """
        # 1. Hyperparameters
        lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
        kernel_size = trial.suggest_int("kernel_size", 2, 5)
        dropout = trial.suggest_float("dropout", 0.1, 0.4)
        
        input_dim = data.shape[1] if data.ndim > 1 else 1
        hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64])
        num_channels = [hidden_dim, hidden_dim]

        # 2. Instantiate
        try:
            agent = SpecialistAgent(
                input_dim=input_dim,
                output_dim=input_dim, 
                num_channels=num_channels,
                kernel_size=kernel_size,
                dropout=dropout,
                learning_rate=lr
            )
            agent.to(device)
        except Exception as e:
            return float('inf')

        # 3. Data Prep
        seq_len = 30
        if len(data) > seq_len + 1:
            x = torch.FloatTensor(data[-seq_len-1:-1]).unsqueeze(0).to(device)
            y = torch.FloatTensor(data[-seq_len:]).unsqueeze(0).to(device)
        else:
            return float('inf')

        # 4. Train Step
        try:
            loss = agent.train_step((x, y))
            return loss if np.isfinite(loss) else float('inf')
        except Exception as e:
            return float('inf')
        finally:
            del agent