# SYNAPSE - A Gateway of Intelligent Perception for Traffic Management
# Copyright (C) 2025 Noxfort Systems
#
# File: src/agents/base_agent.py
# Author: Gabriel Moraes
# Date: 2025-12-25

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Any

# Import Interface for type checking compliance
from src.domain.interfaces import IAgent

# FIX V3: BaseAgent agora herda de nn.Module.
# Isso unifica a interface de 'to()', 'cuda()', 'cpu()', 'state_dict()'
# e resolve conflitos de MRO (Method Resolution Order) e dispositivos.
class BaseAgent(nn.Module, ABC, IAgent):
    """
    Abstract Base Class for all Neural Agents in SYNAPSE.
    
    Refactored V3 (Unified Hierarchy):
    - Inherits from nn.Module to natively handle device movement (CPU/GPU).
    - Supports both Inheritance (self is model) and Composition (self has model).
    """

    def __init__(self, model: nn.Module = None, name: str = "UnknownAgent"):
        # Inicializa o nn.Module primeiro (Crucial para registro de parâmetros)
        super().__init__()
        
        self.name = name
        self.device = torch.device("cpu")
        
        # Gestão de Modelo (Herança vs Composição)
        # Se um modelo externo for passado e não for nós mesmos, armazenamos.
        if model is not None and model is not self:
            self._external_model = model
            # Registra como submódulo para que .to() funcione automaticamente
            self.add_module("_external_model_ref", model)
        else:
            self._external_model = None

    @property
    def model(self) -> nn.Module:
        """
        Retorna o modelo ativo.
        """
        if self._external_model:
            return self._external_model
        return self

    def to(self, device: torch.device) -> 'BaseAgent':
        """
        Move o agente e seus componentes para o dispositivo alvo.
        """
        self.device = device
        
        # 1. Move a própria estrutura (nn.Module base)
        # Isso move todos os parâmetros registrados em self (ex: self.encoder no Linguist)
        super().to(device)
        
        # 2. Se houver composição (modelo externo), move ele também
        if self._external_model:
            self._external_model.to(device)
            
        return self

    def load_weights(self, path: str):
        """Standard interface for loading model state."""
        try:
            # Carrega no dispositivo configurado
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint)
        except Exception as e:
            # Otimização: Evitar print excessivo em loops de optuna
            pass

    def save_weights(self, path: str):
        """Standard interface for saving model state."""
        try:
            torch.save(self.model.state_dict(), path)
        except Exception as e:
            print(f"[{self.name}] Failed to save weights: {e}")

    @abstractmethod
    def inference(self, input_data: Any) -> Any:
        """Primary inference method."""
        pass
    
    @abstractmethod
    def train_step(self, batch_data: Any) -> float:
        """Single training step returning loss."""
        pass