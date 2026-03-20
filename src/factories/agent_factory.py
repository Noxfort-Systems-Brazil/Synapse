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
# File: src/factories/agent_factory.py
# Author: Gabriel Moraes
# Date: 2026-02-16

from typing import Dict, Any, Optional

# Import Agents
from src.agents.specialist_agent import SpecialistAgent
from src.agents.coordinator_agent import CoordinatorAgent
from src.agents.fuser_agent import FuserAgent
from src.agents.imputer_agent import ImputerAgent
from src.agents.corrector_agent import CorrectorAgent
from src.agents.peak_classifier_agent import PeakClassifierAgent
from src.agents.linguist_agent import LinguistAgent
from src.agents.auditor_agent import AuditorAgent
from src.agents.jurist_agent import JuristAgent

class AgentFactory:
    """
    The Central Agent Fabricator & Registry.
    
    Refactored V5 (Jurist Integration):
    - Now manages the JuristAgent (LLM) for XAI tasks.
    - Provides a complete 'One-Stop-Shop' for all Neural Agents in SYNAPSE.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Args:
            config: Global system configuration dict (usually loaded from yaml).
        """
        self.config = config or {}
        
        # Registry: { source_id: { 'specialist': ..., 'jurist': ..., etc } }
        self.registry: Dict[str, Dict[str, Any]] = {}

    # --- Registry Management Methods (Stateful) ---

    def get_or_create_linguist(self, source_id: str) -> LinguistAgent:
        """Retrieves/Creates LinguistAgent (The Gatekeeper)."""
        agent = self._get_from_registry(source_id, 'linguist')
        if agent: return agent
            
        p_ling = self.config.get('linguist', {})
        new_agent = LinguistAgent(
            model_name=p_ling.get('model_name', "distilroberta-base"),
            learning_rate=p_ling.get('lr', 1e-4)
        )
        self._register(source_id, 'linguist', new_agent)
        return new_agent

    def get_or_create_specialist(self, source_id: str) -> SpecialistAgent:
        """Retrieves/Creates SpecialistAgent (The Operational Model)."""
        agent = self._get_from_registry(source_id, 'specialist')
        if agent: return agent
            
        p_spec = self.config.get('specialist', {})
        input_dim = p_spec.get('input_dim', 1) 
        output_dim = p_spec.get('output_dim', 1)
        
        new_agent = self.create_specialist(self.config, input_dim, output_dim)
        self._register(source_id, 'specialist', new_agent)
        return new_agent

    def get_or_create_auditor(self, source_id: str) -> AuditorAgent:
        """Retrieves/Creates AuditorAgent (The Security Guard)."""
        agent = self._get_from_registry(source_id, 'auditor')
        if agent: return agent

        p_aud = self.config.get('auditor', {})
        input_len = p_aud.get('input_len', 60)

        new_agent = self.create_auditor(self.config, input_len)
        self._register(source_id, 'auditor', new_agent)
        return new_agent

    def get_or_create_jurist(self, source_id: str) -> JuristAgent:
        """
        Retrieves/Creates JuristAgent (The Judge/Explainer).
        Note: The Jurist is usually global, but we register per source 
        if we want isolated contexts or to follow the pattern. 
        For now, we create a new instance (the agent handles its own VRAM loading).
        """
        agent = self._get_from_registry(source_id, 'jurist')
        if agent: return agent

        new_agent = self.create_jurist(self.config)
        self._register(source_id, 'jurist', new_agent)
        return new_agent

    def _get_from_registry(self, source_id: str, agent_type: str) -> Optional[Any]:
        if source_id in self.registry:
            return self.registry[source_id].get(agent_type)
        return None

    def _register(self, source_id: str, agent_type: str, agent: Any):
        if source_id not in self.registry:
            self.registry[source_id] = {}
        self.registry[source_id][agent_type] = agent

    # --- Static Builders (Stateless) ---

    @staticmethod
    def create_specialist(config: Dict[str, Any], input_dim: int, output_dim: int) -> SpecialistAgent:
        p_spec = config.get('specialist', {})
        num_channels = [16, 32]
        kernel_size = 2
        
        if p_spec:
            if 'num_levels' in p_spec and 'base_channel' in p_spec:
                num_channels = [p_spec['base_channel']] * p_spec['num_levels']
            kernel_size = p_spec.get('kernel_size', kernel_size)
            
        return SpecialistAgent(
            input_dim=input_dim,
            output_dim=output_dim,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=p_spec.get('dropout', 0.2),
            learning_rate=p_spec.get('lr', 0.001)
        )

    @staticmethod
    def create_auditor(config: Dict[str, Any], input_len: int) -> AuditorAgent:
        p_aud = config.get('auditor', {})
        return AuditorAgent(
            input_len=input_len,
            J=p_aud.get('J', 2),
            Q=p_aud.get('Q', 1),
            latent_dim=p_aud.get('latent_dim', 16),
            learning_rate=p_aud.get('lr', 1e-3)
        )

    @staticmethod
    def create_jurist(config: Dict[str, Any]) -> JuristAgent:
        """
        Creates a Jurist Agent (LLM Wrapper).
        """
        p_jurist = config.get('jurist', {})
        model_id = p_jurist.get('model_id', "Qwen/Qwen2.5-1.5B-Instruct")
        
        return JuristAgent(
            model_id=model_id
        )

    @staticmethod
    def create_coordinator(config: Dict[str, Any], input_dim: int, hidden_dim: int, output_dim: int) -> CoordinatorAgent:
        p_coord = config.get('coordinator', {})
        return CoordinatorAgent(
            in_channels=input_dim,
            hidden_channels=p_coord.get('hidden_channels', hidden_dim),
            out_channels=output_dim,
            heads=p_coord.get('heads', 4),
            dropout=p_coord.get('dropout', 0.6),
            learning_rate=p_coord.get('lr', 0.005)
        )

    @staticmethod
    def create_fuser(config: Dict[str, Any], num_variates: int, seq_len: int, pred_len: int) -> FuserAgent:
        p_fuser = config.get('fuser', {})
        return FuserAgent(
            num_variates=num_variates,
            seq_len=seq_len,
            pred_len=pred_len,
            d_model=p_fuser.get('d_model', 512),
            n_heads=p_fuser.get('n_heads', 8),
            layers=p_fuser.get('layers', 2),
            learning_rate=p_fuser.get('lr', 0.0001)
        )

    @staticmethod
    def create_imputer(config: Dict[str, Any], feature_dim: int) -> ImputerAgent:
        p_imp = config.get('imputer', {})
        return ImputerAgent(
            feature_dim=feature_dim,
            hidden_dim=p_imp.get('hidden_dim', 24),
            num_layers=p_imp.get('num_layers', 3),
            learning_rate=p_imp.get('lr', 0.001)
        )

    @staticmethod
    def create_corrector(config: Dict[str, Any], input_dim: int) -> CorrectorAgent:
        p_corr = config.get('corrector', {})
        return CorrectorAgent(
            input_dim=input_dim,
            hidden_dim=p_corr.get('hidden_dim', 64),
            latent_dim=p_corr.get('latent_dim', 16),
            kernel_size=p_corr.get('kernel_size', 3),
            learning_rate=p_corr.get('lr', 0.001)
        )

    @staticmethod
    def create_classifier(config: Dict[str, Any], input_dim: int, output_dim: int) -> PeakClassifierAgent:
        p_clf = config.get('classifier', {})
        return PeakClassifierAgent(
            input_dim=input_dim,
            hidden_dim=p_clf.get('hidden_dim', 64),
            output_dim=output_dim,
            learning_rate=p_clf.get('lr', 0.001)
        )