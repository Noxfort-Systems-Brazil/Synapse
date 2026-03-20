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
# File: src/workers/xai_worker.py
# Author: Gabriel Moraes
# Date: 2025-12-03
#
# Refactored V4 (Ephemeral Thread Architecture):
# - Replaced persistent QThread loop with ON-DEMAND ephemeral threads.
# - Each analysis request creates a QThread, runs Captum + Jurist, emits result, dies.
# - Jurist LLM is cached at the manager level (loaded on first demand, reused).
# - No idle thread consuming resources when XAI is not requested.

import torch
import numpy as np
import traceback
import json
from datetime import datetime
from PyQt6.QtCore import QThread, QObject, pyqtSignal, pyqtSlot

# Import Captum
from captum.attr import IntegratedGradients

# Import Agents
from src.agents.auditor_agent import AuditorAgent
from src.agents.jurist_agent import JuristAgent
from src.agents.specialist_agent import SpecialistAgent
from src.agents.fuser_agent import FuserAgent


class _XAITask(QThread):
    """
    Ephemeral thread that runs ONE XAI analysis and dies.
    
    Lifecycle: Created → run() → result_ready emitted → finished → deleteLater.
    """
    
    result_ready = pyqtSignal(dict)

    def __init__(self, payload: dict, device, jurist, model_config: dict):
        super().__init__()
        self.payload = payload
        self.device = device
        self.jurist = jurist  # Shared reference, NOT owned by this thread
        self.model_config = model_config
        
        # Shadow models (disposable)
        self.shadow_auditor = None
        self.shadow_tcn = None
        self.shadow_fuser = None
        
        # Auto-cleanup
        self.finished.connect(self.deleteLater)

    def run(self):
        """Thread entry point. Executes Captum analysis + Jurist generation."""
        target = self.payload['target']
        print(f"[XAI] 🧠 Analyzing '{target}' (ephemeral thread)...")
        
        try:
            # 1. Setup Model Wrapper
            model_wrapper = None
            if target == 'auditor': model_wrapper = self._prepare_auditor()
            elif target == 'tcn': model_wrapper = self._prepare_tcn()
            elif target == 'fuser': model_wrapper = self._prepare_fuser()
            
            if not model_wrapper:
                raise ValueError(f"Unknown target model: {target}")

            # 2. Captum Analysis (Integrated Gradients)
            input_list = self.payload['input_vector']
            input_tensor = torch.tensor([input_list], dtype=torch.float32, device=self.device)
            input_tensor.requires_grad = True
            
            ig = IntegratedGradients(model_wrapper)
            baseline = torch.zeros_like(input_tensor)
            
            attributions, delta = ig.attribute(
                inputs=input_tensor,
                baselines=baseline,
                return_convergence_delta=True
            )
            attr_list = attributions.cpu().detach().numpy().flatten().tolist()
            
            # 3. Semantic Report (via shared Jurist)
            feature_names = self.payload.get('feature_names', [])
            if not feature_names:
                feature_names = [f"F{i}" for i in range(len(attr_list))]
            
            attribution_map = dict(zip(feature_names, attr_list))
            sig_map = {k: v for k, v in attribution_map.items() if abs(v) > 0.001}
            if not sig_map: sig_map = attribution_map

            report_text = "Generating..."
            
            if self.jurist and self.jurist.is_loaded:
                try:
                    report_text = self.jurist.generate_report(
                        tensor_data=sig_map,
                        timestamp=self.payload['timestamp'],
                        locale="pt_BR"
                    )
                except Exception as e_gen:
                    print(f"[XAI] Generation Error: {e_gen}")
                    traceback.print_exc()
                    report_text = f"Error generating text: {str(e_gen)}"
            else:
                report_text = "Jurist Agent not available (Check 'Model Vault')."

            # 4. Emit Result
            result = {
                "type": "XAI_RESULT",
                "target": target,
                "request_id": self.payload['request_id'],
                "timestamp": self.payload['timestamp'],
                "input_vector": input_list,
                "attributions": attr_list,
                "convergence_delta": delta.item(),
                "semantic_text": report_text
            }
            self.result_ready.emit(result)

        except Exception as e:
            print(f"[XAI] Process Error: {e}")
            traceback.print_exc()
            self.result_ready.emit({
                "type": "XAI_RESULT",
                "target": target,
                "request_id": self.payload['request_id'],
                "timestamp": self.payload['timestamp'],
                "input_vector": [],
                "attributions": [],
                "convergence_delta": 0.0,
                "semantic_text": f"Analysis Process Failed: {str(e)}"
            })

    # --- Shadow Model Wrappers (disposable) ---
    
    def _prepare_auditor(self):
        dim = len(self.payload['input_vector'])
        self.shadow_auditor = AuditorAgent(input_dim=dim)
        self.shadow_auditor.model.to(self.device)
        self.shadow_auditor.model.eval()
        return self._auditor_loss_wrapper

    def _prepare_tcn(self):
        self.shadow_tcn = SpecialistAgent(
            input_dim=self.model_config.get('feature_dim', 1),
            output_dim=32,
            num_channels=[16, 32]
        )
        self.shadow_tcn.to(self.device)
        self.shadow_tcn.tcn.eval()
        self.shadow_tcn.decoder.eval()
        return self._tcn_wrapper

    def _prepare_fuser(self):
        self.shadow_fuser = FuserAgent(
            num_variates=len(self.payload['feature_names']),
            seq_len=60, pred_len=10
        )
        self.shadow_fuser.to(self.device)
        self.shadow_fuser.model.eval()
        return self._fuser_wrapper

    def _auditor_loss_wrapper(self, inputs):
        reconstructed = self.shadow_auditor.model(inputs)
        squared_diff = (inputs - reconstructed) ** 2
        return torch.sum(squared_diff, dim=1).unsqueeze(1)

    def _tcn_wrapper(self, inputs):
        batch_size = inputs.size(0)
        seq_len = inputs.size(1)
        x = inputs.view(batch_size, 1, seq_len)
        tcn_out = self.shadow_tcn.tcn(x)
        emb = self.shadow_tcn.decoder(tcn_out.transpose(1, 2))
        last_emb = emb[:, -1, :]
        return torch.sum(last_emb, dim=1).unsqueeze(1)

    def _fuser_wrapper(self, inputs):
        return torch.sum(inputs, dim=1).unsqueeze(1)


class XAIWorker(QObject):
    """
    On-Demand XAI Manager.
    
    Refactored V4 (Ephemeral Architecture):
    - NOT a QThread itself. It's a QObject that SPAWNS ephemeral threads.
    - Each submit_request() creates a _XAITask thread that dies after completion.
    - Jurist LLM is loaded lazily on first use and cached here.
    - No idle thread consuming resources.
    """
    
    result_ready = pyqtSignal(dict)

    def __init__(self, model_config: dict):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_config = model_config
        
        # Cached Jurist (loaded on first demand)
        self.jurist = None
        self._jurist_loading = False
        
        # Track active tasks (prevent garbage collection)
        self._active_tasks = []

    def submit_request(self, target_type: str, input_vector: list, feature_names: list, 
                       model_state: dict = None, error: float = 0.0):
        """Spawns an ephemeral thread for one XAI analysis."""
        
        # Lazy-load Jurist on first demand
        if self.jurist is None and not self._jurist_loading:
            self._load_jurist()
        
        payload = {
            "timestamp": datetime.now().isoformat(),
            "target": target_type,
            "input_vector": input_vector,
            "feature_names": feature_names,
            "model_state": model_state,
            "error": error,
            "request_id": f"req_{target_type}_{int(datetime.now().timestamp())}"
        }
        
        # Create ephemeral thread
        task = _XAITask(payload, self.device, self.jurist, self.model_config)
        task.result_ready.connect(self.result_ready)
        task.finished.connect(lambda: self._cleanup_task(task))
        
        self._active_tasks.append(task)
        task.start()
        
        print(f"[XAI] Spawned ephemeral thread for '{target_type}' analysis.")

    def _load_jurist(self):
        """Loads the Jurist LLM (cached for reuse across requests)."""
        self._jurist_loading = True
        print("[XAI] Loading Jurist Agent (first demand)...")
        try:
            self.jurist = JuristAgent()
            self.jurist.load_resources()
            if self.jurist.is_loaded:
                print("[XAI] ✅ Jurist Agent loaded successfully.")
            else:
                print("[XAI] ⚠️ Jurist Agent failed to load resources.")
                self.jurist = None
        except Exception as e:
            print(f"[XAI] Jurist Init Error: {e}")
            traceback.print_exc()
            self.jurist = None
        finally:
            self._jurist_loading = False

    def _cleanup_task(self, task):
        """Remove finished task from active list."""
        if task in self._active_tasks:
            self._active_tasks.remove(task)

    def unload_resources(self):
        """Frees Jurist VRAM when no longer needed."""
        if self.jurist:
            self.jurist.unload_resources()
            self.jurist = None