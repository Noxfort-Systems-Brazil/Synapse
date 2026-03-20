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
# File: src/services/optimizer_service.py
# Author: Gabriel Moraes
# Date: 2026-03-02

import os
import gc
import torch
import optuna
import numpy as np
from typing import Optional, Dict, Callable, Any

# Geometric data handling for GATv2 Fix
try:
    from torch_geometric.data import Data
except ImportError:
    Data = None

from PyQt6.QtCore import QObject, pyqtSignal

# --- Utils ---
from src.utils.logging_setup import logger

# --- Optimization Sub-Modules (SRP Refactoring) ---
from src.optimization.data_loader import DataLoader
from src.optimization.callbacks import DerivativeConvergenceCallback

# Split Strategies
from src.optimization.strategies_flow import FlowStrategies
from src.optimization.strategies_quality import QualityStrategies
from src.optimization.strategies_semantic import SemanticStrategies

# Cartographer Strategy (Safe Import)
try:
    from src.optimization.strategies_spatial import SpatialStrategies
    from src.services.map_service import MapService
    CARTOGRAPHER_STRATEGY_AVAILABLE = True
except ImportError:
    CARTOGRAPHER_STRATEGY_AVAILABLE = False

class OptimizerService(QObject):
    """
    Manages Hyperparameter Optimization (AutoML) using Optuna.
    
    Refactored V34 (Architecture Nomenclature Fix):
    - Acts purely as an Orchestrator.
    - Delegates Training Logic to specialized Flow/Quality/Semantic strategies.
    - Accurately prepares temporal windows for heavy agents like TimesNet to prevent silent hangs.
    """
    
    training_finished = pyqtSignal(dict)
    optimization_finished = pyqtSignal()

    def __init__(self): 
        super().__init__()
        self.safety_max_trials = 500 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.base_dir = os.path.join(os.path.expanduser("~"), "Documentos", "Synapse")
        self.checkpoint_dir = os.path.join(self.base_dir, "Checkpoint")
        self.checkpoint_file = os.path.join(self.checkpoint_dir, "best_hparams.pth")
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _force_cleanup(self):
        """
        PROTOCOL: TOTAL CLEANUP
        Forces Python GC and PyTorch CUDA Cache to release all resources.
        Critical for avoiding OOM between sequential phases.
        """
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        logger.info("[OptimizerService] 🧹 Resources Flushed (RAM + VRAM).")

    def run(self, map_file_path: Optional[str] = None, history_file_path: Optional[str] = None):
        """
        Main execution pipeline.
        Sequentially optimizes all agents based on priority.
        """
        logger.info(f"[OptimizerService] 🧠 Starting Calculus-Based AutoML...")
        
        # 1. Initial State Check
        self._force_cleanup()

        if os.path.exists(self.checkpoint_file):
            logger.info(f"[OptimizerService] 💾 Checkpoint found. Skipping optimization.")
            self.optimization_finished.emit()
            return 

        # 2. Data Loading (Delegated)
        if not history_file_path or not os.path.exists(history_file_path):
            logger.error("[OptimizerService] ❌ CRITICAL: No User Parquet File provided.")
            return

        univ_data = DataLoader.load_parquet_data(history_file_path)
        if univ_data is None: return

        map_graph = None
        if map_file_path and os.path.exists(map_file_path):
            map_graph = DataLoader.load_sumo_map(map_file_path)

        final_results = {}

        # --- EXECUTION PHASES ---

        # Phase 1: Spatial (Coordinator) -> FlowStrategies
        if map_graph:
            logger.info("[OptimizerService] 🌐 Tuning Coordinator (GATv2)...")
            
            if isinstance(map_graph, dict):
                if Data is not None:
                    try:
                        logger.info("[OptimizerService] 🛠️ Converting Map Dict to PyTorch Geometric Data...")
                        map_graph = Data(**map_graph)
                    except Exception as e:
                        logger.warning(f"[OptimizerService] ⚠️ Failed to convert map_graph to Data: {e}")
                else:
                    logger.warning("[OptimizerService] ⚠️ PyTorch Geometric not found. GATv2 training may fail.")

            final_results['coordinator'] = self._optimize_phase(
                lambda t: FlowStrategies.coordinator_strategy(t, map_graph, self.device),
                slope_threshold=1e-3
            )
        
        # Phase 1b: Spatial Alignment (Cartographer) -> SpatialStrategies
        if map_file_path and CARTOGRAPHER_STRATEGY_AVAILABLE:
            logger.info("[OptimizerService] 🗺️ Tuning Cartographer (Sinkhorn+GATv2)...")
            try:
                ms = MapService()
                if ms.load_network(map_file_path):
                    graph_data = {'edges': ms.edges, 'nodes': ms.nodes}
                    final_results['cartographer'] = self._optimize_phase(
                        lambda t: SpatialStrategies.cartographer_strategy(t, graph_data, self.device),
                        slope_threshold=1e-3
                    )
                    
                    # Save diploma if training succeeded
                    if final_results.get('cartographer'):
                        diploma_dir = os.path.join(self.base_dir, "weights")
                        os.makedirs(diploma_dir, exist_ok=True)
                        diploma_path = os.path.join(diploma_dir, "cartographer.safetensors")
                        try:
                            from src.models.sinkhorn_cross_attention import SinkhornCrossAttention
                            best_p = final_results['cartographer']
                            best_model = SinkhornCrossAttention(
                                d_model=best_p.get('cart_d_model', 64),
                                n_heads=best_p.get('cart_n_heads', 4),
                                n_gat_layers=best_p.get('cart_n_gat_layers', 2),
                                sinkhorn_iters=best_p.get('cart_sinkhorn_iters', 10),
                                dropout=best_p.get('cart_dropout', 0.1),
                                temperature=best_p.get('cart_temperature', 0.1),
                            )
                            best_model.save_diploma(diploma_path)
                            logger.info(f"[OptimizerService] 🎓 Cartographer diploma saved: {diploma_path}")
                        except Exception as e:
                            logger.warning(f"[OptimizerService] ⚠️ Diploma save failed: {e}")
                else:
                    logger.warning("[OptimizerService] ⚠️ MapService failed to load network for Cartographer.")
            except Exception as e:
                logger.warning(f"[OptimizerService] ⚠️ Cartographer training skipped: {e}")
        
        # Phase 2: Temporal (Fuser) -> FlowStrategies
        logger.info("[OptimizerService] 🔮 Tuning Fuser (iTransformer)...")
        final_results['fuser'] = self._optimize_phase(
            lambda t: FlowStrategies.fuser_strategy(t, univ_data, self.device)
        )

        # Phase 3: Local Patterns (Specialist) -> FlowStrategies
        logger.info("[OptimizerService] 🎯 Tuning Specialist (TCN)...")
        final_results['specialist'] = self._optimize_phase(
            lambda t: FlowStrategies.specialist_strategy(t, univ_data, self.device)
        )

        # Phase 4: Security (Auditor) -> QualityStrategies
        logger.info("[OptimizerService] 🛡️ Tuning Auditor (Wavelet AE + OCC)...")
        final_results['auditor'] = self._optimize_phase(
            lambda t: QualityStrategies.auditor_strategy(t, univ_data, self.device)
        )

        # Phase 5: Resilience (Imputer) -> QualityStrategies
        logger.info("[OptimizerService] 🧬 Tuning Imputer (TimeGAN)...")
        final_results['imputer'] = self._optimize_phase(
            lambda t: QualityStrategies.imputer_strategy(t, univ_data, self.device),
            slope_threshold=1e-6,
            window=30
        )

        # Phase 6: Quality (Corrector) -> QualityStrategies
        logger.info("[OptimizerService] 🧹 Tuning Corrector (VAE-TCN)...")
        final_results['corrector'] = self._optimize_phase(
            lambda t: QualityStrategies.corrector_strategy(t, univ_data, self.device)
        )
        
        # Phase 7: Semantics (Linguist) -> SemanticStrategies
        logger.info("[OptimizerService] 🗣️ Tuning Linguist (DistilRoBERTa + TCN-AE)...")
        final_results['linguist'] = self._optimize_phase(
            lambda t: SemanticStrategies.linguist_strategy(t, univ_data, self.device)
        )

        # Phase 8: Classification (Peak Classifier) -> SemanticStrategies
        logger.info("[OptimizerService] 📊 Tuning Peak Classifier (TimesNet)...")
        
        # Extract a small representative sliding window block so TimesNet (seq_len=96)
        # can train in seconds instead of dragging the epoch for hours.
        seq_len = 96
        sample_limit = 64  # Fast batch for HPO calibration
        
        if isinstance(univ_data, np.ndarray):
            series_1d = np.mean(univ_data, axis=1) if univ_data.ndim > 1 else univ_data
        else:
            series_1d = np.array(univ_data).flatten()

        threshold = np.mean(series_1d) + np.std(series_1d)
        
        needed_length = seq_len + sample_limit
        if len(series_1d) > needed_length:
            series_1d = series_1d[-needed_length:]
            
        x_windows = []
        y_labels = []
        
        for i in range(len(series_1d) - seq_len):
            window = series_1d[i : i + seq_len]
            target_val = series_1d[i + seq_len]
            
            x_windows.append(window)
            y_labels.append(1.0 if target_val > threshold else 0.0)
            
        if not x_windows: # Fallback safety
            x_windows = [np.random.randn(seq_len)]
            y_labels = [1.0]
            
        inputs = np.array(x_windows, dtype=np.float32)
        targets = np.array(y_labels, dtype=np.float32)
        
        # Ensure at least one positive class to prevent BCE loss collapse
        if targets.sum() == 0: targets[0] = 1.0

        final_results['classifier'] = self._optimize_phase(
            lambda t: SemanticStrategies.classifier_strategy(t, inputs, targets, self.device)
        )

        # --- FINALIZATION ---
        self._force_cleanup()
        
        try:
            torch.save(final_results, self.checkpoint_file)
            logger.info(f"[OptimizerService] ✅✅ CHECKPOINT SAVED. Optimization Complete.")
            self.optimization_finished.emit()
        except Exception as e:
            logger.error(f"[OptimizerService] ❌ Failed to save checkpoint: {e}")

    def _optimize_phase(self, objective_fn: Callable, slope_threshold=1e-5, window=25) -> Dict[str, Any]:
        """
        Generic Optimization Loop.
        Handles Study creation, Callbacks, and Resource Cleanup.
        """
        study = optuna.create_study(direction="minimize")
        
        callback = DerivativeConvergenceCallback(
            slope_threshold=slope_threshold, 
            window_size=window, 
            min_trials=35,
            signal_emitter=self.training_finished.emit
        )
        
        try:
            study.optimize(objective_fn, n_trials=self.safety_max_trials, callbacks=[callback])
            best_params = study.best_params
        except Exception as e:
            logger.error(f"[OptimizerService] Phase Failed: {e}")
            best_params = {}
        
        # Cleanup
        del study
        self._force_cleanup()
        
        return best_params