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
# File: src/agents/peak_classifier_agent.py
# Author: Gabriel Moraes
# Date: 2026-03-02

import os
import json
import logging
import gc
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture

# SYNAPSE Local Models
from src.models.itransformer_lite import iTransformerLite
from src.models.timesnet import TimesNet
from src.models.distilroberta import DistilRobertaSemanticExtractor

logger = logging.getLogger(__name__)

class PeakClassifierAgent(nn.Module):
    """
    Peak Classifier Agent (Extrator de Sazonalidade)
    Implements the ADAGIO pipeline for peak detection with Native Semantic Discovery
    and Active VRAM Management (Chunking & Garbage Collection).
    """

    def __init__(self, itransformer_config: dict, timesnet_config: dict, output_path: str = "peak_schedule.json"):
        super(PeakClassifierAgent, self).__init__()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_path = output_path
        
        # --- EXTREME MLOPS: VRAM PROTECTION ---
        # We override the potentially massive sequence length sent by the Stage.
        # We force a safe chunk size (e.g., 2048 or 4096) that fits easily in 6GB VRAM.
        self.chunk_size = 2048
        itransformer_config['seq_len'] = self.chunk_size
        itransformer_config['pred_len'] = self.chunk_size
        timesnet_config['seq_len'] = self.chunk_size
        timesnet_config['pred_len'] = self.chunk_size
        
        self.semantic_extractor = None
        
        # Step 1: iTransformer - The Conciliator
        self.itransformer = iTransformerLite(
            num_sensors=itransformer_config.get('num_variates', 2),
            d_model=itransformer_config.get('d_model', 32),
            n_heads=itransformer_config.get('n_heads', 2)
        ).to(self.device)
        self.itransformer.eval()
        
        # Step 2: TimesNet - The Time Machine
        timesnet_config.pop("num_kernels", None)
        self.timesnet = TimesNet(**timesnet_config).to(self.device)
        
        # Step 3: Gaussian Mixture Model - The Probabilistic Judge
        self.gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)

        # Step 4: Auxiliary Classifier Head for HPO Tuning (Optuna)
        # TimesNet projects back to enc_in (which is 1 since stress_signal is 1D).
        self.hpo_classifier = nn.Linear(timesnet_config.get('enc_in', 1), 1).to(self.device)
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Optimizer encompassing all neural layers that require tuning
        self.optimizer = torch.optim.Adam([
            {'params': self.timesnet.parameters()},
            {'params': self.hpo_classifier.parameters()}
        ], lr=timesnet_config.get('learning_rate', 1e-3))


        self.day_map = {
            0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday",
            4: "Friday", 5: "Saturday", 6: "Sunday"
        }

    def _get_text_embedding(self, texts: list) -> torch.Tensor:
        """Helper method to extract dense vector embeddings using our local extractor."""
        if self.semantic_extractor is None:
            logger.info("Loading local DistilRobertaSemanticExtractor for Column Discovery...")
            self.semantic_extractor = DistilRobertaSemanticExtractor()
            
        embeddings = []
        for text in texts:
            emb = self.semantic_extractor._get_embedding([text])
            embeddings.append(emb)
        return torch.cat(embeddings, dim=0)

    def _discover_kinematic_columns(self, columns: list) -> tuple:
        """Uses Local DistilRoBERTa to semantically match available dataset columns."""
        if self.semantic_extractor is None:
            logger.info("Loading local DistilRobertaSemanticExtractor for Column Discovery...")
            self.semantic_extractor = DistilRobertaSemanticExtractor()
            
        volume_concept = ["traffic volume, vehicle count, vehicle flow, intensity, amount of cars"]
        speed_concept = ["traffic speed, vehicle velocity, average speed, km/h, mph"]
        
        vol_emb = self._get_text_embedding(volume_concept) 
        spd_emb = self._get_text_embedding(speed_concept)  
        col_embs = self._get_text_embedding(columns)       
        
        vol_scores = F.cosine_similarity(vol_emb.expand_as(col_embs), col_embs, dim=1)
        spd_scores = F.cosine_similarity(spd_emb.expand_as(col_embs), col_embs, dim=1)
        
        best_vol_idx = torch.argmax(vol_scores).item()
        best_spd_idx = torch.argmax(spd_scores).item()
        
        best_vol_col = columns[best_vol_idx]
        best_spd_col = columns[best_spd_idx]
        
        return best_vol_col, best_spd_col

    def _extract_neural_features(self, volume_tensor: torch.Tensor, speed_tensor: torch.Tensor) -> np.ndarray:
        """Passes the raw multivariable data through the neural networks in safe chunks to prevent OOM."""
        self.eval() 
        all_features = []
        total_len = volume_tensor.shape[1]
        
        logger.info(f"Processing {total_len} time steps in chunks of {self.chunk_size}...")
        
        for i in range(0, total_len, self.chunk_size):
            # 1. Sliding Window (Fatiamento)
            vol_chunk = volume_tensor[:, i:i+self.chunk_size]
            spd_chunk = speed_tensor[:, i:i+self.chunk_size]
            
            current_len = vol_chunk.shape[1]
            
            # Pad the last chunk if it's smaller than the expected sequence length
            if current_len < self.chunk_size:
                pad_len = self.chunk_size - current_len
                vol_chunk = F.pad(vol_chunk, (0, pad_len), mode='constant', value=0.0)
                spd_chunk = F.pad(spd_chunk, (0, pad_len), mode='constant', value=0.0)
                
            with torch.no_grad():
                multivariate_input = torch.stack((vol_chunk, spd_chunk), dim=-1).to(self.device)
                
                # Forward Pass
                stress_signal = self.itransformer(multivariate_input)
                timesnet_features = self.timesnet(stress_signal)
                
                # 2. Immediate Offloading to Host RAM
                chunk_result = timesnet_features.mean(dim=-1).cpu().numpy().flatten()
                
                # Truncate padding if it was added
                if current_len < self.chunk_size:
                    chunk_result = chunk_result[:current_len]
                    
                all_features.append(chunk_result)
                
            # 3. Active Garbage Collection (Clear VRAM)
            del multivariate_input, stress_signal, timesnet_features
            torch.cuda.empty_cache()
            gc.collect()
            
        return np.concatenate(all_features)

    def _build_canonical_week(self, df: pd.DataFrame, neural_scores: np.ndarray) -> pd.DataFrame:
        """Groups the historical neural scores by Day of Week and Hour (168 blocks)."""
        df = df.copy()
        df['neural_score'] = neural_scores
        
        if 'timestamp' in df.columns:
            time_col = 'timestamp'
        elif 'event_timestamp' in df.columns:
            time_col = 'event_timestamp'
        else:
            raise ValueError("No valid timestamp column found ('timestamp' or 'event_timestamp').")
            
        df[time_col] = pd.to_datetime(df[time_col])
        df['day_of_week'] = df[time_col].dt.dayofweek
        df['hour'] = df[time_col].dt.hour
        
        canonical_week = df.groupby(['day_of_week', 'hour'])['neural_score'].mean().reset_index()
        return canonical_week

    def _generate_schedule_json(self, canonical_week: pd.DataFrame, predictions: np.ndarray) -> dict:
        """Formats the classified 168 blocks into the required JSON structure."""
        canonical_week['is_peak'] = predictions
        schedule = {day: {} for day in self.day_map.values()}
        
        for day_name in self.day_map.values():
            for h in range(24):
                time_str = f"{h:02d}:00"
                schedule[day_name][time_str] = False
                
        for _, row in canonical_week.iterrows():
            day_name = self.day_map[int(row['day_of_week'])]
            time_str = f"{int(row['hour']):02d}:00"
            schedule[day_name][time_str] = bool(row['is_peak'])
            
        return schedule

    def train_step(self, x_windows: np.ndarray, y_labels: np.ndarray) -> float:
        """
        Executes a forward and backward pass for HPO Hyperparameter Tuning.
        Trains the neural feature extractors against simple binary peak targets.
        """
        self.train()
        
        # 1. Convert to tensors
        x_tensor = torch.tensor(x_windows, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y_labels, dtype=torch.float32).view(-1, 1).to(self.device)
        
        # TimesNet/iTransformer expect [Batch, SeqLen, Channels]
        # x_windows is shape [Batch, SeqLen]. We need bivariate if possible, but during tuning 
        # optimizer_service.py passes a flat 1D sequence window (so Channels=1).
        # We duplicate the channel to match the minimum d_model inputs or 2-channel expectation of stress_projector.
        if len(x_tensor.shape) == 2:
            x_tensor = x_tensor.unsqueeze(-1).expand(-1, -1, 2)
            
        self.optimizer.zero_grad()
        
        # 2. Forward Pass
        with torch.no_grad():
            stress_signal = self.itransformer(x_tensor)
            
        timesnet_features = self.timesnet(stress_signal)
        
        # Global Average Pooling on Sequence Dimension -> [Batch, d_model]
        pooled_features = timesnet_features.mean(dim=1)
        
        # 3. Classify and compute loss
        logits = self.hpo_classifier(pooled_features)
        loss = self.criterion(logits, y_tensor)
        
        # 4. Backward & Step
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item()

    def process_and_classify(self, historical_df: pd.DataFrame):
        """Main pipeline execution."""
        logger.info("Starting Peak Classification Pipeline...")
        
        # 0. Semantic Column Discovery
        available_cols = historical_df.columns.tolist()
        vol_col, spd_col = self._discover_kinematic_columns(available_cols)
        
        logger.info(f"🧠 Semantic NLP identified '{vol_col}' as Volume/Flow.")
        logger.info(f"🧠 Semantic NLP identified '{spd_col}' as Speed/Velocity.")
        
        # 0.5 Data Preparation
        volume_tensor = torch.tensor(historical_df[vol_col].values, dtype=torch.float32).unsqueeze(0)
        speed_tensor = torch.tensor(historical_df[spd_col].values, dtype=torch.float32).unsqueeze(0)
        
        # 1 & 2. Neural Feature Extraction (Now completely VRAM safe)
        logger.info("Extracting neural features via iTransformer and TimesNet...")
        neural_scores = self._extract_neural_features(volume_tensor, speed_tensor)
        
        # 3. Build Canonical Week
        logger.info("Building Canonical Week (Aggregating historical data into 168 blocks)...")
        canonical_week = self._build_canonical_week(historical_df, neural_scores)
        
        # 4. GMM Probabilistic Classification
        logger.info("Applying Gaussian Mixture Model to classify Peaks vs Normal...")
        scores_matrix = canonical_week['neural_score'].values.reshape(-1, 1)
        self.gmm.fit(scores_matrix)
        cluster_labels = self.gmm.predict(scores_matrix)
        
        cluster_means = self.gmm.means_.flatten()
        peak_cluster_index = np.argmax(cluster_means)
        is_peak_array = (cluster_labels == peak_cluster_index)
        
        # 5. Export to JSON
        logger.info("Formatting output and saving to JSON...")
        final_schedule = self._generate_schedule_json(canonical_week, is_peak_array)
        
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(final_schedule, f, indent=4)
            
        logger.info(f"Successfully generated peak schedule at: {self.output_path}")
        return final_schedule