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
# File: src/models/distilroberta.py
# Author: Gabriel Moraes
# Date: 2026-02-28

import torch
import logging
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Tuple
from safetensors.torch import save_file

# Enable Tensor Cores globally for matrix multiplications and cuDNN operations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class DistilRobertaSemanticExtractor:
    def __init__(self, model_name: str = "sentence-transformers/all-distilroberta-v1", similarity_threshold: float = 0.80):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.similarity_threshold = similarity_threshold
        
        self.learned_concepts: Dict[str, torch.Tensor] = {}
        self.concept_examples: Dict[str, List[str]] = {}
        self.concept_counter = 0
        
        self.logger.info(f"Loading agnostic NLP embedding model: {model_name} with AMP & Tensor Cores...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.model.eval()
            self.logger.info(f"Model loaded successfully on {self.device}.")
        except Exception as e:
            self.logger.error(f"Failed to load embedding model: {e}")
            self.model = None

    def _get_embedding(self, texts: List[str]) -> torch.Tensor:
        """Generates a mean-pooled sentence embedding using AMP for accelerated inference."""
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(self.device)
        
        # Use Automatic Mixed Precision (AMP) to leverage Tensor Cores and accelerate inference
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        with torch.no_grad():
            with torch.autocast(device_type=device_type, dtype=torch.float16 if device_type == 'cuda' else torch.bfloat16):
                model_output = self.model(**encoded_input)
            
        attention_mask = encoded_input['attention_mask']
        token_embeddings = model_output[0] 
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask
        
        cluster_embedding = torch.mean(mean_pooled, dim=0, keepdim=True)
        return F.normalize(cluster_embedding, p=2, dim=1)

    def learn_and_map_semantics(self, unique_values: List[str]) -> Tuple[str, bool]:
        """
        Evaluates new strings, compares them to known concepts via cosine similarity.
        Returns the concept ID and a boolean indicating if a NEW concept was created.
        """
        if not self.model or not unique_values:
            return "unknown_concept", False

        try:
            clean_texts = [str(val).strip().lower() for val in unique_values if str(val).strip()]
            if not clean_texts:
                return "empty_data", False

            current_embedding = self._get_embedding(clean_texts)
            best_match_concept = None
            highest_similarity = -1.0

            for concept_id, concept_vector in self.learned_concepts.items():
                # Ensure vectors are compared on the same device
                similarity = F.cosine_similarity(current_embedding, concept_vector.to(self.device)).item()
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    best_match_concept = concept_id

            if highest_similarity >= self.similarity_threshold:
                self.logger.debug(f"Mapped to existing {best_match_concept} (Similarity: {highest_similarity:.2f})")
                
                new_examples = [ex for ex in clean_texts[:3] if ex not in self.concept_examples[best_match_concept]]
                self.concept_examples[best_match_concept].extend(new_examples)
                
                return best_match_concept, False # False means it is NOT a new concept

            self.concept_counter += 1
            new_concept_id = f"Semantic_Concept_{self.concept_counter}"
            
            # Store learned concepts on CPU to preserve VRAM for active model processing
            self.learned_concepts[new_concept_id] = current_embedding.cpu()
            self.concept_examples[new_concept_id] = clean_texts[:5]
            
            self.logger.info(f"Discovered new semantic pattern. Created: {new_concept_id}")
            return new_concept_id, True # True means a new concept was created

        except Exception as e:
            self.logger.error(f"Error during semantic embedding extraction: {e}")
            return "error_in_extraction", False

    def get_ontology_report(self) -> Dict[str, List[str]]:
        """Returns the dictionary of dynamically learned concepts and their typical examples."""
        return self.concept_examples

    def save_ontology(self, filepath: str) -> None:
        """Saves the learned mathematical embeddings to a physical safetensors file."""
        if not self.learned_concepts:
            self.logger.warning("No concepts learned yet. Skipping safetensors export.")
            return
            
        try:
            # Safetensors requires contiguous memory tensors on CPU
            tensors_to_save = {k: v.contiguous().cpu() for k, v in self.learned_concepts.items()}
            save_file(tensors_to_save, filepath)
            self.logger.info(f"Successfully serialized {len(tensors_to_save)} concepts to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save ontology to safetensors: {e}")