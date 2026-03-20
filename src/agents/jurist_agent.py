# SYNAPSE - A Gateway of Intelligent Perception for Traffic Management
# Copyright (C) 2025 Noxfort Labs
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
# File: src/agents/jurist_agent.py
# Author: Gabriel Moraes
# Date: 2026-02-14

import os
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, Any, Optional
import gc

from src.agents.base_agent import BaseAgent

class JuristAgent(BaseAgent):
    """
    The Jurist Agent ('O Juiz').
    
    Responsibilities:
    1. XAI Generation: Explains technical decisions in natural/legal language.
    2. Regulatory Compliance: Maps anomalies to Brazilian Traffic Code (CTB) infractions.
    
    Refactored V5 (Qwen3 1.7B FP16):
    - Uses local Model Vault Qwen3 1.7B.
    - Implements Thinking Mode (CoT) with temperature=0 for reasoning.
    - Adapts output language to UI context.
    """

    def __init__(self, model_id: Optional[str] = None):
        """
        Initializes the agent but DOES NOT load the model immediately.
        """
        # We pass None as model initially because we want lazy loading
        super().__init__(model=None, name="JuristAgent")
        
        if model_id is None:
            # Default to the local Model Vault
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            self.model_id = os.path.join(base_dir, "Model Vault", "qwen3_1.7B")
        else:
            self.model_id = model_id

        self.tokenizer = None
        self.is_loaded = False

    def _get_current_device(self) -> torch.device:
        """
        Introspects the model to find its actual physical location.
        """
        if not self.is_loaded or self.model is None:
            return torch.device("cpu")
            
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def load_resources(self):
        """
        Explicitly loads the Heavy LLM into VRAM in Native FP16 mode.
        """
        if self.is_loaded:
            return

        print(f"[{self.name}] Loading LLM ({self.model_id}) in FP16 Mode...")
        
        try:
            # Tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            
            # Model Loading - NATIVE FP16
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                device_map="auto",
                torch_dtype=torch.float16, # Forced FP16
                trust_remote_code=True
            )
            
            self.is_loaded = True
            print(f"[{self.name}] LLM Loaded successfully (FP16).")
            
        except Exception as e:
            print(f"[{self.name}] Failed to load LLM: {e}")
            self.is_loaded = False

    def unload_resources(self):
        """
        Frees VRAM by moving the model to CPU and deleting references.
        """
        if not self.is_loaded:
            return

        print(f"[{self.name}] Unloading LLM to free VRAM...")
        
        if self.model:
            self.model.cpu()
        
        del self.model
        del self.tokenizer
        
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def inference(self, input_data: Dict[str, Any]) -> str:
        """
        Standard Interface: Generates a verdict.
        Input Dict should contain: 'context', 'anomaly_data', 'source_id', 'language'
        """
        return self.generate_verdict(input_data)

    def generate_verdict(self, context_data: Dict[str, Any]) -> str:
        """
        Constructs the prompt and runs the LLM generation safely.
        """
        if not self.is_loaded:
            self.load_resources()

        device = self._get_current_device()
        
        # 1. Prompt Engineering
        source = context_data.get('source_id', 'Unknown')
        values = context_data.get('values', [])
        status = context_data.get('status', 'NORMAL')
        language = context_data.get('language', 'pt_BR').lower()
        
        if 'en' in language:
            system_prompt = (
                "You are the Jurist Agent of the SYNAPSE system. Your role is to analyze technical "
                "traffic data and issue explanatory reports based on the local Traffic Code. "
                "Be technical, direct, and legal. Justify AI decisions based on the provided vectors.\n"
                "IMPORTANT: You MUST first think step-by-step using <think>...</think> tags, and then provide your final answer."
            )
            user_prompt = (
                f"Incident Report:\n"
                f"- Source: {source}\n"
                f"- Detected State: {status}\n"
                f"- Numerical Data: {values}\n\n"
                f"Analyze this scenario. If there is an anomaly, cite the applicable traffic code article and suggest the control action."
            )
        elif 'fr' in language:
            system_prompt = (
                "Vous êtes l'Agent Juriste du système SYNAPSE. Votre rôle est d'analyser les données techniques "
                "de trafic et d'émettre des rapports explicatifs basés sur le code de la route local. "
                "Soyez technique, direct et juridique. Justifiez les décisions de l'IA sur la base des vecteurs fournis.\n"
                "IMPORTANT: Vous DEVEZ d'abord réfléchir étape par étape à l'aide des balises <think>...</think>, puis fournir votre réponse finale."
            )
            user_prompt = (
                f"Rapport d'incident:\n"
                f"- Source: {source}\n"
                f"- État détecté: {status}\n"
                f"- Données numériques: {values}\n\n"
                f"Analysez ce scénario. S'il y a une anomalie, citez l'article du code de la route applicable et suggérez l'action de contrôle."
            )
        elif 'es' in language:
            system_prompt = (
                "Usted es el Agente Jurista del sistema SYNAPSE. Su función es analizar datos técnicos "
                "de tráfico y emitir informes explicativos basados en el código de circulación local. "
                "Sea técnico, directo y jurídico. Justifique las decisiones de la IA en base a los vectores proporcionados.\n"
                "IMPORTANTE: DEBE primero pensar paso a paso usando las etiquetas <think>...</think>, y luego proporcionar su respuesta final."
            )
            user_prompt = (
                f"Informe de Incidente:\n"
                f"- Fuente: {source}\n"
                f"- Estado Detectado: {status}\n"
                f"- Datos Numéricos: {values}\n\n"
                f"Analice este escenario. Si hay una anomalía, cite el artículo del código de circulación aplicable y sugiera la acción de control."
            )
        elif 'ru' in language:
            system_prompt = (
                "Вы юрист-агент системы SYNAPSE. Ваша задача - анализировать технические данные о дорожном движении "
                "и выдавать пояснительные отчеты на основе местных правил дорожного движения. "
                "Будьте техничны, прямолинейны и юридически точны. Обосновывайте решения ИИ на основе предоставленных векторов.\n"
                "ВАЖНО: Вы ДОЛЖНЫ сначала подумать шаг за шагом, используя теги <think>...</think>, а затем предоставить свой окончательный ответ."
            )
            user_prompt = (
                f"Отчет об инциденте:\n"
                f"- Источник: {source}\n"
                f"- Обнаруженное состояние: {status}\n"
                f"- Числовые данные: {values}\n\n"
                f"Проанализируйте этот сценарий. Если есть аномалия, назовите применимую статью правил дорожного движения и предложите действие по контролю."
            )
        elif 'zh' in language:
            system_prompt = (
                "您是 SYNAPSE 系统的法务代理（Jurist Agent）。您的职责是分析技术交通数据，并根据当地交通法规发布解释性报告。"
                "请保持专业、直接和法律属性。基于提供的向量证明 AI 的决策。\n"
                "重要提示：您必须首先使用 <think>...</think> 标签逐步思考，然后提供您的最终答案。"
            )
            user_prompt = (
                f"事件报告：\n"
                f"- 来源：{source}\n"
                f"- 检测状态：{status}\n"
                f"- 数值数据：{values}\n\n"
                f"请分析此场景。如果存在异常，请引用适用的交通法规条款并建议控制措施。"
            )
        else: # Default to pt_BR
            system_prompt = (
                "Você é o Agente Jurista do sistema SYNAPSE. Sua função é analisar dados técnicos "
                "de tráfego e emitir laudos explicativos baseados no Código de Trânsito Brasileiro (CTB). "
                "Seja técnico, direto e jurídico. Justifique as decisões da IA com base nos vetores fornecidos.\n"
                "IMPORTANTE: Você DEVE primeiro pensar passo-a-passo usando tags <think> e </think>, e em seguida fornecer sua resposta final."
            )
            user_prompt = (
                f"Relatório de Incidente:\n"
                f"- Fonte: {source}\n"
                f"- Estado Detectado: {status}\n"
                f"- Dados Numéricos: {values}\n\n"
                f"Analise este cenário. Se houver anomalia, cite o artigo do CTB aplicável e sugira a ação de controle."
            )
            
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # 2. Tokenization & Formatting
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Ensure inputs are on the same device as the model
        model_inputs = self.tokenizer([text], return_tensors="pt").to(device)

        # 3. Generation (With FP16 optimization enabled by torch_dtype)
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=512,
                temperature=0.0, # Temperature 0 for coherent logical reasoning
                do_sample=False
            )
            
        # 4. Decode
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # 5. Extract Final Output (Hide Thinking)
        response_clean = re.sub(r'(?s)<think>.*?</think>', '', response).strip()
        
        # Fallback if the model forgot to close the tag but still used it
        if '<think>' in response_clean:
            parts = response_clean.split('</think>', 1)
            if len(parts) > 1:
                response_clean = parts[1].strip()
            else:
                # If there's a think tag without a closing one, we might have truncated output.
                # Just remove everything after <think> if we can't find </think>. But the model 
                # might only generate the thought. So we return what we can.
                response_clean = response_clean.replace('<think>', '').strip()
                
        return response_clean

    def train_step(self, batch_data: Any) -> float:
        """
        Jurist Agent does not train in the real-time loop.
        """
        return 0.0