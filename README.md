# SYNAPSE

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20Ubuntu%2022.04%2B-lightgrey)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)]()
[![CUDA](https://img.shields.io/badge/CUDA-12.x-76B900?logo=nvidia&logoColor=white)]()
[![Status](https://img.shields.io/badge/Status-Production%20Ready-green)]()

> **A Gateway of Intelligent Perception for Traffic Management**

---

**SYNAPSE** is a real-time, AI-powered traffic perception platform built by **Noxfort Systems**. It ingests sensor data from urban intersections, fuses it with global traffic intelligence (Waze, TomTom), and produces physics-validated traffic state estimations that are transmitted to the traffic light controller **CARINA** via gRPC.

The system treats traffic as a **macroscopic fluid-dynamic phenomenon** — not as individual vehicles. All processing operates on aggregated metrics: flow (veh/min), average speed (m/s), occupancy (0–1), and queue length.

---

## Neural Architecture Stack

SYNAPSE employs **11 specialized neural models** orchestrated by a multi-agent system across 3 perception levels.

### Level 1 — Local Perception (Per-Sensor)

| Model | Architecture | Purpose |
|-------|-------------|---------|
| **TCN** | Temporal Convolutional Network with dilated causal convolutions, weight normalization, and residual connections. Exponentially increasing dilation (1, 2, 4, 8…) captures long-range dependencies without recurrence | Deep temporal feature extraction per sensor. Each `SpecialistAgent` maintains one TCN that learns the unique temporal signature of its assigned intersection |
| **NeuroSymbolic** | Frozen DistilRoBERTa (82M params) → TCN-Autoencoder. The transformer extracts semantic embeddings from sensor logs; the TCN-AE learns to reconstruct "normal" patterns | Semantic-physical validation. High reconstruction error = anomalous log (correct grammar but impossible physics). Used by the `LinguistAgent` as a gatekeeper before any sensor is trusted |
| **VAE-TCN** | Variational Autoencoder with TCN backbone. Encoder → μ/σ → reparameterization → Decoder. KL divergence regularizes the latent space | Noise filtering and data reconstruction. The `CorrectorAgent` uses the generative prior to clean raw signals while preserving the statistical structure (Golden Dataset integrity) |

### Level 2 — Regional/Spatial Perception

| Model | Architecture | Purpose |
|-------|-------------|---------|
| **GATv2 Lite** | 2-layer Graph Attention Network v2 (Brody et al., 2022) using `torch_geometric`. Multi-head attention (4 heads) over road topology. Nodes = intersections, Edges = roads | Spatial reasoning. The `CoordinatorAgent` propagates information across the road graph, allowing upstream congestion to influence downstream predictions |
| **SinkhornCrossAttention** | Siamese GATv2 encoder (shared weights) → Bidirectional Cross-Attention → Sinkhorn doubly-stochastic normalization (10 iterations). Produces differentiable 1-to-1 node alignment matrices | Map matching for the `CartographerAgent`. Given a GPS trace with ambiguity (e.g., parallel roads), this model resolves which road segment the sensor belongs to. Cascading architecture: low entropy → fast heuristic (CPU, μs); high entropy → neural disambiguation (GPU, ms) |

### Level 3 — Global Perception (Multivariate Fusion)

| Model | Architecture | Purpose |
|-------|-------------|---------|
| **iTransformer** | Inverted Transformer (Liu et al., 2024). Instead of treating time steps as tokens, it treats **each sensor as a token** and embeds its full time series as the feature vector. Standard Transformer Encoder then learns cross-sensor correlations via self-attention | Global traffic state fusion. The `FuserAgent` builds a "Perfect Snapshot" of the present traffic state by attending to all sensor histories simultaneously. This is the primary output model |
| **TimesNet** | FFT frequency analysis → Top-K period extraction → 1D-to-2D reshape → Inception-style 2D convolutions. Captures multi-scale periodic patterns (rush hours, weekly cycles) with native AMP support | Periodic pattern extraction for the `TranslationAgent`. Discovers latent temporal structures (morning/evening peaks, weekend patterns) from raw signal without manual annotation |

### Generative & Security Models

| Model | Architecture | Purpose |
|-------|-------------|---------|
| **TimeGAN** | GRU-based GAN (Yoon et al., NeurIPS 2019). Four components: Embedder, Recovery, Generator (with Supervisor), Discriminator. All operate in latent space with Xavier initialization | Synthetic data generation for the `ImputerAgent`. Fills long data gaps (NaNs) with statistically realistic sequences that preserve the temporal dynamics of real traffic |
| **WaveletAE-OCC** | Wavelet Scattering Transform (Kymatio) → Autoencoder → One-Class Classifier (Deep SVDD). Adaptive threshold via EMA (μ + 2σ covers 95% of normal data) | Anomaly detection for the `AuditorAgent`. The wavelet front-end provides translation-invariant features; the OCC detects distributional drift indicating sensor tampering or failure |
| **DistilRoBERTa** | 82M parameter Transformer (Sanh et al., 2019). Loaded as frozen backbone via HuggingFace `AutoModel` | Shared semantic encoder. Provides contextual embeddings for the `LinguistAgent` (validation) and `TranslationAgent` (concept extraction). Never fine-tuned — only the task-specific heads are trained |

### Explainability

| Model | Architecture | Purpose |
|-------|-------------|---------|
| **Qwen 3 1.7B-Instruct thinking** | 31.7B parameter causal LLM (FP16). Loaded via HuggingFace Transformers | The `JuristAgent` translates mathematical decisions into human-readable legal reports compliant with the Brazilian Traffic Code (CTB). Runs on-demand, not in the inference loop |

---

## Multi-Agent System: MARKVART™

**M**odular **A**daptive **R**easoning **K**ernel with **V**ariational **A**ttention & **R**ecurrent **T**ransformers.

Eleven decoupled agents, each with a single responsibility:

| Agent | Model | Responsibility |
|-------|-------|---------------|
| � **Auditor** | WaveletAE-OCC | Anomaly detection, sensor integrity verification |
| 🗺️ **Cartographer** | SinkhornCrossAttention + FastMapMatcher | GPS → Map Matching (which road is this sensor on?) |
| 🧠 **Coordinator** | GATv2 Lite | Spatial reasoning, propagates traffic state across the city graph |
| 🧹 **Corrector** | VAE-TCN + Z-Score | Noise filtering, Golden Dataset reconstruction |
| 📸 **Fuser** | iTransformer | Spatio-temporal fusion → "Perfect Snapshot" of traffic state |
| 🧩 **Imputer** | TimeGAN | Synthetic time-series generation for missing data |
| ⚖️ **Jurist** | Qwen 3 1.7B-Instruct thinking | XAI: Technical decisions → legal-grade reports |
| 📖 **Linguist** | NeuroSymbolic (DistilRoBERTa + TCN-AE) | Semantic-physical validation of new sensors |
| 🏔️ **PeakClassifier** | DistilRoBERTa + GMM | Classifies temporal peaks (morning/evening rush) |
| 🎯 **Specialist** | TCN + Task Decoders | Per-sensor temporal feature extraction |
| 🔤 **Translation** | DistilRoBERTa + TimesNet + FTT | Semantic extraction from textual logs, periodic pattern discovery |

### Sensor Onboarding Pipeline (Linguist)

When a new sensor connects, it enters a **Zero Trust** quarantine:

```
QUARANTINE → Buffer 60 packets → Train TCN-AE (learn signal grammar)
  → Pass: Validate physics (speed ≥ 0, occupancy ∈ [0,1])
    → Pass: Transfer encoder to Specialist → ACTIVE ✅
    → Fail: REJECTED ❌ (possible injection attack)
  → Fail: Retry with 60 more samples (up to 3 attempts)
```

No sensor produces output until the Linguist validates both its **grammar** (internal consistency) and **physics** (domain constraints).

---

## Operational Phases: ADAGIO™

**A**utomated **D**ata **A**nalysis, **G**eneration & **I**nference **O**rchestration.

| Phase | Mode | What Happens |
|-------|------|-------------|
| **Phase 0** | Optimization | Optuna-based hyperparameter search (Calculus-Based AutoML). Requires ≥1 local + ≥1 global live sensor |
| **Phase 1** | Offline Bootstrap | Golden Dataset generation from historical .parquet. Peak detection using hybrid DistilRoBERTa + iTransformer + GMM pipeline |
| **Phase 2** | Online Runtime | All agents loaded. 4-thread architecture: Neural inference (1s heartbeat), Linguist (standby on quarantine events), XAI (on-demand ephemeral threads), Ingestion (Push gateway port 8080 + Pull HTTP pollers) |

### Mandatory Prerequisites

- **Map file** (`.net.xml`): SUMO network topology defining intersections, roads, and connections
- **Historical data** (`.parquet`): Minimum 7 days of aggregated traffic metrics
- **Live sources**: ≥1 LOCAL sensor (camera/radar at a specific intersection) + ≥1 GLOBAL API (Waze/TomTom covering the city)

---

## Graceful Degradation

Three fallback layers ensure traffic control is **never interrupted**:

| Level | System | Strategy | Latency |
|-------|--------|----------|---------|
| 🧠 **L0** | MARKVART™ | Full neural inference with all 9 agents | ~1s cycles |
| 📊 **L1** | AFB (Baseline Fusion Algorithm) | Statistical fusion (weighted average, median, trimmed mean) over raw sensor data. Per-sensor Z-score pre-filtering via `SensorGuard` | < 10ms |
| 📈 **L2** | MEH (Historical State Module) | Lookup of pre-computed average profiles indexed by (hour, day_of_week) from the Golden Dataset | < 1ms |

Degradation is transparent to CARINA — the output format is identical regardless of which level is active.

---

## Communication: HFT-Link

Enriched output is transmitted to CARINA via **HFT-Link**:
- **Protocol:** gRPC over HTTP/2 with Protobuf serialization
- **Security:** Application-level validation (VPN recommended for production deployment)
- **Latency:** < 250ms end-to-end
- **Reliability:** Auto-reconnect with exponential backoff

---

## Security & Compliance

| Pillar | Implementation |
|--------|---------------|
| **Zero Trust Ingestion** | Every sensor starts in Quarantine. Unknown source IDs are silently rejected. Promotion requires Linguist validation |
| **Physical Validation** | Negative speeds, occupancy > 1.0, or impossible queue lengths automatically invalidate the source |
| **Explainability (XAI)** | Jurist Agent produces CTB-compliant audit reports. All decisions are traceable and reproducible |
| **Data Sovereignty** | No data leaves the municipal territory. Logs enable full retroactive auditing |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **Language** | Python 3.12 |
| **Deep Learning** | PyTorch 2.x, CUDA 12.x, cuDNN, TensorFloat32 |
| **NLP** | HuggingFace Transformers (DistilRoBERTa, Qwen 31.7B-Instruct thinking) |
| **Graph Neural Networks** | PyTorch Geometric (GATv2Conv) |
| **Signal Processing** | Kymatio (Wavelet Scattering), PyTorch FFT |
| **HPO** | Optuna (Tree-structured Parzen Estimator) |
| **UI** | PyQt6 |
| **Communication** | gRPC, Protobuf |
| **Data** | Apache Parquet, SUMO .net.xml |

---

## � Pre-trained Models (Model Vault)

Due to Git file size limits, the pre-trained neural network tensors and PyTorch files are not included in this repository. 
To run SYNAPSE, you must download the foundational models separately.

**Download the primary Language Model here:**
- [HuggingFace Hub (Qwen3-1.7B)](https://huggingface.co/Qwen/Qwen3-1.7B)

**Installation:**
After downloading the model files, place them inside the root directory under the `Model Vault/` folder matching the correct path:
`SYNAPSE_CORE/Model Vault/qwen3_1.7B/`

---

## �📄 License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)** — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <sub>Built with ❤️ by <strong>Noxfort Systems</strong></sub>
</p>