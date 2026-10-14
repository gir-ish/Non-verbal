<div align="center">

# MATA: Non-Verbal Emotion Recognition via Modality-Binding Foundation Models

## Strong Alone, Stronger Together: Synergizing Modality-Binding Foundation Models with Optimal Transport

[![Paper](https://img.shields.io/badge/Paper-IEEE%2010889257-blue)](https://ieeexplore.ieee.org/document/10889257)
[![ArXiv](https://img.shields.io/badge/ArXiv-2409.14221-red)](https://arxiv.org/abs/2409.14221)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-ee4c2c)](https://pytorch.org/)

</div>

---

## 📋 Paper Details

**Title**: Strong Alone, Stronger Together: Synergizing Modality-Binding Foundation Models with Optimal Transport for Non-Verbal Emotion Recognition

**Authors**: Orchid Chetia Phukan, Mohd Mujtaba Akhtar, Girish, Swarup Ranjan Behera, Sishir Kalita, Arun Balaji Buduru, Rajesh Sharma

**Conference**: ICASSP 2025

---

## 🎯 Overview

**MATA** (Modality-binding Audio Temporal Attention) recognizes emotions from non-verbal vocalizations (laughter, sighs, cries, gasps) by synergizing multimodal foundation models through optimal transport-based alignment and intra-modality attention refinement.

### Key Innovation

Hybrid approach combining:
- **Modality-Binding Foundation Models** - Joint audio-visual-text pre-training
- **Optimal Transport Alignment** - Instance-level alignment of features
- **Temporal Attention** - Intra-modality temporal interaction
- **Cross-Foundation Fusion** - Intelligent combination of multiple models

---

## 🔑 Key Features

🎵 **Non-Verbal Focus** - Emotion from prosodic cues, not speech content

🌐 **Modality-Binding** - Leverages audio-visual-text joint embeddings

🚀 **Optimal Transport** - Mathematical rigor in feature alignment

⚡ **Multi-Foundation** - Combines LanguageBind, ImageBind, and specialized models

📊 **State-of-the-Art** - SOTA on ASVP-ESD, JNV, and VIVAE datasets

---

## 📦 Installation

```bash
git clone https://github.com/gir-ish/Non-verbal.git
cd Non-verbal
pip install -r requirements.txt
```

---

## 🚀 Quick Start

```python
from mata import MATAModel

# Initialize MATA model
model = MATAModel(num_emotions=8)

# Load non-verbal audio
features = extract_features('laughter.wav')

# Predict emotion
predictions = model(features)
print(f"Emotion: {predictions['emotion']}")
print(f"Confidence: {predictions['confidence']:.3f}")
```

---

## 📊 Results

**Non-Verbal Emotion Recognition Performance:**

| Dataset | Accuracy | F1-Score |
|---------|----------|----------|
| ASVP-ESD | **76.47%** | **70.35%** |
| JNV | **77.40%** | **76.19%** |
| VIVAE | **75.12%** | **74.63%** |

---

## 📁 Repository Structure

```
Non-verbal/
├── README.md
├── LICENSE
├── setup.py
├── requirements.txt
│
├── mata/
│   ├── __init__.py
│   ├── models.py              # MATA model architecture
│   ├── foundation_models.py    # Multi-foundation integration
│   ├── transport.py            # Optimal transport alignment
│   ├── attention.py            # Temporal attention module
│   ├── fusion.py               # Cross-foundation fusion
│   └── utils.py
│
├── data/
│   ├── __init__.py
│   ├── loaders.py              # Dataset loading
│   └── preprocessor.py         # Feature extraction
│
├── configs/
│   └── default_config.yaml
│
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
│
└── tests/
    └── test_models.py
```

---

## 🏗️ MATA Architecture

```
┌─────────────────────────────┐
│ Non-Verbal Audio Input      │
│ (Laughter, Sighs, Cries)    │
└────────────┬────────────────┘
             │
   ┌─────────┴──────────┐
   │                    │
┌──▼───────┐      ┌────▼─────┐
│LanguageBind     ImageBind   │
│ (Audio-Text)   (Audio-Visual)
└──┬───────┘      └────┬──────┘
   │                   │
   └────────┬──────────┘
            │
   ┌────────▼────────┐
   │ Optimal Transport│
   │   Alignment     │
   └────────┬────────┘
            │
   ┌────────▼────────┐
   │ Temporal Attention
   │    Module       │
   └────────┬────────┘
            │
   ┌────────▼────────┐
   │ Cross-Foundation│
   │    Fusion       │
   └────────┬────────┘
            │
   ┌────────▼────────┐
   │ Emotion Classifier
   └────────┬────────┘
            │
   ┌────────▼────────┐
   │ Emotion Output  │
   └─────────────────┘
```

---

## 🛠️ Core Components

### 1. Modality-Binding Foundation Models
- **LanguageBind**: Audio-text joint embeddings
- **ImageBind**: Audio-visual joint embeddings
- Captures cross-modal emotion cues

### 2. Optimal Transport Alignment
- Instance-level feature distribution alignment
- Wasserstein distance minimization
- Mathematically principled fusion

### 3. Temporal Attention
- Captures temporal dependencies in non-verbal vocalizations
- Multi-head self-attention over time
- Refines intra-modality interactions

### 4. Cross-Foundation Fusion
- Intelligent weighting of different foundation models
- Attention-based model selection
- Residual connections for stability

---

## 📈 Training

```bash
python scripts/train.py \
  --config configs/default_config.yaml \
  --data-dir /path/to/nonverbal/datasets \
  --output-dir ./checkpoints
```

---

## 📊 Evaluation

```bash
python scripts/evaluate.py \
  --checkpoint ./checkpoints/best_model.pt \
  --test-data /path/to/test/set
```

---

## 🔍 Inference

```bash
python scripts/inference.py \
  --audio /path/to/nonverbal_audio.wav \
  --checkpoint ./checkpoints/best_model.pt
```

---

## 📚 Citation

```bibtex
@inproceedings{phukan2025mata,
  title={Strong Alone, Stronger Together: Synergizing Modality-Binding Foundation Models with Optimal Transport for Non-Verbal Emotion Recognition},
  author={Phukan, Orchid Chetia and Akhtar, Mohd Mujtaba and Girish and Behera, Swarup Ranjan and Kalita, Sishir and Buduru, Arun Balaji and Sharma, Rajesh},
  booktitle={Proceedings of ICASSP 2025},
  year={2025}
}
```

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

<div align="center">

**Advanced Non-Verbal Emotion Recognition Through Multimodal Foundation Model Synergy**

[⬆ Back to Top](#mata-non-verbal-emotion-recognition-via-modality-binding-foundation-models)

</div>