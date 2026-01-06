# SFP-xLSTM_demo1
SFP_xLSTM: A_fusion_fault_diagnosis_framework_of_sparse_focus_modulation_and_xLSTM_for_vibration_signals

Implementation of the paper: **"SFP-xLSTM: A fusion fault diagnosis framework of sparse focus modulation and xLSTM for vibration signals"**

This code base contains the code used in the paper, including the baseline model for comparison and improvement, the paper model demo, and some databases. It also includes other experimental parts and result pictures not mentioned in the paper, which will be used for subsequent research and deepening.
## 📋 Overview

SFP-xLSTM is a novel fault diagnosis framework that integrates:
- **Gramian Angular Difference Fields (GADF)** for signal-to-image encoding
- **Sparse Focus Modulation (SFPM)** for multi-scale feature extraction
- **Extended LSTM (xLSTM)** for long-term temporal dependency modeling

### Key Features
- ✅ 98.12% accuracy on CWRU dataset
- ✅ 98.47% accuracy on PU dataset  
- ✅ 87.83% accuracy at -4 dB SNR (strong noise robustness)
- ✅ 5.6 ms inference time (real-time capable)
- ✅ ~2.3M parameters (lightweight)

## 🏗️ Architecture

```
Input Signal (1D) 
    ↓
┌─────────────────────┐
│   GADF Encoding     │  → 2D Image (256×256)
└─────────────────────┘
    ↓
┌─────────────────────┐
│   SFPM Module       │  → Multi-scale Features
│  (L=3, ρ=0.3)       │
└─────────────────────┘
    ↓
┌─────────────────────┐
│   Adaptive Pool     │  → Sequence (P²×C)
└─────────────────────┘
    ↓
┌─────────────────────┐
│   xLSTM Module      │  → Temporal Features
│  (sLSTM + mLSTM)    │
└─────────────────────┘
    ↓
┌─────────────────────┐
│   Classifier        │  → Fault Categories
└─────────────────────┘
```


## 🔧 Installation

```bash
# Clone repository
git clone https://github.com/your-repo/sfp-xlstm.git
cd sfp-xlstm

# Install dependencies
pip install torch numpy scikit-learn tqdm matplotlib
```

## 📊 Model Parameters (Table 4 from Paper)

| Module | Parameter | Value |
|--------|-----------|-------|
| GAF | Image Size | 256 × 256 |
| GAF | Encoding Method | GADF |
| SFPM | Number of Focus Layers (L) | 3 |
| SFPM | Base Kernel Size (k) | 3 |
| SFPM | Dilation Rate Growth Factor | 2 |
| SFPM | Sparsity Rate (ρ) | 0.3 |
| xLSTM | Hidden Dimension | 256 |
| xLSTM | Number of Memory Subspaces (Ns) | 4 |
| xLSTM | Dropout Rate | 0.2 |
| Training | Batch Size | 32 |
| Training | Learning Rate | 0.003 |
| Training | Epochs | 150 |
| Training | Optimizer | Adam |

## 🚀 Quick Start

### 1. Basic Model Usage

```python
from models.sfp_xlstm import create_model, GADFEncoder
import torch

# Create model with paper parameters
model = create_model(num_classes=10)

# Prepare input (GADF image)
x = torch.randn(1, 1, 256, 256)  # (batch, channel, height, width)

# Forward pass
output = model(x)
prediction = torch.argmax(output, dim=1)
```

### 2. GADF Encoding

```python
import numpy as np
from models.sfp_xlstm import GADFEncoder

# Create encoder
encoder = GADFEncoder(image_size=256)

# Convert 1D signal to 2D GADF image
signal = np.random.randn(1024)  # 1D vibration signal
gadf_image = encoder.encode(signal)  # 256×256 GADF matrix
```

### 3. Training

```python
from train import run_experiment

# Run experiment on CWRU dataset (Task A)
run_experiment(task='A', num_classes=10, num_runs=10)

# Run with noise (Task C)
run_experiment(task='C', num_classes=10, snr_db=-4, num_runs=10)
```

## 📈 Experimental Results

### Task A: CWRU Dataset

| Method | Accuracy (%) | Precision (%) | Recall (%) | F1 (%) | Time (ms) |
|--------|-------------|---------------|------------|--------|-----------|
| 1D-CNN | 92.45 ± 1.23 | 91.87 ± 1.34 | 92.13 ± 1.18 | 92.00 ± 1.21 | 2.3 |
| LSTM | 91.28 ± 1.56 | 90.65 ± 1.67 | 91.02 ± 1.45 | 90.83 ± 1.52 | 3.8 |
| Shift-deformable | 96.73 ± 0.65 | 96.41 ± 0.68 | 96.58 ± 0.62 | 96.49 ± 0.64 | 11.3 |
| **SFP-xLSTM (Ours)** | **98.12 ± 0.42** | **98.03 ± 0.45** | **98.08 ± 0.41** | **98.05 ± 0.43** | **5.6** |

### Task C: Noise Robustness (DataCastle Dataset)

| Method | 10 dB | 5 dB | 0 dB | -4 dB |
|--------|-------|------|------|-------|
| 1D-CNN | 85.23% | 78.56% | 68.92% | 52.34% |
| Shift-deformable | 91.45% | 86.78% | 79.12% | 66.34% |
| **SFP-xLSTM (Ours)** | **98.63%** | **97.12%** | **94.56%** | **87.83%** |

## 🔬 Key Formulas

### GADF Encoding (Eq. 3)
```
G_GADF[i,j] = sin(θᵢ - θⱼ)
```

### Sparsification (Eq. 12-13)
```
S = K ⊙ TopK(M)
K = max(⌊ρ·H·W⌋, K_min)
```

### xLSTM Matrix Memory Update (Eq. 21)
```
C_{t+1} = f_t·C_t + i_t·v_t·k_t^T
```

### Composite Loss (Eq. 36)
```
L = L_CE + λ₁·L_sparse + λ₂·L_temporal
```

## 📖 Citation

```bibtex
@article{guan2025sfpxlstm,
  title={SFP-xLSTM: A fusion fault diagnosis framework of sparse focus modulation and xLSTM for vibration signals},
  author={Guan, Yubo and Li, Peng and Zhao, Aiying and Wang, Shilin},
  journal={TBD},
  year={2025}
}


## 🙏 Acknowledgments

This work was supported by:
- Lanzhou Science and Technology Plan Project (2025-GN-1, 2025-3-002)
- Science and Technology Program of Gansu Province (24JRRA287)
