# Reliability-Weighted Supervision with Class Frequency Correction for Pediatric Brain Tumor Segmentation

<div align="center">
  <h3>Reliability-Weighted Supervision with Class Frequency Correction for Pediatric Brain Tumor Segmentation</h3>
  
  [![BraTS](https://img.shields.io/badge/BraTS-2026-0066cc.svg)](https://www.synapse.org/#!Synapse:syn53708249/wiki/627703)
  [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
  [![Python](https://img.shields.io/badge/Python-3.8+-3776ab.svg)](https://www.python.org/)
  [![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)](https://pytorch.org/)
</div>

## 📖 Overview

This repository contains the official implementation of **URW-PEDs** (Uncertainty Reliability Weighting for Pediatric brain tumors), a framework for pediatric brain tumor segmentation that adapts training supervision using epistemic uncertainty and class frequency correction to handle annotation ambiguity and severe class imbalance.

### Key Contributions

- **Reliability-Weighted Supervision**: Combines epistemic uncertainty (from cross-validation fold disagreement) and inverse class frequency correction to calibrate per-voxel loss contributions.
- **Addresses Two Failure Modes**: 
  - *Uncertain supervision* in ambiguous boundaries → down-weighted via epistemic uncertainty.
  - *Blind failure* in underrepresented classes → up-weighted via frequency correction.
- **No Architectural Changes**: Built on nnU-Net v2 without modifications to network, preprocessing, or inference.
- **State-of-the-Art Results**: Achieves significant improvements on BraTS 2026 Pediatric validation set, with CC showing +20.3% improvement in DSC.

### 🎯 Performance Highlights

| Method | DSC (CC) | DSC (ET) | DSC (All Lesions) |
|--------|----------|----------|-------------------|
| nnU-Net Baseline | 0.0984 | 0.4808 | 0.9414 |
| **URW-PEDs (Ours)** | **0.1184** | **0.5081** | **0.9518** |
| **Improvement** | **+20.3%** | **+5.7%** | **+1.1%** |

*Note: ED (edema) shows 0.0000 due to extreme underrepresentation (only 818 voxels in training set). This is a data limitation, not a failure of the method.*

## 📋 Table of Contents

- [Overview](#-overview)
- [Methodology](#-methodology)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results](#-results)
- [Repository Structure](#-repository-structure)
- [Citation](#-citation)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

## 🧠 Methodology

### Problem Statement

Pediatric brain tumor segmentation faces unique challenges:
- **Rarity of cases**: Limited annotated data.
- **Heterogeneous morphology**: High variability across tumor subtypes.
- **Annotation uncertainty**: Significant inter-rater disagreement, especially in subregions like Cystic Component (CC) and Edema (ED).
- **Severe class imbalance**: Some subregions (e.g., ED) are drastically underrepresented.

### Our Approach

#### 1. Epistemic Uncertainty Estimation

We train 5-fold cross-validation models on nnU-Net, then perform full-dataset inference with all five fold models to obtain per-voxel softmax probabilities. The epistemic uncertainty map for class `c` is the inter-fold variance:

$$u_i^c = Var_k[p_i^(k,c)]$$

High variance indicates regions where models disagree → likely annotation-uncertain boundaries.

A brain mask $M_i$ derived from the union of foreground predictions suppresses spurious variance outside the brain.

#### 2. Reliability Weight Computation

For each case $i$ and class $c$:

- **Absent class detection**: If the class is absent in ground truth (during training) or has mean variance below threshold $τ = 1e-6$ (test-time), assign minimum weight `ε = 0.05`.
- **Epistemic weight**: Normalize mean variance across present classes and invert:

  $$w_i^{c,epis} = 1 - normalized_variance_i^c$$
  
- **Frequency correction**: Compute inverse class frequency from mean true positive voxel counts:
  
  $$w^{c,freq} = (1/f_c) / max(1/f_c')$$
  
- **Combined weight**: Blend both signals with balancing parameter $α = 0.5$:
  
  $$w_i^c = α * w_i^{c,epis} + (1-α) * w^{c,freq}$$
  
  Clipped to $[ε, 1.0]$.

#### 3. Weighted Loss Function

The reliability weight is broadcast to all voxels of class $c$ in a training patch, and applied to both cross-entropy and Dice loss:


$$L_rel = 1/|P| Σ(w_i^c * L_CE) + 1/C Σ(1 - (2Σ(w_i^c * p^c * y^c) + ε)/(Σ(w_i^c * (p^c + y^c)) + ε))$$


The weights are fixed after computation; they are not updated online.

### 🧪 Key Findings

| Configuration | Mean Dice | Δ from Baseline |
|---------------|-----------|-----------------|
| nnU-Net (E1) | 0.4251 | — |
| Pure Epistemic (E2a) | 0.4550 | +0.0299 |
| **Epistemic + Frequency (E2b)** | **0.4709** | **+0.0458** |
| Pure Aleatoric (E4a) | 0.4283 | +0.0032 |
| Aleatoric + Frequency (E4b) | 0.4330 | +0.0079 |
| Spatial Epistemic (E3a) | 0.4471 | +0.0220 |
| Spatial Epistemic + Freq (E3b) | 0.4520 | +0.0269 |

**Key Insight**: Combination of epistemic uncertainty and frequency correction provides complementary benefits. Spatial weighting underperforms due to within-patch gradient destabilization.

## 🚀 Installation

### Prerequisites

- Python 3.10+
- CUDA 11.0+ (for GPU acceleration)
- NVIDIA GPU with 16GB+ VRAM (recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/zephyr-9598/BraTS-2026-PEDs.git
cd BraTS-2026-PEDs

# Create conda environment
conda create -n brats_ped python=3.10
conda activate brats_ped

# Install dependencies
pip install -r requirements.txt

# Install nnU-Net v2
pip install nnunetv2
```

### Dependencies

```
torch>=1.10.0
nnunetv2>=2.0.0
numpy>=1.21.0
scipy>=1.7.0
SimpleITK>=2.2.0
batchgenerators>=0.25
```

## 📊 Usage

### Data Preparation

1. Organize your BraTS 2026 Pediatric dataset in the following structure:

```
dataset/
├── imagesTr/
│   ├── case_0000.nii.gz  # T1
│   ├── case_0001.nii.gz  # T1ce
│   ├── case_0002.nii.gz  # T2
│   └── case_0003.nii.gz  # FLAIR
├── labelsTr/
│   └── case.nii.gz        # Segmentation labels (ET=1, NET=2, CC=3, ED=4)
└── dataset.json           # Dataset configuration
```

2. Preprocess the dataset using nnU-Net:

```bash
nnUNetv2_plan_and_preprocess -d DATASET_ID -verify_dataset_integrity
```

### Training

#### Step 1: Train Baseline 5-Fold Models

```bash
Use the nnUNetv2_train command to train the baseline
```

#### Step 2: Compute Reliability Weights

```bash
Use the custom trainer mode at E3b
nnUNetv2_train DATASET_ID configuration fold -tr nnUNet_reliabilityweights
```

#### Step 3: Train with Reliability-Weighted Supervision

```bash
# Train with epistemic + frequency weighting (E2b - best)
Use the custom trainer mode at E3b
nnUNetv2_train DATASET_ID configuration fold -tr nnUNet_reliabilityweights

# Training with aleatoric weights can be done using the aleatoric script if required
```

### Inference

```bash
Use the inference command generated at inference.txt
```

## 📈 Results

### Official BraTS 2026 Validation Results

| Label | Global DSC (Baseline) | Global DSC (Ours) | Δ |
|-------|-----------------------|-------------------|----|
| **ET (1)** | 0.4808 | **0.5081** | **+5.7%** |
| **NET (2)** | 0.9139 | **0.9248** | **+1.2%** |
| **CC (3)** | 0.0984 | **0.1184** | **+20.3%** |
| **ED (4)** | 0.0000 | 0.0000 | — |
| **ET+NET+CC** | 0.9413 | **0.9563** | **+1.6%** |
| **All Lesions** | 0.9414 | **0.9518** | **+1.1%** |

*ED score remains zero due to extreme underrepresentation in training data (only 818 voxels). This is a data limitation, not a method failure.*


## 📁 Repository Structure

```
BraTS-2026-PEDs/
├── src/
│   ├── custom_trainer.py         # Custom nnUNetTrainer with combined weighting
│   ├── uncertainty.py            # Epistemic and aleatoric uncertainty estimation
│   ├── loss.py                   # Reliability-weighted loss functions
│   ├── weights.py                # Combined weight computation
│   └── utils.py                  # Utility functions
├── scripts/
│   ├── compute_weights.py        # Compute reliability weights from fold models
│   ├── train.py                  # Training script
│   ├── inference.py              # Inference script
│   └── evaluate.py               # Evaluation script
├── configs/
│   ├── baseline.yaml             # Baseline nnU-Net configuration
│   ├── epistemic.yaml            # Epistemic-only weighting
│   └── combined.yaml             # Epistemic + frequency correction (recommended)
├── experiments/
│   ├── baseline/                 # Baseline experiment results
│   └── combined/                 # Combined weighting experiment results
├── assets/
│   ├── qualitative_comparison.png
│   └── architecture.png
├── tests/
│   ├── test_uncertainty.py
│   ├── test_loss.py
│   └── test_weights.py
├── requirements.txt
├── setup.py
├── LICENSE
└── README.md
```

## 📝 Citation

If you find this work useful for your research, please cite:

```bibtex
@article{paravila2026reliability,
  title={Reliability-Weighted Supervision with Class Frequency Correction for Pediatric Brain Tumor Segmentation},
  author={Paravila, Ajesh Saviour},
  journal={BraTS 2026 Challenge},
  year={2026},
  url={https://github.com/zephyr-9598/BraTS-2026-PEDs}
}
```

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **National Health Research Institutes (NHRI), Taiwan**: For providing computational resources and support
- **Dr. Maxim Solovchuk**: For guidance and supervision
- **BraTS 2026 Organizers**: For organizing the challenge and providing the dataset
- **nnU-Net Team**: For their excellent framework

## 🔗 Related Links

- [BraTS 2026 Challenge](https://www.synapse.org/#!Synapse:syn53708249/wiki/627703)
- [nnU-Net GitHub](https://github.com/MIC-DKFZ/nnUNet)
- [BraTS-PEDs Paper](https://arxiv.org/abs/2404.15009)

## 📧 Contact

**Ajesh Saviour Paravila**
- Email: ajesh.saviour@gmail.com
- GitHub: [@zephyr-9598](https://github.com/zephyr-9598)
- Institute: Institute of Biomedical Engineering and Nanomedicine, National Health Research Institutes, Taiwan

---

<div align="center">
  Made with ❤️ by Ajesh Saviour Paravila
</div>
