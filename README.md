# Mamba-EdgeNet

Official implementation of:

**Mamba-EdgeNet: Learnable Edge-Guided State Space Model for Skin Lesion Segmentation**

Jiaming Xu, Qi Mao, Lei Qiu, Yu Chen, Gongshuang Tao

Biomedical Physics & Engineering Express, IOP Publishing, 2026.

**DOI:** https://doi.org/10.1088/2057-1976/aeada8

## Overview

Mamba-EdgeNet is a skin lesion segmentation framework that combines a
ResNet50-based encoder with state-space modeling and learnable edge guidance.
The implementation includes the EfficientMamba2D module, learnable edge
enhancement, shallow and deep edge modeling, and gated edge-semantic fusion.

## Repository Structure

```text
Mamba-EdgeNet/
├── mamba_edgenet.py
├── ef_albp_torch.py
├── requirements.txt
└── README.md

- mamba_edgenet.py: Mamba-EdgeNet model, training, and evaluation code.
- ef_albp_torch.py: differentiable EF-ALBP implementation used by the
  learnable edge enhancer.
- requirements.txt: Python dependencies.
Environment
The experiments reported in the paper were conducted using:
- Python 3.10
- CUDA 12.1
- PyTorch 2.6.0.dev20241112+cu121
- torchvision 0.20.0.dev20241112+cu121
- mamba-ssm 2.2.5
- causal-conv1d 1.5.1
The exact PyTorch build used in our experimental environment was a development
build. Users may use a compatible PyTorch/CUDA environment when reproducing
the code.
Installation
Install PyTorch and torchvision according to your CUDA environment first.
Then install the remaining dependencies:
pip install -r requirements.txt

mamba-ssm and causal-conv1d should be installed using versions compatible
with the local PyTorch and CUDA configuration.
Data
The datasets used in the paper are not redistributed in this repository.
The experiments use ISIC 2016, ISIC 2017, ISIC 2018, and PH2. Please obtain
the datasets from their official sources and adapt the dataset paths to your
local directory structure when necessary.
For the default ISIC 2016 loader, the expected structure is:
data/isic2016/
├── train/
│   ├── data/
│   └── mask/
└── test/
    ├── data/
    └── mask/

Usage
The default input resolution is 256 × 256.
Training:
python mamba_edgenet.py --mode train --root_path ./data/isic2016

Testing:
python mamba_edgenet.py --mode test --root_path ./data/isic2016

Checkpoints and outputs are stored under:
./outputs/MambaEdgeNet/res50_mamba/

The dataset loaders included in the source code reflect the directory
structures used in our experiments. Users may adapt these loaders and paths
to their own dataset organization.
Citation
If you find this work useful, please cite:
@article{Xu2026MambaEdgeNet,
  title   = {Mamba-EdgeNet: Learnable Edge-Guided State Space Model for Skin Lesion Segmentation},
  author  = {Xu, Jiaming and Mao, Qi and Qiu, Lei and Chen, Yu and Tao, Gongshuang},
  journal = {Biomedical Physics \& Engineering Express},
  year    = {2026},
  doi     = {10.1088/2057-1976/aeada8}
}

Acknowledgment
This repository provides the research implementation associated with the
Mamba-EdgeNet paper. Dataset copyrights remain with their respective owners.
