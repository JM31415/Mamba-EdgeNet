# Mamba-EdgeNet

Official implementation of:

**Mamba-EdgeNet: Learnable Edge-Guided State Space Model for Skin Lesion Segmentation**

Jiaming Xu, Qi Mao, Lei Qiu, Yu Chen, Gongshuang Tao

Biomedical Physics & Engineering Express, IOP Publishing, 2026.

**DOI:** https://doi.org/10.1088/2057-1976/aeada8

## Overview

Mamba-EdgeNet is a skin lesion segmentation framework that combines a ResNet50-based encoder with state-space modeling and learnable edge guidance.

The implementation includes:

- EfficientMamba2D with resolution-adaptive spatial scanning;
- a learnable edge enhancer based on EF-ALBP;
- shallow edge feature extraction;
- deep edge refinement;
- gated fusion of edge-aware and semantic features.

The released code contains the model architecture, training pipeline, evaluation code, and the differentiable EF-ALBP implementation used in the paper.

## Repository Structure

```text
Mamba-EdgeNet/
├── mamba_edgenet.py
├── ef_albp_torch.py
├── requirements.txt
└── README.md
```

- `mamba_edgenet.py`: Mamba-EdgeNet model, training, and evaluation code.
- `ef_albp_torch.py`: Differentiable EF-ALBP implementation used by the learnable edge enhancer.
- `requirements.txt`: Python dependencies.

## Environment

The experiments reported in the paper were conducted using:

- Python 3.10
- CUDA 12.1
- PyTorch 2.6.0.dev20241112+cu121
- torchvision 0.20.0.dev20241112+cu121
- mamba-ssm 2.2.5
- causal-conv1d 1.5.1

The exact PyTorch build used in our experimental environment was a development build.

Users may use a compatible PyTorch, CUDA, `mamba-ssm`, and `causal-conv1d` environment when reproducing the code.

## Installation

Install PyTorch and torchvision according to your CUDA environment first.

Then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

`mamba-ssm` and `causal-conv1d` should be installed using versions compatible with the local PyTorch and CUDA configuration.

The main Python dependencies are listed in `requirements.txt`.

## Data

The datasets used in the paper are not redistributed in this repository.

The experiments use:

- ISIC 2016
- ISIC 2017
- ISIC 2018
- PH2

Please obtain these datasets from their official sources.

The dataset loaders included in `mamba_edgenet.py` reflect the directory structures used in our experiments. Users may adapt the dataset paths to their own local organization when necessary.

For the default ISIC 2016 loader, the expected directory structure is:

```text
data/isic2016/
├── train/
│   ├── data/
│   └── mask/
└── test/
    ├── data/
    └── mask/
```

Images are expected under the `data` folders and the corresponding segmentation masks under the `mask` folders.

## Usage

The default input resolution is `256 × 256`.

### Training

```bash
python mamba_edgenet.py --mode train --root_path ./data/isic2016
```

### Testing

```bash
python mamba_edgenet.py --mode test --root_path ./data/isic2016
```

By default, checkpoints and outputs are stored under:

```text
./outputs/MambaEdgeNet/res50_mamba/
```

The testing mode expects the trained checkpoint:

```text
res50_mamba_best_model.pth
```

under the corresponding output directory.

Users who only need the model architecture may directly reuse the model definitions in `mamba_edgenet.py` and adapt the data-loading and training code to their own pipeline.

## EF-ALBP

The file `ef_albp_torch.py` contains the differentiable edge-aware filtering and ALBP operations used by the learnable edge-enhancement branch.

The main functions used by Mamba-EdgeNet are:

```python
edge_aware_filtering_torch(...)
albp_torch(...)
```

These operations are integrated into the learnable edge-enhancement module in `mamba_edgenet.py`.

## Reproducibility Notes

The released implementation preserves the model structure and training configuration used in the paper.

Some dataset paths may need to be adapted according to the local dataset organization.

Because `mamba-ssm` and `causal-conv1d` are sensitive to the installed PyTorch and CUDA versions, users should ensure that compatible builds are installed for their environment.

The original experimental environment used:

```text
PyTorch 2.6.0.dev20241112+cu121
torchvision 0.20.0.dev20241112+cu121
CUDA 12.1
mamba-ssm 2.2.5
causal-conv1d 1.5.1
```

## Citation

If you find this work useful, please cite:

```bibtex
@article{Xu2026MambaEdgeNet,
  title   = {Mamba-EdgeNet: Learnable Edge-Guided State Space Model for Skin Lesion Segmentation},
  author  = {Xu, Jiaming and Mao, Qi and Qiu, Lei and Chen, Yu and Tao, Gongshuang},
  journal = {Biomedical Physics \& Engineering Express},
  year    = {2026},
  doi     = {10.1088/2057-1976/aeada8}
}
```

Paper:

https://doi.org/10.1088/2057-1976/aeada8

## Acknowledgment

This repository provides the research implementation associated with the Mamba-EdgeNet paper.

Dataset copyrights remain with their respective owners.
