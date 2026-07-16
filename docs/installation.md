# Installation

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.1, PyTorch Lightning ≥ 2.2, MONAI ≥ 1.3
- A CUDA-capable GPU is recommended for training and full-volume inference; the
  package also runs on CPU (for example, the smoke tests).

## Install from source

```bash
git clone https://github.com/Removirt/LightningMedSeg3D.git
cd LightningMedSeg3D
pip install -e .
```

Installing the package registers the `lightning-medseg3d` console command.

To include the development tools (tests, linters, type checker):

```bash
pip install -e ".[dev]"
```

## Data layout

The two supported cohorts are placed under `./data`:

```text
data/
├── BTCV/
│   ├── imagesTr/*.nii.gz
│   └── labelsTr/*.nii.gz
└── Task03_Liver/
    ├── imagesTr/*.nii.gz
    └── labelsTr/*.nii.gz
```

The datasets are public but must be obtained from their providers; see
[Datasets](datasets.md).

## Pre-trained weights

Trained checkpoints for all nine architectures (BTCV and MSD Task03) are archived
on Zenodo: <https://doi.org/10.5281/zenodo.21037952>.
