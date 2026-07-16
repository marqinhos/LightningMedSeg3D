<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/_static/wordmark-dark.svg">
    <img src="docs/_static/wordmark-light.svg" alt="LightningMedSeg3D" width="520"/>
  </picture>
</p>

[![License](https://img.shields.io/badge/license-AGPL--3.0-blue.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-readthedocs-8CA1AF.svg)](https://lightningmedseg3d.readthedocs.io)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch Lightning](https://img.shields.io/badge/lightning-CLI-purple.svg)](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html)
[![MONAI](https://img.shields.io/badge/MONAI-1.3%2B-orange.svg)](https://monai.io/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21037952.svg)](https://doi.org/10.5281/zenodo.21037952)

> **📦 Pre-trained weights** — the trained weights for all nine architectures
> (BTCV and MSD Task03) are archived on Zenodo and can be downloaded from
> **<https://doi.org/10.5281/zenodo.21037952>**.

Reproducible 3D medical image segmentation benchmark built on top of the
[PyTorch Lightning CLI](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html)
distribution and the [MONAI](https://monai.io/) framework. Companion code
for the paper "_Comprehensive Evaluation of 3D CNN and Transformer
Architectures for 3D Liver Segmentation in Medical Imaging: A Reproducible,
Externally Validated Benchmark Built on a PyTorch Lightning CLI Pipeline_"
(Informatics in Medicine Unlocked, under revision).

> This repository **replaces** the legacy
> [`MedicalLiverSegmentationToolKit`](https://github.com/Removirt/MedicalLiverSegmentationToolKit)
> code distribution. The previous ad-hoc `train.py`, `train_sequential.py`,
> `predict_sequential.py` and `metrics_sequential.py` scripts have been
> deprecated in favour of a single Lightning CLI entry point.

## Why a Lightning CLI distribution?

* **One entry point, four subcommands** — `fit`, `validate`, `test`,
  `predict` are exposed by Lightning out of the box, so every experiment
  reported in the paper is reproducible with a single command and a
  single YAML file.
* **Architecture, dataset, optimiser and trainer are decoupled** through
  the standard `LightningModule` / `LightningDataModule` interfaces.
* **YAML-driven configuration** removes hidden command-line flags from
  the legacy distribution and makes every reported number auditable.
* **Lightning-managed callbacks** (checkpointing, learning-rate
  monitoring, TensorBoard logging) are configured declaratively.

## Implemented architectures

All nine networks are built from the reference PyTorch implementations
vendored under `lightning_medseg3d.models.backbones` — faithful copies of
the original [`MedicalLiverSegmentationToolKit`](https://github.com/Removirt/MedicalLiverSegmentationToolKit)
models, so the topologies match exactly the ones that produced the
reported results. They are instantiated by name through
`lightning_medseg3d.models.factory.build_model`, with the default
hyper-parameters taken from the toolkit's BTCV configuration files.

| Family | Architecture | `architecture` key | Implementation |
| --- | --- | --- | --- |
| CNN | 3D U-Net | `unet` | `backbones.unet.UNet` (`block='SingleConv'`) |
| CNN | 3D V-Net | `vnet` | `backbones.vnet.VNet` |
| CNN | 3D Res-UNet | `resunet` | `backbones.unet.UNet` (`block='BasicBlock'`) |
| CNN | 3D UNet++ | `unetpp` | `backbones.unetpp.UNetPlusPlus` |
| CNN | 3D Attention-UNet | `attention_unet` | `backbones.attention_unet.AttentionUNet` |
| Transformer | 3D UNETR | `unetr` | `backbones.unetr.UNETR` |
| Transformer | 3D SwinUNETR | `swin_unetr` | `backbones.swin_unetr.SwinUNETR` |
| Transformer | MedFormer 3D | `medformer` | `backbones.medformer.MedFormer` |
| Transformer | 3D SegFormer | `segformer` | `backbones.segformer.SegFormer3D` |

> MONAI is still used for the training utilities (losses, metrics,
> sliding-window inference and data transforms); the UNETR and SwinUNETR
> backbones additionally reuse MONAI's transformer building blocks, as in
> the original toolkit.

## Supported datasets

| Cohort | Task | Subjects | Source |
| --- | --- | --- | --- |
| BTCV | Liver-only | 50 | <https://zenodo.org/records/1169361> |
| MSD Task03 | Liver + hepatic tumour | 201 | <http://medicaldecathlon.com> |

## Installation

```bash
git clone https://github.com/Removirt/LightningMedSeg3D.git
cd LightningMedSeg3D
pip install -e .
```

Datasets must be placed under `./data/BTCV/{imagesTr,labelsTr}` and
`./data/Task03_Liver/{imagesTr,labelsTr}`.

## Reproducing the paper

Every result reported in the paper is reproduced from a single command:

```bash
# BTCV — train SwinUNETR
lightning-medseg3d fit --config configs/btcv/swin_unetr.yaml

# BTCV — evaluate on held-out test set
lightning-medseg3d test --config configs/btcv/swin_unetr.yaml \
    --ckpt_path runs/btcv/swin_unetr/checkpoints/last.ckpt

# MSD Task03 — train Attention-UNet (liver + tumour)
lightning-medseg3d fit --config configs/msd_task03/attention_unet.yaml

# MSD Task03 — run inference on the test set (predicted volumes returned per batch)
lightning-medseg3d predict --config configs/msd_task03/attention_unet.yaml \
    --ckpt_path runs/msd_task03/attention_unet/checkpoints/last.ckpt
```

To swap architectures, simply switch the YAML file. To override any
single hyperparameter without editing the YAML:

```bash
lightning-medseg3d fit --config configs/btcv/swin_unetr.yaml \
    --trainer.max_epochs 200 --data.batch_size 1
```

## Pre-trained weights

The trained model weights for all nine architectures, for both BTCV and MSD
Task03, are archived on Zenodo: <https://doi.org/10.5281/zenodo.21037952>.

## Citation

If you use this toolkit, please cite:

```bibtex
@article{FdezGonzalez2026LiverBenchmark,
  title   = {Comprehensive Evaluation of 3D CNN and Transformer Architectures for 3D Liver Segmentation in Medical Imaging: A Reproducible, Externally Validated Benchmark Built on a PyTorch Lightning CLI Pipeline},
  author  = {Fdez-Gonzalez, Marcos and Nodar-Corral, Lois and Fdez-Vidal, Xose R. and Estevez-Fernandez, Sergio and Comesana, Enrique},
  journal = {Informatics in Medicine Unlocked},
  year    = {2026}
}
```

For software citation (version, license, archived weights DOI), see [`CITATION.cff`](CITATION.cff).
