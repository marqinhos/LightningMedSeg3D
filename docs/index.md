# LightningMedSeg3D

```{raw} html
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="_static/wordmark-dark.svg">
    <img src="_static/wordmark-light.svg" alt="LightningMedSeg3D" width="480">
  </picture>
</p>
```

**A configuration-driven PyTorch Lightning framework for reproducible 3D medical
image segmentation benchmarking.**

LightningMedSeg3D trains, evaluates and runs inference for nine convolutional and
transformer architectures under one shared, seed-controlled protocol, driven
entirely by human-readable YAML files and a single console command. It is built
on the [PyTorch Lightning CLI](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html)
and [MONAI](https://monai.io/).

```{code-block} bash
pip install -e .
lightning-medseg3d fit --config configs/btcv/swin_unetr.yaml
```

::::{grid} 2
:gutter: 3

:::{grid-item-card} 🚀 One command per experiment
`fit`, `validate`, `test` and `predict` are exposed by a single entry point;
every run is defined by one YAML file.
:::

:::{grid-item-card} 🧩 Nine architectures, one protocol
U-Net, V-Net, Res-UNet, UNet++, Attention-UNet, UNETR, SwinUNETR, MedFormer and
SegFormer, all built by name through a common factory.
:::

:::{grid-item-card} 🔁 Reproducible by construction
Seed-controlled 70/10/20 splits, a shared preprocessing pipeline and a resolved
config saved with every run.
:::

:::{grid-item-card} 📦 Archived weights
Trained checkpoints for all architectures are archived on
[Zenodo](https://doi.org/10.5281/zenodo.21037952).
:::
::::

```{toctree}
:maxdepth: 2
:caption: Getting started
:hidden:

installation
quickstart
configuration
```

```{toctree}
:maxdepth: 2
:caption: Reference
:hidden:

architectures
datasets
reproducibility
api
```

```{toctree}
:maxdepth: 1
:caption: Project
:hidden:

contributing
```

## Citation

If you use LightningMedSeg3D, please cite the companion benchmark study (see the
repository `CITATION.cff`). The framework is released under the
**GNU Affero General Public License v3.0 (or later)**.
