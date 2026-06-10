"""PyTorch Lightning CLI entry point for LightningMedSeg3D.

This module exposes a single, fully declarative entry point (``main``) that
relies on :class:`lightning.pytorch.cli.LightningCLI` to expose ``fit``,
``validate``, ``test`` and ``predict`` subcommands. The full experiment
description (model, datamodule, optimiser, trainer flags, callbacks) is
driven by a YAML file placed under ``configs/`` -- no per-architecture
Python entry script is required.

Replacement for the legacy ``train.py`` / ``train_sequential.py`` /
``predict_sequential.py`` / ``metrics_sequential.py`` scripts of the prior
``MedicalLiverSegmentationToolKit`` distribution.
"""
from __future__ import annotations

from lightning.pytorch.cli import LightningCLI

from lightning_medseg3d.datamodules.segmentation_datamodule import SegmentationDataModule
from lightning_medseg3d.models.segmentation_module import SegmentationModule


def main() -> None:
    LightningCLI(
        model_class=SegmentationModule,
        datamodule_class=SegmentationDataModule,
        subclass_mode_model=False,
        subclass_mode_data=False,
        save_config_kwargs={"overwrite": True},
        seed_everything_default=42,
    )


if __name__ == "__main__":
    main()
