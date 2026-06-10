"""Lightning module wrapper for 3D segmentation networks.

All nine architectures (U-Net, V-Net, Res-UNet, Attention-UNet, UNet++,
UNETR, SwinUNETR, MedFormer, SegFormer) are accessed by name and built
through ``models.factory.build_model``. The optimiser, learning-rate
schedule and loss are exposed as constructor arguments so that they can
be configured entirely from YAML via the PyTorch Lightning CLI.
"""
from __future__ import annotations

from typing import Any, Optional

import lightning.pytorch as pl
import torch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric, SurfaceDiceMetric
from monai.transforms import AsDiscrete, Compose, EnsureType
from torch import nn

from lightning_medseg3d.models.factory import build_model


class SegmentationModule(pl.LightningModule):
    """LightningModule for 3D voxel-wise multi-class segmentation."""

    def __init__(
        self,
        architecture: str = "swin_unetr",
        in_channels: int = 1,
        num_classes: int = 2,
        roi_size: tuple[int, int, int] = (96, 96, 96),
        sw_batch_size: int = 4,
        sw_overlap: float = 0.5,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        scheduler: str = "cosine",
        warmup_epochs: int = 5,
        max_epochs: int = 500,
        loss: str = "dice_ce",
        class_weights: Optional[list[float]] = None,
        nsd_tolerance_mm: float = 2.0,
    ) -> None:
        super().__init__()
        # ``save_hyperparameters`` exposes every constructor arg in
        # ``self.hparams`` and to Lightning's logger.
        self.save_hyperparameters()

        self.net: nn.Module = build_model(
            architecture=architecture,
            in_channels=in_channels,
            num_classes=num_classes,
            roi_size=roi_size,
        )

        if loss == "dice_ce":
            self.loss_fn = DiceCELoss(
                to_onehot_y=True,
                softmax=True,
                include_background=False,
                lambda_dice=1.0,
                lambda_ce=1.0,
                weight=torch.tensor(class_weights) if class_weights else None,
            )
        else:
            raise ValueError(f"Unknown loss type {loss!r}")

        self.post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=num_classes)])
        self.post_label = Compose([EnsureType(), AsDiscrete(to_onehot=num_classes)])

        self.dice_metric = DiceMetric(include_background=False, reduction="mean")
        self.hd_metric = HausdorffDistanceMetric(include_background=False, percentile=95.0)
        self.nsd_metric = SurfaceDiceMetric(
            include_background=False,
            class_thresholds=[nsd_tolerance_mm] * (num_classes - 1),
        )

    # ------------------------------------------------------------------
    # forward / shared step
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def _shared_step(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        images, labels = batch["image"], batch["label"]
        logits = self.net(images)
        loss = self.loss_fn(logits, labels)
        return loss, logits, labels

    # ------------------------------------------------------------------
    # train / val / test
    # ------------------------------------------------------------------
    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss, _, _ = self._shared_step(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch["image"].shape[0])
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        images, labels = batch["image"], batch["label"]
        logits = sliding_window_inference(
            inputs=images,
            roi_size=tuple(self.hparams.roi_size),
            sw_batch_size=self.hparams.sw_batch_size,
            predictor=self.net,
            overlap=self.hparams.sw_overlap,
        )
        loss = self.loss_fn(logits, labels)

        preds = [self.post_pred(p) for p in logits]
        ys = [self.post_label(y) for y in labels]
        self.dice_metric(y_pred=preds, y=ys)
        self.hd_metric(y_pred=preds, y=ys)
        self.nsd_metric(y_pred=preds, y=ys)

        self.log("val/loss", loss, on_epoch=True, prog_bar=True, batch_size=images.shape[0])

    def on_validation_epoch_end(self) -> None:
        dice = self.dice_metric.aggregate().mean().item()
        hd = self.hd_metric.aggregate().mean().item()
        nsd = self.nsd_metric.aggregate().mean().item()
        self.dice_metric.reset()
        self.hd_metric.reset()
        self.nsd_metric.reset()
        self.log_dict(
            {"val/dice": dice, "val/hd95": hd, "val/nsd": nsd},
            prog_bar=True,
        )

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self) -> None:
        self.on_validation_epoch_end()

    def predict_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> dict[str, Any]:
        images = batch["image"]
        logits = sliding_window_inference(
            inputs=images,
            roi_size=tuple(self.hparams.roi_size),
            sw_batch_size=self.hparams.sw_batch_size,
            predictor=self.net,
            overlap=self.hparams.sw_overlap,
        )
        return {
            "logits": logits,
            "preds": logits.argmax(dim=1),
            "meta": batch.get("meta", None),
        }

    # ------------------------------------------------------------------
    # optimisation
    # ------------------------------------------------------------------
    def configure_optimizers(self) -> dict[str, Any]:
        optimiser = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        if self.hparams.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimiser, T_max=self.hparams.max_epochs
            )
        elif self.hparams.scheduler == "none":
            return {"optimizer": optimiser}
        else:
            raise ValueError(f"Unknown scheduler {self.hparams.scheduler!r}")
        return {"optimizer": optimiser, "lr_scheduler": scheduler}
