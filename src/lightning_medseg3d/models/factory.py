"""Architecture factory.

Centralises the construction of the nine 3D networks evaluated in the
paper. Every network is instantiated from the reference PyTorch
implementations vendored under :mod:`lightning_medseg3d.models.backbones`,
which are faithful copies of the *Medical Liver Segmentation Toolkit*
models. The default hyper-parameters below mirror the toolkit's BTCV
configuration files so that the topologies match the ones that produced
the reported results.
"""
from __future__ import annotations

from typing import Sequence

from torch import nn


def build_model(
    architecture: str,
    in_channels: int,
    num_classes: int,
    roi_size: Sequence[int],
) -> nn.Module:
    """Instantiate an architecture by name.

    Parameters
    ----------
    architecture
        One of ``unet``, ``vnet``, ``resunet``, ``unetpp``,
        ``attention_unet``, ``unetr``, ``swin_unetr``, ``medformer``,
        ``segformer``.
    in_channels
        Number of input channels (1 for CT).
    num_classes
        Total number of classes including background.
    roi_size
        Sliding-window ROI used by the network (96^3 in our benchmark).
    """
    arch = architecture.lower()
    img_size = tuple(roi_size)

    # ------------------------------------------------------------------
    # Convolutional U-shaped networks
    # ------------------------------------------------------------------
    if arch == "unet":
        # Original 3D U-Net (config/btcv/unet_3d.yaml).
        from lightning_medseg3d.models.backbones import UNet

        return UNet(
            in_ch=in_channels,
            base_ch=32,
            scale=[[1, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
            kernel_size=[[1, 3, 3], [2, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
            num_classes=num_classes,
            block="SingleConv",
            norm="in",
        )

    if arch in {"resunet", "res_unet"}:
        # Same U-Net topology with residual ``BasicBlock`` units
        # (config/btcv/resunet_3d.yaml).
        from lightning_medseg3d.models.backbones import UNet

        return UNet(
            in_ch=in_channels,
            base_ch=32,
            scale=[[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
            kernel_size=[[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
            num_classes=num_classes,
            block="BasicBlock",
            norm="in",
        )

    if arch == "vnet":
        # V-Net (config/btcv/vnet_3d.yaml).
        from lightning_medseg3d.models.backbones import VNet

        return VNet(
            inChans=in_channels,
            outChans=num_classes,
            scale=[[1, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
            baseChans=16,
        )

    if arch in {"unetpp", "unet++", "unet_plus_plus"}:
        # Nested U-Net++ (config/btcv/unet++_3d.yaml).
        from lightning_medseg3d.models.backbones import UNetPlusPlus

        return UNetPlusPlus(
            in_ch=in_channels,
            base_ch=32,
            scale=[[1, 2, 2], [1, 2, 2], [2, 2, 2], [2, 2, 2]],
            kernel_size=[[1, 3, 3], [1, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
            num_classes=num_classes,
            block="BasicBlock",
            norm="in",
        )

    if arch in {"attention_unet", "attn_unet"}:
        # Attention-UNet with residual blocks (config/btcv/attention_unet_3d.yaml).
        from lightning_medseg3d.models.backbones import AttentionUNet

        return AttentionUNet(
            in_ch=in_channels,
            base_ch=32,
            scale=[[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
            kernel_size=[[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
            num_classes=num_classes,
            block="BasicBlock",
            norm="in",
        )

    # ------------------------------------------------------------------
    # Transformer / hybrid networks
    # ------------------------------------------------------------------
    if arch == "unetr":
        # UNETR (config/btcv/unetr_3d.yaml).
        from lightning_medseg3d.models.backbones import UNETR

        return UNETR(
            in_channels=in_channels,
            out_channels=num_classes,
            img_size=img_size,
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            pos_embed="perceptron",
            norm_name="instance",
            conv_block=True,
            res_block=True,
            dropout_rate=0.0,
        )

    if arch in {"swin_unetr", "swinunetr"}:
        # SwinUNETR (config/btcv/swin_unetr_3d.yaml).
        from lightning_medseg3d.models.backbones import SwinUNETR

        return SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=num_classes,
            feature_size=48,
            num_heads=(3, 6, 12, 24),
            use_checkpoint=True,
        )

    if arch == "medformer":
        # MedFormer (config/btcv/medformer_3d.yaml).
        from lightning_medseg3d.models.backbones import MedFormer

        return MedFormer(
            in_chan=in_channels,
            num_classes=num_classes,
            base_chan=32,
            map_size=[4, 4, 4],
            conv_block="BasicBlock",
            kernel_size=[[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
            scale=[[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
            norm="in",
            act="gelu",
            proj_type="depthwise",
            aux_loss=False,
        )

    if arch in {"segformer", "segformer3d"}:
        # SegFormer 3D (config/btcv/segformer_3d.yaml).
        from lightning_medseg3d.models.backbones import SegFormer3D

        return SegFormer3D(
            in_channels=in_channels,
            sr_ratios=[4, 2, 1, 1],
            embed_dims=[32, 64, 160, 256],
            patch_kernel_size=[7, 3, 3, 3],
            patch_stride=[4, 2, 2, 2],
            patch_padding=[3, 1, 1, 1],
            mlp_ratios=[4, 4, 4, 4],
            num_heads=[1, 2, 5, 8],
            depths=[2, 2, 2, 2],
            decoder_head_embedding_dim=256,
            num_classes=num_classes,
            decoder_dropout=0.0,
        )

    raise ValueError(f"Unknown architecture {architecture!r}")
