"""Lightning DataModule shared by BTCV and MSD Task03.

Handles dataset partitioning, MONAI preprocessing pipeline, on-the-fly
data augmentation, sliding-window inference loaders and the construction
of stratified splits documented in the paper.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import lightning.pytorch as pl
import numpy as np
from monai.data import CacheDataset, DataLoader, list_data_collate
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
    RandCropByLabelClassesd,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
    ToTensord,
)


SUPPORTED_DATASETS = {"btcv", "msd_task03"}


class SegmentationDataModule(pl.LightningDataModule):
    """Unified BTCV / MSD Task03 datamodule with reproducible stratified splits."""

    def __init__(
        self,
        dataset: str = "btcv",
        data_root: str = "./data",
        roi_size: tuple[int, int, int] = (96, 96, 96),
        intensity_range: tuple[float, float] = (-175.0, 250.0),
        spacing: tuple[float, float, float] = (1.5, 1.5, 2.0),
        batch_size: int = 2,
        num_workers: int = 4,
        train_frac: float = 0.7,
        val_frac: float = 0.1,
        test_frac: float = 0.2,
        seed: int = 42,
        cache_rate: float = 1.0,
        samples_per_volume: int = 4,
    ) -> None:
        super().__init__()
        if dataset not in SUPPORTED_DATASETS:
            raise ValueError(f"dataset must be one of {SUPPORTED_DATASETS}, got {dataset!r}")
        if abs(train_frac + val_frac + test_frac - 1.0) > 1e-6:
            raise ValueError("train/val/test fractions must sum to 1.0")
        self.save_hyperparameters()

    # ------------------------------------------------------------------
    # split construction
    # ------------------------------------------------------------------
    def _enumerate_cases(self) -> list[dict[str, str]]:
        """Locate (image, label) pairs for the configured dataset."""
        root = Path(self.hparams.data_root)
        if self.hparams.dataset == "btcv":
            img_dir = root / "BTCV" / "imagesTr"
            lab_dir = root / "BTCV" / "labelsTr"
            pattern = "*.nii.gz"
        else:  # msd_task03
            img_dir = root / "Task03_Liver" / "imagesTr"
            lab_dir = root / "Task03_Liver" / "labelsTr"
            pattern = "*.nii.gz"

        images = sorted(p for p in img_dir.glob(pattern) if not p.name.startswith("._"))
        cases = []
        for image_path in images:
            label_path = lab_dir / image_path.name
            if label_path.exists():
                cases.append({"image": str(image_path), "label": str(label_path)})
        if not cases:
            raise FileNotFoundError(
                f"No (image,label) pairs found in {img_dir} -- "
                f"download the {self.hparams.dataset.upper()} cohort first."
            )
        return cases

    def _split(self, cases: list[dict[str, str]]) -> tuple[list, list, list]:
        rng = np.random.default_rng(self.hparams.seed)
        idx = np.arange(len(cases))
        rng.shuffle(idx)
        n = len(cases)
        n_train = int(round(self.hparams.train_frac * n))
        n_val = int(round(self.hparams.val_frac * n))
        train_idx, val_idx, test_idx = (
            idx[:n_train],
            idx[n_train : n_train + n_val],
            idx[n_train + n_val :],
        )
        return (
            [cases[i] for i in train_idx],
            [cases[i] for i in val_idx],
            [cases[i] for i in test_idx],
        )

    # ------------------------------------------------------------------
    # transforms
    # ------------------------------------------------------------------
    def _preprocess(self) -> Compose:
        lo, hi = self.hparams.intensity_range
        return Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=self.hparams.spacing,
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRanged(keys=["image"], a_min=lo, a_max=hi, b_min=0.0, b_max=1.0, clip=True),
                CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True),
                SpatialPadd(keys=["image", "label"], spatial_size=self.hparams.roi_size),
            ]
        )

    def _train_transforms(self) -> Compose:
        # 50/35/15 split between tumour-centred, liver-centred and background
        # samples mirrors the multifocal-lesion sampling used in the MSD
        # tumour evaluation reported in the paper.
        return Compose(
            [
                *self._preprocess().transforms,
                RandCropByLabelClassesd(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=self.hparams.roi_size,
                    ratios=[0.15, 0.35, 0.50] if self.hparams.dataset == "msd_task03" else [0.15, 0.85],
                    num_classes=3 if self.hparams.dataset == "msd_task03" else 2,
                    num_samples=self.hparams.samples_per_volume,
                ),
                RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.1),
                RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.1),
                RandFlipd(keys=["image", "label"], spatial_axis=[2], prob=0.1),
                RandRotate90d(keys=["image", "label"], prob=0.1, max_k=3),
                RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
                ToTensord(keys=["image", "label"]),
            ]
        )

    def _eval_transforms(self) -> Compose:
        return Compose([*self._preprocess().transforms, ToTensord(keys=["image", "label"])])

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------
    def setup(self, stage: Optional[str] = None) -> None:
        cases = self._enumerate_cases()
        train_cases, val_cases, test_cases = self._split(cases)

        if stage in (None, "fit"):
            self.train_ds = CacheDataset(
                data=train_cases,
                transform=self._train_transforms(),
                cache_rate=self.hparams.cache_rate,
                num_workers=self.hparams.num_workers,
            )
            self.val_ds = CacheDataset(
                data=val_cases,
                transform=self._eval_transforms(),
                cache_rate=self.hparams.cache_rate,
                num_workers=self.hparams.num_workers,
            )
        if stage in (None, "test", "validate", "predict"):
            self.test_ds = CacheDataset(
                data=test_cases,
                transform=self._eval_transforms(),
                cache_rate=self.hparams.cache_rate,
                num_workers=self.hparams.num_workers,
            )

    def _loader(self, ds, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=self.hparams.batch_size,
            shuffle=shuffle,
            num_workers=self.hparams.num_workers,
            collate_fn=list_data_collate,
            pin_memory=True,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_ds, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_ds, shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        return self._loader(self.test_ds, shuffle=False)
