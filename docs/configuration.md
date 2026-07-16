# Configuration

Each experiment is a single YAML file with three top-level sections, `trainer`,
`model` and `data`, plus a global `seed_everything`. The Lightning CLI maps each
section onto the constructor of the `Trainer`, the `SegmentationModule` and the
`SegmentationDataModule` respectively.

```yaml
seed_everything: 42

trainer:
  accelerator: gpu
  precision: 16-mixed
  max_epochs: 800
  callbacks:
    - class_path: lightning.pytorch.callbacks.ModelCheckpoint
      init_args: {monitor: val/dice, mode: max, save_top_k: 3, save_last: true}
    - class_path: lightning.pytorch.callbacks.LearningRateMonitor
      init_args: {logging_interval: epoch}
  logger:
    - class_path: lightning.pytorch.loggers.TensorBoardLogger
      init_args: {save_dir: ./runs/btcv/swin_unetr, name: tb}

model:
  architecture: swin_unetr     # factory key (see Architectures)
  in_channels: 1
  num_classes: 2               # includes background
  roi_size: [96, 96, 96]
  sw_batch_size: 4
  sw_overlap: 0.5
  learning_rate: 1.0e-3
  weight_decay: 0.05
  scheduler: cosine            # "cosine" or "none"
  warmup_epochs: 5             # linear warm-up before cosine annealing
  max_epochs: 800
  loss: dice_ce
  nsd_tolerance_mm: 2.0

data:
  dataset: btcv                # "btcv" or "msd_task03"
  data_root: ./data
  roi_size: [96, 96, 96]
  intensity_range: [-175.0, 250.0]
  spacing: [1.5, 1.5, 2.0]
  batch_size: 2
  train_frac: 0.7
  val_frac: 0.1
  test_frac: 0.2
  seed: 42
  samples_per_volume: 4
```

## Model parameters

```{list-table}
:header-rows: 1
:widths: 25 75

* - Key
  - Meaning
* - `architecture`
  - Factory key selecting one of the nine networks.
* - `num_classes`
  - Number of output classes, including background.
* - `roi_size`
  - Sliding-window ROI used for training and inference.
* - `sw_overlap`
  - Sliding-window overlap fraction at inference.
* - `scheduler`
  - `cosine` (linear warm-up of `warmup_epochs` then cosine annealing) or `none`.
* - `loss`
  - Currently `dice_ce` (MONAI `DiceCELoss`, background excluded).
* - `class_weights`
  - Optional per-class loss weights for imbalanced targets.
* - `nsd_tolerance_mm`
  - Boundary tolerance (mm) for the Normalised Surface Dice metric.
```

## Overrides

Any field can be overridden without editing the file:

```bash
lightning-medseg3d fit --config configs/btcv/unet.yaml \
  --model.learning_rate 5e-4 --trainer.max_epochs 300
```

The fully resolved configuration is written back next to each run, so every
experiment remains auditable.
