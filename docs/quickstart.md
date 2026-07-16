# Quickstart

Every experiment is a subcommand plus a configuration file.

## Train

```bash
# Train SwinUNETR on BTCV (liver-only)
lightning-medseg3d fit --config configs/btcv/swin_unetr.yaml
```

Training writes checkpoints, TensorBoard logs and the fully resolved
configuration under the run directory declared in the YAML file.

## Evaluate

```bash
lightning-medseg3d test \
  --config configs/btcv/swin_unetr.yaml \
  --ckpt_path runs/btcv/swin_unetr/checkpoints/last.ckpt
```

`test` reports the Dice Similarity Coefficient (DSC), the 95th-percentile
Hausdorff Distance (HD95) and the Normalised Surface Dice (NSD) on the held-out
test split.

## Predict

```bash
lightning-medseg3d predict \
  --config configs/msd_task03/attention_unet.yaml \
  --ckpt_path runs/msd_task03/attention_unet/checkpoints/last.ckpt
```

`predict` runs sliding-window inference and returns, per batch, the predicted
logits and the discrete label map. To persist predictions as NIfTI files, attach
a Lightning [`BasePredictionWriter`](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.BasePredictionWriter.html)
callback in the configuration.

## Switch architecture or dataset

Switching architecture or dataset only requires switching the configuration
file; single fields can be overridden on the command line:

```bash
lightning-medseg3d fit \
  --config configs/btcv/swin_unetr.yaml \
  --trainer.max_epochs 200 --data.batch_size 1
```
