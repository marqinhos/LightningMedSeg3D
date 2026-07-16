# Reproducing LightningMedSeg3D

This folder is a [Code Ocean](https://codeocean.com)-compatible **reproducible
capsule** for LightningMedSeg3D. It gives a reviewer a one-click, GPU-free way
to confirm that the software installs and runs end to end.

## Layout

```
reproducible_capsule/
├── environment/Dockerfile   # CPU-only Python 3.10 image; installs the package
├── code/
│   ├── run                  # reproducible-run entry point (Code Ocean calls this)
│   └── demo_predict.py      # builds all 9 architectures + runs inference (CPU)
├── data/README.md           # how to add real weights (Zenodo) and datasets
├── results/                 # outputs are written here (Code Ocean mounts /results)
└── metadata/metadata.yml    # capsule metadata
```

## What the reproducible run does

`code/run` executes `code/demo_predict.py`, which:

1. builds every one of the nine architectures (`unet`, `vnet`, `resunet`,
   `unetpp`, `attention_unet`, `unetr`, `swin_unetr`, `medformer`, `segformer`)
   through `lightning_medseg3d.models.factory.build_model` and asserts each
   returns an output of shape `(1, num_classes, 32, 32, 32)`; and
2. runs MONAI sliding-window inference with the 3D U-Net on a synthetic
   `96³` volume and writes:
   - `results/demo_prediction.nii.gz` — the predicted label map, and
   - `results/demo_summary.json` — per-architecture shape report and the
     inference summary.

No GPU, dataset or downloaded checkpoint is required, so the run is deterministic
and hardware-independent (`torch.manual_seed(42)`).

## Run it on Code Ocean

Upload this folder as a capsule (or import the GitHub repository), keep
`environment/Dockerfile` as the environment and `code/run` as the reproducible
run, then press **Reproducible Run**. Outputs appear under `/results`.

## Run it locally with Docker

```bash
cd reproducible_capsule
docker build -t lightningmedseg3d-capsule -f environment/Dockerfile .
docker run --rm -v "$PWD/results:/results" lightningmedseg3d-capsule bash code/run
```

## Run it locally without Docker

```bash
pip install "torch>=2.1" && pip install lightning-medseg3d   # or: pip install -e ..
RESULTS_DIR=./results python code/demo_predict.py
```

## Reproducing the published benchmark

The synthetic demo validates the software, not the reported accuracy. To
regenerate the published numbers, add the archived weights and the public
cohorts as described in [`data/README.md`](data/README.md) and run the
`lightning-medseg3d test`/`predict` commands from the main README.
