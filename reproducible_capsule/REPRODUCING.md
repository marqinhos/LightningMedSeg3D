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
   returns an output of shape `(1, num_classes, 64, 64, 64)`; and
2. runs MONAI sliding-window inference with the 3D U-Net on a synthetic
   `96³` volume and writes:
   - `results/demo_prediction.nii.gz` — the predicted label map, and
   - `results/demo_summary.json` — per-architecture shape report and the
     inference summary.

No GPU, dataset or downloaded checkpoint is required, so the run is deterministic
and hardware-independent (`torch.manual_seed(42)`).

## Run it on Code Ocean

Code Ocean does **not** accept a raw `FROM python:3.10-slim` (or any public
Docker Hub image) as the capsule's base image — it requires building on top of
one of its own registry images (`registry.codeocean.com/codeocean/...`),
selected through its UI. `environment/Dockerfile` in this folder is written for
local `docker build` (see below) and cannot be uploaded to Code Ocean as-is; it
will fail with *"Invalid Dockerfile: Only base Docker images from Code Ocean
are supported."*

To set up the same environment on Code Ocean:

1. Create the capsule (**Import from GitHub**, pointed at this
   `reproducible_capsule/` subfolder, or **Upload** its contents directly).
2. Open the **Environment** tab and pick a **Base Image** from Code Ocean's own
   catalogue (search "Python", choose a Python 3.10/3.11 Ubuntu image).
3. Add the following as **Post-Install** commands (paste as shell commands, or
   use the "+ Add package" pip helper):

   ```bash
   pip install "torch>=2.1,<2.4" --index-url https://download.pytorch.org/whl/cpu
   pip install "lightning-medseg3d @ git+https://github.com/Removirt/LightningMedSeg3D.git@main" pytest
   ```

   (Once the package is published on PyPI, replace the second line with
   `pip install "lightning-medseg3d==1.0.0"`.)
4. Under **Reproducible Run**, set the command to `bash code/run`.
5. Press **Rebuild**, then **Reproducible Run**. Outputs appear under
   `/results`.

If you prefer to keep editing a literal Dockerfile, switch the Environment tab
to **Advanced → Dockerfile**: Code Ocean pre-fills the correct
`registry.codeocean.com/...` `FROM` line for the base image you picked in step
2 — leave that line untouched and add the two `RUN pip install` lines above
beneath it.

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
