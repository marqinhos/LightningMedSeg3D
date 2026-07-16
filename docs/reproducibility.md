# Reproducibility

LightningMedSeg3D is designed so that every reported experiment is regenerable
from declared inputs.

- **Single command, single file.** Each run is fully specified by one YAML file
  and one subcommand; the fully resolved configuration is written back with the
  run.
- **Fixed seeds.** `seed_everything` (default 42) and a seeded NumPy generator
  make data splitting and training deterministic.
- **Declared splits.** The 70/10/20 train/validation/test partition is identical
  across architectures for a given seed and fraction triple.
- **Shared protocol.** All architectures use the same module, data module, loss,
  sliding-window inferer and metric stack.
- **Archived weights.** Trained checkpoints for all architectures are archived on
  Zenodo (DOI [10.5281/zenodo.21037952](https://doi.org/10.5281/zenodo.21037952)),
  so published predictions can be reproduced without retraining.

## Reproducible capsule

A Code Ocean–compatible capsule under `reproducible_capsule/` runs, GPU-free, a
synthetic end-to-end demonstration that builds all nine architectures and writes
a NIfTI prediction and a JSON summary. See `reproducible_capsule/REPRODUCING.md`.

```bash
cd reproducible_capsule
docker build -t lightningmedseg3d-capsule -f environment/Dockerfile .
docker run --rm -v "$PWD/results:/results" lightningmedseg3d-capsule bash code/run
```

## Metrics

Validation and testing report three complementary metrics, aggregated per epoch:

- **DSC** — Dice Similarity Coefficient (overlap; higher is better).
- **HD95** — 95th-percentile Hausdorff Distance (boundary; lower is better).
- **NSD** — Normalised Surface Dice at a configurable tolerance (`nsd_tolerance_mm`).
