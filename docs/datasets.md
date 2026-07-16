# Datasets

The data module supports two public abdominal-CT cohorts, selected by the
`dataset` field.

```{list-table}
:header-rows: 1
:widths: 20 50 15 15

* - Cohort
  - Task (reported target)
  - Classes
  - `dataset`
* - BTCV
  - Liver (multi-organ CT)
  - 2
  - `btcv`
* - MSD Task03
  - Liver (+ hepatic tumour during training)
  - 3
  - `msd_task03`
```

On MSD Task03 the tumour label is used during training to help resolve the liver
in the presence of lesions; the number of classes includes the background.

## Obtaining the data

- **BTCV** — <https://www.synapse.org/#!Synapse:syn3193805>
- **MSD Task03 (Liver)** — <http://medicaldecathlon.com>

Place the volumes under `data/BTCV/{imagesTr,labelsTr}` and
`data/Task03_Liver/{imagesTr,labelsTr}` (see [Installation](installation.md)).

## Preprocessing

The shared preprocessing pipeline (applied identically at every stage) reorients
to RAS, resamples to a configurable voxel spacing (default `1.5 × 1.5 × 2.0` mm),
clips intensities to a soft-tissue window (default `[-175, 250]` HU) with linear
normalisation to `[0, 1]`, crops the foreground and pads to the ROI. Training
adds label-aware patch sampling, random flips, random 90° rotations and random
intensity shifts.

## Splits

Partitioning uses a seeded NumPy generator, so the 70/10/20
train/validation/test split is deterministic and identical across architectures
for a given `seed`.
