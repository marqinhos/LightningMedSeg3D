# Contributing

Contributions are welcome. The framework is designed to be extended at
well-defined seams.

## Development setup

```bash
git clone https://github.com/Removirt/LightningMedSeg3D.git
cd LightningMedSeg3D
pip install -e ".[dev]"
pytest -q            # forward-shape smoke tests for all architectures
ruff check src tests
```

Continuous integration (GitHub Actions) installs the package on Python 3.10 and
3.11, lints with `ruff` and runs the test suite.

## Add an architecture

1. Add the backbone under `src/lightning_medseg3d/models/backbones/`.
2. Register it by name in `build_model` (`models/factory.py`).
3. (Optional) ship a YAML config under `configs/<dataset>/`.

Because `SegmentationModule` accesses the network only through `build_model`, no
other code changes are needed.

## Add a dataset

Extend the case enumeration in
`datamodules/segmentation_datamodule.py` and, where needed, the label-aware
sampling ratios.

## Extension points

The `losses/`, `metrics/` and `callbacks/` subpackages are reserved namespaces
for project-specific extensions (for example a NIfTI prediction-writer callback
or a per-case CSV exporter).

## License

By contributing you agree that your contributions are licensed under the
**GNU Affero General Public License v3.0 (or later)**.
