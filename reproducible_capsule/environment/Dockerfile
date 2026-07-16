# Code Ocean-compatible environment for the LightningMedSeg3D reproducible capsule.
# CPU-only so the capsule is small, portable and runs without a GPU.
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip

# CPU build of PyTorch keeps the image small and hardware-independent.
RUN pip install "torch>=2.1,<2.4" --index-url https://download.pytorch.org/whl/cpu

# Install LightningMedSeg3D. Pin to a tagged release for exact reproducibility;
# `@main` is used here as a placeholder until v1.0.0 is tagged / published to PyPI.
#   Alternative (once on PyPI):  pip install "lightning-medseg3d==1.0.0"
RUN pip install "lightning-medseg3d @ git+https://github.com/Removirt/LightningMedSeg3D.git@main" \
                pytest

WORKDIR /code
