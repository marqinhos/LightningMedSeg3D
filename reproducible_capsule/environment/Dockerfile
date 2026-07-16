# Environment for the LightningMedSeg3D reproducible capsule.
# CPU-only so the capsule is small, portable and runs without a GPU.
#
# NOTE for Code Ocean: this Dockerfile is meant for local `docker build` (see
# REPRODUCING.md). Code Ocean rejects a public Docker Hub base image such as
# `python:3.10-slim` uploaded directly ("Invalid Dockerfile: Only base Docker
# images from Code Ocean are supported"). On Code Ocean, pick a Python base
# image via the Environment tab's Base Image picker instead, and add the
# `pip install` lines below as Post-Install commands -- see REPRODUCING.md
# for the exact steps.
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
