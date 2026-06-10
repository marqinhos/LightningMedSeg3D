"""Model registry and factory."""
from lightning_medseg3d.models.factory import build_model
from lightning_medseg3d.models.segmentation_module import SegmentationModule

__all__ = ["build_model", "SegmentationModule"]
