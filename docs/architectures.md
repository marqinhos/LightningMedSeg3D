# Architectures

All networks are instantiated by name through
`lightning_medseg3d.models.factory.build_model`. Adding an architecture requires
only registering it in the factory; the training and data code are unchanged.

```{list-table}
:header-rows: 1
:widths: 14 30 26 30

* - Family
  - Architecture
  - `architecture` key
  - Backbone class
* - CNN
  - 3D U-Net
  - `unet`
  - `UNet` (`SingleConv`)
* - CNN
  - 3D V-Net
  - `vnet`
  - `VNet`
* - CNN
  - 3D Res-UNet
  - `resunet`
  - `UNet` (`BasicBlock`)
* - CNN
  - 3D UNet++
  - `unetpp`
  - `UNetPlusPlus`
* - CNN
  - 3D Attention-UNet
  - `attention_unet`
  - `AttentionUNet`
* - Transformer
  - 3D UNETR
  - `unetr`
  - `UNETR`
* - Transformer
  - 3D SwinUNETR
  - `swin_unetr`
  - `SwinUNETR`
* - Hybrid
  - MedFormer 3D
  - `medformer`
  - `MedFormer`
* - Transformer
  - 3D SegFormer
  - `segformer`
  - `SegFormer3D`
```

The backbones are faithful, vendored PyTorch implementations, so topologies match
those used to produce the reported benchmark results. The U-Net and Res-UNet
share one implementation and differ only by residual block type; UNETR and
SwinUNETR reuse MONAI transformer building blocks.

## Example

```python
from lightning_medseg3d.models.factory import build_model

model = build_model(
    architecture="swin_unetr",
    in_channels=1,
    num_classes=2,
    roi_size=(96, 96, 96),
)
```
