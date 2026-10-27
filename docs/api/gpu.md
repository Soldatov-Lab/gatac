# GPU helpers — `gatac.gpu`

The `gatac.gpu` namespace holds the workarounds GATAC itself uses for CuPy and
CUDA driver behaviours at genome scale. They are public so that
[extensions](../extensions.md) built on GATAC can avoid the same problems:
`bincount` never takes CuPy 14's CUB histogram path, which crashes past ~5M
bins (a genome-wide 500 bp tile matrix has 6.2M), and `to_device` copies large
host arrays in 64 MB chunks, so the WSL2 driver does not keep a host buffer
the size of the copy.

```{eval-rst}
.. currentmodule:: gatac.gpu

.. autosummary::
   :toctree: generated/
   :nosignatures:

   bincount
   to_device
```
