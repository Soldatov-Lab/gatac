# Peaks — `gatac.peak`

Call ATAC peaks per cell-type group using the MACS3 algorithm {cite:p}`Zhang2008` under the hood,
merge them into a non-overlapping set, count fragments over peaks, and
identify differentially accessible peaks between groups using a GPU-
accelerated binomial test with Benjamini–Hochberg correction {cite:p}`Benjamini1995`.

```{eval-rst}
.. currentmodule:: gatac.peak

.. autosummary::
   :toctree: generated/
   :nosignatures:

   call
   merge
   make_matrix
   markers
   get_markers
```
