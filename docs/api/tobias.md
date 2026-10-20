# Footprinting — `gatac.tobias`

GPU port of the [TOBIAS](https://github.com/loosolab/TOBIAS) pipeline
{cite:p}`Bentsen2020`.
`footprint` corrects the Tn5 insertion bias of each pseudobulk
(TOBIAS `ATACorrect`) and computes footprint scores (`ScoreBigwig`), reading
cut sites straight from the fragments parquet — no BAM or bigwig files.
`bindetect` scans the peaks for motifs and tests for differential binding
between groups (`BINDetect`). The binding sites depend only on the peaks and
motifs: `bindetect_sites` builds them once, and `bindetect(..., sites=)` reuses
them for every set of pseudobulks footprinted on the same peaks.

Every stage matches TOBIAS 0.17.5 exactly when given the same inputs. End to
end, the DWM bias score differs from TOBIAS only in its last bits (floating-point
summation order). That is about half the deviation TOBIAS itself shows when
its bias score is perturbed by 1e-13. `tobias_compat=True` (the default) also
reproduces TOBIAS's quirks; `tobias_compat=False` fixes them:

* the bias model depends on the BAM read length (`read_length`, 50 by
  default);
* for MEME motifs the reverse-strand matrix is rebuilt from
  `round(p × nsites)`, which is coarse for `nsites=1` files such as cisBP
  {cite:p}`Weirauch2014`;
* the per-peak log2FC average actually groups by binding site.

```{eval-rst}
.. currentmodule:: gatac.tobias

.. autosummary::
   :toctree: generated/
   :nosignatures:

   footprint
   bindetect_sites
   bindetect
   FootprintTracks
   SiteSet
   BINDetectResult
```
