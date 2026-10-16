# Motifs — `gatac.motif`

Read motifs from MEME-format files, test for over-representation in peak
sets, run GSEA on motif rankings, and score per-cell TF activity with
chromVAR.

```{eval-rst}
.. currentmodule:: gatac.motif

.. autosummary::
   :toctree: generated/
   :nosignatures:

   read
   DNAMotif
   sample_gc_matched_background
   enrichment
   presence_matrix
   enrichment_regression
   gsea
```

## chromVAR

Compute transcription-factor activity deviation scores following the chromVAR
algorithm.  All compute-intensive steps are executed on GPU.

```{eval-rst}
.. autosummary::
   :toctree: generated/
   :nosignatures:

   chromvar
   compute_peak_bias
   sample_bg_peaks
   scan
   compute_deviations
```

To run the four steps individually, see the docstring of
`compute_deviations` (which lists them end-to-end).
