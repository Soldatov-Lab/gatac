# Plotting — `gatac.pl`

The `gatac.pl` namespace draws the standard diagnostic plots with matplotlib:
the fragment size distribution of a Parquet fragment file, the TSS enrichment
and fragment count QC panels from {func}`gatac.pp.compute_metrics`, and the
top enriched motifs per peak set from {func}`gatac.motif.enrichment`.

```{eval-rst}
.. currentmodule:: gatac.pl

.. autosummary::
   :toctree: generated/
   :nosignatures:

   fragment_size_distribution
   qc_metrics
   motif_enrichment
```
