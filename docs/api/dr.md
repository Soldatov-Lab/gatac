# Dimensionality reduction — `gatac.dr`

Compute a spectral decomposition of the cell × feature matrix — the standard
entry point for UMAP and clustering in ATAC-seq workflows (`spectral`, after
SnapATAC2 {cite:p}`Zhang2024`) — or run TF-IDF followed by a truncated SVD, the
latent semantic indexing {cite:p}`Cusanovich2015` of Signac {cite:p}`Stuart2021`
and ArchR {cite:p}`Granja2021` (`lsi`, with `project_lsi` to place new cells on
a fitted embedding).

`lsi` reproduces `ArchR:::.computeLSI` component for component, including its
three `LSIMethod` variants, its depth-outlier hold-out, and the
depth-correlation dimension filter. `iterative_lsi` is the port of
`ArchR::addIterativeLSI`: it clusters on a first embedding, keeps the features
whose per-cluster accessibility varies most, and redoes the decomposition on
those — so each round selects for the structure the previous round found. It
requires cuGraph for the clustering step; `lsi` does not.

```{eval-rst}
.. currentmodule:: gatac.dr

.. autosummary::
   :toctree: generated/
   :nosignatures:

   spectral
   lsi
   iterative_lsi
   project_lsi
   model_from_adata
```

The individual stages of the LSI pipeline are importable from their module,
for building a custom iterative scheme or inspecting one step in isolation
(`from gatac.dr.lsi import scale_dims`). `scale_dims` is ArchR's `scaleDims` —
note it standardises each *cell* across its dimensions, not each dimension
across cells.

```{eval-rst}
.. currentmodule:: gatac.dr.lsi

.. autosummary::
   :toctree: generated/
   :nosignatures:

   feature_accessibility
   initial_features
   accessibility_pool
   cluster_var_features
   scale_dims
```
