# Preprocessing — `gatac.pp`

The `gatac.pp` namespace covers the preprocessing pipeline: computing quality
metrics, filtering barcodes, and building count matrices from the fragment
files written by [`gatac.io`](io.md).

---

## Quality metrics & filtering

Compute TSS enrichment score and fragment-level statistics entirely on GPU
using a streaming approach, filter barcodes by quality thresholds, and detect
doublet / multiplet cells with the AMULET Poisson method {cite:p}`Thibodeau2021`.  `filter_fragments`
accepts a pre-computed metrics DataFrame or CSV and a pandas `DataFrame.query` string;
`detect_doublets` returns per-cell p/q values and an `is_doublet` flag.

```{eval-rst}
.. currentmodule:: gatac.pp

.. autosummary::
   :toctree: generated/
   :nosignatures:

   compute_metrics
   filter_fragments
   detect_doublets
```

---

## Matrix processing

Build cell × feature matrices from QC-filtered fragments, and post-process the
resulting `.h5ad` files.  Includes fixed-width genomic bins (`make_tile_matrix`,
compatible with SnapATAC2's count strategy {cite:p}`Zhang2024`), gene activity over a GTF annotation
— either SnapATAC2-style paired-insertion counts (`make_gene_matrix`) or
ArchR-style distance-weighted gene scores (`make_gene_score_matrix`, a port of
`addGeneScoreMatrix` {cite:p}`Granja2021`) — and operations on existing `.h5ad` files: combining
samples (`combine`) and selecting the most accessible genomic features
(`select_features`).

Given a list of fragment files, `make_tile_matrix` builds one tile matrix of
all of them without writing per-sample matrices: `scan_tiles` reads the
files once for per-tile statistics, `select_features` chooses tiles from
them, and `build_tile_matrix` reads the files again to build the matrix of
the chosen tiles only. Call these yourself to choose tiles another way.

```{eval-rst}
.. autosummary::
   :toctree: generated/
   :nosignatures:

   make_tile_matrix
   scan_tiles
   build_tile_matrix
   make_gene_matrix
   make_gene_score_matrix
   select_features
   combine
```
