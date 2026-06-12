# Preprocessing — `gatac.pp`

The `gatac.pp` namespace covers the full preprocessing pipeline: reading raw
fragment files, computing quality metrics, filtering barcodes, and building
count matrices.

---

## Fragment I/O

Convert raw fragment TSV.GZ files to columnar Parquet for efficient GPU
streaming.  Parquet preserves row-group structure so GATAC can process files
larger than GPU memory.

```{eval-rst}
.. currentmodule:: gatac.pp

.. autosummary::
   :toctree: generated/
   :nosignatures:

   make_parquet
   make_parquet_batch
   read_fragments_parquet
```

---

## Quality metrics & filtering

Compute TSS enrichment score and fragment-level statistics entirely on GPU
using a streaming approach, then filter barcodes by quality thresholds.
`filter_fragments` accepts a pre-computed metrics DataFrame or CSV and a
Polars query string.

```{eval-rst}
.. autosummary::
   :toctree: generated/
   :nosignatures:

   compute_metrics
   filter_fragments
```

---

## Matrix processing

Build cell × feature matrices from QC-filtered fragments, and post-process the
resulting `.h5ad` files.  Includes fixed-width genomic bins (`make_tile_matrix`,
compatible with SnapATAC2's count strategy), gene activity scores over a GTF
annotation (`make_gene_matrix`), and operations on existing `.h5ad` files:
combining samples (`combine`) and selecting the most accessible genomic
features across one or many matrices (`select_features`, `select_features_multi`).

```{eval-rst}
.. autosummary::
   :toctree: generated/
   :nosignatures:

   make_tile_matrix
   make_gene_matrix
   select_features
   select_features_multi
   combine
```
