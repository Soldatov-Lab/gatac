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

## Quality metrics

Compute TSS enrichment score and fragment-level statistics entirely on GPU
using a streaming approach — the full fragment file never needs to reside in
GPU memory simultaneously.

```{eval-rst}
.. autosummary::
   :toctree: generated/
   :nosignatures:

   compute_metrics
```

---

## Fragment filtering

Filter barcodes by quality thresholds.  Accepts a pre-computed metrics
DataFrame or CSV and a Polars query string.

```{eval-rst}
.. autosummary::
   :toctree: generated/
   :nosignatures:

   filter_fragments
   cleanup_gpu_memory
```

---

## Tile matrix

Bin the genome into fixed-size windows and accumulate fragment insertions per
cell.  The default count strategy (`"unique"`) is compatible with SnapATAC2.

```{eval-rst}
.. autosummary::
   :toctree: generated/
   :nosignatures:

   make_tile_matrix
   HG38
   HG19
   MM10
   MM39
```

---

## Gene activity matrix

Score gene activity by counting paired fragment insertions over promoter and
gene-body regions defined by a GTF annotation.

```{eval-rst}
.. autosummary::
   :toctree: generated/
   :nosignatures:

   make_gene_matrix
```

---

## h5ad processing

Operations on existing cell × feature `.h5ad` matrices: combine multiple
samples into a single file, and select the most accessible genomic features
across one or many matrices.

```{eval-rst}
.. autosummary::
   :toctree: generated/
   :nosignatures:

   select_features
   select_features_multi
   combine
```
