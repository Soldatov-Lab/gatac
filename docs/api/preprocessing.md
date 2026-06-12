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
   combine
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
   load_tss_from_gtf
```

### Computed metrics

| Column | Description |
|--------|-------------|
| `tsse_score` | TSS enrichment score |
| `n_unique` | Number of unique fragments per barcode |
| `duplicate_fraction` | Fraction of duplicate fragments |
| `mito_fraction` | Fraction of mitochondrial fragments |

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
```

### Count strategies

| Strategy | Description |
|----------|-------------|
| `"unique"` | Count unique insertions per tile (SnapATAC2-compatible, **default**) |
| `"count"` | Count all fragment insertions |
| `"binarize"` | Binary accessibility (0/1) |

### Built-in genomes

GATAC ships chromosome-size dictionaries for common reference genomes:

```{eval-rst}
.. autosummary::
   :toctree: generated/
   :nosignatures:

   HG38
   HG19
   MM10
   MM39
```

You can also pass a genome name string directly:

```python
adata = ga.pp.make_tile_matrix("pbmc.parquet", chrom_sizes="hg38")
```

`make_tile_matrix` also accepts `.h5ad` and 10x `.h5` matrices when their
feature names contain genomic intervals such as `chr1:100-200` (or
`chr1;100-200`). In that mode, GATAC keeps only interval-like features and
aggregates them into fixed tiles by overlap.

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

## Feature selection

Select the most accessible genomic features from a tile matrix.  GPU-
accelerated quantile-based selection following the ArchR approach for binary
matrices.

```{eval-rst}
.. autosummary::
   :toctree: generated/
   :nosignatures:

   select_features
   select_features_multi
```
