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

### Usage example

```python
import gatac as ga

# Single file
out = ga.pp.make_parquet("pbmc.tsv.gz")

# Batch with per-sample barcode prefixes
paths = ga.pp.make_parquet_batch(
    ["sampleA.tsv.gz", "sampleB.tsv.gz"],
    barcode_prefix=["A_", "B_"],
)
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

### Usage example

```python
tss = ga.pp.load_tss_from_gtf("GRCh38.gtf.gz")

metrics = ga.pp.compute_metrics(
    "pbmc.parquet",
    tss_df=tss,
    min_unique_frags=100,
    exclude_chroms=["chrM", "M"],
)

# Plot QC
import scanpy as sc
import pandas as pd
obs = metrics.to_pandas().set_index("barcode")
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

### Usage example

```python
ga.pp.filter_fragments(
    "pbmc.parquet",
    metrics="pbmc_metrics.csv",
    filter_query="tsse_score > 5 and n_unique > 1000",
    output_parquet="pbmc_filtered.parquet",
)
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

Or a custom dict:

```python
adata = ga.pp.make_tile_matrix(
    "pbmc.parquet",
    chrom_sizes={"chr1": 248956422, "chr2": 242193529, ...},
)
```

### Usage example

```python
adata = ga.pp.make_tile_matrix(
    "pbmc_filtered.parquet",
    chrom_sizes="hg38",
    tile_size=500,
    min_fragments_per_cell=200,
    exclude_chroms=["chrM", "chrY"],
)
print(adata)  # AnnData object with n_obs × n_vars
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

### Usage example

```python
adata_gene = ga.pp.make_gene_matrix(
    "pbmc_filtered.parquet",
    gene_anno="GRCh38.gtf.gz",
    id_type="gene",
    upstream=2000,
    downstream=0,
    include_gene_body=True,
)
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

### Usage example

```python
# Single AnnData
ga.pp.select_features(adata, n_features=500_000)

# Multi-sample streaming
ga.pp.select_features_multi(
    ["sampleA.h5ad", "sampleB.h5ad"],
    output_path="combined.h5ad",
    n_features=500_000,
)
```
