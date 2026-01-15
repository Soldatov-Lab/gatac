# GATAC

GPU-accelerated ATAC-seq processing toolkit.

## Installation

```bash
uv sync
```

## CLI Usage

```bash
# Convert TSV.GZ to Parquet
gatac convert fragments.tsv.gz

# Generate tile matrix
gatac tile fragments.parquet -t 5000 -m 100

# Feature selection
gatac features tile_matrix.h5ad -n 500000
```

## Python API

```python
from gatac import make_parquet, make_tile_matrix, select_features

# Convert to parquet
make_parquet("fragments.tsv.gz")

# Create tile matrix
adata = make_tile_matrix("fragments.parquet")

# Select features
select_features(adata, n_features=500000)
```
