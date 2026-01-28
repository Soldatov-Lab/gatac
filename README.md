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

# Compute quality metrics (TSSe, etc.)
gatac metrics fragments.parquet -g annotations.gtf -o fragments_metrics.csv

# Compute quality metrics (TSSe, etc.)
gatac metrics fragments.parquet -g annotations.gtf --batch-size 32

# Generate tile matrix (m=min_unique_fragments)
gatac tile fragments.parquet -g hg38 -t 500 -m 100

# Generate tile matrix with quality filtering using metrics CSV
gatac tile fragments.parquet -g hg38 --metrics metrics.csv --filter "tsse_score > 5"

# Feature selection (single file)
gatac features tile_matrix.h5ad -n 500000

# Feature selection (multi-file streaming)
gatac features "data/*.h5ad" -o combined.h5ad -n 500000
```

## Python API

```python
from gatac import (
    make_parquet, 
    make_tile_matrix, 
    select_features, 
    select_features_multi,
    read_fragments_parquet,
    load_tss_from_gtf, 
    compute_metrics
)

# Convert to parquet
make_parquet("fragments.tsv.gz")

# Compute metrics (Streaming Polars GPU)
tss = load_tss_from_gtf("annotations.gtf")
metrics = compute_metrics("fragments.parquet", tss)

# Create tile matrix (pre-filtered by unique fragments)
adata = make_tile_matrix(
    "fragments.parquet", 
    chrom_sizes="hg38", 
    min_fragments_per_cell=500,
    barcode_prefix="Sample1_"
)

# Create tile matrix using pre-computed metrics for filtering
# metrics can be a path to a CSV or a cuDF DataFrame
adata = make_tile_matrix(
    "fragments.parquet", 
    chrom_sizes="hg38",
    metrics=metrics, 
    filter_query="tsse_score > 5"
)

# Select features (single AnnData)
select_features(adata, n_features=50000)

# Select features (Multi-file streaming)
select_features_multi(
    input_paths=["sample1.h5ad", "sample2.h5ad"],
    output_path="combined.h5ad",
    n_features=500000
)
```

