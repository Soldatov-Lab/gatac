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

# Out-of-core streaming metrics (for datasets larger than VRAM)
gatac metrics fragments.parquet -g annotations.gtf --streaming --batch-size 5

# Generate tile matrix (m=min_unique_fragments)
gatac tile fragments.parquet -g hg38 -t 500 -m 100

# Generate tile matrix with quality filtering using metrics CSV
gatac tile fragments.parquet -g hg38 --metrics metrics.csv --filter "tsse_score > 5"

# Feature selection
gatac features tile_matrix.h5ad -n 50000
```

## Python API

```python
from gatac import (
    make_parquet, 
    make_tile_matrix, 
    select_features, 
    read_fragments_parquet,
    load_tss_from_gtf, 
    compute_metrics,
    load_tss_from_gtf_polars,
    compute_metrics_streaming
)

# Convert to parquet
make_parquet("fragments.tsv.gz")

# Compute metrics (In-memory cuDF)
tss = load_tss_from_gtf("annotations.gtf")
fragments = read_fragments_parquet("fragments.parquet")
metrics = compute_metrics(fragments, tss)

# Compute metrics (Streaming Polars GPU, for large datasets)
tss_lf = load_tss_from_gtf_polars("annotations.gtf")
metrics_streaming = compute_metrics_streaming("fragments.parquet", tss_lf)

# Create tile matrix (pre-filtered by unique fragments)
adata = make_tile_matrix("fragments.parquet", chrom_sizes="hg38", min_fragments_per_cell=500)

# Create tile matrix using pre-computed metrics for filtering
# metrics can be a path to a CSV or a cuDF DataFrame
adata = make_tile_matrix(
    "fragments.parquet", 
    chrom_sizes="hg38",
    metrics=metrics, 
    filter_query="tsse_score > 5"
)

# Select features
select_features(adata, n_features=50000)
```

