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
gatac metrics fragments.parquet -g annotations.gtf --batch-size 64

# Generate tile matrix (m=min_unique_fragments)
gatac tile fragments.parquet -g hg38 -t 500 -m 100

# Generate tile matrix with quality filtering using metrics CSV
gatac tile fragments.parquet -g hg38 --metrics metrics.csv --filter "tsse_score > 5"

# Generate gene activity matrix
gatac gene fragments.parquet -g annotations.gtf

# Generate gene activity matrix with filtering
gatac gene fragments.parquet -g annotations.gtf --metrics metrics.csv --filter "tsse_score > 5"

# Feature selection (single file)
gatac features tile_matrix.h5ad -n 500000

# Feature selection (multi-file streaming)
gatac features "data/*.h5ad" -o combined.h5ad -n 500000
```

## Python API

```python
import gatac as ga

# Convert to parquet
ga.pp.make_parquet("fragments.tsv.gz")

# Compute metrics (Streaming Polars GPU)
tss = ga.pp.load_tss_from_gtf("annotations.gtf")
metrics = ga.pp.compute_metrics("fragments.parquet", tss)

# Create tile matrix (pre-filtered by unique fragments)
adata = ga.pp.make_tile_matrix(
    "fragments.parquet", 
    chrom_sizes="hg38", 
    min_fragments_per_cell=500,
    barcode_prefix="Sample1_"
)

# Create tile matrix using pre-computed metrics for filtering
# metrics can be a path to a CSV or a cuDF DataFrame
adata = ga.pp.make_tile_matrix(
    "fragments.parquet", 
    chrom_sizes="hg38",
    metrics=metrics, 
    filter_query="tsse_score > 5"
)

# Create gene activity matrix
adata_gene = ga.pp.make_gene_matrix(
    "fragments.parquet",
    gene_anno="annotations.gtf",
    upstream=2000,
    include_gene_body=True
)

# Create gene activity matrix with filtering
adata_gene = ga.pp.make_gene_matrix(
    "fragments.parquet",
    gene_anno="annotations.gtf",
    metrics=metrics,
    filter_query="tsse_score > 5"
)

# Select features (single AnnData)
ga.pp.select_features(adata, n_features=50000)

# Select features (Multi-file streaming)
ga.pp.select_features_multi(
    input_paths=["sample1.h5ad", "sample2.h5ad"],
    output_path="combined.h5ad",
    n_features=500000
)
```

