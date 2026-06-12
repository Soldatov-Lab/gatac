# GATAC: GPU-Accelerated scATACseq Analysis

GATAC is a GPU-accelerated toolkit for end-to-end ATAC-seq data processing.
Starting from raw fragment files, it produces analysis-ready sparse matrices
(tile, peak, and gene activity) that integrate directly with the
[scverse](https://scverse.org) ecosystem — AnnData, Scanpy, and SnapATAC2.
Computations are offloaded to the GPU via [RAPIDS cuDF](https://rapids.ai),
[CuPy](https://cupy.dev), and [cuML](https://docs.rapids.ai/api/cuml/stable/),
delivering 10–50× speedups over CPU-only workflows on typical single-cell
datasets.

GATAC workflow takes advantage of [Apache Parquet](https://parquet.apache.org) file format. Parquet's columnar layout and
built-in compression make it an ideal staging format for GPU-based pipelines:
columns are read independently, so only the genomic coordinates actually needed
for a given operation are transferred to device memory. RAPIDS cuDF can read
Parquet directly into GPU memory with zero CPU round-trips, enabling streaming
aggregation over datasets that far exceed the size of available GPU RAM.

## Installation

```bash
uv sync --extra cuda12
```

## CLI Usage

```bash
# Convert TSV.GZ to Parquet
gatac convert fragments.tsv.gz

# Compute quality metrics (TSSe, etc.)
gatac metrics fragments.parquet -g annotations.gtf -o fragments_metrics.csv

# Compute quality metrics (TSSe, etc.)
gatac metrics fragments.parquet -g annotations.gtf --batch-size 64

# Filter fragments by minimum fragment count
gatac filter fragments.parquet -m 500 -o fragments_filtered.parquet

# Filter fragments using metrics and quality threshold
gatac filter fragments.parquet --metrics metrics.csv --filter "tsse_score > 5" -o fragments_filtered.parquet

# Filter fragments with genome-based chromosome filtering
gatac filter fragments.parquet -g hg38 -m 500 -o fragments_filtered.parquet

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

# Merge h5ad files (file-by-file)
gatac combine sample1.h5ad sample2.h5ad -o combined.h5ad
```

## Python API

```python
import gatac as ga

# Convert to parquet
ga.pp.make_parquet("fragments.tsv.gz")

# Compute metrics (Streaming Polars GPU)
metrics = ga.pp.compute_metrics("fragments.parquet", "annotations.gtf")

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

# Filter fragments by minimum fragment count
ga.pp.filter_fragments(
    "fragments.parquet",
    min_fragments_per_cell=500
)

# Filter fragments using metrics with quality threshold
ga.pp.filter_fragments(
    "fragments.parquet",
    metrics=metrics,
    filter_query="tsse_score > 5"
)

# Filter multiple files with genome-based chromosome filtering
ga.pp.filter_fragments(
    ["sample1.parquet", "sample2.parquet"],
    metrics=metrics,
    filter_query="tsse_score > 5 and n_unique > 1000",
    chrom_sizes="hg38"
)

# Select features (single AnnData)
ga.pp.select_features(adata, n_features=50000)

# Select features (Multi-file streaming)
ga.pp.select_features_multi(
    input_paths=["sample1.h5ad", "sample2.h5ad"],
    output_path="combined.h5ad",
    n_features=500000
)

# Merge multiple AnnData files
ga.pp.combine(
    input_paths=["sample1.h5ad", "sample2.h5ad"],
    output_path="merged.h5ad"
)
```

