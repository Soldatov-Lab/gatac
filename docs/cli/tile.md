# `gatac tile`

Build a cell × genomic-tile count matrix from either an ATAC-seq fragment
Parquet file or an interval matrix stored as `.h5ad` or 10x `.h5`. The output
is an **AnnData** (`.h5ad`) object compatible with Scanpy and SnapATAC2.

---

## Synopsis

```
gatac tile <input.parquet|input.h5ad|input.h5> -g <genome|chr_sizes>
           [-o OUTPUT] [-t TILE_SIZE] [-m MIN_FRAGS]
           [-e CHROMS ...] [--metrics METRICS] [--filter QUERY]
           [--count-strategy STRATEGY] [--barcode-prefix PREFIX]
           [--low-memory]
```

---

## Arguments

### Positional

| Argument | Description |
|----------|-------------|
| `input` | Path to a fragment `.parquet`, interval-matrix `.h5ad`, or 10x `.h5` file |

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `-g`, `--genome` | **required** | Genome name (`hg38`, `mm10`, …) or path to a 2-column TSV of `chrom\tsize` |
| `-o`, `--output` | `<input>_tile.h5ad` | Output h5ad path |
| `-t`, `--tile-size` | `5000` | Tile / bin size in base pairs |
| `-m`, `--min-fragments` | `100` | Minimum unique fragments per barcode |
| `-e`, `--exclude-chroms` | `chrM chrY M Y` | Chromosomes to exclude (space-separated) |
| `--metrics` | — | Metrics CSV for quality-based cell filtering |
| `--filter` | — | Polars query string applied to metrics |
| `--count-strategy` | `unique` | Counting strategy: `unique`, `count`, or `binarize` |
| `--barcode-prefix` | — | String prepended to barcodes |
| `--low-memory` | off | Enable low-memory (single row-group) mode |

---

## Count strategies

| Strategy | Description | Compatibility |
|----------|-------------|---------------|
| `unique` | Unique fragment insertions per tile (deduplicated) | SnapATAC2-compatible (**default**) |
| `count` | All fragment insertions (not deduplicated) | — |
| `binarize` | Binary accessibility (0 or 1) | ArchR-compatible |

For `.h5ad` and 10x `.h5` inputs, GATAC detects interval-like features from
`var_names` using names such as `chr1:100-200` or `chr1;100-200`. Those
features are aggregated into fixed tiles by overlap. In that mode,
`unique` and `count` both preserve the original matrix values, while
`binarize` clips the final tile matrix to `0/1`.

---

## Built-in genomes

Pass a genome name string to use built-in chromosome sizes:

| Name | Assembly |
|------|----------|
| `hg38` / `GRCh38` | Human GRCh38 (Gencode v41) |
| `hg19` / `GRCh37` | Human GRCh37 |
| `mm10` / `GRCm38` | Mouse GRCm38 (Gencode vM25) |
| `mm39` / `GRCm39` | Mouse GRCm39 (Gencode vM30) |

---

## Examples

### Quick start

```bash
gatac tile pbmc.parquet -g hg38 -t 500 -m 100
```

### From a 10x peak matrix

```bash
gatac tile filtered_peak_bc_matrix.h5 -g hg38 -t 500 -o pbmc_tile.h5ad
```

### With quality filtering

```bash
gatac tile pbmc.parquet -g hg38 \
    --metrics pbmc_metrics.csv \
    --filter "tsse_score > 5 and n_unique > 1000" \
    -o pbmc_tile.h5ad
```

### Custom tile size, exclude sex chromosomes

```bash
gatac tile pbmc.parquet -g hg38 \
    -t 1000 \
    -e chrM chrY \
    -o pbmc_tile_1kb.h5ad
```

### Binarized matrix (ArchR-style)

```bash
gatac tile pbmc.parquet -g hg38 --count-strategy binarize
```

### Low-memory mode for constrained GPUs

```bash
gatac tile pbmc.parquet -g hg38 --low-memory -o pbmc_tile.h5ad
```

---

## Python equivalent

```python
import gatac as ga

adata = ga.pp.make_tile_matrix(
    "pbmc.parquet",
    chrom_sizes="hg38",
    tile_size=500,
    min_fragments_per_cell=100,
    exclude_chroms=["chrM", "chrY"],
    metrics="pbmc_metrics.csv",
    filter_query="tsse_score > 5 and n_unique > 1000",
    count_strategy="unique",
)
adata.write_h5ad("pbmc_tile.h5ad")
```

---

## Output AnnData structure

| Slot | Content |
|------|---------|
| `adata.X` | Sparse cell × tile count matrix (`scipy.sparse.csr_matrix`) |
| `adata.obs` | Barcode metadata (barcode string as index) |
| `adata.var` | Tile metadata: `chrom`, `start`, `end` |
