# `gatac tile`

Build a cell × genomic-tile count matrix from either an ATAC-seq fragment
Parquet file or an interval matrix stored as `.h5ad` or 10x `.h5`. Several
fragment files make one combined matrix of all their cells. The output
is an **AnnData** (`.h5ad`) object compatible with Scanpy and SnapATAC2.

**`gatac tile` includes the feature selection.** With `-n N`, only the `N`
most accessible tiles are kept, and the full tile matrix is never built:
the fragment files are read once to measure every tile's accessibility,
the top `N` tiles are chosen, and the files are read again to build the
matrix of those tiles only. On a large cohort, the full matrix of every
tile can be several times larger than the selected one, too large to load;
`-n` avoids it altogether. Without `-n`, every tile is kept.

---

## Synopsis

```
gatac tile <input.parquet|input.h5ad|input.h5> [input.parquet ...] -g <genome|chr_sizes>
           [-o OUTPUT] [-t TILE_SIZE] [-m MIN_FRAGS] [-n N_FEATURES]
           [-e CHROMS ...] [--metrics METRICS] [--filter QUERY]
           [--count-strategy STRATEGY] [--barcode-prefix PREFIX]
           [--low-memory]
```

---

## Arguments

### Positional

| Argument | Description |
|----------|-------------|
| `input` | Path to a fragment `.parquet`, interval-matrix `.h5ad`, or 10x `.h5` file; or several fragment `.parquet` files |

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `-g`, `--genome` | **required** | Genome name (`hg38`, `mm10`, …) or path to a 2-column TSV of `chrom\tsize` |
| `-o`, `--output` | `<input>_tile_matrix.h5ad` | Output h5ad path (required for several inputs) |
| `-t`, `--tile-size` | `5000` | Tile / bin size in base pairs |
| `-m`, `--min-fragments` | `100` | Minimum unique fragments per barcode |
| `-n`, `--n-features` | all tiles | Keep only the N most accessible tiles, without building the full matrix (fragment input); see [Feature selection](#feature-selection) |
| `-e`, `--exclude-chroms` | `chrM chrY M Y` | Chromosomes to exclude (space-separated) |
| `--metrics` | — | Metrics CSV for quality-based cell filtering |
| `--filter` | — | Pandas `DataFrame.query` string applied to metrics |
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

## Feature selection

`-n N` keeps the `N` most accessible tiles, chosen as
{func}`gatac.pp.select_features` chooses them:

- **counts** (`unique`, `count`): tiles ranked by their total count over all
  cells, after dropping empty tiles and the 0.5 % least and most accessible
  (SnapATAC2's quantile filter);
- **binary** (`binarize`): tiles ranked by the number of cells with signal
  (ArchR's top-N).

The selection is made on all cells of all input files together. The tile
size (`-t`) sets the genome binning first, and `-n` then picks among those
tiles: 5 kb tiles cover hg38 in about 600,000 tiles, 500 bp tiles in about
6 million.

The output keeps, for every tile, the statistics the selection used
(`n_cells`, `count`, `count_sq` in `var`), so another selection can be made
later from `var` without reading the matrix.

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

### Top 500,000 tiles of 5 kb (SnapATAC2's default)

```bash
gatac tile pbmc.parquet -g hg38 -t 5000 -n 500000 -o pbmc_tile_5kb_top500k.h5ad
```

### Top 25,000 tiles of 500 bp, binarized (ArchR's iterative LSI default)

```bash
gatac tile pbmc.parquet -g hg38 -t 500 -n 25000 \
    --count-strategy binarize -o pbmc_tile_500bp_top25k.h5ad
```

### Several samples in one matrix

```bash
gatac tile sample1.parquet sample2.parquet sample3.parquet -g hg38 \
    -t 5000 -n 50000 -o combined.h5ad
```

The tiles are chosen on the cells of all samples together, and no
per-sample matrix is written. Cell names repeated across files are made
unique (`-1`, `-2`, ...); `--barcode-prefix` prefixes them all. Filter the
cells of each sample first (`gatac filter`), since `--metrics` applies the
same table to every file.

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

# Top 500,000 tiles of 5 kb, without building the full matrix
adata = ga.pp.make_tile_matrix("pbmc.parquet", chrom_sizes="hg38",
                               tile_size=5000, n_features=500_000)

# Several samples, without per-sample matrices
adata = ga.pp.make_tile_matrix(
    ["sample1.parquet", "sample2.parquet"],
    chrom_sizes="hg38",
    tile_size=5000,
    n_features=50_000,
    output_path="combined.h5ad",
)

# Or step by step, to choose the tiles another way
tiles = ga.pp.scan_tiles(["sample1.parquet", "sample2.parquet"], chrom_sizes="hg38")
ga.pp.select_features(tiles, n_features=500_000)     # or set tiles.var["selected"]
adata = ga.pp.build_tile_matrix(tiles, output_path="combined.h5ad")
```

---

## Output AnnData structure

| Slot | Content |
|------|---------|
| `adata.X` | Sparse cell × tile count matrix (`scipy.sparse.csr_matrix`) |
| `adata.obs` | Barcode metadata (barcode string as index) |
| `adata.var` | Tile metadata: `chrom`, `start`, `end`; tile statistics over all cells: `n_cells`, `count`, `count_sq`; with `-n`, `selected` and `accessibility_count` |
| `adata.uns["tile_stats"]` | Number of cells the statistics were summed over |
