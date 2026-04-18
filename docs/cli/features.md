# `gatac features`

GPU-accelerated selection of the most accessible genomic features from one or
more tile-matrix h5ad files.  Uses quantile-based filtering following the
ArchR approach for binary matrices.

---

## Synopsis

```
gatac features <input.h5ad> [-n N_FEATURES]
               [-o OUTPUT] [--no-binarize]
               [--filter-lower QUANTILE] [--filter-upper QUANTILE]
```

---

## Arguments

### Positional

| Argument | Description |
|----------|-------------|
| `input.h5ad` | Path(s) or glob to h5ad tile-matrix file(s) |

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `-n`, `--n-features` | `500000` | Number of features to retain |
| `-o`, `--output` | in-place or `<input>_features.h5ad` | Output path |
| `--no-binarize` | — | Skip binarization of the output matrix |
| `--filter-lower` | `0.005` | Lower quantile cutoff (removes very rare features) |
| `--filter-upper` | `0.005` | Upper quantile cutoff (removes ubiquitously open features) |

---

## Algorithm

For **binary matrices** (standard tile matrices), GATAC:

1. Computes per-feature accessibility counts (sum of binarized matrix).
2. Filters out features in the lower `filter_lower_quantile` and upper
   `filter_upper_quantile` quantile of accessibility.
3. Selects the top `n_features` most accessible features from the remaining
   set.

For **count matrices**, the same procedure applies but on raw counts.

---

## Multi-file streaming

When multiple h5ad files (or a glob) are provided, GATAC performs **streaming
feature selection**:

- Feature counts are accumulated file-by-file without loading all data into
  memory simultaneously.
- A single set of `n_features` is selected across the union of all features.
- Output is a combined h5ad file.

This is the recommended approach for large multi-sample studies.

---

## Examples

### Single file (in-place)

```bash
gatac features tile_matrix.h5ad -n 500000
```

### Single file, save to new path

```bash
gatac features tile_matrix.h5ad -n 200000 -o tile_selected.h5ad
```

### Multi-sample streaming

```bash
gatac features "data/*.h5ad" -n 500000 -o combined_selected.h5ad
```

### Loose filtering (keep more rare/ubiquitous features)

```bash
gatac features tile_matrix.h5ad \
    -n 500000 \
    --filter-lower 0.001 \
    --filter-upper 0.001
```

---

## Python equivalent

```python
import gatac as ga

# Single file
ga.pp.select_features(adata, n_features=500_000)

# Multi-file streaming
ga.pp.select_features_multi(
    ["sampleA.h5ad", "sampleB.h5ad"],
    output_path="combined.h5ad",
    n_features=500_000,
    binarize=True,
)
```

---

## Output AnnData changes

| Slot | Content |
|------|---------|
| `adata.var["selected"]` | Boolean mask of selected features |
| `adata.var["accessibility_count"]` | Per-feature accessibility count |

After feature selection, downstream tools automatically subset to
`adata.var["selected"] == True` unless the full matrix is requested.
