# `gatac doublets`

Detect doublet / multiplet cells in a single-cell ATAC-seq sample using the
[AMULET](https://github.com/UcarLab/AMULET) Poisson method of
Thibodeau et al. (2021).  Cells with significantly more overlapping
insertions than expected are flagged as doublets.

---

## Synopsis

```
gatac doublets <input.parquet> -g <genome> [-o OUTPUT]
                 [-m MIN_FRAGS] [--q Q] [--q-rep Q_REP]
                 [--expected-overlap N] [--max-insert N]
                 [--min-overlap N] [--repeat-filter BED]
                 [-j THREADS]
```

---

## Arguments

### Positional

| Argument | Description |
|----------|-------------|
| `input.parquet` | Path to the fragment Parquet file |

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `-g`, `--genome` | **required** | Genome name (`hg38`, `mm10`, …) or path to a chromosome sizes file |
| `-o`, `--output` | `<input>_doublets.csv` | Output CSV path |
| `-m`, `--min-fragments` | `100` | Minimum unique fragments per cell to include |
| `--q` | `0.01` | FDR threshold for doublet calling |
| `--q-rep` | `0.01` | FDR threshold for inferring repetitive regions |
| `--expected-overlap` | `2` | Expected number of reads overlapping (Poisson mean) |
| `--max-insert` | `900` | Maximum fragment insert size in bp |
| `--min-overlap` | `1` | Minimum overlap length in bp |
| `--repeat-filter` | — | BED file of known repetitive regions to exclude |
| `-j`, `--threads` | `1` | Parallel workers for overlap detection |

---

## Output

A CSV with one row per cell:

| Column | Description |
|--------|-------------|
| `cell_id` | Cell barcode |
| `p_value` | Poisson p-value for the observed overlap count |
| `q_value` | Benjamini–Hochberg adjusted p-value |
| `is_doublet` | `True` if `q_value < --q` |

---

## Examples

### Basic usage

```bash
gatac doublets pbmc.parquet -g hg38 -o pbmc_doublets.csv
```

### Stricter threshold

```bash
gatac doublets pbmc.parquet -g hg38 --q 0.001 -o pbmc_doublets.csv
```

### Exclude known repetitive regions

```bash
gatac doublets pbmc.parquet -g hg38 --repeat-filter hg38_repeats.bed
```

---

## Python equivalent

```python
import gatac as ga

result = ga.pp.detect_doublets(
    "pbmc.parquet",
    chrom_sizes="hg38",
    min_fragments=100,
    q_threshold=0.01,
)
print(result.head())
#   cell_id   p_value   q_value  is_doublet
# 0  AAAC...  0.8321   0.9412     False
# 1  AAAC...  0.0001   0.0034     True
# ...

# Drop doublets from an AnnData
doublets = set(result.loc[result["is_doublet"], "cell_id"])
adata = adata[~adata.obs_names.isin(doublets)].copy()
```

---

## Notes

AMULET's Poisson model assumes a uniform single-copy background and is
designed for **autosomes only**.  Make sure the supplied `chrom_sizes`
covers the autosomes of your species (the built-in `hg38`, `hg19`, `mm10`,
`mm39` aliases do).
