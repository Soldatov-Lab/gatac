# `gatac combine`

Merge multiple AnnData (`.h5ad`) files into a single file.  Uses a
memory-efficient file-by-file streaming strategy to handle large multi-sample
datasets without requiring all data to be resident in memory simultaneously.

---

## Synopsis

```
gatac combine <input1.h5ad> [input2.h5ad ...] -o <output.h5ad>
```

---

## Arguments

### Positional

| Argument | Description |
|----------|-------------|
| `input` | Two or more h5ad file paths (glob patterns supported) |

### Options

| Flag | Description |
|------|-------------|
| `-o`, `--output` | **Required.** Output h5ad path |

---

## Behaviour

- **Dtype optimisation**: determines the smallest integer dtype that can
  represent the maximum value across all inputs (e.g. `uint8`, `uint16`).
- **Duplicate barcodes**: if the same barcode appears in multiple files, a
  suffix (`_1`, `_2`, …) is appended automatically.
- **Variable alignment**: all inputs must share the same `var` index (e.g.
  the same genes). For tile matrices, `gatac tile` with several fragment
  files builds the combined matrix directly, without per-sample files.

---

## Examples

### Merge two samples

```bash
gatac combine sampleA.h5ad sampleB.h5ad -o combined.h5ad
```

### Merge all samples in a directory

```bash
gatac combine data/*.h5ad -o combined.h5ad
```

---

## Python equivalent

```python
import gatac as ga
from pathlib import Path

ga.pp.combine(
    [Path("sampleA.h5ad"), Path("sampleB.h5ad")],
    output_path=Path("combined.h5ad"),
)
```

---

## Recommended workflow for multi-sample studies

The [Overview](../overview.md) shows how these steps connect, through
clustering and peak calling.

```bash
# 1. Convert fragments, cells named by sample (sample1_<barcode>):
#    samples in one matrix must not share cell names
#    ({sample} is written as is: gatac fills it in for each file, s1_, s2_, ...)
gatac convert "samples/*.tsv.gz" --output-dir parquets/ --barcode-prefix "{sample}_"

# 2. Compute metrics per sample
for f in parquets/*.parquet; do
    gatac metrics "$f" -g GRCh38.gtf.gz -o "metrics/$(basename $f .parquet)_metrics.csv"
done

# 3. Keep the cells passing QC, per sample
for f in parquets/*.parquet; do
    name=$(basename $f .parquet)
    gatac filter "$f" --metrics "metrics/${name}_metrics.csv" \
        --filter "tsse_score > 5" -g hg38 -o "filtered/${name}.parquet"
done

# 4. One tile matrix of every sample, 500,000 most accessible tiles
gatac tile filtered/*.parquet -g hg38 -n 500000 -o combined_tiles.h5ad

# 5. One gene matrix of every sample
gatac gene filtered/*.parquet -g GRCh38.gtf.gz -o combined_genes.h5ad
```

`gatac tile`, `gatac gene` and `gatac genescore` build one matrix of
several fragment files directly; `gatac combine` merges `.h5ad` files made
separately. Either way, the cells of different samples must have distinct
names. Files already converted without prefixes get them from
[`gatac rename`](rename), in place:

```bash
for f in parquets/*.parquet; do
    gatac rename "$f" --barcode-prefix "$(basename $f .parquet)_"
done
```
