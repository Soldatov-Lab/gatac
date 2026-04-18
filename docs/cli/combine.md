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
  the same set of tiles).  Use `gatac features` with multi-file mode to
  produce aligned outputs first.

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

```bash
# 1. Convert fragments
gatac convert "samples/*.tsv.gz" --output-dir parquets/

# 2. Compute metrics per sample
for f in parquets/*.parquet; do
    gatac metrics "$f" -g GRCh38.gtf.gz -o "metrics/$(basename $f .parquet)_metrics.csv"
done

# 3. Build tile matrices per sample
for f in parquets/*.parquet; do
    name=$(basename $f .parquet)
    gatac tile "$f" -g hg38 \
        --metrics "metrics/${name}_metrics.csv" \
        --filter "tsse_score > 5" \
        -o "tiles/${name}.h5ad"
done

# 4. Feature selection + combine
gatac features "tiles/*.h5ad" -n 500000 -o combined.h5ad
```
