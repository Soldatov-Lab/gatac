# `gatac rename`

Set the barcode prefix of a fragment Parquet file. A cell's name is its
barcode preceded by the file's prefix, and every GATAC step names it so.
The prefix is stored once, in the file footer, so only the footer is
rewritten: renaming takes milliseconds whatever the file's size, and the
barcodes of the original `fragments.tsv.gz` are kept.

Use it to give each sample's cells their own names before combining
samples converted without prefixes: a matrix of several files stops with
an error when the files share cell names.

---

## Synopsis

```
gatac rename <input.parquet> --barcode-prefix PREFIX [-o OUTPUT]
```

---

## Arguments

### Positional

| Argument | Description |
|----------|-------------|
| `input.parquet` | Fragment Parquet file (from `gatac convert` or `gatac filter`), changed in place |

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--barcode-prefix` | **required** | New prefix, which replaces the file's; `""` removes it |
| `-o`, `--output` | — | Write a copy with the new prefix instead of changing the input |

---

## Examples

### Name each sample's cells by its file

```bash
for f in parquets/*.parquet; do
    gatac rename "$f" --barcode-prefix "$(basename $f .parquet)_"
done
```

Cells of `parquets/sample1.parquet` are then named `sample1_<barcode>`.
When converting, `gatac convert "samples/*.tsv.gz" --barcode-prefix "{sample}_"`
names them so directly.

### Rename a copy

```bash
gatac rename sample1.parquet --barcode-prefix S1_ -o sample1_named.parquet
```

---

## Notes

- Name the cells before computing QC metrics: the metrics are keyed by cell
  name, and `gatac filter` matches them on the input file's names. A
  filtered file keeps its input's prefix, or takes `gatac filter
  --barcode-prefix`.
- `gatac convert` of a `.parquet` file writes the barcodes of the original
  file, without the prefix.

---

## Python equivalent

```python
import gatac as ga

ga.io.set_barcode_prefix("sample1.parquet", "sample1_")
ga.io.FragmentFile("sample1.parquet").barcode_prefix  # 'sample1_'
```
