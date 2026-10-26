# Overview

GATAC goes from the fragment files of a set of samples to a cell × peak
matrix. Each sample's fragments are converted once to a Parquet file, and
the cells that pass QC are written to a filtered Parquet file. From there,
each matrix is generated from the filtered Parquet files and saved as a
single combined file containing all samples. The filtered files remain
available for subsequent steps, including peak calling and other downstream
analyses.

```{raw} html
:file: _static/pipeline_overview.svg
```

---

## Steps

### 1. Fragment files to Parquet

`gatac convert` ({func}`gatac.io.make_parquet_batch`) writes one Parquet
file per `fragments.tsv.gz`. Chromosomes and barcodes are stored as
integer ids, with their names stored once per file, so that the GPU never
handles strings while streaming fragments; see
[Fragment files](#fragment-files).

A cell's name is its barcode preceded by the sample's **barcode prefix**,
and every later step names it so: the QC metrics, the filtered files, the
matrices and the peak steps, which find a cell's fragments by its name.
The prefix is stored once, in the file's footer, and the barcodes
themselves stay those of the `fragments.tsv.gz`, which `gatac convert`
writes back from the Parquet file. The prefix is given when converting,
where `{sample}`, written as is, is replaced by each file's sample name, the
part of the input paths that differs between the files:

```bash
gatac convert "samples/*.tsv.gz" --output-dir parquets/ --barcode-prefix "{sample}_"
# samples/s1.tsv.gz → parquets/s1.parquet, cells s1_<barcode>
# samples/s2.tsv.gz → parquets/s2.parquet, cells s2_<barcode>
```

or set at any time afterwards with `gatac rename`
({func}`gatac.io.set_barcode_prefix`), which rewrites the footer only and
takes milliseconds:

```bash
gatac rename parquets/s1.parquet --barcode-prefix s1_
```

- One sample needs no prefix.
- Samples that go into one matrix must not share cell names, as 10x
  samples from the same barcode whitelist do: give each its own prefix,
  e.g. `sample1_`. Building a matrix of samples that share names stops
  with an error before any work.

### 2. QC and filtering

`gatac metrics` ({func}`gatac.pp.compute_metrics`) gives per-cell QC
metrics (TSS enrichment, unique fragments, duplicate and mitochondrial
fractions) as a CSV file or a DataFrame, and `gatac doublets`
({func}`gatac.pp.detect_doublets`) flags doublets. `gatac filter`
({func}`gatac.pp.filter_fragments`) writes, per sample, a Parquet file with
the fragments of the cells that pass a filtering query on the metrics.

The filtered files are the input of every later step that reads fragments.

### 3. Tile matrix

`gatac tile` ({func}`gatac.pp.make_tile_matrix`) builds a tile matrix from
a fragment file; given several files, it builds one matrix of all the
samples. As the next step is most often feature selection, the tile step
can also write a matrix of the `n` most accessible tiles only (`-n`,
`n_features`), which is faster and more memory efficient, especially for
very large cohorts. This process consists of two passes over the filtered
Parquet files, with the selection in between:

1. **Scan** ({func}`gatac.pp.scan_tiles`): counts, per tile, the cells and
   insertions of every sample, without building a matrix.
2. **Select** ({func}`gatac.pp.select_features`): ranks the tiles from these
   statistics and keeps the top `n`.
3. **Build** ({func}`gatac.pp.build_tile_matrix`): counts the cells over the
   selected tiles only.

In the generated AnnData, `obs["source_file"]` records the file each cell
comes from, which is how the peak steps find a cell's fragments.

### 4. Embedding and clusters

{func}`gatac.dr.spectral` embeds the cells from the tile matrix, on the GPU
even when the matrix is larger than its memory. A kNN graph and Leiden on
the embedding (`rapids_singlecell` or Scanpy) produce the clusters.

### 5. Peaks per cluster

{func}`gatac.peak.call` reads the fragments of each cluster's cells from the
filtered files and calls peaks with MACS3 {cite:p}`Zhang2008`;
{func}`gatac.peak.merge` merges them into one set of fixed-width peaks.

### 6. Peak matrix

{func}`gatac.peak.make_matrix` counts the insertions of every cell over the
merged peaks, from the filtered files. The peak matrix is the input of
{func}`gatac.peak.markers`, motif enrichment and chromVAR
([`gatac.motif`](api/motif.md)).

### Gene matrix

`gatac gene` ({func}`gatac.pp.make_gene_matrix`) counts insertions over
promoters and gene bodies; given several filtered files, it builds one
matrix of all the samples, with `obs["source_file"]` as for the tile
matrix. It gives gene activity for annotating the clusters. `gatac
genescore` ({func}`gatac.pp.make_gene_score_matrix`) gives ArchR's gene
scores in the same way.

---

## Example

Steps 1 to 3 and the gene matrix with the CLI:

```bash
# 1. Convert fragments, cells named by sample (sample1_<barcode>)
#    ({sample} is written as is: gatac fills it in for each file, s1_, s2_, ...)
gatac convert "samples/*.tsv.gz" --output-dir parquets/ --barcode-prefix "{sample}_"

# 2. QC metrics, then the cells passing QC, per sample
for f in parquets/*.parquet; do
    name=$(basename $f .parquet)
    gatac metrics "$f" -g GRCh38.gtf.gz -o "metrics/${name}.csv"
    gatac filter "$f" --metrics "metrics/${name}.csv" \
        --filter "tsse_score > 5" -g hg38 -o "filtered/${name}.parquet"
done

# 3. One tile matrix of every sample, 500,000 most accessible tiles
gatac tile filtered/*.parquet -g hg38 -n 500000 -o tiles.h5ad

# One gene matrix of every sample
gatac gene filtered/*.parquet -g GRCh38.gtf.gz -o genes.h5ad
```

Steps 4 to 6 in Python:

```python
import anndata as ad
import gatac as ga
import rapids_singlecell as rsc

adata = ad.read_h5ad("tiles.h5ad")

# 4. Embedding and clusters
ga.dr.spectral(adata, n_comps=30)
rsc.pp.neighbors(adata, use_rep="X_spectral", n_neighbors=30)
rsc.tl.leiden(adata, resolution=0.5)

# 5. Peaks per cluster; a directory: cells are found by obs["source_file"]
ga.peak.call(adata, groupby="leiden", parquet_path="filtered/", genome="hg38")
ga.peak.merge(adata, chrom_sizes="hg38")

# 6. Peak matrix
peaks = ga.peak.make_matrix(adata, parquet_path="filtered/", genome="hg38")
```

The [tutorials](tutorials/index) run these steps on one sample.

---

## Run time

These steps were timed on a cohort of 79 samples (225 GB of gzipped
fragment files, 530,592 cells after filtering), against SnapATAC2
{cite:p}`Zhang2024` 2.9.0 (commit
[`55dde65`](https://github.com/scverse/SnapATAC2/commit/55dde65eb0e487724794aaa275bf9792fcb90bfd))
running the same steps. Each tool had 20 CPU cores, and GATAC one A100 GPU.
Each tool called peaks on its own Leiden clusters. SnapATAC2's `tl.macs3`
(MACS3 3.0.4) ran with 10 workers, as 20 ran out of memory: each worker
holds the pileup of one cluster. Hover over a segment for both tools'
times.

::::{tab-set}

:::{tab-item} To the clusters

```{raw} html
:file: _static/runtime_clustering.svg
```

:::

:::{tab-item} To the peak matrix

```{raw} html
:file: _static/runtime_peaks.svg
```

:::

::::

| Stage | SnapATAC2 | GATAC | Speedup |
|-------|-----------|-------|---------|
| Import | 1 h 18 min | 37 min | ×2.1 |
| QC + filter | 9.3 min | 4.7 min | ×2.0 |
| Tiles + selection | 42 min | 4.6 min | ×9.3 |
| Embedding + clusters | 36 min | 35 s | ×62 |
| Gene matrix | 21 min | 2.3 min | ×8.9 |
| Peak calling | 6 h 20 min | 16 min | ×24 |
| Peak matrix | 14 min | 1.4 min | ×10 |

Import, which writes the Parquet files, is about half of GATAC's time to the
peak matrix: it decompresses the gzipped fragment files on the CPU. It is done
once per sample, and every later step reads the Parquet files.
For GATAC, "Tiles + selection" includes loading the tile matrix, and
"Embedding + clusters" includes the kNN graph and Leiden clustering.
Single-function comparisons on PBMC 5k, which can be rerun, are in
[Reproducibility](reproducibility).

---

## Fragment files

Every step that reads fragments streams them from Parquet files to the
GPU. GATAC's Parquet files provide multiple optimizations, such as storing
each chromosome and barcode as an integer id, with the barcode names and
fragment counts in a table inside the same file, saving space and
increasing GPU throughput.

The layout, the encodings and how to read the files outside GATAC are
described in [Parquet fragment format](api/io.md#parquet-fragment-format).
