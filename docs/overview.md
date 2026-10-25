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
file per `fragments.tsv.gz`, with an optional prefix on the barcodes so that
cells stay unique across samples. Chromosomes and barcodes are stored as
integer ids, with their names stored once per file, so that the GPU never
handles strings while streaming fragments; see
[Fragment files](#fragment-files).

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
promoters and gene bodies, one filtered file at a time; `gatac combine`
({func}`gatac.pp.combine`) merges the samples. It gives gene activity for
annotating the clusters.

---

## Example

Steps 1 to 3 and the gene matrix with the CLI:

```bash
# 1. Convert fragments
gatac convert "samples/*.tsv.gz" --output-dir parquets/

# 2. QC metrics, then the cells passing QC, per sample
for f in parquets/*.parquet; do
    name=$(basename $f .parquet)
    gatac metrics "$f" -g GRCh38.gtf.gz -o "metrics/${name}.csv"
    gatac filter "$f" --metrics "metrics/${name}.csv" \
        --filter "tsse_score > 5" -g hg38 -o "filtered/${name}.parquet"
done

# 3. One tile matrix of every sample, 500,000 most accessible tiles
gatac tile filtered/*.parquet -g hg38 -n 500000 -o tiles.h5ad

# Gene matrices per sample, merged
for f in filtered/*.parquet; do
    gatac gene "$f" -g GRCh38.gtf.gz -o "genes/$(basename $f .parquet).h5ad"
done
gatac combine genes/*.h5ad -o genes.h5ad
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

## Fragment files

Every step that reads fragments streams them from Parquet files to the
GPU. GATAC's Parquet files provide multiple optimizations, such as storing
each chromosome and barcode as an integer id, with the barcode names and
fragment counts in a table inside the same file, saving space and
increasing GPU throughput.

The layout, the encodings and how to read the files outside GATAC are
described in [Parquet fragment format](api/io.md#parquet-fragment-format).
